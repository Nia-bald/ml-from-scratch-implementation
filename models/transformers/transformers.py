
import torch
from torch import nn
from torch.nn import functional as F


class SingularSelfAttentionHead(nn.Module):

    def __init__(self, input_dimension, output_dimension, key_dimension, query_dimension):
        super().__init__()
        self.W_q = nn.Linear(input_dimension, query_dimension )
        self.W_k = nn.Linear(input_dimension, key_dimension )
        self.W_v = nn.Linear(input_dimension, output_dimension)
    
    def forward(self, x_batch):
        Q = self.W_q(x_batch)
        K = self.W_k(x_batch)
        V = self.W_v(x_batch)

        attention_matrix = Q@K.transpose(-2, -1)
        attention_matrix = attention_matrix.masked_fill(torch.tril(torch.ones_like(attention_matrix), diagonal=-1).bool(), float('-inf'))
        return F.softmax(attention_matrix, dim = -2)@V


class LayerNorm(nn.Module):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x_batch):
        x_mean = x_batch.mean(dim=-1)
        x_std = x_batch.std(dim=-1)
        x_normalized = (x_batch - x_mean.unsqueeze(-1))/(x_std.unsqueeze(-1) + self.epsilon)
        return x_normalized


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, number_of_heads, input_dimension, key_dimension, query_dimension):
        super().__init__()
        self.heads = nn.ModuleList([
                        SingularSelfAttentionHead(input_dimension, input_dimension//number_of_heads, key_dimension, query_dimension)
                        for _ in range(number_of_heads)
                    ])
        self.feedforward = nn.Linear(input_dimension, input_dimension)
        self.layer_norm = LayerNorm(epsilon=0.0001)
    
    def forward(self, x_batch):
        combined_self_attention = torch.cat([head(x_batch) for head in self.heads], dim = -1)
        x_batch = x_batch + combined_self_attention
        x_batch = self.layer_norm(x_batch)
        x_batch = x_batch + self.feedforward(x_batch)
        return self.layer_norm(x_batch)



class Transformers(nn.Module):
    def __init__(self,vocab_size, context_length, attention_blocks, heads_per_block, embedding_dimension, key_dimension, query_dimension):
        super().__init__()
        self.context_length = context_length
        self.static_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.position_embedding = nn.Embedding(context_length, embedding_dimension)
        self.multi_head_self_attention_layers = nn.Sequential(*[MultiHeadSelfAttention(heads_per_block, embedding_dimension, key_dimension, query_dimension) for _ in range(attention_blocks)])
        self.linear_vocab_dimension = nn.Linear(embedding_dimension, vocab_size)
    def forward(self, x_batch):
        x_embedding = self.static_embedding(x_batch)
        x_pos = self.position_embedding(torch.arange(self.context_length))
        x_batch = x_embedding + x_pos
        x_batch = self.multi_head_self_attention_layers(x_batch)
        x_batch = self.linear_vocab_dimension(x_batch)
        return x_batch

        
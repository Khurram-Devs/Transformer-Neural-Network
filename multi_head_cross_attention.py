from self_attention import scaled_dot_product
import torch
from torch import nn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        print(f"x.size(): {x.size()}")
        kv = self.kv_layer(x)
        print(f"kv.size(): {kv.size()}")
        q = self.q_layer(y)
        print(f"q.size(): {q.size()}")
        kv = kv.reshape(
            batch_size, max_sequence_length, self.num_heads, 2 * self.head_dim
        )
        q = q.reshape(batch_size, max_sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, max_sequence_length, d_model)
        out = self.linear_layer(values)
        print(f"out after passing through linear layer: {out.size()}")
        return out

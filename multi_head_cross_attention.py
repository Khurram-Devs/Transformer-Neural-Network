import torch
from torch import nn
from global_functions import scaled_dot_product


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, src_seq_len, _ = x.size()
        _, tgt_seq_len, _ = y.size()

        kv = self.kv_layer(x) 
        kv = kv.view(batch_size, src_seq_len, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1) 

        q = self.q_layer(y)
        q = q.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        context, attention = scaled_dot_product(q, k, v, mask=mask)  

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, tgt_seq_len, self.d_model)

        output = self.output_layer(context)

        return output

import torch
from torch import nn, Tensor
from global_functions import scaled_dot_product


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor = None) -> Tensor:
        b, s_len, _ = x.size()
        _, t_len, _ = y.size()

        kv = self.kv_layer(x)
        kv = kv.view(b, s_len, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        q = self.q_layer(y)
        q = q.view(b, t_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        context, _ = scaled_dot_product(q, k, v, mask)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(b, t_len, self.d_model)

        return self.output_layer(context)
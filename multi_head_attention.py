import torch
import torch.nn as nn
from torch import Tensor
from global_functions import scaled_dot_product


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        b, t, _ = x.size()

        qkv = self.qkv_layer(x)
        qkv = qkv.view(b, t, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        context, _ = scaled_dot_product(q, k, v, mask)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(b, t, self.d_model)

        return self.output_layer(context)
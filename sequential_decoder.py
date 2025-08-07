from torch import nn, Tensor
from typing import Optional


class SequentialDecoder(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y

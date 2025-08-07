from torch import nn, Tensor
from typing import Optional


class SequentialEncoder(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: Tensor, self_attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, self_attention_mask)
        return x

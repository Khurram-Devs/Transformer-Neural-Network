from torch import nn, Tensor
from typing import Optional


class SequentialDecoder(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        encoder_output: Tensor,
        decoder_input: Tensor,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = decoder_input
        for layer in self.layers:
            x = layer(encoder_output, x, self_attention_mask, cross_attention_mask)
        return x
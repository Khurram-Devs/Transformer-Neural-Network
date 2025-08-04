from sequential_decoder import SequentialDecoder
from decoder_layer import DecoderLayer
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, mask):
        y = self.layers(x, y, mask)
        return y

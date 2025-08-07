import torch
from torch import nn, Tensor
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from positionwise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, self_attention_mask: Tensor = None) -> Tensor:
        res1 = x
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + res1)

        res2 = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + res2)

        return x
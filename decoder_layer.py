import torch
from torch import nn, Tensor
from typing import Optional
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from multi_head_cross_attention import MultiHeadCrossAttention
from positionwise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float = 0.1
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalization([d_model])
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(
        self,
        encoder_output: Tensor,
        decoder_input: Tensor,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        res1 = decoder_input
        x = self.self_attention(decoder_input, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + res1)

        res2 = x
        x = self.cross_attention(encoder_output, x, mask=cross_attention_mask)
        x = self.dropout2(x)
        x = self.norm2(x + res2)

        res3 = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + res3)

        return x

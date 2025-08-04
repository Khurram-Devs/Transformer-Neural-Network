from layer_normalization import LayerNormalization
from mutlihead_attention import MultiHeadAttention
from multi_head_cross_attention import MultiHeadCrossAttention
from positionwise_feed_forward import PositionwiseFeedForward
import torch
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.droFpout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=decoder_mask)
        print("DROP OUT 1")
        y = self.dropout1(y)
        print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1(y + _y)

        _y = y
        print("CROSS ATTENTION")
        y = self.encoder_decoder_attention(x, y, mask=None)
        print("DROP OUT 2")
        y = self.dropout2(y)
        print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2(y + _y)

        _y = y
        print("FEED FORWARD 1")
        y = self.ffn(y)
        print("DROP OUT 3")
        y = self.dropout3(y)
        print("ADD + LAYER NORMALIZATION 3")
        y = self.norm3(y + _y)
        return y

import torch
from torch import nn
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from positionwise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden_dim=ffn_hidden, dropout=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, self_attention_mask=None):
        residual1 = x
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual1)

        residual2 = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual2)

        return x

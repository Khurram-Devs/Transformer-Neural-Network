from sequential_decoder import SequentialDecoder
from decoder_layer import DecoderLayer
from sentence_embedding import SentenceEmbedding
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        num_heads,
        drop_prob,
        num_layers,
        max_sequence_length,
        language_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length,
            d_model,
            language_to_index,
            START_TOKEN,
            END_TOKEN,
            PADDING_TOKEN,
        )
        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token
    ):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

from encoder_layer import EncoderLayer
from sentence_embedding import SentenceEmbedding
from sequential_encoder import SequentialEncoder
import torch
from torch import nn


class Encoder(nn.Module):
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
        self.layers = SequentialEncoder(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

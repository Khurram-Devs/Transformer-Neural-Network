import torch
from torch import nn
from encoder_layer import EncoderLayer
from sentence_embedding import SentenceEmbedding
from sequential_encoder import SequentialEncoder


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

        self.embedding = SentenceEmbedding(
            max_sequence_length=max_sequence_length,
            d_model=d_model,
            language_to_index=language_to_index,
            START_TOKEN=START_TOKEN,
            END_TOKEN=END_TOKEN,
            PADDING_TOKEN=PADDING_TOKEN
        )

        self.encoder_layers = SequentialEncoder(*[
            EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)
        ])

    def forward(self, input_sentences, self_attention_mask=None, start_token=True, end_token=True):
        embedded = self.embedding(input_sentences, start_token, end_token)
        encoded = self.encoder_layers(embedded, self_attention_mask)
        return encoded

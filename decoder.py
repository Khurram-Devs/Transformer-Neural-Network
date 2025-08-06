import torch
from torch import nn
from sequential_decoder import SequentialDecoder
from decoder_layer import DecoderLayer
from sentence_embedding import SentenceEmbedding


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

        self.embedding = SentenceEmbedding(
            max_sequence_length=max_sequence_length,
            d_model=d_model,
            language_to_index=language_to_index,
            START_TOKEN=START_TOKEN,
            END_TOKEN=END_TOKEN,
            PADDING_TOKEN=PADDING_TOKEN
        )

        self.decoder_layers = SequentialDecoder(*[
            DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        encoder_output,
        decoder_input,
        self_attention_mask=None,
        cross_attention_mask=None,
        start_token=True,
        end_token=True
    ):
        embedded = self.embedding(decoder_input, start_token, end_token)
        decoded = self.decoder_layers(encoder_output, embedded, self_attention_mask, cross_attention_mask)
        return decoded
    
    def get_embedding(self):
        return self.embedding

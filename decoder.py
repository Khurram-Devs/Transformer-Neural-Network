import torch
from torch import nn, Tensor
from typing import Optional
from sequential_decoder import SequentialDecoder
from decoder_layer import DecoderLayer
from sentence_embedding import SentenceEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        language_to_index: dict,
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str,
    ):
        super().__init__()

        self.embedding = SentenceEmbedding(
            max_sequence_length=max_sequence_length,
            d_model=d_model,
            language_to_index=language_to_index,
            START_TOKEN=START_TOKEN,
            END_TOKEN=END_TOKEN,
            PADDING_TOKEN=PADDING_TOKEN,
        )

        self.decoder_layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        encoder_output: Tensor,
        decoder_input: list,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
        start_token: bool = True,
        end_token: bool = True,
    ) -> Tensor:
        embedded = self.embedding(decoder_input, start_token, end_token)
        decoded = self.decoder_layers(
            encoder_output, embedded, self_attention_mask, cross_attention_mask
        )
        return decoded

    def get_embedding(self) -> SentenceEmbedding:
        return self.embedding
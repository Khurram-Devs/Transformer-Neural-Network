import torch
from torch import nn, Tensor
from encoder_layer import EncoderLayer
from sentence_embedding import SentenceEmbedding
from sequential_encoder import SequentialEncoder


class Encoder(nn.Module):
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

        self.encoder_layers = SequentialEncoder(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_sentences: list,
        self_attention_mask: Tensor = None,
        start_token: bool = True,
        end_token: bool = True,
    ) -> Tensor:
        embedded = self.embedding(input_sentences, start_token, end_token)
        encoded = self.encoder_layers(embedded, self_attention_mask)
        return encoded
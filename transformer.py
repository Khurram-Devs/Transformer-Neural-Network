import torch
from torch import nn, Tensor
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        es_vocab_size: int,
        english_to_index: dict,
        spanish_to_index: dict,
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            drop_prob=drop_prob,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            language_to_index=english_to_index,
            START_TOKEN=START_TOKEN,
            END_TOKEN=END_TOKEN,
            PADDING_TOKEN=PADDING_TOKEN,
        )

        self.decoder = Decoder(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            drop_prob=drop_prob,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            language_to_index=spanish_to_index,
            START_TOKEN=START_TOKEN,
            END_TOKEN=END_TOKEN,
            PADDING_TOKEN=PADDING_TOKEN,
        )

        self.output_layer = nn.Linear(d_model, es_vocab_size)

    def forward(
        self,
        x: list,
        y: list,
        encoder_self_attention_mask: Tensor = None,
        decoder_self_attention_mask: Tensor = None,
        decoder_cross_attention_mask: Tensor = None,
        enc_start_token: bool = False,
        enc_end_token: bool = False,
        dec_start_token: bool = False,
        dec_end_token: bool = False,
    ) -> Tensor:
        enc_out = self.encoder(
            x,
            self_attention_mask=encoder_self_attention_mask,
            start_token=enc_start_token,
            end_token=enc_end_token,
        )

        dec_out = self.decoder(
            enc_out,
            y,
            self_attention_mask=decoder_self_attention_mask,
            cross_attention_mask=decoder_cross_attention_mask,
            start_token=dec_start_token,
            end_token=dec_end_token,
        )

        return self.output_layer(dec_out)

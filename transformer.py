import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        num_heads,
        drop_prob,
        num_layers,
        max_sequence_length,
        es_vocab_size,
        english_to_index,
        spanish_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
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
        x,
        y,
        encoder_self_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
        enc_start_token=False,
        enc_end_token=False,
        dec_start_token=False,
        dec_end_token=False,
    ):
        encoder_output = self.encoder(
            x,
            self_attention_mask=encoder_self_attention_mask,
            start_token=enc_start_token,
            end_token=enc_end_token,
        )

        decoder_output = self.decoder(
            encoder_output,
            y,
            self_attention_mask=decoder_self_attention_mask,
            cross_attention_mask=decoder_cross_attention_mask,
            start_token=dec_start_token,
            end_token=dec_end_token,
        )

        logits = self.output_layer(decoder_output)
        return logits

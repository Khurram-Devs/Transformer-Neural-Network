from encoder import Encoder
from decoder import Decoder
import torch
from torch import nn


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
            d_model,
            ffn_hidden,
            num_heads,
            drop_prob,
            num_layers,
            max_sequence_length,
            english_to_index,
            START_TOKEN,
            END_TOKEN,
            PADDING_TOKEN,
        )
        self.decoder = Decoder(
            d_model,
            ffn_hidden,
            num_heads,
            drop_prob,
            num_layers,
            max_sequence_length,
            spanish_to_index,
            START_TOKEN,
            END_TOKEN,
            PADDING_TOKEN,
        )
        self.linear = nn.Linear(d_model, es_vocab_size)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

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
        x = self.encoder(
            x,
            encoder_self_attention_mask,
            start_token=enc_start_token,
            end_token=enc_end_token,
        )
        out = self.decoder(
            x,
            y,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
            start_token=dec_start_token,
            end_token=dec_end_token,
        )
        out = self.linear(out)
        return out

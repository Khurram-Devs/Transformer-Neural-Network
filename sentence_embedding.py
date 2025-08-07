import torch
from torch import nn, Tensor
from positional_encoding import PositionalEncoding
from global_functions import get_device


class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        d_model: int,
        language_to_index: dict,
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.language_to_index = language_to_index

        self.START_IDX = language_to_index.get(START_TOKEN, 0)
        self.END_IDX = language_to_index.get(END_TOKEN, 0)
        self.PAD_IDX = language_to_index.get(PADDING_TOKEN, 0)

        self.vocab_size = len(language_to_index)
        self.embedding = nn.Embedding(
            self.vocab_size, d_model, padding_idx=self.PAD_IDX
        )
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)

    def batch_tokenize(
        self, batch, start_token: bool = True, end_token: bool = True
    ) -> Tensor:
        device = next(self.parameters()).device
        batch_size = len(batch)
        token_ids = torch.full(
            (batch_size, self.max_sequence_length),
            fill_value=self.PAD_IDX,
            dtype=torch.long,
            device=device,
        )

        for i, sentence in enumerate(batch):
            if isinstance(sentence, str):
                tokens = list(sentence)
            else:
                tokens = sentence
                
            indices = [self.language_to_index.get(token, self.PAD_IDX) for token in tokens]

            if start_token:
                indices = [self.START_IDX] + indices
            if end_token:
                indices = indices + [self.END_IDX]

            indices = indices[: self.max_sequence_length]
            token_ids[i, : len(indices)] = torch.tensor(indices, dtype=torch.long, device=device)

        return token_ids

    def forward(
        self, batch, start_token: bool = True, end_token: bool = True
    ) -> Tensor:
        x = self.batch_tokenize(batch, start_token, end_token)
        x = self.embedding(x)
        x = self.position_encoder(x)
        return self.dropout(x)
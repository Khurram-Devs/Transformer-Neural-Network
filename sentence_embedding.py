import torch
from torch import nn
from positional_encoding import PositionalEncoding
from global_functions import get_device


class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        max_sequence_length,
        d_model,
        language_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.device = get_device()
        self.max_sequence_length = max_sequence_length
        self.language_to_index = language_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

        self.pad_idx = self.language_to_index.get(PADDING_TOKEN, 0)
        self.start_idx = self.language_to_index.get(START_TOKEN, 0)
        self.end_idx = self.language_to_index.get(END_TOKEN, 0)

        self.vocab_size = len(language_to_index)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)

    def batch_tokenize(self, batch, start_token=True, end_token=True):
        def tokenize(sentence):
            indices = [self.language_to_index.get(token, self.pad_idx) for token in list(sentence)]
            if start_token:
                indices.insert(0, self.start_idx)
            if end_token:
                indices.append(self.end_idx)
            indices = indices[:self.max_sequence_length]
            padding_needed = self.max_sequence_length - len(indices)
            indices += [self.pad_idx] * padding_needed
            return torch.tensor(indices, device=self.device)

        tokenized_batch = [tokenize(sentence) for sentence in batch]
        return torch.stack(tokenized_batch)

    def forward(self, batch, start_token=True, end_token=True):
        x = self.batch_tokenize(batch, start_token, end_token)
        x = self.embedding(x)
        x = self.position_encoder(x) 
        return self.dropout(x)

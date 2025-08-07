import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
from typing import Optional, Tuple


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scaled_dot_product(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention = F.softmax(scores, dim=-1)
    output = torch.matmul(attention, v)
    return output, attention


def create_masks(eng_batch, es_batch, max_len, pad_idx, neg_inf=-1e9):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()

    enc_pad_mask = torch.full((num_sentences, max_len, max_len), False)
    dec_self_mask = torch.full((num_sentences, max_len, max_len), False)
    dec_cross_mask = torch.full((num_sentences, max_len, max_len), False)

    for i in range(num_sentences):
        eng_len = len(eng_batch[i])
        es_len = len(es_batch[i])
        eng_pad = np.arange(eng_len + 1, max_len)
        es_pad = np.arange(es_len + 1, max_len)

        enc_pad_mask[i, :, eng_pad] = True
        enc_pad_mask[i, eng_pad, :] = True
        dec_self_mask[i, :, es_pad] = True
        dec_self_mask[i, es_pad, :] = True
        dec_cross_mask[i, :, eng_pad] = True
        dec_cross_mask[i, es_pad, :] = True

    return (
        torch.where(enc_pad_mask, neg_inf, 0),
        torch.where(look_ahead_mask + dec_self_mask, neg_inf, 0),
        torch.where(dec_cross_mask, neg_inf, 0),
    )


def is_valid_tokens(sentence: list, vocab: set) -> bool:
    return all(token in vocab for token in sentence)


def is_valid_length(sentence: list, max_sequence_length: int) -> bool:
    return len(sentence) < (max_sequence_length - 2)

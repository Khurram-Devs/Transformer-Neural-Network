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
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        scores = scores.masked_fill(mask == 0, -1e9)

    attention = F.softmax(scores, dim=-1)
    attention = torch.nan_to_num(attention, nan=0.0)
    output = torch.matmul(attention, v)
    return output, attention


def create_masks(eng_batch, es_batch, max_len, pad_idx):
    """Fixed mask creation with proper tensor handling"""
    device = get_device()
    num_sentences = len(eng_batch)
    
    look_ahead_mask = torch.triu(torch.ones(max_len, max_len, device=device), diagonal=1).bool()

    enc_pad_mask = torch.zeros((num_sentences, max_len, max_len), dtype=torch.bool, device=device)
    dec_self_mask = torch.zeros((num_sentences, max_len, max_len), dtype=torch.bool, device=device)
    dec_cross_mask = torch.zeros((num_sentences, max_len, max_len), dtype=torch.bool, device=device)

    for i in range(num_sentences):
        eng_tokens = list(eng_batch[i]) if isinstance(eng_batch[i], str) else eng_batch[i]
        es_tokens = list(es_batch[i]) if isinstance(es_batch[i], str) else es_batch[i]
        
        eng_len = len(eng_tokens)
        es_len = len(es_tokens)
        
        if eng_len < max_len:
            enc_pad_mask[i, :, eng_len:] = True
            enc_pad_mask[i, eng_len:, :] = True
            
        if es_len < max_len:
            dec_self_mask[i, :, es_len:] = True
            dec_self_mask[i, es_len:, :] = True
            dec_cross_mask[i, :, eng_len:] = True
            dec_cross_mask[i, es_len:, :] = True

    dec_self_mask = dec_self_mask | look_ahead_mask.unsqueeze(0)

    return enc_pad_mask, dec_self_mask, dec_cross_mask


def is_valid_tokens(sentence: list, vocab: set) -> bool:
    """Fixed to handle both string and list inputs"""
    if isinstance(sentence, str):
        tokens = list(sentence)
    else:
        tokens = sentence
    return all(token in vocab for token in tokens)


def is_valid_length(sentence, max_sequence_length: int) -> bool:
    """Fixed to handle both string and list inputs"""
    if isinstance(sentence, str):
        length = len(sentence)
    else:
        length = len(sentence)
    return length < (max_sequence_length - 2)
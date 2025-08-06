import torch
import math
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention = F.softmax(scores, dim=-1)
    output = torch.matmul(attention, v)
    return output, attention



def is_valid_tokens(sentence, vocab):
    return all(token in vocab for token in sentence)


def is_valid_length(sentence, max_sequence_length):
    return len(sentence) < (max_sequence_length - 2)

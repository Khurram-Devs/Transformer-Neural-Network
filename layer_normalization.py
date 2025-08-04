import torch
from torch import nn


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean \n ({mean.size()}): \n {mean}")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation \n ({std.size()}): \n {std}")
        y = (inputs - mean) / std
        print(f"y \n ({y.size()}): \n {y}")
        out = self.gamma * y + self.beta
        print(f"out \n ({out.size()}): \n {out}")
        return out


# Input

batch_size = 3
sentence_length = 5
embeddin_dim = 8
inputs = torch.randn(sentence_length, batch_size, embeddin_dim)

print(f"input \n ({inputs.size()}): \n {inputs}")

layer_norm = LayerNormalization(inputs.size()[-2:])
out = layer_norm.forward(inputs)

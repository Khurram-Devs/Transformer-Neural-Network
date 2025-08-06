import torch
from torch import nn


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = tuple(-i for i in range(1, len(self.gamma.shape) + 1))
        mean = inputs.mean(dim=dims, keepdim=True)
        variance = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta

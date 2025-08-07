import torch
from torch import nn, Tensor


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        if isinstance(parameters_shape, int):
            parameters_shape = (parameters_shape,)

        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
        self.normalized_dims = tuple(-i for i in range(1, len(parameters_shape) + 1))

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=self.normalized_dims, keepdim=True)
        variance = ((inputs - mean) ** 2).mean(dim=self.normalized_dims, keepdim=True)
        norm = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * norm + self.beta
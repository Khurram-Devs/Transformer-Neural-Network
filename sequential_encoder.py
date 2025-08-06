from torch import nn
class SequentialEncoder(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, self_attention_mask):
        for layer in self.layers:
            x = layer(x, self_attention_mask)
        return x

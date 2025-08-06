from torch import nn
class SequentialDecoder(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y, self_attention_mask=None, cross_attention_mask=None):
        for layer in self.layers:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y

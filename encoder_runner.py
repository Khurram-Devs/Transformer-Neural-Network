from encoder import Encoder
import torch

d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)

x = torch.randn((batch_size, max_sequence_length, d_model))
out = encoder(x)
print(out)

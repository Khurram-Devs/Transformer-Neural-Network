from decoder import Decoder
import torch

d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

x = torch.randn((batch_size, max_sequence_length, d_model))
y = torch.randn((batch_size, max_sequence_length, d_model))
mask = torch.full([max_sequence_length, max_sequence_length], float("-inf"))
mask = torch.triu(mask, diagonal=1)
decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = decoder(x, y, mask)
print(out)

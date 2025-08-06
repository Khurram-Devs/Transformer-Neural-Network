import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from transformer import Transformer
from global_functions import is_valid_length, is_valid_tokens

english_file = "./english.txt"
spanish_file = "./spanish.txt"
TOTAL_SENTENCES = 10000
max_sequence_length = 128
batch_size = 64
d_model = 256
ffn_hidden = 1024
num_heads = 4
drop_prob = 0.1
num_layers = 4
num_epochs = 20
learning_rate = 5e-4 
NEG_INFTY = -1e9

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

spanish_vocabulary = [
    START_TOKEN, " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-",
    ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", "<", "=", ">", "?",
    "@", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "á", "é", "í", "ó", "ú", "ñ", "ü", "¡", "¿", "{", "|", "}", "~", PADDING_TOKEN, END_TOKEN,
]
english_vocabulary = [
    START_TOKEN, " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-",
    ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", "<", "=", ">", "?",
    "@", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "{", "|", "}", "~", PADDING_TOKEN, END_TOKEN,
]

index_to_spanish = {k: v for k, v in enumerate(spanish_vocabulary)}
spanish_to_index = {v: k for k, v in enumerate(spanish_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

with open(english_file, "r") as file:
    english_sentences = file.readlines()
with open(spanish_file, "r") as file:
    spanish_sentences = file.readlines()

english_sentences = [s.strip().lower() for s in english_sentences[:TOTAL_SENTENCES]]
spanish_sentences = [s.strip().lower() for s in spanish_sentences[:TOTAL_SENTENCES]]

valid_indices = [
    i for i in range(len(english_sentences))
    if is_valid_length(english_sentences[i], max_sequence_length)
    and is_valid_length(spanish_sentences[i], max_sequence_length)
    and is_valid_tokens(spanish_sentences[i], spanish_vocabulary)
]

english_sentences = [english_sentences[i] for i in valid_indices]
spanish_sentences = [spanish_sentences[i] for i in valid_indices]

class TextDataset(Dataset):
    def __init__(self, english_sentences, spanish_sentences):
        self.english_sentences = english_sentences
        self.spanish_sentences = spanish_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.spanish_sentences[idx]

dataset = TextDataset(english_sentences, spanish_sentences)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = Transformer(
    d_model,
    ffn_hidden,
    num_heads,
    drop_prob,
    num_layers,
    max_sequence_length,
    len(spanish_vocabulary),
    english_to_index,
    spanish_to_index,
    START_TOKEN,
    END_TOKEN,
    PADDING_TOKEN,
).to(device)

for param in transformer.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=spanish_to_index[PADDING_TOKEN], reduction="none")

def create_masks(eng_batch, es_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.triu(torch.ones(max_sequence_length, max_sequence_length), diagonal=1).bool()

    encoder_padding_mask = torch.full((num_sentences, max_sequence_length, max_sequence_length), False)
    decoder_padding_mask_self = torch.full((num_sentences, max_sequence_length, max_sequence_length), False)
    decoder_padding_mask_cross = torch.full((num_sentences, max_sequence_length, max_sequence_length), False)

    for i in range(num_sentences):
        eng_len = len(eng_batch[i])
        es_len = len(es_batch[i])
        eng_pad = np.arange(eng_len + 1, max_sequence_length)
        es_pad = np.arange(es_len + 1, max_sequence_length)

        encoder_padding_mask[i, :, eng_pad] = True
        encoder_padding_mask[i, eng_pad, :] = True
        decoder_padding_mask_self[i, :, es_pad] = True
        decoder_padding_mask_self[i, es_pad, :] = True
        decoder_padding_mask_cross[i, :, eng_pad] = True
        decoder_padding_mask_cross[i, es_pad, :] = True

    return (
        torch.where(encoder_padding_mask, NEG_INFTY, 0),
        torch.where(look_ahead_mask + decoder_padding_mask_self, NEG_INFTY, 0),
        torch.where(decoder_padding_mask_cross, NEG_INFTY, 0),
    )

transformer.train()
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    for batch_num, (eng_batch, es_batch) in enumerate(train_loader):
        transformer.train()

        (
            enc_mask,
            dec_self_mask,
            dec_cross_mask,
        ) = create_masks(eng_batch, es_batch)

        optimizer.zero_grad()
        outputs = transformer(
            eng_batch,
            es_batch,
            enc_mask.to(device),
            dec_self_mask.to(device),
            dec_cross_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=True,
        )

        labels = transformer.decoder.get_embedding().batch_tokenize(
            es_batch, start_token=False, end_token=True
        )

        loss = criterion(
            outputs.view(-1, len(spanish_vocabulary)),
            labels.view(-1),
        )

        valid = labels.view(-1) != spanish_to_index[PADDING_TOKEN]
        loss = loss.sum() / valid.sum()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_num % 100 == 0 or batch_num == 0:
            print(f"Batch {batch_num} | Loss: {loss.item():.4f}")
            print(f"English: {eng_batch[0]}")
            print(f"Target:  {es_batch[0]}")
            prediction = torch.argmax(outputs[0], axis=1)
            predicted_sentence = ""
            for idx in prediction:
                word = index_to_spanish[idx.item()]
                if word == END_TOKEN:
                    break
                predicted_sentence += word
            print(f"Predicted: {predicted_sentence}")
            print("-" * 40)

transformer.eval()

def translate(eng_sentence):
    eng_sentence = (eng_sentence.lower(),)
    es_sentence = ("",)
    for _ in range(max_sequence_length):
        (
            enc_mask,
            dec_self_mask,
            dec_cross_mask,
        ) = create_masks(eng_sentence, es_sentence)
        with torch.no_grad():
            predictions = transformer(
                eng_sentence,
                es_sentence,
                enc_mask.to(device),
                dec_self_mask.to(device),
                dec_cross_mask.to(device),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=False,
            )
        next_token_logits = predictions[0][len(es_sentence[0])]
        next_token_index = torch.argmax(next_token_logits).item()
        next_token = index_to_spanish[next_token_index]
        es_sentence = (es_sentence[0] + next_token,)
        if next_token == END_TOKEN:
            break
    return es_sentence[0]

test_input = "should we go to the college?"
result = translate(test_input)
print(f"\nTranslation:\nInput: {test_input}\nOutput: {result}")

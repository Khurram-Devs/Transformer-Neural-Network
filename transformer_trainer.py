from transformer import Transformer
from global_functions import is_valid_length
from global_functions import is_valid_tokens
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

english_file = "./english.txt"
spanish_file = "./spanish.txt"

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

spanish_vocabulary = [
    START_TOKEN,
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "á",
    "é",
    "í",
    "ó",
    "ú",
    "ñ",
    "ü",
    "¡",
    "¿",
    "{",
    "|",
    "}",
    "~",
    PADDING_TOKEN,
    END_TOKEN,
]
english_vocabulary = [
    START_TOKEN,
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
    PADDING_TOKEN,
    END_TOKEN,
]

index_to_spanish = {k: v for k, v in enumerate(spanish_vocabulary)}
spanish_to_index = {v: k for k, v in enumerate(spanish_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {k: v for k, v in enumerate(english_vocabulary)}

with open(english_file, "r") as file:
    english_sentences = file.readlines()
with open(spanish_file, "r") as file:
    spanish_sentences = file.readlines()

TOTAL_SENTENCES = 2000
english_sentences = english_sentences[:TOTAL_SENTENCES]
spanish_sentences = spanish_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip("\n").lower() for sentence in english_sentences]
spanish_sentences = [sentence.rstrip("\n") for sentence in spanish_sentences]

max_sequence_length = 200

valid_sentence_indices = []
for index in range(len(spanish_sentences)):
    spanish_sentences, english_sentences = (
        spanish_sentences[index],
        english_sentences[index],
    )
    if (
        is_valid_length(spanish_sentences, max_sequence_length)
        and is_valid_length(english_sentences, max_sequence_length)
        and is_valid_tokens(spanish_sentences, spanish_vocabulary)
    ):
        valid_sentence_indices.append(index)

spanish_sentences = [spanish_sentences[i] for i in valid_sentence_indices]
english_sentences = [english_sentences[i] for i in valid_sentence_indices]

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
es_vocab_size = len(spanish_vocabulary)

transformer = Transformer(
    d_model,
    ffn_hidden,
    num_heads,
    drop_prob,
    num_layers,
    max_sequence_length,
    es_vocab_size,
    english_to_index,
    spanish_to_index,
    START_TOKEN,
    END_TOKEN,
    PADDING_TOKEN,
)


class TextDataset(Dataset):
    def __init__(self, enlish_sentences, spanish_sentences):
        self.english_sentences = english_sentences
        self.spanish_sentences = spanish_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.spanish_sentences[idx]


dataset = TextDataset(english_sentences, spanish_sentences)

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    if batch_num > 3:
        break

criterian = nn.CrossEntropLoss(
    ignore_index=spanish_to_index[PADDING_TOKEN], reduction="none"
)

for params in transformer.parameters():
    if params.dims() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NEG_INFTY = -1e9


def create_masks(eng_batch, es_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False
    )
    decoder_padding_mask_self_attention = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False
    )
    decoder_padding_mask_cross_attention = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False
    )

    for idx in range(num_sentences):
        eng_sentence_length, es_sentence_length = len(eng_batch[idx]), len(
            es_batch[idx]
        )
        eng_chars_to_padding_mask = np.arange(
            eng_sentence_length + 1, max_sequence_length
        )
        es_chars_to_padding_mask = np.arange(
            es_sentence_length + 1, max_sequence_length
        )
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, es_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, es_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, es_chars_to_padding_mask, :] = True
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(
        look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0
    )
    decoder_cross_attention_mask = torch.where(
        decoder_padding_mask_cross_attention, NEG_INFTY, 0
    )
    return (
        encoder_self_attention_mask,
        decoder_self_attention_mask,
        decoder_cross_attention_mask,
    )


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, es_batch = batch
        (
            encoder_self_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        ) = create_masks(eng_batch, es_batch)
        optim.zero_grad()
        es_predictions = transformer(
            eng_batch,
            es_batch,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=True,
        )
        labels = transformer.decoder.sentence_embedding.batch_tokenize(
            es_batch, start_token=False, end_token=True
        )
        loss = criterian(
            es_predictions.view(-1, es_vocab_size).to(device),
            labels.view(-1).to(device),
        ).to(device)
        valid_indicies = torch.where(
            labels.view(-1) == spanish_to_index[PADDING_TOKEN], False, True
        )
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Spanish Translation: {es_batch[0]}")
            es_sentence_predicted = torch.argmax(es_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in es_sentence_predicted:
                if idx == spanish_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_spanish[idx.item()]
            print(f"Spanish Prediction: {predicted_sentence}")

            transformer.eval()
            es_sentence = ("",)
            eng_sentence = ("should we go to the college?",)
            for word_counter in range(max_sequence_length):
                (
                    encoder_self_attention_mask,
                    decoder_self_attention_mask,
                    decoder_cross_attention_mask,
                ) = create_masks(eng_sentence, es_sentence)
                predictions = transformer(
                    eng_sentence,
                    es_sentence,
                    encoder_self_attention_mask.to(device),
                    decoder_self_attention_mask.to(device),
                    decoder_cross_attention_mask.to(device),
                    enc_start_token=False,
                    enc_end_token=False,
                    dec_start_token=True,
                    dec_end_token=False,
                )
                next_token_prob_distribution = predictions[0][word_counter]
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_spanish[next_token_index]
                es_sentence = (es_sentence[0] + next_token,)
                if next_token == END_TOKEN:
                    break
            print(
                f"Evaluation translation (should we go to the college?) : {es_sentence}"
            )
            print("-----------------------------")

transformer.eval()


def translate(eng_sentence):
    eng_sentence = (eng_sentence,)
    es_sentence = ("",)
    for word_counter in range(max_sequence_length):
        (
            encoder_self_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        ) = create_masks(eng_sentence, es_sentence)
        predictions = transformer(
            eng_sentence,
            es_sentence,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=False,
        )
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_spanish[next_token_index]
        es_sentence = (es_sentence[0] + next_token,)
        if next_token == END_TOKEN:
            break
        return es_sentence[0]

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from transformer import Transformer
from global_functions import create_masks, is_valid_length, is_valid_tokens, get_device

english_file = "./english.txt"
spanish_file = "./spanish.txt"
TOTAL_SENTENCES = 1000
max_sequence_length = 128
batch_size = 64
d_model = 256
ffn_hidden = 1024
num_heads = 4
drop_prob = 0.1
num_layers = 4
num_epochs = 10
learning_rate = 5e-4
model_path = "transformer_es.pt"
device = get_device()

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

index_to_spanish = {i: ch for i, ch in enumerate(spanish_vocabulary)}
spanish_to_index = {ch: i for i, ch in enumerate(spanish_vocabulary)}
index_to_english = {i: ch for i, ch in enumerate(english_vocabulary)}
english_to_index = {ch: i for i, ch in enumerate(english_vocabulary)}

with open(english_file, "r", encoding="utf-8") as file:
    english_sentences = [
        line.strip().lower() for line in file.readlines()[:TOTAL_SENTENCES]
    ]

with open(spanish_file, "r", encoding="utf-8") as file:
    spanish_sentences = [
        line.strip().lower() for line in file.readlines()[:TOTAL_SENTENCES]
    ]

valid_indices = [
    i
    for i in range(len(english_sentences))
    if is_valid_length(english_sentences[i], max_sequence_length)
    and is_valid_length(spanish_sentences[i], max_sequence_length)
    and is_valid_tokens(spanish_sentences[i], spanish_vocabulary)
]

english_sentences = [english_sentences[i] for i in valid_indices]
spanish_sentences = [spanish_sentences[i] for i in valid_indices]


class TextDataset(Dataset):
    def __init__(self, en, es):
        self.en = en
        self.es = es

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        return self.en[idx], self.es[idx]


train_loader = DataLoader(
    TextDataset(english_sentences, spanish_sentences),
    batch_size=batch_size,
    shuffle=True,
)

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

if os.path.exists(model_path):
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.eval()
    print("✅ Model loaded.")
else:
    print("🚀 Training model from scratch...")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(
        ignore_index=spanish_to_index[PADDING_TOKEN], reduction="none"
    )

    transformer.train()
    for epoch in range(num_epochs):
        print(f"\n🌍 Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for batch_num, (en, es) in enumerate(train_loader):
            enc_mask, dec_self_mask, dec_cross_mask = create_masks(
                en, es, max_sequence_length, spanish_to_index[PADDING_TOKEN]
            )
            optimizer.zero_grad()
            outputs = transformer(
                en,
                es,
                enc_mask.to(device),
                dec_self_mask.to(device),
                dec_cross_mask.to(device),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True,
            )
            labels = transformer.decoder.get_embedding().batch_tokenize(
                es, start_token=False, end_token=True
            )
            loss = criterion(outputs.view(-1, len(spanish_vocabulary)), labels.view(-1))
            valid = labels.view(-1) != spanish_to_index[PADDING_TOKEN]
            loss = loss.sum() / valid.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_num % 10 == 0:
                print(f"Batch {batch_num} | Loss: {loss.item():.4f}")
        print(f"✅ Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader):.4f}")

    torch.save(transformer.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")


def translate(eng_sentence: str) -> str:
    transformer.eval()
    eng_sentence = (eng_sentence.lower(),)
    generated = ["<START>"]

    for _ in range(max_sequence_length):
        es_string = "".join([tok for tok in generated if tok != START_TOKEN])
        enc_mask, dec_self_mask, dec_cross_mask = create_masks(
            eng_sentence,
            (es_string,),
            max_sequence_length,
            spanish_to_index[PADDING_TOKEN],
        )

        with torch.no_grad():
            logits = transformer(
                eng_sentence,
                (es_string,),
                enc_mask.to(device),
                dec_self_mask.to(device),
                dec_cross_mask.to(device),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=False,
            )

        next_token_id = torch.argmax(logits[0][len(es_string)]).item()
        next_token = index_to_spanish[next_token_id]
        generated.append(next_token)

        if next_token == END_TOKEN:
            break

    result = "".join([tok for tok in generated if tok not in (START_TOKEN, END_TOKEN)])
    print("Generated Tokens:", generated)
    return result

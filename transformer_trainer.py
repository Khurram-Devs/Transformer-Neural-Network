import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from transformer import Transformer
from global_functions import create_masks, is_valid_length, is_valid_tokens, get_device

english_file = "./english.txt"
spanish_file = "./spanish.txt"
TOTAL_SENTENCES = 100
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

try:
    with open(english_file, "r", encoding="utf-8") as file:
        english_sentences = [
            line.strip().lower() for line in file.readlines()[:TOTAL_SENTENCES]
        ]

    with open(spanish_file, "r", encoding="utf-8") as file:
        spanish_sentences = [
            line.strip().lower() for line in file.readlines()[:TOTAL_SENTENCES]
        ]
except FileNotFoundError as e:
    print(f"Error: {e}")
    english_sentences = ["hello world", "i am fine", "how are you"] * 34
    spanish_sentences = ["hola mundo", "estoy bien", "como estas"] * 34

valid_indices = []
for i in range(min(len(english_sentences), len(spanish_sentences))):
    eng_sentence = english_sentences[i]
    spa_sentence = spanish_sentences[i]
    
    if (is_valid_length(eng_sentence, max_sequence_length) and 
        is_valid_length(spa_sentence, max_sequence_length) and
        is_valid_tokens(spa_sentence, set(spanish_vocabulary)) and
        is_valid_tokens(eng_sentence, set(english_vocabulary))):
        valid_indices.append(i)

english_sentences = [english_sentences[i] for i in valid_indices]
spanish_sentences = [spanish_sentences[i] for i in valid_indices]

print(f"Valid sentences: {len(english_sentences)}")


class TextDataset(Dataset):
    def __init__(self, en, es):
        self.en = en
        self.es = es

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        return self.en[idx], self.es[idx]


if len(english_sentences) < batch_size:
    batch_size = min(len(english_sentences), 4)
    print(f"Adjusted batch size to: {batch_size}")

train_loader = DataLoader(
    TextDataset(english_sentences, spanish_sentences),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
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
    try:
        transformer.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        transformer.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("🚀 Training model from scratch...")
        os.remove(model_path)
        transformer = None

if transformer is None or not os.path.exists(model_path):
    if transformer is None:
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
    
    print("🚀 Training model from scratch...")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(
        ignore_index=spanish_to_index[PADDING_TOKEN], 
        reduction="mean",
        label_smoothing=0.1
    )

    transformer.train()
    for epoch in range(num_epochs):
        print(f"\n🌍 Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        num_batches = 0
        
        for batch_num, (en, es) in enumerate(train_loader):
            try:
                enc_mask, dec_self_mask, dec_cross_mask = create_masks(
                    en, es, max_sequence_length, spanish_to_index[PADDING_TOKEN]
                )
                
                enc_mask = enc_mask.to(device)
                dec_self_mask = dec_self_mask.to(device)
                dec_cross_mask = dec_cross_mask.to(device)
                
                optimizer.zero_grad()
                
                outputs = transformer(
                    en,
                    es,
                    encoder_self_attention_mask=enc_mask,
                    decoder_self_attention_mask=dec_self_mask,
                    decoder_cross_attention_mask=dec_cross_mask,
                    enc_start_token=False,
                    enc_end_token=False,
                    dec_start_token=True,
                    dec_end_token=False,
                )
                
                target_tokens = []
                for sentence in es:
                    tokens = list(sentence) + ["<END>"]
                    target_tokens.append(tokens)
                
                labels = transformer.decoder.get_embedding().batch_tokenize(
                    ["".join(tokens) for tokens in target_tokens], 
                    start_token=False, 
                    end_token=False
                )
                
                seq_len = min(outputs.size(1), labels.size(1))
                outputs = outputs[:, :seq_len, :]
                labels = labels[:, :seq_len]
                
                outputs_flat = outputs.contiguous().view(-1, len(spanish_vocabulary))
                labels_flat = labels.contiguous().view(-1)
                
                mask = (labels_flat != spanish_to_index[PADDING_TOKEN])
                if mask.sum() == 0:
                    continue
                    
                valid_outputs = outputs_flat[mask]
                valid_labels = labels_flat[mask]
                
                loss = criterion(valid_outputs, valid_labels)
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in batch {batch_num}")
                    continue
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                
                if batch_num % 10 == 0:
                    print(f"Batch {batch_num} | Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"✅ Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"❌ Epoch {epoch + 1} | No valid batches processed")

    try:
        torch.save(transformer.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def translate(eng_sentence: str, temperature: float = 0.8, max_length: int = 50) -> str:
    """Fixed translation function with proper autoregressive generation"""
    if not eng_sentence.strip():
        return ""
        
    transformer.eval()
    device = next(transformer.parameters()).device
    
    eng_sentence = eng_sentence.lower().strip()
    
    generated_ids = [spanish_to_index["<START>"]]
    
    with torch.no_grad():
        source_tokens = transformer.encoder.embedding.batch_tokenize(
            [eng_sentence], start_token=False, end_token=False
        )
        
        batch_size, seq_len = source_tokens.shape
        enc_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            pad_positions = (source_tokens[i] == english_to_index[PADDING_TOKEN]).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                first_pad = pad_positions[0].item()
                enc_mask[i, :, first_pad:] = True
                enc_mask[i, first_pad:, :] = True
        
        embedded_source = transformer.encoder.embedding.forward(
            [eng_sentence], start_token=False, end_token=False
        )
        encoder_output = transformer.encoder.encoder_layers(embedded_source, enc_mask)
        
        for step in range(max_length):
            current_seq = torch.tensor([generated_ids], dtype=torch.long, device=device)
            target_len = len(generated_ids)
            
            dec_self_mask = torch.triu(torch.ones(target_len, target_len, device=device), diagonal=1).bool()
            dec_self_mask = dec_self_mask.unsqueeze(0)
            
            source_len = embedded_source.shape[1]
            dec_cross_mask = torch.zeros(1, target_len, source_len, dtype=torch.bool, device=device)
            
            pad_positions = (source_tokens[0] == english_to_index[PADDING_TOKEN]).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                first_pad = pad_positions[0].item()
                dec_cross_mask[0, :, first_pad:] = True
            
            current_embedded = transformer.decoder.embedding.embedding(current_seq)
            current_embedded = transformer.decoder.embedding.position_encoder(current_embedded)
            current_embedded = transformer.decoder.embedding.dropout(current_embedded)
            
            decoder_output = current_embedded
            for layer in transformer.decoder.decoder_layers.layers:
                decoder_output = layer(
                    encoder_output, decoder_output, dec_self_mask, dec_cross_mask
                )
            
            logits = transformer.output_layer(decoder_output[0, -1, :])
            
            if temperature > 0:
                logits = logits / temperature
                top_k = 10
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
                next_token_id = top_k_indices[next_token_idx].item()
            else:
                next_token_id = torch.argmax(logits).item()
            
            next_token = index_to_spanish.get(next_token_id, "<END>")
            
            if next_token == "<END>" or next_token_id == spanish_to_index.get("<END>", -1):
                break
            elif next_token == "<PADDING>" or next_token_id == spanish_to_index.get("<PADDING>", -1):
                break
            
            generated_ids.append(next_token_id)
            
            if len(generated_ids) > max_length + 1:
                break
    
    result_tokens = []
    for token_id in generated_ids[1:]:
        token = index_to_spanish.get(token_id, "")
        if token not in ["<START>", "<END>", "<PADDING>"]:
            result_tokens.append(token)
    
    result = "".join(result_tokens).strip()
    return result if result else "translation_failed"
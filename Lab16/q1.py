## Implement sequence to sequence learning using LSTM and GRU. You can use the the English to Hindi & Hindi to English
## corpus dataset -

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# ----------------------------------------------------------------------
# 0. SETTINGS (change these to experiment)
# ----------------------------------------------------------------------
CSV_PATH = "Hindi_English_Truncated_Corpus.csv"
NUM_SAMPLES = 20000     # how many sentence pairs to use (more = better, slower)
MAX_LEN = 15            # drop sentences longer than this many words
MAX_VOCAB = 8000        # keep only the most common words
EMBED_DIM = 128         # size of word vectors
HIDDEN_DIM = 256        # size of LSTM/GRU memory
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Special tokens
PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"

np.random.seed(42)
torch.manual_seed(42)

# Use the GPU if you have one, otherwise the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# LOADING THE DATA
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["english_sentence", "hindi_sentence"])
    df = df.sample(n=min(NUM_SAMPLES * 2, len(df)), random_state=42)
    english_raw = df["english_sentence"].astype(str).tolist()
    hindi_raw = df["hindi_sentence"].astype(str).tolist()
    print("Loaded dataset:", len(english_raw), "sentence pairs")
else:
    print(f"'{CSV_PATH}' not found")
    pairs = [
        ("hello", "नमस्ते"), ("thank you", "धन्यवाद"),
        ("how are you", "आप कैसे हैं"), ("i am fine", "मैं ठीक हूँ"),
        ("what is your name", "आपका नाम क्या है"),
        ("my name is raj", "मेरा नाम राज है"),
        ("i love india", "मैं भारत से प्यार करता हूँ"),
        ("good morning", "सुप्रभात"), ("good night", "शुभ रात्रि"),
        ("where are you going", "आप कहाँ जा रहे हैं"),
        ("i am going home", "मैं घर जा रहा हूँ"),
        ("he is a good boy", "वह एक अच्छा लड़का है"),
        ("she is a good girl", "वह एक अच्छी लड़की है"),
        ("i like water", "मुझे पानी पसंद है"),
        ("please help me", "कृपया मेरी मदद करें"),
        ("what time is it", "कितने बजे हैं"),
    ] * 20  # repeat so there is something to train on
    english_raw = [p[0] for p in pairs]
    hindi_raw = [p[1] for p in pairs]


# CLEAN THE TEXT
def clean_english(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # keep letters, digits, spaces
    return re.sub(r"\s+", " ", text).strip()


def clean_hindi(text):
    text = text.strip()
    # keep only Devanagari letters (and spaces)
    text = re.sub(r"[^\u0900-\u0963\u0966-\u097F\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


english_sents, hindi_sents = [], []
for e, h in zip(english_raw, hindi_raw):
    e, h = clean_english(e), clean_hindi(h)
    if not e or not h:
        continue
    # keep only short sentences (faster and easier to learn)
    if len(e.split()) <= MAX_LEN and len(h.split()) <= MAX_LEN:
        english_sents.append(e)
        hindi_sents.append(h)

english_sents = english_sents[:NUM_SAMPLES]
hindi_sents = hindi_sents[:NUM_SAMPLES]
print("Pairs after cleaning/filtering:", len(english_sents))
print("Example:", english_sents[0], "->", hindi_sents[0])


# BUILD VOCABULARIES (word -> number, number -> word)
def build_vocab(sentences, max_vocab):
    counts = {}
    for s in sentences:
        for w in s.split():
            counts[w] = counts.get(w, 0) + 1
    common = sorted(counts, key=counts.get, reverse=True)[: max_vocab - 4]
    itos = [PAD, UNK, START, END] + common          # index -> word
    stoi = {w: i for i, w in enumerate(itos)}       # word -> index
    return stoi, itos


eng_stoi, eng_itos = build_vocab(english_sents, MAX_VOCAB)
hin_stoi, hin_itos = build_vocab(hindi_sents, MAX_VOCAB)
ENG_VOCAB, HIN_VOCAB = len(eng_itos), len(hin_itos)
print("English vocab size:", ENG_VOCAB, "| Hindi vocab size:", HIN_VOCAB)


def sentence_to_ids(sentence, stoi):
    return [stoi.get(w, stoi[UNK]) for w in sentence.split()]


def pad(ids, length):
    """Fill with zeros (the <pad> id) so every row has the same length."""
    return ids + [0] * (length - len(ids))


# ----------------------------------------------------------------------
# TURN SENTENCES INTO TENSORS
# ----------------------------------------------------------------------
# Teacher forcing: while training, the decoder is fed the CORRECT previous
# Hindi word and must predict the next one.
#   decoder input : <start> w1 w2 w3
#   decoder target:  w1 w2 w3 <end>
enc_in, dec_in, dec_out = [], [], []
for e, h in zip(english_sents, hindi_sents):
    h_ids = sentence_to_ids(h, hin_stoi)
    enc_in.append(pad(sentence_to_ids(e, eng_stoi), MAX_LEN))
    dec_in.append(pad([hin_stoi[START]] + h_ids, MAX_LEN + 1))
    dec_out.append(pad(h_ids + [hin_stoi[END]], MAX_LEN + 1))

enc_in = torch.tensor(enc_in, dtype=torch.long)
dec_in = torch.tensor(dec_in, dtype=torch.long)
dec_out = torch.tensor(dec_out, dtype=torch.long)

# Train / test split (90% / 10%)
split = int(0.9 * len(enc_in))
train_data = TensorDataset(enc_in[:split], dec_in[:split], dec_out[:split])
test_data = TensorDataset(enc_in[split:], dec_in[split:], dec_out[split:])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
print("Train:", len(train_data), "| Test:", len(test_data))


# MODEL
def make_rnn(cell_type):
    """Return the recurrent layer class: nn.LSTM or nn.GRU."""
    return nn.LSTM if cell_type == "LSTM" else nn.GRU

class Encoder(nn.Module):
    def __init__(self, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(ENG_VOCAB, EMBED_DIM, padding_idx=0)
        self.rnn = make_rnn(cell_type)(EMBED_DIM, HIDDEN_DIM, batch_first=True)

    def forward(self, src):
        # src: (batch, src_len) of word ids
        lengths = (src != 0).sum(dim=1).cpu()          # real length of each sentence
        embedded = self.embedding(src)                 # (batch, src_len, EMBED_DIM)
        # "Packing" tells the RNN to ignore the <pad> positions
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        # hidden is the "memory" of the sentence:
        #   LSTM -> a tuple (h, c)     GRU -> just h
        return hidden


class Decoder(nn.Module):
    def __init__(self, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(HIN_VOCAB, EMBED_DIM, padding_idx=0)
        self.rnn = make_rnn(cell_type)(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, HIN_VOCAB)     # memory -> score for every Hindi word

    def forward(self, tokens, hidden):
        # tokens: (batch, steps) of Hindi word ids
        embedded = self.embedding(tokens)
        output, hidden = self.rnn(embedded, hidden)    # start from the encoder's memory
        logits = self.fc(output)                       # (batch, steps, HIN_VOCAB)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, cell_type):
        super().__init__()
        self.encoder = Encoder(cell_type)
        self.decoder = Decoder(cell_type)

    def forward(self, src, dec_input):
        hidden = self.encoder(src)
        logits, _ = self.decoder(dec_input, hidden)
        return logits


# TRAINING AND EVALUATION FUNCTIONS

# ignore_index=0 -> padding positions do not count in the loss
loss_fn = nn.CrossEntropyLoss(ignore_index=0)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for src, d_in, d_out in loader:
        src, d_in, d_out = src.to(device), d_in.to(device), d_out.to(device)

        optimizer.zero_grad()
        logits = model(src, d_in)                                  # forward pass
        loss = loss_fn(logits.reshape(-1, HIN_VOCAB), d_out.reshape(-1))
        loss.backward()                                            # backpropagation
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # avoid exploding gradients
        optimizer.step()                                           # update weights

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for src, d_in, d_out in loader:
            src, d_in, d_out = src.to(device), d_in.to(device), d_out.to(device)
            logits = model(src, d_in)
            loss = loss_fn(logits.reshape(-1, HIN_VOCAB), d_out.reshape(-1))
            total_loss += loss.item()

            # word-level accuracy (ignoring padding)
            predictions = logits.argmax(dim=-1)
            mask = d_out != 0
            correct += ((predictions == d_out) & mask).sum().item()
            total += mask.sum().item()
    return total_loss / len(loader), correct / total


# TRANSLATE A SENTENCE (greedy decoding: always pick the most likely word)
def translate(sentence, model):
    model.eval()
    sentence = clean_english(sentence)
    ids = sentence_to_ids(sentence, eng_stoi)[:MAX_LEN]
    if len(ids) == 0:
        return ""
    src = torch.tensor([pad(ids, MAX_LEN)], dtype=torch.long).to(device)

    with torch.no_grad():
        hidden = model.encoder(src)                       # English -> memory
        token = torch.tensor([[hin_stoi[START]]], device=device)
        result = []
        for _ in range(MAX_LEN + 1):
            logits, hidden = model.decoder(token, hidden)  # one step at a time
            next_id = logits[0, -1].argmax().item()
            if next_id == hin_stoi[END]:
                break
            result.append(hin_itos[next_id])
            token = torch.tensor([[next_id]], device=device)  # feed prediction back in
    return " ".join(result)


# TRAIN BOTH MODELS AND COMPARE
histories = {}
models = {}

for cell_type in ["LSTM", "GRU"]:
    print("\n" + "=" * 60)
    print(f"Training seq2seq with {cell_type}")
    print("=" * 60)

    model = Seq2Seq(cell_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, test_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    histories[cell_type] = history
    models[cell_type] = model

# PLOT LOSS / ACCURACY
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, h in histories.items():
    axes[0].plot(h["val_loss"], label=f"{name} val loss")
    axes[1].plot(h["val_accuracy"], label=f"{name} val accuracy")
axes[0].set_title("Validation loss (lower is better)")
axes[1].set_title("Validation accuracy (higher is better)")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.savefig("lstm_vs_gru.png")
print("\nSaved plot to lstm_vs_gru.png")

# SAMPLE TRANSLATIONS
print("\n" + "=" * 60)
print("Sample translations from the held-out test set")
print("=" * 60)
for i in range(5):
    print(f"\nEnglish   : {english_sents[split + i]}")
    print(f"Actual    : {hindi_sents[split + i]}")
    for name, model in models.items():
        print(f"{name:<10}: {translate(english_sents[split + i], model)}")

print("\nTry your own sentences:")
for text in ["how are you", "thank you", "what is your name"]:
    print(f"\nEnglish   : {text}")
    for name, model in models.items():
        print(f"{name:<10}: {translate(text, model)}")
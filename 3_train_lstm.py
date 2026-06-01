import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from collections import Counter
import joblib
import os
import json
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Config ────────────────────────────────────────────────
MAX_VOCAB  = 20000
MAX_LEN    = 150
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
BATCH_SIZE = 128
EPOCHS     = 3
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv('processed/train.csv')
val_df   = pd.read_csv('processed/val.csv')
test_df  = pd.read_csv('processed/test.csv')

# ── Fix empty texts ───────────────────────────────────────
train_df['cleaned'] = train_df['cleaned'].fillna('unknown').astype(str)
val_df['cleaned']   = val_df['cleaned'].fillna('unknown').astype(str)
test_df['cleaned']  = test_df['cleaned'].fillna('unknown').astype(str)

train_df = train_df[train_df['cleaned'].str.strip() != ''].reset_index(drop=True)
val_df   = val_df[val_df['cleaned'].str.strip()   != ''].reset_index(drop=True)
test_df  = test_df[test_df['cleaned'].str.strip() != ''].reset_index(drop=True)

print(f"After cleaning: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# ── Build vocabulary ──────────────────────────────────────
print("Building vocabulary...")
all_text = ' '.join(train_df['cleaned'].tolist())
words    = all_text.split()
counter  = Counter(words)
vocab    = ['<PAD>', '<UNK>'] + [w for w, _ in counter.most_common(MAX_VOCAB - 2)]
word2idx = {w: i for i, w in enumerate(vocab)}

os.makedirs('processed', exist_ok=True)
with open('processed/vocab.json', 'w') as f:
    json.dump(word2idx, f)
print(f"Vocabulary size: {len(word2idx)}")

# ── Encode & pad ──────────────────────────────────────────
def encode(text, word2idx, max_len):
    tokens = text.split()[:max_len]
    if len(tokens) == 0:
        tokens = ['unknown']
    ids    = [word2idx.get(t, 1) for t in tokens]
    length = max(len(ids), 1)
    ids   += [0] * (max_len - length)
    return ids, length

# ── Dataset class ─────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, df):
        self.texts  = df['cleaned'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, length = encode(self.texts[idx], word2idx, MAX_LEN)
        return (
            torch.tensor(ids,    dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

train_loader = DataLoader(NewsDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(NewsDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(NewsDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)

# ── Bi-LSTM Model ─────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed   = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden)).squeeze(1)

model = BiLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training setup ────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# ── Accuracy helper ───────────────────────────────────────
def get_accuracy(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, lengths, labels in loader:
            ids, lengths, labels = ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            preds = torch.sigmoid(model(ids, lengths)) >= 0.5
            correct += (preds == labels.bool()).sum().item()
            total   += labels.size(0)
    return correct / total

# ── Training loop ─────────────────────────────────────────
best_val_acc = 0
history = {'train_loss': [], 'val_acc': []}

print("\n" + "="*50)
print("TRAINING STARTED")
print("="*50)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (ids, lengths, labels) in enumerate(train_loader):
        ids, lengths, labels = ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(ids, lengths)
        loss   = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    val_acc  = get_accuracy(val_loader)
    scheduler.step(avg_loss)

    history['train_loss'].append(avg_loss)
    history['val_acc'].append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS} => Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'processed/best_model.pt')
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")

# ── Final test accuracy ───────────────────────────────────
model.load_state_dict(torch.load('processed/best_model.pt', map_location=DEVICE))
test_acc = get_accuracy(test_loader)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Test Accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\n🎉 Training complete! Model saved to processed/best_model.pt")

# ── Save history ──────────────────────────────────────────
joblib.dump(history, 'processed/history.pkl')
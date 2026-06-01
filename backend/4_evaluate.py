import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from collections import Counter
import json
import joblib

# ── Config (must match training) ──────────────────────────
MAX_VOCAB  = 20000
MAX_LEN    = 150
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
BATCH_SIZE = 256
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load vocab ────────────────────────────────────────────
with open('processed/vocab.json', 'r') as f:
    word2idx = json.load(f)

# ── Encode function ───────────────────────────────────────
def encode(text, word2idx, max_len):
    tokens = str(text).split()[:max_len]
    if len(tokens) == 0:
        tokens = ['unknown']
    ids    = [word2idx.get(t, 1) for t in tokens]
    length = max(len(ids), 1)
    ids   += [0] * (max_len - length)
    return ids, length

# ── Dataset ───────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, df):
        self.texts  = df['cleaned'].fillna('unknown').tolist()
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

# ── Model ─────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0,
                                  batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed   = pack_padded_sequence(embedded, lengths.cpu(),
                                         batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden)).squeeze(1)

# ── Load model ────────────────────────────────────────────
model = BiLSTM(len(word2idx), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
model.load_state_dict(torch.load('processed/best_model.pt', map_location=DEVICE))
model.eval()
print("✅ Model loaded!")

# ── Load test data ────────────────────────────────────────
test_df     = pd.read_csv('processed/test.csv')
test_loader = DataLoader(NewsDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

# ── Get predictions ───────────────────────────────────────
all_preds  = []
all_probs  = []
all_labels = []

with torch.no_grad():
    for ids, lengths, labels in test_loader:
        ids, lengths = ids.to(DEVICE), lengths.to(DEVICE)
        logits = model(ids, lengths)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().astype(int))

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# ── Classification report ─────────────────────────────────
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_labels, all_preds,
                             target_names=['Real News', 'Fake News']))

# ── ROC AUC ───────────────────────────────────────────────
auc = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC Score: {auc:.4f}")

# ── Plot confusion matrix ─────────────────────────────────
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# ── Precision-Recall curve ────────────────────────────────
plt.subplot(1, 3, 2)
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

# ── Confidence distribution ───────────────────────────────
plt.subplot(1, 3, 3)
real_probs = all_probs[all_labels == 0]
fake_probs = all_probs[all_labels == 1]
plt.hist(real_probs, bins=30, alpha=0.7, color='#2ecc71', label='Real')
plt.hist(fake_probs, bins=30, alpha=0.7, color='#e74c3c', label='Fake')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Confidence Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('evaluation_plots.png', dpi=150)
plt.show()
print("\n✅ Plots saved as evaluation_plots.png")
print("\n🎉 Evaluation complete!")
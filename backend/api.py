from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import json
import os
import re
import nltk
import httpx
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Config ────────────────────────────────────────────────
MAX_LEN    = 150
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
DEVICE     = torch.device('cpu')

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # newsapi.org key

# ── Load vocab ────────────────────────────────────────────
with open('processed/vocab.json', 'r') as f:
    word2idx = json.load(f)

# ── Model ─────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  dropout=dropout,
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
model = BiLSTM(len(word2idx), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model.load_state_dict(torch.load('processed/best_model.pt', map_location=DEVICE))
model.eval()
print("✅ Model loaded and ready!")

# ── Text cleaning ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── Encode ────────────────────────────────────────────────
def encode(text):
    tokens = text.split()[:MAX_LEN]
    if len(tokens) == 0:
        tokens = ['unknown']
    ids    = [word2idx.get(t, 1) for t in tokens]
    length = max(len(ids), 1)
    ids   += [0] * (MAX_LEN - length)
    return ids, length

# ── Keyword extraction ────────────────────────────────────
def get_keywords(text, top_n=5):
    words = text.split()
    scored = [(w, word2idx.get(w, 0)) for w in words if w in word2idx]
    scored = sorted(scored, key=lambda x: x[1])
    seen = set()
    keywords = []
    for w, _ in scored:
        if w not in seen and len(w) > 3:
            seen.add(w)
            keywords.append(w)
        if len(keywords) == top_n:
            break
    return keywords

# ── NewsAPI verification ──────────────────────────────────
async def verify_with_newsapi(query: str):
    if not NEWS_API_KEY:
        return 0, []
    try:
        short_query = ' '.join(query.split()[:6])
        short_query = re.sub(r"['\"\(\)\[\]]", '', short_query)
        short_query = re.sub(r'\s+', ' ', short_query).strip()
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": short_query,
            "apiKey": NEWS_API_KEY,
            "pageSize": 5,
            "language": "en",
            "sortBy": "relevancy"
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(url, params=params)
            data = res.json()
            print(f"DEBUG NewsAPI: totalResults={data.get('totalResults')} | status={data.get('status')}")
            articles = data.get("articles", [])
            found = len(articles)
            sources = list(set([
                a["source"]["name"]
                for a in articles
                if a.get("source")
            ]))
            return found, sources
    except Exception as e:
        print(f"DEBUG NewsAPI error: {e}")
        return 0, []

# ── Combine LSTM + NewsAPI scores ─────────────────────────
def compute_final_score(lstm_real_prob: float, news_found: int):
    if news_found >= 3:
        boost = 0.50
    elif news_found == 2:
        boost = 0.40
    elif news_found == 1:
        boost = 0.30
    else:
        boost = 0.0

    final_real = min(lstm_real_prob + boost, 0.99)
    final_fake = round(1 - final_real, 4)
    final_real = round(final_real, 4)
    return final_real, final_fake

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(title="Fake News Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    text: str

class NewsOutput(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    keywords: list
    news_sources_found: int
    verified_sources: list
    verification_note: str

@app.get("/")
def root():
    return {"message": "Fake News Detection API is running!"}

@app.post("/predict", response_model=NewsOutput)
async def predict(news: NewsInput):
    # Step 1 — LSTM prediction
    cleaned = clean_text(news.text)
    ids, length = encode(cleaned)
    x      = torch.tensor([ids],    dtype=torch.long)
    length = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        logit = model(x, length)
        prob  = torch.sigmoid(logit).item()

    lstm_real = 1 - prob
    lstm_fake = prob

    # Step 2 — NewsAPI verification
    if NEWS_API_KEY:
        news_found, sources = await verify_with_newsapi(news.text)
    else:
        news_found, sources = 0, []

    # Step 3 — Combine scores
    final_real, final_fake = compute_final_score(lstm_real, news_found)
    label      = "REAL" if final_real >= 0.5 else "FAKE"
    confidence = round(max(final_real, final_fake) * 100, 2)

    # Step 4 — Verification note
    if not NEWS_API_KEY:
        note = "NewsAPI key not set — LSTM score used"
    elif news_found >= 3:
        note = f"✅ Found in {news_found} real news sources — score boosted"
    elif news_found > 0:
        note = f"⚠️ Found in {news_found} source(s) — slight boost applied"
    else:
        note = "🔍 Not found in recent news — LSTM score used"

    keywords = get_keywords(cleaned)

    return NewsOutput(
        label=label,
        confidence=confidence,
        fake_probability=round(final_fake * 100, 2),
        real_probability=round(final_real * 100, 2),
        keywords=keywords,
        news_sources_found=news_found,
        verified_sources=sources[:3],
        verification_note=note
    )
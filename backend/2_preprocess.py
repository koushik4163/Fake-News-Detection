import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

print("="*50)
print("STEP 1: Loading dataset")
print("="*50)
df = pd.read_csv("WELFake_Dataset.csv")
df['text']  = df['text'].fillna('')
df['title'] = df['title'].fillna('')

# ── Combine title + text into one input ───────────────────
df['content'] = df['title'] + ' ' + df['text']
print(f"Total samples: {len(df)}")

# ── Clean text function ───────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'<.*?>', '', text)                 # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)              # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra spaces
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    return ' '.join(tokens)

print("\n" + "="*50)
print("STEP 2: Cleaning text (this takes 1-2 mins...)")
print("="*50)
df['cleaned'] = df['content'].apply(clean_text)
print("✅ Text cleaning done!")

# ── Show sample cleaned text ──────────────────────────────
print("\nOriginal:")
print(df['content'].iloc[0][:200])
print("\nCleaned:")
print(df['cleaned'].iloc[0][:200])

# ── Split dataset ─────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Splitting dataset")
print("="*50)
X = df['cleaned']
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train size : {len(X_train)}")
print(f"Val size   : {len(X_val)}")
print(f"Test size  : {len(X_test)}")

# ── TF-IDF Vectorizer ─────────────────────────────────────
print("\n" + "="*50)
print("STEP 4: Building TF-IDF vectors")
print("="*50)
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
print("✅ TF-IDF done!")

# ── Save everything ───────────────────────────────────────
print("\n" + "="*50)
print("STEP 5: Saving processed data")
print("="*50)
os.makedirs('processed', exist_ok=True)

joblib.dump(tfidf, 'processed/tfidf_vectorizer.pkl')

# Save splits as CSV
train_df = pd.DataFrame({'cleaned': X_train, 'label': y_train})
val_df   = pd.DataFrame({'cleaned': X_val,   'label': y_val})
test_df  = pd.DataFrame({'cleaned': X_test,  'label': y_test})

train_df.to_csv('processed/train.csv', index=False)
val_df.to_csv('processed/val.csv',     index=False)
test_df.to_csv('processed/test.csv',   index=False)

print("✅ Saved:")
print("   processed/tfidf_vectorizer.pkl")
print("   processed/train.csv")
print("   processed/val.csv")
print("   processed/test.csv")
print("\n🎉 Preprocessing complete! Ready for model training.")
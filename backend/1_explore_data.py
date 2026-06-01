import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load dataset ──────────────────────────────────────────
df = pd.read_csv("WELFake_Dataset.csv")

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Total articles : {len(df)}")
print(f"Columns        : {list(df.columns)}")
print(f"Shape          : {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ── Check for missing values ──────────────────────────────
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum())

# ── Class distribution ────────────────────────────────────
print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)
print(df['label'].value_counts())
print("\n0 = Real News")
print("1 = Fake News")

# ── Text length analysis ──────────────────────────────────
df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')
df['text_length'] = df['text'].apply(lambda x: len(str(x)))
print("\n" + "="*50)
print("TEXT LENGTH STATS")
print("="*50)
print(df.groupby('label')['text_length'].describe())

# ── Plot 1: Class distribution bar chart ─────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
colors = ['#2ecc71', '#e74c3c']
df['label'].value_counts().plot(kind='bar', color=colors, edgecolor='black')
plt.title('Class Distribution')
plt.xlabel('Label (0=Real, 1=Fake)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# ── Plot 2: Text length distribution ─────────────────────
plt.subplot(1, 3, 2)
df[df['label']==0]['text_length'].hist(bins=50, alpha=0.7, color='#2ecc71', label='Real')
df[df['label']==1]['text_length'].hist(bins=50, alpha=0.7, color='#e74c3c', label='Fake')
plt.title('Text Length Distribution')
plt.xlabel('Character Count')
plt.ylabel('Frequency')
plt.legend()

# ── Plot 3: Average text length per class ────────────────
plt.subplot(1, 3, 3)
avg_len = df.groupby('label')['text_length'].mean()
avg_len.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Avg Text Length per Class')
plt.xlabel('Label (0=Real, 1=Fake)')
plt.ylabel('Avg Characters')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=150)
plt.show()
print("\nPlot saved as data_exploration.png")

# ── Sample articles ───────────────────────────────────────
print("\n" + "="*50)
print("SAMPLE REAL NEWS TITLE")
print("="*50)
print(df[df['label']==0]['title'].dropna().iloc[0])

print("\n" + "="*50)
print("SAMPLE FAKE NEWS TITLE")
print("="*50)
print(df[df['label']==1]['title'].dropna().iloc[0])

print("\n✅ Exploration complete!")
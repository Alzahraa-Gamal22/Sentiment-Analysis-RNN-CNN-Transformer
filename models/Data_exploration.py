
# ==========================================
# IMDB Sentiment Analysis - Data Exploration
# ==========================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from html import unescape
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 2. Load Dataset

df = pd.read_csv(
    r"D:\Downloads\Sentiment Analysis\Sentiment-Analysis-RNN-CNN-Transformer\IMDB Dataset.csv"
)

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Reviews:")
print(df.duplicated().sum())


# 3. Remove Duplicates

df = df.drop_duplicates(subset=["review"])

print("\nDataset Shape After Removing Duplicates:", df.shape)
print("Duplicates After Removing:", df.duplicated().sum())


# 4. Sentiment Distribution

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

sns.countplot(data=df, x="sentiment")
plt.title("Sentiment Distribution")
plt.show()


# 5. Text Cleaning

def clean_text(text):
    text = unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df["clean_review"] = df["review"].apply(clean_text)

print("\nOriginal Review:")
print(df["review"].iloc[0][:300])

print("\nCleaned Review:")
print(df["clean_review"].iloc[0][:300])


# 6. Encode Sentiment Labels

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("\nEncoded Labels:")
print(df[["sentiment", "label"]].head())

print("\nLabel Distribution:")
print(df["label"].value_counts())


# 7. Split Dataset

X = df["clean_review"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining Samples:", len(X_train))
print("Validation Samples:", len(X_val))
print("Testing Samples:", len(X_test))


# 8. Tokenization

MAX_WORDS = 20000
MAX_LEN = 300

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

print("\nOriginal Review:")
print(X_train.iloc[0][:200])

print("\nConverted Sequence:")
print(X_train_seq[0][:30])


# 9. Padding Sequences

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_val_pad = pad_sequences(
    X_val_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


# 10. Final Shapes

print("\nTraining Shape:", X_train_pad.shape)
print("Validation Shape:", X_val_pad.shape)
print("Testing Shape:", X_test_pad.shape)

print("\nData Preparation Completed Successfully!")

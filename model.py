
import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv("Fake.csv")[["text"]]
true = pd.read_csv("True.csv")[["text"]]

fake["label"] = 0
true["label"] = 1

# Balance dataset (IMPORTANT FIX)
min_len = min(len(fake), len(true))
fake = fake.sample(min_len, random_state=42)
true = true.sample(min_len, random_state=42)

data = pd.concat([fake, true], ignore_index=True)
data = data.dropna()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# Split
x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Clean AFTER split
x_train = x_train.apply(clean_text)
x_test = x_test.apply(clean_text)

# TF-IDF (improved)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=5,
    ngram_range=(1,2)
)

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Model (FIXED)
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(xv_train, y_train)

# Accuracy
pred = model.predict(xv_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")
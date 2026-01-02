import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "problems_data.jsonl")
)

df = pd.read_json(DATA_PATH, lines=True)


df.fillna("", inplace=True)

df["combined_text"] = (
    df["title"] + " " +
    df["description"] + " " +
    df["input_description"] + " " +
    df["output_description"]
)

df["combined_text"] = df["combined_text"].str.lower()

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_text = vectorizer.fit_transform(df["combined_text"])

# 1. Text length
df["text_length"] = df["combined_text"].apply(len)

# 2. Math symbol count
df["math_symbol_count"] = df["combined_text"].apply(
    lambda x: sum(1 for c in x if c in "$\\_^<>=")
)

# 3. Keyword count (topic-based)
KEYWORDS = [
    "graph", "tree", "dfs", "bfs",
    "dp", "dynamic programming",
    "greedy", "binary search",
    "segment tree", "fenwick",
    "flow", "matching", "fft", "bitmask"
]

def keyword_count(text):
    return sum(text.count(k) for k in KEYWORDS)

df["keyword_count"] = df["combined_text"].apply(keyword_count)

# 4. Algorithm hint count
ALGO_HINTS = [
    "dynamic programming", "dp", "greedy",
    "binary search", "shortest path",
    "flow", "matching", "segment tree",
    "fenwick", "fft", "bitmask"
]

def algo_hint_count(text):
    return sum(text.count(h) for h in ALGO_HINTS)

df["algo_hint_count"] = df["combined_text"].apply(algo_hint_count)

# 5. Constraint count (large numbers)
def constraint_count(text):
    numbers = re.findall(r"\d+", text)
    return sum(1 for n in numbers if int(n) >= 1000)

df["constraint_count"] = df["combined_text"].apply(constraint_count)

# 6. Max constraint value
def max_constraint_value(text):
    numbers = re.findall(r"\d+", text)
    return max([int(n) for n in numbers], default=0)

df["max_constraint_value"] = df["combined_text"].apply(max_constraint_value)

# 7. Constraint-related word count
CONSTRAINT_WORDS = ["constraint", "constraints", "<=", "≤", "bound", "limit"]

def constraint_word_count(text):
    return sum(text.count(w) for w in CONSTRAINT_WORDS)

df["constraint_word_count"] = df["combined_text"].apply(constraint_word_count)

# 8. Sample I/O complexity
def sample_io_length(sample):
    if not isinstance(sample, list):
        return 0
    total = 0
    for s in sample:
        total += len(s.get("input", "")) + len(s.get("output", ""))
    return total

df["sample_io_length"] = df["sample_io"].apply(sample_io_length)

# 9. Math density
def math_density(text):
    if len(text) == 0:
        return 0
    math_chars = sum(1 for c in text if c in "$\\_^<>=")
    return math_chars / len(text)

df["math_density"] = df["combined_text"].apply(math_density)

manual_feature_columns = [
    "text_length",
    "math_symbol_count",
    "math_density",
    "keyword_count",
    "algo_hint_count",
    "constraint_count",
    "max_constraint_value",
    "constraint_word_count",
    "sample_io_length"
]

manual_features = df[manual_feature_columns].values

scaler = StandardScaler()
manual_features_scaled = scaler.fit_transform(manual_features)

X = hstack([X_text, manual_features_scaled])

print("Final feature matrix shape:", X.shape)

__all__ = [
    "vectorizer",
    "scaler",
    "keyword_count",
    "algo_hint_count",
    "constraint_count",
    "max_constraint_value",
    "constraint_word_count",
    "sample_io_length",
    "math_density"
]

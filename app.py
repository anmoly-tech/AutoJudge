import streamlit as st
import pickle
import os
import sys
import numpy as np


BASE_DIR = os.path.dirname(__file__)
CODES_DIR = os.path.join(BASE_DIR, "codes")

sys.path.insert(0, CODES_DIR)
from features import (
    vectorizer,
    scaler,
    keyword_count,
    algo_hint_count,
    constraint_count,
    max_constraint_value,
    constraint_word_count,
    sample_io_length,
    math_density
)

with open("codes/classification/models/classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("codes/regression/models/regressor.pkl", "rb") as f:
    reg = pickle.load(f)

st.set_page_config(page_title="AutoJudge", layout="centered")
st.title(" AutoJudge: Problem Difficulty Predictor")

st.write("Paste a programming problem description to predict its difficulty.")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

if st.button("Predict"):
    combined_text = (desc + " " + inp + " " + out).lower()

    # TF-IDF
    X_text = vectorizer.transform([combined_text])

    # Manual features
    manual = np.array([[
    len(combined_text),                         # text_length
    sum(1 for c in combined_text if c in "$\\_^<>="),  # math_symbol_count
    math_density(combined_text),                # math_density
    keyword_count(combined_text),               # keyword_count
    algo_hint_count(combined_text),             # algo_hint_count
    constraint_count(combined_text),            # constraint_count
    max_constraint_value(combined_text),        # max_constraint_value
    constraint_word_count(combined_text),       # constraint_word_count
    0                                           # sample_io_length (unknown at inference)
]])


    manual_scaled = scaler.transform(manual)

    from scipy.sparse import hstack
    X_final = hstack([X_text, manual_scaled])

    # Predictions
    pred_class = clf.predict(X_final)[0]
    pred_score = reg.predict(X_final)[0]

    st.success(f"Predicted Difficulty Class: **{pred_class.upper()}**")
    st.success(f"Predicted Difficulty Score: **{pred_score:.2f}**")

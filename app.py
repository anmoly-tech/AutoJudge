import streamlit as st
import pickle
import os
import sys
import numpy as np
from scipy.sparse import hstack

BASE_DIR = os.path.dirname(__file__)
CODES_DIR = os.path.join(BASE_DIR, "codes")
sys.path.insert(0, CODES_DIR)

from features import (
    vectorizer,
    scaler,
    keyword_count,
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

# STREAMLIT UI
st.set_page_config(page_title="AutoJudge", layout="centered")
st.title("AutoJudge: Problem Difficulty Predictor")

st.write("Paste a programming problem description to predict its difficulty.")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

# PREDICTION
if st.button("Predict"):
    combined_text = (desc + " " + inp + " " + out).lower()

    # TF-IDF
    X_text = vectorizer.transform([combined_text])

    # Manual features 
    # sample i/o length unkownn at time of inference 
    manual = np.array([[
        len(combined_text),                  
        math_density(combined_text),          
        keyword_count(combined_text),         
        constraint_count(combined_text),      
        max_constraint_value(combined_text),  
        constraint_word_count(combined_text), 
        0                                    
    ]])

    manual_scaled = scaler.transform(manual)

    X_final = hstack([X_text, manual_scaled])

    pred_class = clf.predict(X_final)[0]
    pred_score = reg.predict(X_final)[0]

    st.success(f"Predicted Difficulty Class: **{pred_class.upper()}**")
    st.success(f"Predicted Difficulty Score: **{pred_score:.2f}**")

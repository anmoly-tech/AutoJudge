# AutoJudge: Predicting Programming Problem Difficulty

## Overview
AutoJudge is a machine learning system that automatically predicts the
difficulty of programming problems using **only textual information**.
The system performs:

- **Classification** → Easy / Medium / Hard  
- **Regression** → Numerical difficulty score  

This project removes the need for manual difficulty labeling and demonstrates
how natural language processing and feature engineering can be applied to
competitive programming problems.

---

## Dataset
Each problem in the dataset contains:

- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numeric difficulty score)

Total problems: **~4100**

The dataset is provided and already labeled.

---

## Data Preprocessing
- Missing values handled using empty strings
- Text fields combined into a single `combined_text`
- Text converted to lowercase for normalization

---

## Feature Extraction

### 1. TF-IDF Features
- TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text
  into numerical vectors
- Limited to the top **5000** features
- English stop words removed

### 2. Feature Engineering (Creative Features)
To improve performance and interpretability, domain-specific features were added:

- Text length
- Mathematical symbol count and density
- Algorithm keyword frequency (graph, dp, tree, bfs, dfs, etc.)
- Algorithm hint count
- Constraint count and maximum constraint value
- Constraint-related word frequency
- Sample input/output complexity

These features capture signals that raw text statistics may miss.

---

## Models

### Classification Model
- **Model:** Logistic Regression  
- **Task:** Predict problem difficulty class  
- **Evaluation:**
  - Accuracy ≈ **51%**
  - Confusion Matrix: **[[ 47  64  42]**
                       **[ 24 291  74]**
                       **[ 21 179  81]]**
  - Confusion matrix shows strongest performance on Hard problems
  - Medium problems are the most ambiguous due to subjective labeling

### Regression Model
- **Model:** Ridge Regression  
- **Task:** Predict numerical difficulty score  
- **Evaluation Metrics:**
  - MAE ≈ **1.71**
  - RMSE ≈ **2.06**

Ridge regression was chosen to handle the high-dimensional TF-IDF feature space
and avoid numerical instability.

---

## Web Interface
A Streamlit-based web application allows users to:

- Paste a problem description
- Predict difficulty class and score instantly

Run the app with:
```bash
python3 -m streamlit run app.py

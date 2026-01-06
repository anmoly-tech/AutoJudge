# AutoJudge: Predicting Programming Problem Difficulty

## Project Overview
AutoJudge is a machine learning project that predicts the difficulty of
programming problems using only their textual descriptions. The system performs
two tasks:
- Classification of problems into Easy, Medium, or Hard
- Regression to predict a numerical difficulty score

The goal of this project is to automate difficulty estimation, which is usually
done manually on competitive programming platforms.

---

## Dataset
The dataset contains approximately 4100 programming problems. Each problem
includes the following fields:
- Title
- Problem description
- Input description
- Output description
- Difficulty class (Easy / Medium / Hard)
- Numerical difficulty score

The dataset is pre-labeled and provided for the project.

---

## Data Preprocessing
- Missing values are handled by replacing them with empty strings
- All text fields are combined into a single input
- Text is converted to lowercase for consistency

---

## Feature Extraction and Engineering

### TF-IDF Features
Textual data is converted into numerical features using TF-IDF
(Term Frequency–Inverse Document Frequency). The vectorizer is limited to the
top 5000 features and English stop words are removed.

### Engineered Features
In addition to TF-IDF, domain-specific features are added to better capture
problem complexity:
- Text length
- Mathematical symbol count and density
- Algorithm keyword frequency (graph, dp, tree, etc.)
- Algorithm hint count
- Constraint count and maximum constraint value
- Constraint-related word frequency

---

## Models Used

### Classification Model
- Model: Logistic Regression
- Task: Predict difficulty class (Easy / Medium / Hard)
- Accuracy: ~51%

### Regression Model
- Model: Ridge Regression
- Task: Predict numerical difficulty score
- MAE: ~1.7
- RMSE: ~2.1

Ridge regression is used to ensure numerical stability with high-dimensional
TF-IDF features.

---

## Web Interface
A Streamlit-based web interface allows users to paste a problem description,
input format, and output format. The system then predicts the difficulty class
and difficulty score in real time.

---

## How to Run the Project Locally

```bash
git clone https://github.com/anmoly-tech/AutoJudge.git
cd AutoJudge
pip install -r requirements.txt
python3 -m streamlit run app.py

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
(Term Frequency–Inverse Document Frequency). This representation captures
the importance of words relative to the entire dataset. The vectorizer
is limited to the top 5000 features, and English stop words are removed
to reduce noise.

### Engineered Features
In addition to TF-IDF, several domain-specific features are engineered
to better capture the structural and algorithmic complexity of
programming problems:

- **Text length:** Longer problem descriptions often indicate higher complexity.
- **Mathematical symbol density:** Measures the proportion of mathematical symbols
  in the text, indicating mathematically intensive problems.
- **Algorithm keyword frequency:** Counts occurrences of algorithm-related terms
  such as graph, dp, tree, bfs, and dfs.
- **Constraint count:** Counts the number of large numerical constraints
  mentioned in the problem statement.
- **Maximum constraint value:** Captures the largest numerical bound specified,
  indicating required algorithmic efficiency.
- **Constraint-related word frequency:** Counts occurrences of words such as
  constraint, bound, and limit.
- **Sample input-output length:** Measures the size of sample inputs and outputs,
  reflecting problem complexity.


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
pip3 install -r requirements.txt
python3 -m streamlit run app.py

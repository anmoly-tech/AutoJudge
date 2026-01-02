import os
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

CURRENT_DIR = os.path.dirname(__file__)
CODES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CODES_DIR)

from features import X, df

y = df["problem_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Classification Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)


MODEL_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "classifier.pkl")

with open(model_path, "wb") as f:
    pickle.dump(clf, f)

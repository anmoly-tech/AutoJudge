import os
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

CURRENT_DIR = os.path.dirname(__file__)
CODES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CODES_DIR)

from features import X, df


y = df["problem_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL: RIDGE REGRESSION
# Ridge regression is used instead of plain linear regression
# because TF-IDF creates a very high-dimensional feature space.
# Regularization helps prevent unstable coefficients and
# improves numerical stability.

reg = Ridge(alpha=1.0)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

MODEL_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "regressor.pkl")

with open(model_path, "wb") as f:
    pickle.dump(reg, f)

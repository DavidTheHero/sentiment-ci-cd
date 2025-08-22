from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score

def test_min_accuracy():
    # Load the tiny artifact produced by scripts/train.py
    model = joblib.load(Path("models/model.joblib"))
    # Tiny holdout (adjust as you wish)
    X = ["i love this", "this is bad", "great", "awful"]
    y = [1, 0, 1, 0]
    acc = accuracy_score(y, model.predict(X))
    assert acc >= 0.70, f"Accuracy too low: {acc:.2f}"
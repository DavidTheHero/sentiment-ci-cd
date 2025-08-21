from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score

def test_min_accuracy():
    model = joblib.load(Path("models/model.joblib"))
    X = ["i love this", "this is bad", "great", "awful"]
    y = [1,0,1,0]
    acc = accuracy_score(y, model.predict(X))
    assert acc >= 0.70, f"Accuracy too low: {acc:.2f}"
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

X = ["i love this", "this is bad", "great", "terrible", "amazing", "awful"]
y = [1, 0, 1, 0, 1, 0]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=200))
])
pipe.fit(X, y)

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "models/model.joblib")
print("Saved models/model.joblib")
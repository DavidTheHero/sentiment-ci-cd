from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Sentiment Service", version="0.1.0")

MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError("models/model.joblib not found. Run: python scripts/train.py")

model = joblib.load(MODEL_PATH)

class InReq(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.post("/predict")
def predict(req: InReq):
    label = int(model.predict([req.text])[0])
    proba = getattr(model, "predict_proba", lambda X: [[None, None]])([req.text])[0]
    return {"label": label, "proba": [float(p) if p is not None else None for p in proba]}
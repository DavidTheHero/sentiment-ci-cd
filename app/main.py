from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

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

REQS = Counter("inference_requests_total", "Total inference requests")
LAT  = Histogram("inference_latency_seconds", "Inference latency seconds")

@app.post("/predict")
def predict(req: InReq):
    t0 = time.time()
    REQS.inc()
    label = int(model.predict([req.text])[0])
    proba = getattr(model, "predict_proba", lambda X: [[None, None]])([req.text])[0]
    LAT.observe(time.time() - t0)
    return {"label": label, "proba": [float(p) if p is not None else None for p in proba]}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
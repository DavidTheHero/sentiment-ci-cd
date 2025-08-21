from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import joblib, time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Sentiment Service", version="0.1.0")
model = joblib.load("models/model.joblib")

# --- Metrics ---
REQS = Counter("inference_requests_total", "Total inference requests")
LAT  = Histogram("inference_latency_seconds", "Inference latency seconds")

class InReq(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

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
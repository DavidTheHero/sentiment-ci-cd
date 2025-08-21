from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_basic():
    r = client.post("/predict", json={"text": "i love this"})
    assert r.status_code == 200
    body = r.json()
    assert "label" in body and "proba" in body
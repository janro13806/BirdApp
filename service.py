# service.py
from flask import Flask, request, jsonify
import os, requests, io, time
from PIL import Image

MODEL_ID = "chriamue/bird-species-classifier"
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

app = Flask(__name__)

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "backend": "huggingface-inference-api",
        "model": MODEL_ID,
        "auth": "present" if HF_TOKEN else "missing"
    })

def call_hf(image_bytes: bytes, retries: int = 3, timeout: int = 60):
    delay = 2
    for _ in range(retries):
        r = requests.post(API_URL, headers=HEADERS, data=image_bytes, timeout=timeout)
        if r.status_code == 503:
            time.sleep(delay); delay *= 2; continue
        r.raise_for_status()
        return r.json()
    return {"error": "Model loading timeout"}, 503

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    image_bytes = request.files["file"].read()
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    out = call_hf(image_bytes)
    if isinstance(out, tuple):  # error tuple
        payload, code = out
        return jsonify(payload), code

    if not isinstance(out, list) or not out:
        return jsonify({"error": "Unexpected HF response", "raw": out}), 502

    best = max(out, key=lambda x: x.get("score", 0))
    topk = sorted(out, key=lambda x: x.get("score", 0), reverse=True)[:5]
    return jsonify({
        "predicted_class": best.get("label"),
        "confidence": best.get("score"),
        "topK": topk
    })

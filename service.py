# service.py
from flask import Flask, request, jsonify
import os, requests, io, time, json
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "chriamue/bird-species-classifier")
HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
BASE_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/octet-stream",
    "x-wait-for-model": "true",  # wait for cold start instead of immediate 503
}
if HF_TOKEN:
    BASE_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

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
    """
    Call HF Inference API safely.
    Always return (payload, http_status) and never raise.
    """
    delay = 2
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(API_URL, headers=BASE_HEADERS, data=image_bytes, timeout=timeout)
            # 503 while loading – retry with backoff
            if r.status_code == 503:
                last_err = r.text
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue

            # Any non-2xx: return HF's body for visibility
            if r.status_code < 200 or r.status_code >= 300:
                # Try to parse JSON, else return text
                try:
                    body = r.json()
                except Exception:
                    body = {"raw": r.text}
                body.setdefault("error", "HuggingFace request failed")
                body.setdefault("status_code", r.status_code)
                return body, r.status_code

            # Success: parse JSON
            return r.json(), r.status_code

        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(delay)
            delay = min(delay * 2, 16)

    return {"error": "HuggingFace request exception or timeout", "detail": last_err}, 502

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data with key 'file'."}), 400

    image_bytes = request.files["file"].read()
    # Basic sanity check that it’s an image
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    out, code = call_hf(image_bytes)

    # Propagate HF errors transparently
    if code >= 400:
        return jsonify(out), code

    # Expect a list of {label, score}
    if not isinstance(out, list) or not out or not isinstance(out[0], dict):
        return jsonify({"error": "Unexpected HF response", "raw": out}), 502

    # Compute best/topK safely
    try:
        best = max(out, key=lambda x: float(x.get("score", 0.0)))
        topk = sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:5]
    except Exception:
        return jsonify({"error": "Could not interpret HF scores", "raw": out}), 502

    return jsonify({
        "predicted_class": best.get("label"),
        "confidence": float(best.get("score", 0.0)),
        "topK": topk
    }), 200

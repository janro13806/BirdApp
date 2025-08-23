from flask import Flask, request, jsonify
import os, requests, time, io
from PIL import Image

MODEL_ID = "chriamue/bird-species-classifier"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in App Runner
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
    }), 200

def call_hf_inference(image_bytes: bytes, retries: int = 3, timeout: int = 60):
    # Handles model cold start (503 with "Model is loading")
    delay = 2
    for attempt in range(retries):
        r = requests.post(API_URL, headers=HEADERS, data=image_bytes, timeout=timeout)
        if r.status_code == 503:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()
    return {"error": "Model loading timeout after retries"}, 503

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    image_bytes = f.read()

    # Optional sanity check that it’s an image
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    out = call_hf_inference(image_bytes)
    if isinstance(out, tuple):  # (dict, code) from our error return above
        payload, code = out
        return jsonify(payload), code

    if isinstance(out, dict) and "error" in out:
        # Unexpected HF error payload
        return jsonify(out), 502

    if not isinstance(out, list) or not out:
        return jsonify({"error": "Unexpected response from HF", "raw": out}), 502

    best = max(out, key=lambda x: x.get("score", 0))
    topk = sorted(out, key=lambda x: x.get("score", 0), reverse=True)[:5]
    return jsonify({
        "predicted_class": best.get("label"),
        "confidence": best.get("score"),
        "topK": topk
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

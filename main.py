from flask import Flask, request, jsonify
import os, requests, time, io
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "chriamue/bird-species-classifier")
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in App Runner
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Proper headers for binary upload + better DX on cold start
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/octet-stream",
    "x-wait-for-model": "true",  # let HF keep the request open while the model loads
}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

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
    """
    Call Hugging Face Inference API safely.
    - Never raises; always returns (payload, http_status).
    - Retries on 503 (cold start) and transient network errors with backoff.
    """
    delay = 2
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(API_URL, headers=HEADERS, data=image_bytes, timeout=timeout)

            # Model still loading
            if r.status_code == 503:
                last_err = r.text
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue

            # Any non-2xx: surface HF error body
            if not (200 <= r.status_code < 300):
                try:
                    body = r.json()
                except Exception:
                    body = {"raw": r.text}
                body.setdefault("error", "HuggingFace request failed")
                body.setdefault("status_code", r.status_code)
                return body, r.status_code

            # Success
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

    payload, code = call_hf_inference(image_bytes)
    if code != 200:
        # Propagate HF error details (401 token, 403 quota, 503 loading, etc.)
        return jsonify(payload), code

    # Expect a list of {label, score}
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return jsonify({"error": "Unexpected HF response", "raw": payload}), 502

    try:
        best = max(payload, key=lambda x: float(x.get("score", 0.0)))
        topk = sorted(payload, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:5]
    except Exception:
        return jsonify({"error": "Could not interpret HF scores", "raw": payload}), 502

    return jsonify({
        "predicted_class": best.get("label"),
        "confidence": float(best.get("score", 0.0)),
        "topK": topk
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

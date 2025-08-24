from flask import Flask, request, jsonify
import os, requests, time, io
from PIL import Image
import boto3
from typing import Tuple, Dict, Any, Optional

MODEL_ID = os.environ.get("MODEL_ID", "chriamue/bird-species-classifier")
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in App Runner
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
BIRD_MIN_CONF = float(os.getenv("BIRD_MIN_CONF", "0.85"))

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

def _rek():
    # lazy client; created only when /VerifyBirdImage is hit
    return boto3.client("rekognition", region_name=AWS_REGION)

def _verify_with_rekognition(image_bytes: bytes) -> Tuple[Dict[str, Any], int]:
    """
    Check with Amazon Rekognition if the image contains a bird.
    Returns (uniform_body, http_status).
    uniform_body keys: ok, label, confidence, message, error
    """
    try:
        resp = _rek().detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=25,
            MinConfidence=int(BIRD_MIN_CONF * 100)
        )
    except Exception as e:
        return {
            "ok": False,
            "label": None,
            "confidence": 0.0,
            "message": str(e),
            "error": "rekognition_error"
        }, 502

    labels = resp.get("Labels", [])
    best_conf, best_name = 0.0, None

    for lab in labels:
        name = lab.get("Name", "")
        conf = float(lab.get("Confidence", 0.0)) / 100.0
        parents = {p.get("Name", "").lower() for p in lab.get("Parents", [])}
        is_birdish = name.lower() == "bird" or "bird" in parents  # e.g., "Swallow", "Jay"
        if is_birdish and conf > best_conf:
            best_conf, best_name = conf, name

    if best_name and best_conf >= BIRD_MIN_CONF:
        return {
            "ok": True,
            "label": best_name,
            "confidence": best_conf,
            "message": "",
            "error": None
        }, 200

    return {
        "ok": False,
        "label": None,
        "confidence": best_conf,
        "message": f"Doesn’t look like a bird (max confidence {best_conf:.2f}). Try a closer, sharper photo.",
        "error": "not_bird"
    }, 422

@app.route("/VerifyBirdImage", methods=["POST"])
@app.route("/verifybirdimage", methods=["POST"])      # lowercase alias
@app.route("/verify-bird-image", methods=["POST"])    # kebab alias
def verify_bird_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data with key 'file'."}), 400

    image_bytes = request.files["file"].read()
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    payload, code = _verify_with_rekognition(image_bytes)
    return jsonify(payload), code

@app.get("/__routes")
def list_routes():
    return jsonify(sorted([(r.rule, sorted(list(r.methods))) for r in app.url_map.iter_rules()]))

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

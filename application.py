from flask import Flask, request, jsonify
from PIL import Image
import torch
import os

app = Flask(__name__)

# placeholders for later
processor = None
model = None

@app.before_request
def load_model():
    global processor, model
    if processor is None or model is None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        MODEL_ID = "chriamue/bird-species-classifier"
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Bird Identifier API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # ensure model is loaded
    if processor is None or model is None:
        return jsonify({"error": "Model not ready"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    idx = outputs.logits.argmax(-1).item()
    label = model.config.id2label[idx]
    return jsonify({"predicted_class": label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
 

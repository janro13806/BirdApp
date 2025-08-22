from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

app = Flask(__name__)

# Load model and processor once at startup
MODEL_ID = "chriamue/bird-species-classifier"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please use 'file' as the form field."}), 400

    image_file = request.files["file"]
    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    idx = logits.argmax(-1).item()
    label = model.config.id2label[idx]

    return jsonify({"predicted_class": label})

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Bird Identifier API is running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000)
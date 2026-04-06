from __future__ import annotations

import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from classifier import predict_blood_group


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image field"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    raw = file.read()
    if not raw:
        return jsonify({"error": "Empty file"}), 400

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    try:
        result = predict_blood_group(image)
    except ValueError:
        return jsonify({"error": "Unable to process image for blood grouping"}), 400

    return jsonify(result), 200


@app.post("/predict")
def web_predict():
    if "image" not in request.files:
        return render_template("index.html", error="Please upload an image.")
    file = request.files["image"]
    if not file or file.filename == "":
        return render_template("index.html", error="Please select an image file.")

    raw = file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return render_template("index.html", error="Unsupported or invalid image.")

    try:
        result = predict_blood_group(image)
    except ValueError:
        return render_template("index.html", error="Unable to process image for blood grouping.")

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

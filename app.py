"""
app.py
Flask web app for the wound segmentation model.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR  = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

print("Flask app initialised — no ML libraries loaded yet", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    uid         = uuid.uuid4().hex[:10]
    upload_name = f"{uid}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)
    file.save(upload_path)

    try:
        # Deferred import — torch/smp load only on first request
        import cv2
        from model import run_inference

        result = run_inference(upload_path)

        overlay_name = f"{uid}_overlay.png"
        mask_name    = f"{uid}_mask.png"
        heatmap_name = f"{uid}_heatmap.png"

        cv2.imwrite(os.path.join(RESULT_DIR, overlay_name), result["overlay"])
        cv2.imwrite(os.path.join(RESULT_DIR, mask_name),    result["mask"])
        cv2.imwrite(os.path.join(RESULT_DIR, heatmap_name), result["heatmap"])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "original_url": url_for("static", filename=f"uploads/{upload_name}"),
        "overlay_url":  url_for("static", filename=f"results/{overlay_name}"),
        "mask_url":     url_for("static", filename=f"results/{mask_name}"),
        "heatmap_url":  url_for("static", filename=f"results/{heatmap_name}"),
        "wound_area_pct": result["wound_area_pct"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

"""
VisionSpec QC – Flask Backend
==============================
Endpoints:
  POST /predict   → Returns label, confidence, heatmap
  POST /heatmap   → Returns only Grad-CAM overlay
  GET  /health    → Health check
  GET  /           → Serves frontend

Run:
  python app.py
"""

import os
import io
import json
import uuid
import base64
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
import cv2

# ─── Local imports ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import load_image_for_inference, IMG_SIZE
from src.gradcam import GradCAM

# ─── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("visionspec")

UPLOAD_DIR   = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Lazy model loading ────────────────────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        import tensorflow as tf
        from src.predict import Predictor

        # Find best available model (prefer mobilenet)
        candidates = [
            "models/mobilenet_final.keras",
            "models/mobilenet_best.keras",
            "models/custom_cnn_final.keras",
            "models/custom_cnn_best.keras",
        ]
        model_path = None
        for c in candidates:
            if Path(c).exists():
                model_path = c
                break

        if model_path is None:
            raise RuntimeError(
                "No trained model found. Run: python src/train.py --model mobilenet"
            )

        log.info(f"Loading model: {model_path}")
        _predictor = Predictor(model_path)
        log.info("Model ready ✓")

    return _predictor


# ─── Helpers ───────────────────────────────────────────────────────────────
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def _validate_image(file) -> bool:
    return Path(file.filename).suffix.lower() in ALLOWED_EXT

def _read_image_from_request(file) -> np.ndarray:
    """Read uploaded file → numpy array (1, 224, 224, 3) float32."""
    pil_img = Image.open(file.stream).convert("RGB")
    img     = np.array(pil_img, dtype=np.uint8)  # Keep as uint8 first
    img     = cv2.resize(img, IMG_SIZE)
    img     = img.astype(np.float32) / 255.0  # Now normalize
    return np.expand_dims(img, axis=0)

def _save_upload(file) -> Path:
    """Save upload with unique name, return path."""
    ext  = Path(file.filename).suffix.lower()
    name = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / name
    file.stream.seek(0)
    file.save(str(path))
    return path


# ─── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the frontend SPA."""
    return send_from_directory("frontend", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: multipart/form-data with key 'image'

    Returns JSON:
    {
      "label"          : "DEFECT" | "PASS",
      "confidence"     : 87.34,          # percent
      "raw_prob"       : 0.8734,
      "heatmap_b64"    : "...",           # base64 PNG
      "comparison_b64" : "...",           # base64 PNG (3-panel)
      "timestamp"      : "..."
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    if not _validate_image(file):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXT}"}), 415

    try:
        predictor = get_predictor()
        img_array = _read_image_from_request(file)
        result    = predictor.predict_array(img_array)

        # Grad-CAM
        heatmap_b64    = predictor.gradcam.overlay_to_base64(img_array)
        comparison_b64 = predictor.gradcam.comparison_to_base64(
            img_array,
            label      = result["label"],
            confidence = result["confidence"] / 100,
        )

        response = {
            **result,
            "heatmap_b64"    : heatmap_b64,
            "comparison_b64" : comparison_b64,
            "timestamp"      : datetime.utcnow().isoformat(),
        }
        log.info(f"[/predict] {result['label']} ({result['confidence']:.1f}%)")
        return jsonify(response)

    except RuntimeError as e:
        log.error(f"Model error: {e}")
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


@app.route("/heatmap", methods=["POST"])
def heatmap_only():
    """
    POST /heatmap
    Returns only the Grad-CAM heatmap as a PNG file (binary response).
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    if not _validate_image(file):
        return jsonify({"error": "Unsupported file type."}), 415

    try:
        predictor = get_predictor()
        img_array = _read_image_from_request(file)
        overlay   = predictor.gradcam.overlay(img_array)          # BGR ndarray
        _, buf    = cv2.imencode(".png", overlay)
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype     = "image/png",
            as_attachment = False,
        )

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ─── Real-time simulation endpoint (SSE or polling) ───────────────────────
@app.route("/stream_predict", methods=["POST"])
def stream_predict():
    """
    Simulates a high-throughput industrial inspection pipeline.
    Returns JSON array of results for up to 5 images.
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided."}), 400

    results = []
    try:
        predictor = get_predictor()
        for file in files[:5]:          # cap at 5 for demo
            if not _validate_image(file):
                continue
            img_array = _read_image_from_request(file)
            result    = predictor.predict_array(img_array)
            result["heatmap_b64"] = predictor.gradcam.overlay_to_base64(img_array)
            results.append(result)

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    return jsonify({"results": results, "count": len(results)})


# ─── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  default=5000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.info("="*50)
    log.info("  VisionSpec QC – API Server")
    log.info(f"  http://{args.host}:{args.port}")
    log.info("="*50)

    app.run(host=args.host, port=args.port, debug=args.debug)

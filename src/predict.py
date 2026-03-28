"""
VisionSpec QC – Prediction Module
====================================
Loads a trained Keras model and runs inference on single images
or directories (batch mode).

Usage:
  python src/predict.py --model models/mobilenet_final.keras \
                        --image path/to/img.jpg

  python src/predict.py --model models/mobilenet_final.keras \
                        --dir   data/splits/test/DEFECT
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import load_image_for_inference
from src.gradcam import GradCAM

# ─── Label helpers ────────────────────────────────────────────────────────────
LABELS     = {0: "PASS", 1: "DEFECT"}
THRESHOLD  = 0.5


def _prob_to_result(prob: float) -> dict:
    """Convert raw sigmoid probability to labelled result dict."""
    label      = "DEFECT" if prob > THRESHOLD else "PASS"
    confidence = float(prob) if label == "DEFECT" else float(1.0 - prob)
    return {
        "label"      : label,
        "confidence" : round(confidence * 100, 2),   # percent
        "raw_prob"   : round(float(prob), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predictor class (used by Flask app)
# ─────────────────────────────────────────────────────────────────────────────
class Predictor:
    """
    Stateful predictor: loads model once, serves many requests efficiently.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model      = self._load(model_path)
        self.gradcam    = GradCAM(self.model)
        print(f"[Predictor] Model loaded: {model_path}")

    @staticmethod
    def _load(path: str) -> tf.keras.Model:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = tf.keras.models.load_model(str(path))
        # Warm-up pass to avoid first-call latency
        dummy = np.zeros((1, 224, 224, 3), dtype="float32")
        model.predict(dummy, verbose=0)
        return model

    def predict(self, image_path: str) -> dict:
        """Predict a single image. Returns result dict."""
        img  = load_image_for_inference(image_path)
        prob = float(self.model.predict(img, verbose=0)[0][0])
        return {**_prob_to_result(prob), "image_path": str(image_path)}

    def predict_array(self, img_array: np.ndarray) -> dict:
        """Predict directly from a (1, H, W, 3) numpy array."""
        prob = float(self.model.predict(img_array, verbose=0)[0][0])
        return _prob_to_result(prob)

    def predict_with_heatmap(self, image_path: str) -> dict:
        """
        Predict + generate Grad-CAM.
        Returns result dict with 'heatmap_b64' and 'comparison_b64' keys.
        """
        img    = load_image_for_inference(image_path)
        prob   = float(self.model.predict(img, verbose=0)[0][0])
        result = _prob_to_result(prob)

        heatmap_b64    = self.gradcam.overlay_to_base64(img)
        comparison_b64 = self.gradcam.comparison_to_base64(
            img,
            label      = result["label"],
            confidence = result["confidence"] / 100,
        )

        return {
            **result,
            "heatmap_b64"    : heatmap_b64,
            "comparison_b64" : comparison_b64,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_directory(model_path: str, dir_path: str) -> list:
    """Run inference on all images in a directory."""
    predictor = Predictor(model_path)
    exts      = {".jpg", ".jpeg", ".png", ".bmp"}
    images    = [p for p in Path(dir_path).rglob("*") if p.suffix.lower() in exts]

    results   = []
    for img_path in images:
        try:
            res = predictor.predict(str(img_path))
            results.append(res)
            print(f"  {img_path.name:<40} → {res['label']:<7} ({res['confidence']:.1f}%)")
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSpec Inference")
    parser.add_argument("--model",  required=True, help="Path to .keras model file")
    parser.add_argument("--image",  default=None,  help="Path to single image")
    parser.add_argument("--dir",    default=None,  help="Directory of images (batch)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also generate Grad-CAM output")
    parser.add_argument("--output", default=None,
                        help="Save prediction JSON to file")
    args = parser.parse_args()

    if args.image:
        predictor = Predictor(args.model)
        if args.heatmap:
            result = predictor.predict_with_heatmap(args.image)
            # Don't print base64 blobs to terminal
            display = {k: v for k, v in result.items()
                       if not k.endswith("_b64")}
            print(json.dumps(display, indent=2))
        else:
            result = predictor.predict(args.image)
            print(json.dumps(result, indent=2))

        if args.output:
            with open(args.output, "w") as fh:
                json.dump({k: v for k, v in result.items()
                           if not k.endswith("_b64")}, fh, indent=2)

    elif args.dir:
        results = predict_directory(args.model, args.dir)
        if args.output:
            with open(args.output, "w") as fh:
                json.dump(results, fh, indent=2)
        print(f"\n[BATCH] Total: {len(results)}")
        labels   = [r["label"] for r in results]
        n_defect = labels.count("DEFECT")
        n_pass   = labels.count("PASS")
        print(f"  PASS   : {n_pass}")
        print(f"  DEFECT : {n_defect}")

    else:
        parser.print_help()

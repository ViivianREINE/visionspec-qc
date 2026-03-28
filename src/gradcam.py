"""
VisionSpec QC – Grad-CAM Module
=================================
Generates Grad-CAM heatmaps to visualise defect regions.

Supports both custom CNN (last conv layer) and MobileNetV2.

Usage:
  from src.gradcam import GradCAM
  cam = GradCAM(model)
  heatmap_img = cam.overlay(image_array)   # Returns BGR numpy array
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64


class GradCAM:
    """
    Class-activation mapping using gradient information.
    Works with any Keras model that has convolutional layers.
    """

    # Known last conv layer names for supported architectures
    LAYER_MAP = {
        "VisionSpec_CNN"         : "conv4_1",
        "VisionSpec_MobileNetV2" : "out_relu",        # MobileNetV2 last activation
    }

    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        self.model      = model
        self.layer_name = layer_name or self._infer_layer()
        self.grad_model = self._build_grad_model()
        print(f"[GradCAM] Using layer: '{self.layer_name}'")

    # ── Layer detection ───────────────────────────────────────────────────────
    def _infer_layer(self) -> str:
        # Try named lookup first
        name = self.LAYER_MAP.get(self.model.name)
        if name:
            try:
                self.model.get_layer(name)
                return name
            except ValueError:
                pass

        # Walk backwards to find last Conv2D
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        # For MobileNetV2 wrapped in a functional model
        for layer in reversed(self.model.layers):
            if hasattr(layer, "layers"):            # nested model
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        return sub.name

        raise ValueError("Could not find a Conv2D layer in the model.")

    def _build_grad_model(self) -> tf.keras.Model:
        """Build a sub-model that outputs (feature_maps, predictions)."""
        try:
            target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            # Layer may be inside a nested base model
            for layer in self.model.layers:
                if hasattr(layer, "get_layer"):
                    try:
                        target_layer = layer.get_layer(self.layer_name)
                        # Rebuild with nested outputs
                        grad_model = tf.keras.Model(
                            inputs  = self.model.inputs,
                            outputs = [layer.get_layer(self.layer_name).output,
                                       self.model.output],
                        )
                        return grad_model
                    except ValueError:
                        continue
            raise

        return tf.keras.Model(
            inputs  = self.model.inputs,
            outputs = [target_layer.output, self.model.output],
        )

    # ── Core Grad-CAM computation ──────────────────────────────────────────────
    def compute_heatmap(self, img_array: np.ndarray,
                        class_idx: int = 0) -> np.ndarray:
        """
        img_array : (1, H, W, 3), float32, values in [0, 1]
        Returns    : normalised heatmap (H, W), float32 in [0, 1]
        """
        with tf.GradientTape() as tape:
            img_tensor   = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = self.grad_model(img_tensor)
            loss = predictions[:, class_idx]

        grads      = tape.gradient(loss, conv_outputs)         # (1, h, w, C)
        pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)
        cam        = tf.reduce_sum(tf.multiply(conv_outputs[0], pooled), axis=-1)

        # ReLU + normalise
        cam = tf.nn.relu(cam).numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    # ── Overlay helpers ────────────────────────────────────────────────────────
    def overlay(self,
                img_array  : np.ndarray,
                alpha      : float = 0.45,
                colormap   : int   = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Generate a heatmap overlay on the original image.

        img_array : (1, H, W, 3) float32 [0,1]  OR  (H, W, 3) uint8
        Returns   : (H, W, 3) uint8 BGR overlay
        """
        # Normalise input to (H, W, 3) uint8
        if img_array.ndim == 4:
            img_disp = (img_array[0] * 255).astype(np.uint8)
        else:
            img_disp = img_array.astype(np.uint8) if img_array.max() > 1 \
                       else (img_array * 255).astype(np.uint8)

        # Convert to BGR for OpenCV ops
        img_bgr  = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
        H, W     = img_bgr.shape[:2]

        # Compute and resize heatmap
        img_float   = img_array if img_array.ndim == 4 \
                      else np.expand_dims(img_disp / 255.0, 0).astype("float32")
        heatmap     = self.compute_heatmap(img_float)
        heatmap_u8  = (heatmap * 255).astype(np.uint8)
        heatmap_rsz = cv2.resize(heatmap_u8, (W, H))
        heatmap_col = cv2.applyColorMap(heatmap_rsz, colormap)

        # Blend
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_col, alpha, 0)
        return overlay                          # BGR uint8

    def overlay_rgb(self, img_array: np.ndarray, **kwargs) -> np.ndarray:
        """Same as overlay but returns RGB."""
        bgr = self.overlay(img_array, **kwargs)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Encode to base64 PNG (for API) ────────────────────────────────────────
    def overlay_to_base64(self, img_array: np.ndarray, **kwargs) -> str:
        """Returns base64-encoded PNG string of the overlay."""
        overlay_bgr = self.overlay(img_array, **kwargs)
        _, buf = cv2.imencode(".png", overlay_bgr)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ── Side-by-side comparison plot ──────────────────────────────────────────
    def comparison_plot(self,
                        img_array   : np.ndarray,
                        label       : str   = "",
                        confidence  : float = None,
                        save_path   : str   = None) -> np.ndarray:
        """
        3-panel plot: Original | Heatmap | Overlay
        Returns the figure as an RGB numpy array.
        """
        if img_array.ndim == 4:
            img_disp = (img_array[0] * 255).astype(np.uint8)
        else:
            img_disp = img_array

        heatmap     = self.compute_heatmap(
            img_array if img_array.ndim == 4
            else np.expand_dims(img_disp / 255.0, 0).astype("float32")
        )
        H, W        = img_disp.shape[:2]
        heatmap_rsz = cv2.resize((heatmap * 255).astype(np.uint8), (W, H))
        heatmap_col = cv2.applyColorMap(heatmap_rsz, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)

        img_bgr     = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(img_bgr, 0.55, heatmap_col, 0.45, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        fig, axes   = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#0d0d1a")

        title_kw = dict(color="white", fontsize=11, fontweight="bold", pad=8)
        panels   = [("Original", img_disp), ("Grad-CAM", heatmap_rgb), ("Overlay", overlay_rgb)]
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, **title_kw)
            ax.axis("off")
            ax.set_facecolor("#0d0d1a")

        sup = label
        if confidence is not None:
            sup += f"  |  Confidence: {confidence*100:.1f}%"
        if sup:
            fig.suptitle(sup, color="#00ffe5", fontsize=13,
                         fontweight="bold", y=1.02)

        plt.tight_layout()

        # Render to numpy
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        fig_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_BGR2RGB)
        plt.close(fig)

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))
            print(f"[GradCAM] Comparison saved → {save_path}")

        return fig_img

    def comparison_to_base64(self, img_array: np.ndarray, **kwargs) -> str:
        """Returns base64 PNG of the 3-panel comparison."""
        rgb = self.comparison_plot(img_array, **kwargs)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", bgr)
        return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from src.preprocessing import load_image_for_inference

    parser = argparse.ArgumentParser(description="Grad-CAM Demo")
    parser.add_argument("--model",  required=True, help="Path to .keras model")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--output", default="notebooks/gradcam_demo.png")
    args = parser.parse_args()

    model  = tf.keras.models.load_model(args.model)
    img    = load_image_for_inference(args.image)
    cam    = GradCAM(model)
    prob   = float(model.predict(img, verbose=0)[0][0])
    label  = "DEFECT" if prob > 0.5 else "PASS"
    conf   = prob if prob > 0.5 else 1 - prob

    cam.comparison_plot(img, label=label, confidence=conf,
                        save_path=args.output)
    print(f"[DEMO] Prediction: {label}  Confidence: {conf*100:.1f}%")
    print(f"[DEMO] Saved → {args.output}")

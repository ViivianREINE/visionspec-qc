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
        "VisionSpec_MobileNetV2" : "mobilenetv2_1.00_224",
    }

    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        self.model      = model
        self.layer_name = layer_name or self._infer_layer()
        self.grad_model = self._build_grad_model()
        print(f"[GradCAM] Using layer: '{self.layer_name}'")

    # ── Layer detection ───────────────────────────────────────────────────────
    def _find_target_layer(self, layer_name: str):
        """Return a model layer by name, searching nested models if needed."""
        try:
            return self.model.get_layer(layer_name)
        except ValueError:
            pass

        for layer in self.model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    return layer.get_layer(layer_name)
                except (ValueError, AttributeError):
                    continue

        return None

    def _infer_layer(self) -> str:
        # Prefer a hard-coded name for known model variants.
        name = self.LAYER_MAP.get(self.model.name)
        if name:
            target = self._find_target_layer(name)
            if target is not None:
                return name

        # Try common MobileNetV2 layer names.
        mobilenet_candidates = [
            "mobilenetv2_1.00_224",
            "out_relu",
            "re_lu_1",
            "activation",
            "relu",
        ]
        for candidate in mobilenet_candidates:
            if self._find_target_layer(candidate) is not None:
                return candidate

        # Prefer the nested MobileNet base model if present.
        for layer in self.model.layers:
            if hasattr(layer, "layers") and "mobilenet" in layer.name.lower():
                output_shape = getattr(layer, "output_shape", None)
                if output_shape is not None and len(output_shape) == 4:
                    return layer.name

        # Walk backwards to find the last Conv2D layer.
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        # Fall back to the last Conv2D inside nested models.
        for layer in reversed(self.model.layers):
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        return sub.name

        raise ValueError("Could not find a suitable layer in the model for Grad-CAM")

    def _build_grad_model(self) -> tf.keras.Model:
        """Build a sub-model that outputs (feature_maps, predictions)."""
        target_layer = self._find_target_layer(self.layer_name)
        if target_layer is None:
            raise ValueError(f"Could not find layer '{self.layer_name}' in model")

        try:
            grad_model = tf.keras.Model(
                inputs  = self.model.inputs,
                outputs = [target_layer.output, self.model.output],
            )
            # Test the model to ensure it works
            test_input = np.zeros((1, 224, 224, 3), dtype="float32")
            _ = grad_model(test_input, training=False)
            return grad_model
        except Exception as e:
            print(f"[GradCAM] Error building grad model: {e}")
            print(f"[GradCAM] Layer name: {self.layer_name}")
            print(f"[GradCAM] Target layer: {type(target_layer).__name__} {getattr(target_layer, 'name', '<unknown>')}\n")
            print(f"[GradCAM] Model layers: {[l.name for l in self.model.layers]}")
            raise

    # ── Core Grad-CAM computation ──────────────────────────────────────────────
    def compute_heatmap(self, img_array: np.ndarray,
                        class_idx: int = 0) -> np.ndarray:
        """
        img_array : (1, H, W, 3), float32, values in [0, 1]
        Returns    : normalised heatmap (H, W), float32 in [0, 1]
        """
        try:
            with tf.GradientTape() as tape:
                img_tensor = tf.cast(img_array, tf.float32)
                tape.watch(img_tensor)
                
                # Call grad_model in training mode for better gradient flow
                conv_outputs, predictions = self.grad_model(img_tensor, training=True)
                loss = predictions[:, class_idx]
            
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is None:
                print("[GradCAM] Warning: gradients are None, using fallback")
                # Fallback: return a simple heatmap based on feature maps
                cam = tf.reduce_mean(tf.abs(conv_outputs), axis=-1)[0].numpy()
            else:
                # Standard Grad-CAM
                pooled = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
                cam = tf.reduce_sum(tf.multiply(conv_outputs[0], pooled), axis=-1)
                cam = tf.nn.relu(cam).numpy()
            
            # Normalize
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = cam / (cam.max() + 1e-8)
            
            return cam
            
        except Exception as e:
            print(f"[GradCAM] Error in compute_heatmap: {e}")
            # Return a default heatmap
            return np.ones((img_array.shape[1], img_array.shape[2]), dtype=np.float32) * 0.5

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

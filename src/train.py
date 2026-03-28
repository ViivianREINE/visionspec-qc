"""
VisionSpec QC – Training Module
================================
Trains two models:
  1. Custom CNN from scratch
  2. MobileNetV2 transfer learning (primary)

Usage:
  python src/train.py --model mobilenet   # recommended
  python src/train.py --model cnn
  python src/train.py --model both
"""

import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import build_splits, get_generators, SEED

# ─── Reproducibility ──────────────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Directories ──────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
LOGS_DIR   = Path("logs")
PLOTS_DIR  = Path("notebooks")
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

IMG_SIZE   = (224, 224)
EPOCHS     = 30
BATCH_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM CNN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def build_custom_cnn(input_shape=(224, 224, 3)) -> Model:
    """
    Custom 6-block CNN designed for binary defect classification.
    Architecture: Conv → BN → ReLU → Pool × 4, then Dense head.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, 3, padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, 3, padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)
    x = layers.Dropout(0.30, name="drop4")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.50, name="drop_fc")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="VisionSpec_CNN")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOBILENETV2 TRANSFER LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def build_mobilenet(input_shape=(224, 224, 3), fine_tune_at: int = 100) -> Model:
    """
    MobileNetV2 with frozen base + custom classification head.
    Optionally fine-tunes layers above `fine_tune_at`.
    """
    base = MobileNetV2(
        input_shape = input_shape,
        include_top = False,
        weights     = "imagenet",
        alpha       = 1.0,
    )
    # Freeze base
    base.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.40, name="drop1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.30, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="VisionSpec_MobileNetV2")
    return model


def unfreeze_for_fine_tuning(model: Model, fine_tune_at: int = 100) -> Model:
    """Unfreeze top layers of MobileNetV2 base for fine-tuning."""
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"[FINETUNE] Unfroze layers {fine_tune_at}+ in MobileNetV2 base.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def get_callbacks(model_name: str, patience: int = 7):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_path  = str(MODELS_DIR / f"{model_name}_best.keras")
    log_dir    = str(LOGS_DIR / f"{model_name}_{timestamp}")

    callbacks = [
        ModelCheckpoint(
            filepath         = ckpt_path,
            monitor          = "val_accuracy",
            save_best_only   = True,
            verbose          = 1,
        ),
        EarlyStopping(
            monitor          = "val_loss",
            patience         = patience,
            restore_best_weights = True,
            verbose          = 1,
        ),
        ReduceLROnPlateau(
            monitor          = "val_loss",
            factor           = 0.5,
            patience         = 4,
            min_lr           = 1e-7,
            verbose          = 1,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]
    return callbacks, ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPILE & TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def compile_model(model: Model, lr: float = 1e-3) -> Model:
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_model(model: Model, train_gen, val_gen, model_name: str,
                epochs: int = EPOCHS, lr: float = 1e-3):
    model = compile_model(model, lr=lr)
    model.summary()

    callbacks, ckpt_path = get_callbacks(model_name)

    history = model.fit(
        train_gen,
        epochs          = epochs,
        validation_data = val_gen,
        callbacks       = callbacks,
        verbose         = 1,
    )
    print(f"\n[TRAIN] ✓ Best model saved to: {ckpt_path}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model: Model, test_gen, model_name: str) -> dict:
    """Run full evaluation and print/save metrics."""
    print(f"\n[EVAL] Evaluating {model_name} on test set…")
    test_gen.reset()

    # Predictions
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = test_gen.classes

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Model       : {model_name}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"{'='*50}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['PASS','DEFECT'])}")

    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

    # Save metrics JSON
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Confusion matrix plot
    _plot_confusion_matrix(y_true, y_pred, model_name)

    return metrics


def _plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["PASS", "DEFECT"],
                yticklabels=["PASS", "DEFECT"], ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    save_path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[EVAL] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, model_name: str):
    """Plot train vs validation accuracy and loss."""
    h = history.history
    epochs = range(1, len(h["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves – {model_name}", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(epochs, h["accuracy"],     "b-o", label="Train Acc",  markersize=4)
    axes[0].plot(epochs, h["val_accuracy"], "r-o", label="Val Acc",    markersize=4)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, h["loss"],     "b-o", label="Train Loss", markersize=4)
    axes[1].plot(epochs, h["val_loss"], "r-o", label="Val Loss",   markersize=4)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = PLOTS_DIR / f"{model_name}_training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(model_type: str = "mobilenet"):
    print("\n" + "="*60)
    print("  VisionSpec QC – Training Pipeline")
    print("="*60)

    # Load data
    split_paths = build_splits()
    train_gen, val_gen, test_gen = get_generators(split_paths, BATCH_SIZE)

    results = {}

    if model_type in ("cnn", "both"):
        print("\n[PIPELINE] ──── Training Custom CNN ────")
        cnn = build_custom_cnn()
        cnn, hist = train_model(cnn, train_gen, val_gen, "custom_cnn")
        plot_training_curves(hist, "custom_cnn")
        metrics = evaluate_model(cnn, test_gen, "custom_cnn")
        cnn.save(str(MODELS_DIR / "custom_cnn_final.keras"))
        results["cnn"] = metrics

    if model_type in ("mobilenet", "both"):
        print("\n[PIPELINE] ──── Training MobileNetV2 ────")
        mob = build_mobilenet()
        mob, hist = train_model(mob, train_gen, val_gen, "mobilenet", lr=1e-3)
        plot_training_curves(hist, "mobilenet")

        # Optional fine-tuning phase
        print("\n[PIPELINE] ──── Fine-tuning MobileNetV2 ────")
        mob = unfreeze_for_fine_tuning(mob)
        mob, hist_ft = train_model(mob, train_gen, val_gen, "mobilenet_ft",
                                   epochs=10, lr=1e-5)
        plot_training_curves(hist_ft, "mobilenet_ft")
        metrics = evaluate_model(mob, test_gen, "mobilenet")
        mob.save(str(MODELS_DIR / "mobilenet_final.keras"))
        results["mobilenet"] = metrics

    print("\n[PIPELINE] ✓ All training complete.")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSpec Training")
    parser.add_argument("--model", choices=["cnn", "mobilenet", "both"],
                        default="mobilenet", help="Which model to train")
    args = parser.parse_args()
    run_pipeline(args.model)

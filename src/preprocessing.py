"""
VisionSpec QC – Preprocessing Module
=====================================
Handles: RAR extraction, dataset loading, augmentation, train/val/test splits.
Author : VisionSpec Team
"""

import os
import random
import shutil
import zipfile
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
DATA_ROOT   = Path("data")
SPLIT_DIR   = DATA_ROOT / "splits"
CLASS_MAP   = {"Positive": "DEFECT", "Negative": "PASS"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_rar(rar_path: str, dest: str = "data/raw") -> Path:
    """
    Extract a .rar file to dest directory.
    Falls back to patool if rarfile is unavailable.
    """
    rar_path = Path(rar_path)
    dest     = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not rar_path.exists():
        raise FileNotFoundError(f"RAR file not found: {rar_path}")

    print(f"[EXTRACT] Extracting {rar_path.name} → {dest}")

    # Try rarfile first (native)
    try:
        import rarfile
        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(str(dest))
        print("[EXTRACT] ✓ Done via rarfile")
        return dest

    except Exception as e_rar:
        print(f"[EXTRACT] rarfile failed ({e_rar}), trying patool…")

    # Fallback: patool (supports more formats)
    try:
        import patoollib
        patoollib.extract_archive(str(rar_path), outdir=str(dest))
        print("[EXTRACT] ✓ Done via patool")
        return dest

    except Exception as e_pat:
        raise RuntimeError(
            f"Could not extract {rar_path}.\n"
            f"rarfile error : {e_rar}\n"
            f"patool error  : {e_pat}\n"
            "Make sure 'unrar' or 'bsdtar' is installed on your system."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def _collect_images(class_dir: Path) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [p for p in class_dir.rglob("*") if p.suffix.lower() in exts]


def build_splits(raw_dir: str = "data/raw", force: bool = False) -> dict:
    """
    Scan raw_dir for Positive/Negative folders, create train/val/test splits
    inside data/splits/{train,val,test}/{DEFECT,PASS}/.
    Returns paths dict.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    if SPLIT_DIR.exists() and not force:
        print("[SPLIT] Splits already exist. Use force=True to rebuild.")
        return _split_paths()

    # Clean old splits
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    for orig_cls, mapped_cls in CLASS_MAP.items():
        src_dir = raw_dir / orig_cls
        # Also search inside subdirectories (extraction may add a subfolder)
        if not src_dir.exists():
            candidates = list(raw_dir.rglob(orig_cls))
            if candidates:
                src_dir = candidates[0]
            else:
                print(f"[SPLIT] WARNING: class dir '{orig_cls}' not found, skipping.")
                continue

        images = _collect_images(src_dir)
        random.shuffle(images)

        n          = len(images)
        n_train    = int(n * TRAIN_SPLIT)
        n_val      = int(n * VAL_SPLIT)

        splits_data = {
            "train": images[:n_train],
            "val"  : images[n_train : n_train + n_val],
            "test" : images[n_train + n_val :],
        }

        print(f"[SPLIT] {orig_cls} → {mapped_cls}: "
              f"{n} images  train={n_train}  val={n_val}  test={n - n_train - n_val}")

        for split_name, split_imgs in splits_data.items():
            dest = SPLIT_DIR / split_name / mapped_cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(split_imgs, desc=f"  {split_name}/{mapped_cls}", leave=False):
                shutil.copy2(img_path, dest / img_path.name)

    print("[SPLIT] ✓ Dataset split complete.")
    return _split_paths()


def _split_paths() -> dict:
    return {
        split: str(SPLIT_DIR / split)
        for split in ("train", "val", "test")
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def get_generators(split_paths: dict, batch_size: int = BATCH_SIZE):
    """
    Returns Keras ImageDataGenerators for train / val / test.
    """
    train_datagen = ImageDataGenerator(
        rescale          = 1.0 / 255.0,
        rotation_range   = 20,
        zoom_range       = 0.15,
        brightness_range = [0.8, 1.2],
        horizontal_flip  = True,
        fill_mode        = "nearest",
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    def _flow(datagen, path, shuffle):
        return datagen.flow_from_directory(
            path,
            target_size = IMG_SIZE,
            batch_size  = batch_size,
            class_mode  = "binary",
            shuffle     = shuffle,
            seed        = SEED,
        )

    train_gen = _flow(train_datagen,        split_paths["train"], shuffle=True)
    val_gen   = _flow(val_test_datagen,     split_paths["val"],   shuffle=False)
    test_gen  = _flow(val_test_datagen,     split_paths["test"],  shuffle=False)

    # Print class indices for reference
    print(f"[GEN] Class indices: {train_gen.class_indices}")
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# 4. SINGLE-IMAGE LOADER (for inference)
# ─────────────────────────────────────────────────────────────────────────────
def load_image_for_inference(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for model prediction.
    Returns shape (1, 224, 224, 3), values [0, 1].
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisionSpec Preprocessing")
    parser.add_argument("--rar",     type=str, help="Path to .rar dataset file")
    parser.add_argument("--raw",     type=str, default="data/raw",
                        help="Directory where RAR is extracted")
    parser.add_argument("--force",   action="store_true",
                        help="Force rebuild of splits")
    args = parser.parse_args()

    if args.rar:
        extract_rar(args.rar, args.raw)

    paths = build_splits(args.raw, force=args.force)
    train_gen, val_gen, test_gen = get_generators(paths)
    print(f"\n[DONE] Generators ready.")
    print(f"  Train batches : {len(train_gen)}")
    print(f"  Val   batches : {len(val_gen)}")
    print(f"  Test  batches : {len(test_gen)}")

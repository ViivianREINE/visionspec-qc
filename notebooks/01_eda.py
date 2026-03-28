"""
VisionSpec QC – Notebook 01: Exploratory Data Analysis
=======================================================
Run: python notebooks/01_eda.py
Outputs plots to notebooks/ directory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from collections import Counter
from tqdm import tqdm

from src.preprocessing import build_splits, _collect_images, DATA_ROOT, CLASS_MAP

OUT = Path("notebooks")
OUT.mkdir(exist_ok=True)

def run_eda():
    print("="*55)
    print("  VisionSpec QC – Exploratory Data Analysis")
    print("="*55)

    # ── 1. Count classes ────────────────────────────────────
    raw_dir = Path("data/raw")
    class_counts = {}
    for orig, mapped in CLASS_MAP.items():
        src = raw_dir / orig
        if not src.exists():
            matches = list(raw_dir.rglob(orig))
            if matches: src = matches[0]
            else: continue
        n = len(_collect_images(src))
        class_counts[mapped] = n
        print(f"  {mapped:10}: {n} images")

    total = sum(class_counts.values())
    print(f"  {'TOTAL':10}: {total} images\n")

    # ── 2. Class distribution bar ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#00f97a", "#ff3a5c"]
    bars    = ax.bar(list(class_counts.keys()), list(class_counts.values()),
                     color=colors, width=0.5, edgecolor="#0d0d1a", linewidth=1.5)
    for bar, val in zip(bars, class_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f"{val:,}\n({val/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_facecolor("#050810")
    fig.patch.set_facecolor("#050810")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#1a2040")
    ax.set_title("Class Distribution", color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", color="#7a8fb5")
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("  [EDA] Class distribution saved.")

    # ── 3. Sample grid ───────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#050810")
    gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.05)

    row_idx = 0
    for orig, mapped in CLASS_MAP.items():
        src = raw_dir / orig
        if not src.exists():
            matches = list(raw_dir.rglob(orig))
            if not matches: continue
            src = matches[0]

        samples = _collect_images(src)[:6]
        for col_idx, img_path in enumerate(samples):
            ax  = fig.add_subplot(gs[row_idx, col_idx])
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(mapped, color="#00ffe5" if mapped == "PASS" else "#ff3a5c",
                             fontsize=9, fontweight="bold", pad=3)
        row_idx += 1

    fig.suptitle("Sample Images per Class", color="white", fontsize=13, fontweight="bold")
    plt.savefig(OUT / "sample_grid.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("  [EDA] Sample grid saved.")

    # ── 4. Pixel intensity distribution ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#050810")
    ax.set_facecolor("#080d1a")

    colors_map = {"PASS": "#00f97a", "DEFECT": "#ff3a5c"}
    for orig, mapped in CLASS_MAP.items():
        src = raw_dir / orig
        if not src.exists():
            matches = list(raw_dir.rglob(orig))
            if not matches: continue
            src = matches[0]

        samples  = _collect_images(src)[:100]
        all_vals = []
        for p in tqdm(samples, desc=f"  Intensity {mapped}", leave=False):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_vals.extend(img.ravel().tolist())

        ax.hist(all_vals, bins=60, alpha=0.6, color=colors_map[mapped],
                label=mapped, density=True)

    ax.legend(facecolor="#050810", labelcolor="white", fontsize=9)
    ax.set_title("Pixel Intensity Distribution", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel Value (0–255)", color="#7a8fb5")
    ax.set_ylabel("Density",            color="#7a8fb5")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#1a2040")
    plt.tight_layout()
    plt.savefig(OUT / "pixel_distribution.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("  [EDA] Pixel distribution saved.")

    print("\n  ✓ EDA complete. Outputs → notebooks/")


if __name__ == "__main__":
    run_eda()

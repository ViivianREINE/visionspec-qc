# 🔬 VisionSpec QC — AI-Based PCB & Surface Defect Detection

> **Production-grade** computer vision pipeline for industrial quality control.  
> MobileNetV2 + Grad-CAM · Flask API · Glassmorphic Web UI

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=flat-square)

---

## 📁 Project Structure

```
visionspec/
├── data/
│   ├── raw/            ← Extracted dataset (Positive / Negative)
│   ├── splits/         ← Auto-generated train / val / test
│   └── uploads/        ← Flask API uploads (temp)
├── notebooks/
│   ├── 01_eda.py       ← Exploratory Data Analysis
│   └── *.png           ← Saved plots
├── src/
│   ├── preprocessing.py  ← RAR extraction, splits, generators
│   ├── train.py          ← CNN + MobileNetV2 training
│   ├── predict.py        ← Inference (single + batch)
│   └── gradcam.py        ← Grad-CAM heatmap generation
├── models/             ← Saved .keras model weights
├── frontend/
│   └── index.html      ← Elite glassmorphic web UI
├── logs/               ← TensorBoard logs
├── app.py              ← Flask API server
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> For RAR extraction: `pip install rarfile patool`  
> Also install system tool: **WinRAR** (Windows) or `apt install unrar` (Linux/WSL)

### 2. Extract dataset

```bash
python src/preprocessing.py \
  --rar "D:\Infotact_Internships\W-2\Concrete Crack Images for Classification.rar" \
  --raw data/raw
```

### 3. Train models

```bash
# MobileNetV2 (recommended, ~20 min on GPU)
python src/train.py --model mobilenet

# Custom CNN from scratch
python src/train.py --model cnn

# Both
python src/train.py --model both
```

### 4. Start Flask API

```bash
python app.py
```

Open **http://localhost:5000** in your browser → full UI loads.

---

## 🧠 Models

| Model | Architecture | Params | Expected Accuracy |
|-------|-------------|--------|-------------------|
| Custom CNN | 4-block Conv + Dense | ~3M | ~93% |
| MobileNetV2 | ImageNet pretrained + custom head | ~2.2M trainable | ~98% |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict + Grad-CAM (JSON) |
| `POST` | `/heatmap` | Heatmap only (PNG) |
| `POST` | `/stream_predict` | Batch (up to 5 images) |

### Example: cURL

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@sample.jpg" | python -m json.tool
```

### Response

```json
{
  "label"          : "DEFECT",
  "confidence"     : 94.7,
  "raw_prob"       : 0.947,
  "heatmap_b64"    : "<base64 PNG>",
  "comparison_b64" : "<base64 3-panel PNG>",
  "timestamp"      : "2024-01-15T14:23:07"
}
```

---

## 🔥 Grad-CAM

Generate standalone heatmap from CLI:

```bash
python src/gradcam.py \
  --model models/mobilenet_final.keras \
  --image sample.jpg \
  --output notebooks/gradcam_out.png
```

---

## 🧪 Inference

Single image:
```bash
python src/predict.py \
  --model models/mobilenet_final.keras \
  --image sample.jpg \
  --heatmap
```

Batch directory:
```bash
python src/predict.py \
  --model models/mobilenet_final.keras \
  --dir   data/splits/test/DEFECT \
  --output results.json
```

---

## 📊 EDA

```bash
python notebooks/01_eda.py
```

Generates:
- `notebooks/class_distribution.png`
- `notebooks/sample_grid.png`
- `notebooks/pixel_distribution.png`

---

## 🔬 TensorBoard

```bash
tensorboard --logdir logs/
```

---

## 📋 Label Mapping

| Folder | → | Label |
|--------|---|-------|
| `Positive/` | → | `DEFECT` |
| `Negative/` | → | `PASS` |

---

## ⚙️ Configuration

Edit constants at the top of each module:

| File | Key Settings |
|------|-------------|
| `src/preprocessing.py` | `IMG_SIZE`, `BATCH_SIZE`, `TRAIN_SPLIT`, `SEED` |
| `src/train.py` | `EPOCHS`, `BATCH_SIZE` |
| `app.py` | model priority list |

---

## 🛠️ VS Code Tips

- Open `visionspec/` as workspace root
- Install extensions: **Python**, **Pylance**, **Thunder Client** (API testing)
- Run `python app.py` in integrated terminal
- Use Thunder Client to test `/predict` endpoint

---

## 📦 Deliverables

- [x] `src/preprocessing.py` — RAR extraction + splits + augmentation
- [x] `src/train.py` — CNN + MobileNetV2 training
- [x] `src/predict.py` — Inference module
- [x] `src/gradcam.py` — Grad-CAM heatmaps
- [x] `app.py` — Flask REST API
- [x] `frontend/index.html` — Elite glassmorphic UI
- [x] `requirements.txt`
- [x] `README.md`

---

*Built for the Infotact AI Internship — Week 1–6 Project*

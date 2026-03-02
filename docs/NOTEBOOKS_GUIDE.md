# 📘 Egyptian ID OCR - Complete Notebook Guide

## Overview

This project contains **4 Jupyter notebooks** that guide you through the complete Egyptian ID OCR pipeline, from dataset creation to model deployment.

---

## 📊 Quick Start: Two-Stage Detection (NEW!)

**Recommended for new users!**

A new **two-stage YOLO detection pipeline** using NASO7Y pre-trained models is now available:

```bash
# Process full dataset (3.4 minutes)
python scripts/process_full_dataset_two_stage.py

# Results: 57,685 field crops from 16,720 images
```

**Start here:** `notebooks/04_two_stage_detection.ipynb`

---

## 📓 Notebook Sequence

```
┌─────────────────────────────────────────────────────────────┐
│  Notebook 01: Building the Dataset                          │
│  ↓                                                          │
│  Notebook 02: Label and Train                               │
│  ↓                                                          │
│  Notebook 03: Evaluate and Deploy                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ NEW: Notebook 04: Two-Stage Detection (Standalone)   │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Notebook Details

### 1. 🏗️ Notebook 01: Building the Dataset

**File:** `notebooks/01_build_dataset.ipynb`

**Purpose:** Crop individual fields from ID card images using YOLO labels.

**Input:**
- Raw images: `train/`, `valid/`, `test/`
- YOLO labels: `train/labels/`, `valid/labels/`, `test/labels/`

**Output:**
- Cropped fields: `rec/images/`
- Metadata: `crops_metadata.csv`

**When to use:**
- ✅ You have custom YOLO labels
- ✅ You want to use your own trained models
- ✅ You need fine-grained control over cropping

**When NOT to use:**
- ❌ Use Notebook 04 instead for faster, cleaner results

---

### 2. 🏷️ Notebook 02: Label and Train

**File:** `notebooks/02_label_and_train.ipynb`

**Purpose:** Extract text from cropped fields and train PaddleOCR.

**Input:**
- `crops_metadata.csv` (from Notebook 01)
- `rec/images/` (cropped field images)

**Output:**
- `crops_labeled.csv` (OCR-labeled data)
- `rec/train.txt` (PaddleOCR training format)
- Trained PaddleOCR model

**OCR Engines Supported:**
- QARI OCR (recommended)
- QARI AirLLM (low VRAM)
- Bakri OCR
- Bakri AirLLM (low VRAM)
- Gemini API
- AirLLM (72B models)

**Two-Stage Dataset:**
```python
# Use two-stage detection results
crops_df = pd.read_csv(ROOT / "crops_metadata_two_stage.csv")
```

---

### 3. 📈 Notebook 03: Evaluate and Deploy

**File:** `notebooks/03_evaluate_and_deploy.ipynb`

**Purpose:** Evaluate model accuracy, export ONNX, test API.

**Input:**
- Trained PaddleOCR model
- `rec/test.txt` (test set)

**Output:**
- `onnx/rec_sim.onnx` (optimized model)
- Evaluation report (CER, WER, Exact Match)

**Evaluation Metrics:**
- Character Error Rate (CER)
- Word Error Rate (WER)
- Exact Match Accuracy

**Two-Stage Evaluation:**
```python
# Evaluate two-stage detection results
test_df = pd.read_csv(ROOT / "crops_metadata_two_stage.csv")
test_df = test_df[test_df['split'] == 'test']
```

---

### 4. 🎯 Notebook 04: Two-Stage Detection (NEW!)

**File:** `notebooks/04_two_stage_detection.ipynb`

**Purpose:** Complete two-stage detection pipeline demonstration.

**Input:**
- Raw ID card images
- NASO7Y pre-trained models

**Output:**
- Cropped field images: `rec/images/two_stage/`
- Metadata: `crops_metadata_two_stage.csv`

**Pipeline:**
```
Full Image → Card Detection → Card Crop → Field Detection → OCR
```

**Features:**
- Interactive testing
- Class mapping translation (NASO7Y → Project format)
- Full dataset processing
- Results visualization

**Results:**
| Split | Images | Successful | Field Crops |
|-------|--------|------------|-------------|
| Train | 15,669 | 11,415 (72.9%) | 54,362 |
| Valid | 948 | 670 (70.7%) | 3,131 |
| Test | 103 | 39 (37.9%) | 192 |
| **Total** | **16,720** | **12,124 (72.5%)** | **57,685** |

---

## 🚀 Recommended Workflow

### Option A: Two-Stage Detection (Recommended for New Users)

```bash
# Step 1: Process dataset with two-stage detection
python scripts/process_full_dataset_two_stage.py

# Step 2: Open Notebook 04 to explore results
jupyter notebook notebooks/04_two_stage_detection.ipynb

# Step 3: Label crops with OCR (Notebook 02)
python scripts/label_crops.py --method qari-airllm

# Step 4: Train PaddleOCR (Notebook 02, Section 7)

# Step 5: Evaluate and deploy (Notebook 03)
```

### Option B: Custom YOLO Labels

```bash
# Step 1: Build dataset from your labels (Notebook 01)
jupyter notebook notebooks/01_build_dataset.ipynb

# Step 2: Label crops with OCR (Notebook 02)
jupyter notebook notebooks/02_label_and_train.ipynb

# Step 3: Train PaddleOCR (Notebook 02)

# Step 4: Evaluate and deploy (Notebook 03)
jupyter notebook notebooks/03_evaluate_and_deploy.ipynb
```

---

## 📁 File Structure

```
egyption_id_ready/
├── notebooks/
│   ├── 01_build_dataset.ipynb       # Dataset creation (YOLO labels)
│   ├── 02_label_and_train.ipynb     # OCR labeling + training
│   ├── 03_evaluate_and_deploy.ipynb # Evaluation + ONNX export
│   └── 04_two_stage_detection.ipynb # Two-stage pipeline (NEW!)
├── scripts/
│   ├── download_weights.py          # Download NASO7Y models
│   ├── download_models.py           # Download OCR models
│   ├── process_full_dataset_two_stage.py  # Full dataset processing
│   └── label_crops.py               # OCR labeling script
├── weights/
│   ├── card_detection.pt            # Card detection model
│   ├── field_detection.pt           # Field detection model
│   └── nid_detection.pt             # NID digit detection
├── rec/
│   ├── images/                      # Original crops
│   └── images/two_stage/            # Two-stage crops (NEW!)
├── crops_metadata.csv               # Original metadata
├── crops_metadata_two_stage.csv     # Two-stage metadata (NEW!)
└── docs/
    ├── PROCESSING_RESULTS.md        # Full processing results
    ├── CLASS_MAPPING.md             # Class translation guide
    ├── NASO7Y_CLASSES.md            # NASO7Y model classes
    └── TWO_STAGE_DETECTION.md       # Two-stage documentation
```

---

## 🔧 Setup Requirements

### Minimum Requirements

```bash
pip install -r requirements-detection.txt
```

### Full Requirements (including training)

```bash
pip install -r requirements.txt
```

### Download Models

```bash
# Download NASO7Y YOLO models
python scripts/download_weights.py

# Download OCR models
python scripts/download_models.py
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| `docs/PROCESSING_RESULTS.md` | Full two-stage processing results |
| `docs/CLASS_MAPPING.md` | NASO7Y ↔ Project class translation |
| `docs/NASO7Y_CLASSES.md` | NASO7Y model class reference |
| `docs/TWO_STAGE_DETECTION.md` | Two-stage detection guide |

---

## 🆘 Troubleshooting

### Issue: No detections in two-stage pipeline

**Solution:**
- Check image quality
- Verify models downloaded: `ls weights/`
- Try different confidence threshold: `--conf-threshold 0.3`

### Issue: OCR returns empty text

**Solution:**
- Check crop quality: `ls rec/images/two_stage/`
- Try different OCR engine: `--method bakri-airllm`
- Verify GPU availability

### Issue: Out of memory during training

**Solution:**
- Use AirLLM models (low VRAM)
- Reduce batch size in config
- Use 4-bit quantization: `--use-4bit`

---

## 📞 Support

For issues or questions:
1. Check documentation in `docs/`
2. Review notebook examples
3. Test with small dataset first

---

**Last Updated:** March 2, 2025  
**Status:** ✅ Production Ready

# Bakri OCR with AirLLM - Usage Guide

## Overview

This guide explains how to use **Bakri OCR** (`bakrianoo/arabic-legal-documents-ocr-1.0`) with **AirLLM** for layer-wise inference, enabling the model to run on GPUs with as little as **4GB VRAM**.

## What is AirLLM?

AirLLM enables running large language models on consumer GPUs by loading layers sequentially instead of all at once. This trades inference speed for reduced memory usage.

**Benefits:**
- ✅ Run Bakri OCR on 4GB GPU (vs 12GB+ required for full loading)
- ✅ Optional 4-bit quantization for even lower VRAM
- ✅ Same accuracy as full model loading

**Trade-offs:**
- ⚠️ Slower inference (~2-5 seconds per image vs ~500ms)
- ⚠️ Best for offline labeling, not real-time inference

---

## Quick Start

### 1. Install Dependencies

```bash
pip install airllm>=2.15.0
```

### 2. Label Crops with Bakri AirLLM

```bash
# Basic usage (default cache directory)
python scripts/label_crops.py --method bakri-airllm

# With custom cache directory
python scripts/label_crops.py --method bakri-airllm \
  --bakri-airllm-cache ./model/airllm_cache_bakri

# With 4-bit quantization (for very low VRAM)
python scripts/label_crops.py --method bakri-airllm --use-4bit

# Increase layers per batch for faster inference (uses more VRAM)
python scripts/label_crops.py --method bakri-airllm \
  --layers-per-batch 4
```

---

## Python API

### Basic Usage

```python
from src.ocr_engines.bakri_airllm_ocr import BakriAirLLMOCR

# Initialize with AirLLM layer-wise inference
ocr = BakriAirLLMOCR(
    model_name="bakrianoo/arabic-legal-documents-ocr-1.0",
    use_4bit=False,              # Set True for 4-bit quantization
    cache_dir="./model/airllm_cache_bakri",
    layers_per_batch=2,          # Higher = faster but more VRAM
)

# Extract text from a cropped field
text = ocr.extract("rec/images/crop_001.jpg", field_name="name")
print(f"Extracted: {text}")
```

### With 4-bit Quantization (Low VRAM)

```python
ocr = BakriAirLLMOCR(
    model_name="bakrianoo/arabic-legal-documents-ocr-1.0",
    use_4bit=True,               # Enable 4-bit quantization
    cache_dir="./model/airllm_cache_bakri",
    layers_per_batch=1,          # Keep low for minimal VRAM
)
```

### Batch Labeling

```python
import pandas as pd
from src.ocr_engines.bakri_airllm_ocr import BakriAirLLMOCR

# Load crops metadata
df = pd.read_csv("crops_metadata.csv")

# Initialize OCR
ocr = BakriAirLLMOCR()

# Label all crops
ocr.label_crops(df, base_dir=".")

# Save results
df.to_csv("crops_labeled.csv", index=False, encoding="utf-8-sig")
```

---

## Configuration Options

### Environment Variables (via YAML config)

Edit `configs/airllm_config.yml`:

```yaml
# Bakri OCR model with AirLLM
BAKRI_AIRLLM_MODEL: "bakrianoo/arabic-legal-documents-ocr-1.0"

# Cache directory for sharded model files
BAKRI_AIRLLM_CACHE_DIR: "./model/airllm_cache_bakri"

# Quantization (True for <4GB VRAM)
AIRLLM_USE_4BIT: false

# Layers per batch (higher = faster, more VRAM)
BAKRI_AIRLLM_LAYERS_PER_BATCH: 2
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--method bakri-airllm` | Use Bakri OCR with AirLLM | - |
| `--bakri-airllm-cache` | Cache directory for sharded model | `./model/airllm_cache_bakri` |
| `--layers-per-batch` | Layers to keep in GPU memory | `1` |
| `--use-4bit` | Enable 4-bit quantization | `False` |

---

## Performance Guidelines

### VRAM Requirements

| Configuration | Minimum VRAM | Inference Time |
|---------------|--------------|----------------|
| Full load (no AirLLM) | 12GB+ | ~500ms |
| AirLLM (layers_per_batch=4) | 8GB | ~2s |
| AirLLM (layers_per_batch=2) | 6GB | ~3s |
| AirLLM (layers_per_batch=1) | 4GB | ~5s |
| AirLLM + 4-bit (layers_per_batch=1) | 3GB | ~6s |

### Tuning for Your GPU

**For 4GB GPU:**
```bash
python scripts/label_crops.py --method bakri-airllm \
  --use-4bit \
  --layers-per-batch 1
```

**For 6GB GPU:**
```bash
python scripts/label_crops.py --method bakri-airllm \
  --layers-per-batch 2
```

**For 8GB+ GPU:**
```bash
python scripts/label_crops.py --method bakri-airllm \
  --layers-per-batch 4
```

---

## Comparison with Other Methods

| Method | Model Size | VRAM Required | Speed | Accuracy |
|--------|------------|---------------|-------|----------|
| **Bakri AirLLM** | 4B | 4GB+ | Slow | High |
| **Bakri (standard)** | 4B | 12GB+ | Fast | High |
| **QARI-OCR** | 2B | 8GB+ | Medium | Medium-High |
| **AirLLM (72B)** | 72B | 4GB+ | Very Slow | Very High |
| **Gemini API** | - | 0GB (cloud) | Fast | Very High |

---

## Troubleshooting

### Error: `ImportError: No module named 'airllm'`

```bash
pip install airllm>=2.15.0
```

### Error: `CUDA out of memory`

Reduce layers per batch or enable 4-bit quantization:

```bash
python scripts/label_crops.py --method bakri-airllm \
  --layers-per-batch 1 \
  --use-4bit
```

### Error: `Model loading fails`

Ensure you have HuggingFace authentication if the model is gated:

```bash
huggingface-cli login
```

Or use the fallback (standard transformers with 4-bit):

```python
# AirLLM import will fail gracefully and fall back automatically
ocr = BakriAirLLMOCR(use_4bit=True)
```

---

## Best Practices

1. **Use for offline labeling**: AirLLM is slow but accurate - perfect for creating ground truth datasets.

2. **Cache models locally**: Models are downloaded and sharded to the cache directory. Subsequent runs will be faster.

3. **Combine with QARI for validation**: Use QARI-OCR for fast initial labeling, then validate ambiguous cases with Bakri AirLLM.

4. **Monitor VRAM usage**: Use `nvidia-smi` to watch GPU memory and adjust `layers_per_batch` accordingly.

5. **Preprocess images**: Bakri OCR expects grayscale images resized to max 1024px width - this is handled automatically.

---

## Example Workflow

```bash
# Step 1: Build dataset (crop fields from ID images)
python scripts/build_dataset.py

# Step 2: Label with Bakri AirLLM (offline, high accuracy)
python scripts/label_crops.py --method bakri-airllm \
  --layers-per-batch 2 \
  --splits train valid

# Step 3: Review and edit labels if needed
# (Open crops_labeled.csv and manually correct any errors)

# Step 4: Train PaddleOCR model with labeled data
# (Follow notebooks/02_label_and_train.ipynb)
```

---

## See Also

- `src/ocr_engines/bakri_airllm_ocr.py` - Implementation details
- `configs/airllm_config.yml` - Configuration options
- `scripts/label_crops.py` - Labeling script
- `notebooks/02_label_and_train.ipynb` - Complete training pipeline

#!/usr/bin/env python3
"""Test script for QARI AirLLM OCR fix."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Find a sample image to test
import pandas as pd

crops_df = pd.read_csv(ROOT / "crops_metadata.csv")
sample = crops_df[crops_df["split"] == "train"].iloc[0]
img_path = ROOT / sample["image_path"]

print(f"📋 Testing with sample:")
print(f"   Field: {sample['field']}")
print(f"   Image: {img_path}")
print(f"   Exists: {img_path.exists()}")
print()

# Test QARI AirLLM
print("🔬 Testing QARI AirLLM OCR...")
from src.ocr_engines.qari_airllm_ocr import QariAirLLMOCR

qari_airllm = QariAirLLMOCR(
    model_name="NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
    use_4bit=False,
    cache_dir=str(ROOT / "model" / "airllm_cache_qari"),
    layers_per_batch=2,
)

text = qari_airllm.extract(str(img_path), sample["field"])
print()
print(f"📝 Result:")
print(f"   Field: {sample['field']}")
print(f"   Text : '{text}'")
print()

if text.strip():
    print("✅ SUCCESS: QARI AirLLM is working!")
else:
    print("❌ FAILED: QARI AirLLM returned empty text")

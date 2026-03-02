#!/usr/bin/env python3
"""Test script for Bakri AirLLM OCR fix."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Find a national_id sample to test
import pandas as pd

crops_df = pd.read_csv(ROOT / "crops_metadata.csv")
sample = crops_df[crops_df["field"] == "national_id"].iloc[0]
img_path = ROOT / sample["image_path"]

print(f"📋 Testing with national_id sample:")
print(f"   Field: {sample['field']}")
print(f"   Image: {img_path}")
print(f"   Exists: {img_path.exists()}")
print()

# Test Bakri AirLLM
print("🔬 Testing Bakri AirLLM OCR...")
from src.ocr_engines.bakri_airllm_ocr import BakriAirLLMOCR

bakri_airllm = BakriAirLLMOCR(
    model_name="bakrianoo/arabic-legal-documents-ocr-1.0",
    use_4bit=False,
    cache_dir=str(ROOT / "model" / "airllm_cache_bakri"),
    layers_per_batch=2,
)

text = bakri_airllm.extract(str(img_path), sample["field"])
print()
print(f"📝 Result:")
print(f"   Field: {sample['field']}")
print(f"   Text : '{text}'")
print()

if text.strip():
    print("✅ SUCCESS: Bakri AirLLM is working!")
else:
    print("❌ FAILED: Bakri AirLLM returned empty text")

#!/usr/bin/env python3
"""
Download OCR Models
===================
Downloads EasyOCR models for Arabic and English text recognition.

EasyOCR will automatically download models on first run, but this script
pre-downloads them to a specific cache directory for offline use.
"""

import os
from pathlib import Path


def main():
    """Download OCR models."""
    
    print("📦 Egyptian ID OCR - OCR Model Downloader")
    print("=" * 50)
    
    # Try to import easyocr
    try:
        import easyocr
    except ImportError:
        print("❌ EasyOCR not installed!")
        print("   Install with: pip install easyocr")
        return
    
    # Set cache directory
    cache_dir = Path(__file__).parent.parent / "models_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Model cache directory: {cache_dir}")
    print()
    
    print("⏳ Initializing EasyOCR (this will download models on first run)...")
    print("   Languages: Arabic (ar) + English (en)")
    print()
    
    try:
        # This will download models if not already cached
        reader = easyocr.Reader(
            ["ar", "en"],
            gpu=False,  # Use CPU for download
            quantize=True,
            model_storage_directory=str(cache_dir),
            verbose=False,
        )
        
        print("✅ OCR models downloaded successfully!")
        print()
        print("📋 Model files location:")
        print(f"   {cache_dir}")
        print()
        print("📊 Next steps:")
        print("   1. Test OCR: python -c \"import easyocr; r=easyocr.Reader(['ar','en']); print(r.readtext('test'))\"")
        print("   2. Label crops: python scripts/label_crops.py --method qari")
        
    except Exception as e:
        print(f"❌ Failed to download OCR models: {e}")
        print()
        print("   You can manually initialize EasyOCR in Python:")
        print("   >>> import easyocr")
        print("   >>> reader = easyocr.Reader(['ar', 'en'])")


if __name__ == "__main__":
    main()

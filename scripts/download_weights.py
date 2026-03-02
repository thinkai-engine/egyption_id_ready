#!/usr/bin/env python3
"""
Download Weights for Egyptian ID OCR
=====================================
Downloads pre-trained YOLO models from NASO7Y project for:
1. Card Detection - Detects Egyptian ID card in image
2. Field Detection - Detects fields (name, NID, address, serial) on card
3. NID Digit Detection - Detects individual digits in National ID

Models sourced from: https://github.com/NASO7Y/OCR_Egyptian_ID
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


# Model download URLs (from NASO7Y project)
MODELS = {
    "card_detection.pt": {
        "url": "https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id_card.pt",
        "description": "Card Detection - Detects Egyptian ID card in image",
        "class": "id_card",
    },
    "field_detection.pt": {
        "url": "https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_odjects.pt",
        "description": "Field Detection - Detects fields (name, NID, address, serial)",
        "classes": ["job_title", "photo", "expiry_date", "birth_date", "religion", 
                    "name", "address", "national_id", "marital_status", "gender",
                    "governorate", "husband_name", "issue_date", "serial_number"],
    },
    "nid_detection.pt": {
        "url": "https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id.pt",
        "description": "NID Digit Detection - Detects individual digits in National ID",
        "class": "nid_digits",
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with progress bar."""
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=dest_path.name) as t:
            urllib.request.urlretrieve(url, filename=str(dest_path),
                                       reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    """Download all model weights."""
    
    # Create weights directory
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("📦 Egyptian ID OCR - Model Weight Downloader")
    print("=" * 50)
    print(f"📂 Download directory: {weights_dir}")
    print()
    
    # Track download stats
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    for model_name, model_info in MODELS.items():
        model_path = weights_dir / model_name
        
        # Check if already exists
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"⏭️  {model_name}: Already exists ({size_mb:.1f} MB)")
            print(f"    {model_info['description']}")
            stats["skipped"] += 1
            print()
            continue
        
        # Download
        print(f"⬇️  Downloading: {model_name}")
        print(f"    {model_info['description']}")
        print(f"    URL: {model_info['url']}")
        
        if download_file(model_info["url"], model_path):
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"✅ Downloaded successfully ({size_mb:.1f} MB)")
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1
        
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Download Summary:")
    print(f"   ✅ Downloaded: {stats['downloaded']}")
    print(f"   ⏭️  Skipped:    {stats['skipped']}")
    print(f"   ❌ Failed:     {stats['failed']}")
    print()
    
    if stats["failed"] == 0 and stats["downloaded"] > 0:
        print("🎉 All models downloaded successfully!")
        print()
        print("📋 Next steps:")
        print("   1. Download OCR models: python scripts/download_models.py")
        print("   2. Test detection:     python test_two_stage_detection.py <image>")
        print("   3. Run pipeline:       python scripts/label_crops.py --method qari")
    elif stats["failed"] > 0:
        print("⚠️  Some downloads failed. Please check your internet connection.")
        print("   You can manually download from:")
        for model_name, model_info in MODELS.items():
            if not (weights_dir / model_name).exists():
                print(f"   - {model_info['url']}")


if __name__ == "__main__":
    main()

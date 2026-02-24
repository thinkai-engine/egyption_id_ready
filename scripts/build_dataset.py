"""
Build Dataset — Orchestrate field cropping across all splits.
================================================================
Usage:
    python scripts/build_dataset.py
"""

import sys
from pathlib import Path

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from crop_builder import build_crops_from_split

# ── Configuration ─────────────────────────────────────────────
SPLITS = {
    "train": ROOT / "train",
    "valid": ROOT / "valid",
    "test":  ROOT / "test",
}

OUTPUT_DIR = ROOT / "rec" / "images"
METADATA_CSV = ROOT / "crops_metadata.csv"

# ── Optional: load YOLO detector for images without labels ────
detector = None
YOLO_ONNX = ROOT / "model" / "field_detector.onnx"
if YOLO_ONNX.exists():
    try:
        from field_detector import YOLOFieldDetector
        detector = YOLOFieldDetector(str(YOLO_ONNX))
    except Exception as e:
        print(f"⚠️  YOLO detector not loaded: {e}")
        print("    → Only images with existing labels will be processed.")


def main():
    print("=" * 55)
    print("   📸 Egyptian ID OCR — Build Dataset")
    print("=" * 55)

    all_dfs = []
    for split, path in SPLITS.items():
        if not (path / "images").exists():
            print(f"⚠️  Skipping {split} — no images/ directory")
            continue
        df = build_crops_from_split(
            split=split,
            split_path=path,
            output_dir=OUTPUT_DIR,
            detector=detector,
        )
        all_dfs.append(df)

    if not all_dfs:
        print("❌ No data processed!")
        return

    crops_df = pd.concat(all_dfs, ignore_index=True)
    crops_df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"   📊 Dataset Summary")
    print(f"{'=' * 55}")
    print(f"   Total crops  : {len(crops_df):,}")

    for split in SPLITS:
        n = (crops_df["split"] == split).sum()
        print(f"   {split:<12}: {n:>6,} crops")

    print(f"\n   Fields:")
    for field in crops_df["field"].unique():
        n = (crops_df["field"] == field).sum()
        print(f"     {field:<18}: {n:>6,}")

    print(f"\n   Saved → {METADATA_CSV}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

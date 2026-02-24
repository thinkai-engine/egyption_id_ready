"""
Prepare PaddleOCR Label Files
==============================
Clean + reverse text labels → write train.txt / val.txt / test.txt.

Usage:
    python scripts/prepare_paddle_labels.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from text_cleaner import prepare_paddle_label

LABELED_CSV = ROOT / "crops_labeled.csv"
OUTPUT = {
    "train": ROOT / "rec" / "train.txt",
    "valid": ROOT / "rec" / "val.txt",
    "test":  ROOT / "rec" / "test.txt",
}


def build_arabic_dict(df: pd.DataFrame, dict_path: Path):
    """Generate arabic_dict.txt from actual data characters."""
    all_text = "".join(df["label_clean"].dropna().tolist())
    unique_chars = sorted(set(all_text))
    # Remove blank/space duplicates, keep space as explicit entry
    unique_chars = [c for c in unique_chars if c.strip() or c == " "]

    with open(dict_path, "w", encoding="utf-8") as f:
        for c in unique_chars:
            f.write(c + "\n")

    print(f"   Dictionary: {len(unique_chars)} characters → {dict_path}")


def main():
    if not LABELED_CSV.exists():
        print("❌ crops_labeled.csv not found. Run label_crops.py first!")
        return

    df = pd.read_csv(LABELED_CSV)

    # ── Clean and reverse ─────────────────────────────────────
    if "label_text" not in df.columns:
        print("❌ No label_text column found!")
        return

    df["label_clean"] = df["label_text"].apply(
        lambda x: prepare_paddle_label(str(x)) if pd.notna(x) else ""
    )

    # ── Build dictionary from actual data ─────────────────────
    build_arabic_dict(df, ROOT / "arabic_dict.txt")

    # ── Write split files ─────────────────────────────────────
    (ROOT / "rec").mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 45}")
    print(f"   📄 PaddleOCR Label Files")
    print(f"{'=' * 45}")

    total_written = 0
    for split, txt_path in OUTPUT.items():
        subset = df[
            (df["split"] == split) &
            (df["label_clean"].str.len() > 0)
        ]

        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                f.write(f"{row['image_path']}\t{row['label_clean']}\n")

        print(f"   {split:<8}: {len(subset):>6,} samples → {txt_path.name}")
        total_written += len(subset)

    # ── Stats ─────────────────────────────────────────────────
    labeled = (df["label_clean"].str.len() > 0).sum()
    unlabeled = len(df) - labeled

    print(f"\n   Total written : {total_written:,}")
    print(f"   Unlabeled     : {unlabeled:,}")

    # Text length stats
    lens = df[df["label_clean"].str.len() > 0]["label_clean"].str.len()
    if not lens.empty:
        print(f"\n   Text Length:")
        print(f"     Mean : {lens.mean():.1f} chars")
        print(f"     Max  : {lens.max()}")
        print(f"     >40  : {(lens > 40).sum()} "
              f"(check max_text_length in config)")

    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()

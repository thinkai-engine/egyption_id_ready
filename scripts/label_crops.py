"""
Label Crops — Extract text from all cropped fields.
=====================================================
Usage:
    python scripts/label_crops.py                         # default: qari
    python scripts/label_crops.py --method gemini
    python scripts/label_crops.py --method bakri
    python scripts/label_crops.py --method bakri-airllm   # Bakri with AirLLM (low VRAM)
    python scripts/label_crops.py --method airllm         # AirLLM (72B model)
    python scripts/label_crops.py --method both
"""

import sys, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

METADATA_CSV = ROOT / "crops_metadata.csv"
OUTPUT_CSV = ROOT / "crops_labeled.csv"


def extract_text(img_path, field, method, engines):
    """Extract text using the chosen method."""
    result = {"field": field, "image_path": str(img_path)}

    if method == "qari":
        result["label_text"] = engines["qari"].extract(str(img_path), field)

    elif method == "gemini":
        result["label_text"] = engines["gemini"].extract(str(img_path), field)

    elif method == "bakri":
        result["label_text"] = engines["bakri"].extract(str(img_path), field)

    elif method == "bakri-airllm":
        result["label_text"] = engines["bakri_airllm"].extract(str(img_path), field)

    elif method == "airllm":
        result["label_text"] = engines["airllm"].extract(str(img_path), field)

    elif method == "both":
        q = engines["qari"].extract(str(img_path), field)
        g = engines["gemini"].extract(str(img_path), field)
        result["label_text"] = q
        result["qari_text"] = q
        result["gemini_text"] = g
        result["texts_match"] = q.strip() == g.strip()

    return result


def main():
    parser = argparse.ArgumentParser(description="Label crops with OCR")
    parser.add_argument(
        "--method", choices=["qari", "gemini", "bakri", "bakri-airllm", "airllm", "both"],
        default="qari", help="OCR engine to use"
    )
    parser.add_argument(
        "--gemini-key", default="", help="Gemini API key"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid"],
        help="Splits to label (test is for evaluation only)"
    )
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Use 4-bit quantization for QARI (saves VRAM)"
    )
    parser.add_argument(
        "--airllm-model", default="Qwen/Qwen2-VL-72B-Instruct",
        help="AirLLM model name or path (default: Qwen2-VL-72B-Instruct)"
    )
    parser.add_argument(
        "--airllm-cache", default="./model/airllm_cache",
        help="Cache directory for AirLLM sharded model"
    )
    parser.add_argument(
        "--bakri-airllm-cache", default="./model/airllm_cache_bakri",
        help="Cache directory for Bakri AirLLM sharded model"
    )
    parser.add_argument(
        "--layers-per-batch", type=int, default=1,
        help="Number of layers to keep in GPU memory at once (higher = faster, more VRAM)"
    )
    args = parser.parse_args()

    # ── Load metadata ─────────────────────────────────────────
    if not METADATA_CSV.exists():
        print("❌ crops_metadata.csv not found. Run build_dataset.py first!")
        return

    df = pd.read_csv(METADATA_CSV)
    if "label_text" not in df.columns:
        df["label_text"] = ""

    # ── Load existing progress ────────────────────────────────
    done_set = set()
    if OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)
        done_set = set(done_df["image_path"].tolist())
        # Merge existing labels
        for _, row in done_df.iterrows():
            mask = df["image_path"] == row["image_path"]
            if mask.any() and row.get("label_text", ""):
                df.loc[mask, "label_text"] = row["label_text"]
        print(f"▶️  Resuming — {len(done_set)} already labeled")

    # ── Filter to unlabeled in target splits ──────────────────
    subset = df[
        df["split"].isin(args.splits) &
        (~df["image_path"].isin(done_set)) &
        (df["label_text"] == "")
    ]
    print(f"📋 Pending: {len(subset):,} crops using [{args.method.upper()}]")

    if subset.empty:
        print("✅ All crops already labeled!")
        return

    # ── Load engines ──────────────────────────────────────────
    engines = {}
    if args.method in ("qari", "both"):
        from src.ocr_engines.qari_ocr import QariOCR
        engines["qari"] = QariOCR(use_4bit=args.use_4bit)

    if args.method in ("bakri", "both"):
        from src.ocr_engines.bakri_ocr import BakriOCR
        engines["bakri"] = BakriOCR(use_4bit=args.use_4bit)

    if args.method in ("gemini", "both"):
        from src.ocr_engines.gemini_ocr import GeminiOCR
        if not args.gemini_key:
            print("❌ Gemini API key required! Use --gemini-key YOUR_KEY")
            return
        engines["gemini"] = GeminiOCR(api_key=args.gemini_key)

    if args.method in ("airllm",):
        from src.ocr_engines.airllm_ocr import AirLLMOCR
        engines["airllm"] = AirLLMOCR(
            model_name=args.airllm_model,
            use_4bit=args.use_4bit,
            cache_dir=args.airllm_cache,
        )

    if args.method in ("bakri-airllm",):
        from src.ocr_engines.bakri_airllm_ocr import BakriAirLLMOCR
        engines["bakri_airllm"] = BakriAirLLMOCR(
            model_name="bakrianoo/arabic-legal-documents-ocr-1.0",
            use_4bit=args.use_4bit,
            cache_dir=args.bakri_airllm_cache,
            layers_per_batch=args.layers_per_batch,
        )

    # ── Process ───────────────────────────────────────────────
    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
        img_path = ROOT / row["image_path"]
        if not img_path.exists():
            continue

        try:
            result = extract_text(
                img_path, row["field"], args.method, engines
            )
            for k, v in result.items():
                if k in df.columns:
                    df.at[idx, k] = v
                elif k not in ("field", "image_path"):
                    df[k] = ""
                    df.at[idx, k] = v
        except Exception as e:
            print(f"⚠️  Error on {row['image_path']}: {e}")
            continue

        # Save progress every 50 crops
        if idx % 50 == 0:
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        # Rate limit for Gemini and AirLLM (slower inference)
        if args.method in ("gemini", "both"):
            time.sleep(0.4)
        elif args.method in ("airllm", "bakri-airllm"):
            time.sleep(0.1)  # Small delay for layer-wise inference

    # ── Final save ────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    labeled = (df["label_text"].str.len() > 0).sum()
    total = len(df)
    print(f"\n✅ Done! Labeled: {labeled:,}/{total:,}")
    print(f"   Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

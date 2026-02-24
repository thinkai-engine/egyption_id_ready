"""
Evaluate OCR Model — CER / WER / Exact Match
=============================================
Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --test-txt ./rec/test.txt
"""

import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm
import editdistance


def cer(pred: str, gt: str) -> float:
    """Character Error Rate — lower is better."""
    if not gt:
        return 0.0
    return editdistance.eval(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    """Word Error Rate."""
    pred_w, gt_w = pred.split(), gt.split()
    if not gt_w:
        return 0.0
    return editdistance.eval(pred_w, gt_w) / len(gt_w)


def evaluate_from_file(
    test_txt: str,
    base_dir: Path,
    rec_model_dir: str = None,
) -> pd.DataFrame:
    """
    Evaluate PaddleOCR model on test split.

    Parameters
    ----------
    test_txt : path to test label file (image_path \t reversed_label)
    base_dir : base directory for image paths
    """
    with open(test_txt, encoding="utf-8") as f:
        lines = [l.strip() for l in f if "\t" in l]

    if not lines:
        print("⚠️  No test samples found!")
        return pd.DataFrame()

    # Try loading PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_gpu=False,
            rec_model_dir=rec_model_dir or str(
                ROOT / "output" / "egyptian_id_rec" / "best_accuracy"
            ),
            rec_char_dict_path=str(ROOT / "arabic_dict.txt"),
            lang="ar",
            show_log=False,
        )
    except ImportError:
        print("❌ PaddleOCR not installed. Install with: pip install paddleocr")
        return pd.DataFrame()

    records = []
    for line in tqdm(lines, desc="Evaluating"):
        img_rel, label_rev = line.split("\t", 1)
        gt = label_rev[::-1]  # reverse back to normal Arabic

        img_path = base_dir / img_rel
        if not img_path.exists():
            continue

        result = ocr.ocr(str(img_path), cls=False)
        if result and result[0]:
            pred_raw, conf = result[0][0][1]
            pred = pred_raw[::-1]
        else:
            pred, conf = "", 0.0

        field = img_path.stem.split("_")[-1]

        records.append({
            "image": img_path.name,
            "field": field,
            "gt": gt,
            "pred": pred,
            "conf": round(float(conf), 3),
            "cer": cer(pred, gt),
            "wer": wer(pred, gt),
            "exact": pred.strip() == gt.strip(),
        })

    return pd.DataFrame(records)


def print_report(test_df: pd.DataFrame):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 55}")
    print(f"      📊 OCR Evaluation Report")
    print(f"{'=' * 55}")

    fields = test_df["field"].unique()
    for field in sorted(fields):
        sub = test_df[test_df["field"] == field]
        avg_cer = sub["cer"].mean()
        avg_wer = sub["wer"].mean()
        exact_acc = sub["exact"].mean() * 100
        status = "✅" if avg_cer < 0.05 else ("⚠️" if avg_cer < 0.15 else "❌")
        print(f"\n{status} {field}")
        print(f"   CER:          {avg_cer:.3f}  ({(1-avg_cer)*100:.1f}%)")
        print(f"   WER:          {avg_wer:.3f}")
        print(f"   Exact Match:  {exact_acc:.1f}%")
        print(f"   Samples:      {len(sub)}")

    # Overall
    overall_cer = test_df["cer"].mean()
    overall_wer = test_df["wer"].mean()
    overall_acc = test_df["exact"].mean() * 100

    print(f"\n{'─' * 55}")
    print(f"   Overall CER    : {overall_cer:.3f} "
          f"({(1-overall_cer)*100:.1f}% accuracy)")
    print(f"   Overall WER    : {overall_wer:.3f}")
    print(f"   Exact Match    : {overall_acc:.1f}%")
    print(f"   Total Samples  : {len(test_df):,}")
    print(f"{'=' * 55}")


def analyze_errors(test_df: pd.DataFrame, top_n: int = 15):
    """Analyze worst prediction errors."""
    errors = test_df[~test_df["exact"]].copy()
    if errors.empty:
        print("\n🎉 No errors found!")
        return

    errors["char_errors"] = errors.apply(
        lambda r: editdistance.eval(r["pred"], r["gt"]), axis=1
    )

    print(f"\n❌ Error Analysis ({len(errors)} errors / {len(test_df)} total)")
    print(f"{'=' * 55}")

    # Error rate by field
    print("\n   Error Rate by Field:")
    for field in sorted(test_df["field"].unique()):
        f_all = test_df[test_df["field"] == field]
        f_err = errors[errors["field"] == field]
        if f_all.empty:
            continue
        rate = len(f_err) / len(f_all) * 100
        bar = "█" * int(rate / 5)
        print(f"     {field:<18} {rate:5.1f}%  {bar}")

    # Worst errors
    print(f"\n   Top {top_n} Worst Errors:")
    for _, row in errors.nlargest(top_n, "char_errors").iterrows():
        print(f"     [{row['field']}]")
        print(f"       GT  : {row['gt']}")
        print(f"       PRED: {row['pred']}")
        print(f"       CER : {row['cer']:.2f}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-txt",
        default=str(ROOT / "rec" / "test.txt"),
    )
    parser.add_argument("--rec-model-dir", default=None)
    parser.add_argument("--top-errors", type=int, default=15)
    args = parser.parse_args()

    test_df = evaluate_from_file(
        args.test_txt, ROOT, args.rec_model_dir
    )

    if test_df.empty:
        return

    print_report(test_df)
    analyze_errors(test_df, args.top_errors)

    out_path = ROOT / "test_evaluation.csv"
    test_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved → {out_path}")


if __name__ == "__main__":
    main()

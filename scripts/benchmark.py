"""
Speed Benchmark — Measure inference latency on CPU.
=====================================================
Usage:
    python scripts/benchmark.py
"""

import sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def main():
    from src.inference import EgyptianIDOCR

    # ── Configuration ─────────────────────────────────────────
    REC_ONNX = ROOT / "onnx" / "rec_sim.onnx"
    DET_ONNX = ROOT / "model" / "field_detector.onnx"
    DICT = ROOT / "arabic_dict.txt"
    N_RUNS = 100

    if not REC_ONNX.exists():
        print(f"❌ Recognition model not found: {REC_ONNX}")
        print("   Run fine-tuning and ONNX export first.")
        return

    # ── Load model ────────────────────────────────────────────
    ocr = EgyptianIDOCR(
        det_onnx=str(DET_ONNX),
        rec_onnx=str(REC_ONNX),
        dict_path=str(DICT),
        use_gpu=False,
    )

    # ── Create dummy crops (simulating 7 ID fields) ───────────
    crops = {}
    field_sizes = {
        "name": (48, 320),
        "national_id": (48, 280),
        "birth_date": (48, 180),
        "address": (48, 400),
        "governorate": (48, 160),
        "gender": (48, 100),
        "expiry_date": (48, 180),
    }

    for field, (h, w) in field_sizes.items():
        crops[field] = np.random.randint(
            0, 255, (h, w, 3), dtype=np.uint8
        )

    n_fields = len(crops)

    # ── Benchmark ─────────────────────────────────────────────
    # Warm-up
    for _ in range(5):
        for crop in crops.values():
            ocr.recognize(crop)

    # Timed runs
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for crop in crops.values():
            ocr.recognize(crop)
        times.append((time.perf_counter() - t0) * 1000)

    # ── Report ────────────────────────────────────────────────
    print(f"\n{'=' * 45}")
    print(f"  ⚡ Benchmark ({N_RUNS} runs, {n_fields} fields)")
    print(f"{'=' * 45}")
    print(f"  Per card  avg : {statistics.mean(times):.1f} ms")
    print(f"  Per card  p50 : {statistics.median(times):.1f} ms")
    print(f"  Per card  p95 : {sorted(times)[int(N_RUNS * 0.95)]:.1f} ms")
    print(f"  Per field avg : {statistics.mean(times) / n_fields:.1f} ms")
    print(f"  Throughput    : {1000 / statistics.mean(times):.1f} cards/sec")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()

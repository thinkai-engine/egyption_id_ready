# %% [markdown]
# # 🇪🇬 Egyptian ID OCR — Notebook 3: التقييم + التصدير + الـ API
#
# هذا الـ Notebook يقيّم الدقة، يصدّر ONNX، ويختبر الـ API.
#
# **المدخلات**: trained model + `rec/test.txt`  
# **المخرجات**: `onnx/rec_sim.onnx` + evaluation report

# %% [markdown]
# ## 1. إعداد البيئة

# %%
import sys, os, time, subprocess
from pathlib import Path

ROOT = Path(os.getcwd())
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import editdistance
%matplotlib inline

# %% [markdown]
# ## 2. تصدير ONNX

# %%
CONFIG = ROOT / "configs" / "egyptian_id_rec.yml"
BEST_MODEL = ROOT / "output" / "egyptian_id_rec" / "best_accuracy"
INFERENCE_DIR = ROOT / "inference" / "rec"
ONNX_DIR = ROOT / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

print("📦 Step 1: Exporting inference model...")
subprocess.run([
    "python", "tools/export_model.py",
    "-c", str(CONFIG),
    "-o", f"Global.pretrained_model={BEST_MODEL}",
         f"Global.save_inference_dir={INFERENCE_DIR}",
], cwd=str(ROOT / "PaddleOCR") if (ROOT / "PaddleOCR").exists() else str(ROOT))

# %%
print("📦 Step 2: Converting to ONNX...")
subprocess.run([
    "paddle2onnx",
    "--model_dir", str(INFERENCE_DIR),
    "--model_filename", "inference.pdmodel",
    "--params_filename", "inference.pdiparams",
    "--save_file", str(ONNX_DIR / "rec.onnx"),
    "--opset_version", "11",
])

print("📦 Step 3: Optimizing...")
subprocess.run([
    "python", "-m", "onnxsim",
    str(ONNX_DIR / "rec.onnx"),
    str(ONNX_DIR / "rec_sim.onnx"),
])

# حجم النموذج
orig = (ONNX_DIR / "rec.onnx").stat().st_size / 1024 / 1024
opt = (ONNX_DIR / "rec_sim.onnx").stat().st_size / 1024 / 1024
print(f"\n✅ ONNX Export Done!")
print(f"   Original  : {orig:.1f} MB")
print(f"   Optimized : {opt:.1f} MB  ({(1-opt/orig)*100:.0f}% smaller)")

# %% [markdown]
# ## 3. تقييم الدقة على Test Set

# %%
def cer(pred: str, gt: str) -> float:
    if not gt: return 0.0
    return editdistance.eval(pred, gt) / len(gt)

def wer(pred: str, gt: str) -> float:
    pw, gw = pred.split(), gt.split()
    if not gw: return 0.0
    return editdistance.eval(pw, gw) / len(gw)

# %%
# تحميل الـ test data
test_txt = ROOT / "rec" / "test.txt"

with open(test_txt, encoding="utf-8") as f:
    test_lines = [l.strip() for l in f if "\t" in l]

print(f"📋 Test samples: {len(test_lines)}")

# %%
# تحميل النموذج
from inference import EgyptianIDOCR

ocr = EgyptianIDOCR(
    det_onnx=str(ROOT / "model" / "field_detector.onnx"),
    rec_onnx=str(ONNX_DIR / "rec_sim.onnx"),
    dict_path=str(ROOT / "arabic_dict.txt"),
    use_gpu=False,   # التقييم على CPU
)

# %%
# Evaluate
records = []
for line in tqdm(test_lines, desc="Evaluating"):
    img_rel, label_rev = line.split("\t", 1)
    gt = label_rev[::-1]  # عكس الـ label للمقارنة
    
    img_path = ROOT / img_rel
    if not img_path.exists():
        continue
    
    crop = cv2.imread(str(img_path))
    pred, conf = ocr.recognize(crop)
    field = img_path.stem.split("_")[-1]
    
    records.append({
        "image": img_path.name,
        "field": field,
        "gt": gt,
        "pred": pred,
        "conf": round(conf, 3),
        "cer": cer(pred, gt),
        "wer": wer(pred, gt),
        "exact": pred.strip() == gt.strip(),
    })

test_df = pd.DataFrame(records)
test_df.to_csv(ROOT / "test_evaluation.csv", index=False, encoding="utf-8-sig")
print(f"✅ Evaluated: {len(test_df)} samples")

# %% [markdown]
# ## 4. تقرير الدقة

# %%
print(f"{'=' * 55}")
print(f"      📊 OCR Evaluation Report")
print(f"{'=' * 55}")

for field in sorted(test_df["field"].unique()):
    sub = test_df[test_df["field"] == field]
    avg_cer = sub["cer"].mean()
    exact_acc = sub["exact"].mean() * 100
    status = "✅" if avg_cer < 0.05 else ("⚠️" if avg_cer < 0.15 else "❌")
    print(f"\n{status} {field}")
    print(f"   CER: {avg_cer:.3f} ({(1-avg_cer)*100:.1f}%) | Exact: {exact_acc:.1f}% | n={len(sub)}")

print(f"\n{'─' * 55}")
print(f"   Overall CER : {test_df['cer'].mean():.3f} ({(1-test_df['cer'].mean())*100:.1f}%)")
print(f"   Exact Match : {test_df['exact'].mean()*100:.1f}%")
print(f"{'=' * 55}")

# %%
# رسم بياني
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# CER per field
field_cer = test_df.groupby("field")["cer"].mean().sort_values()
colors = ["#4CAF50" if v < 0.05 else "#FFC107" if v < 0.15 else "#F44336" for v in field_cer]
field_cer.plot(kind="barh", ax=axes[0], color=colors)
axes[0].set_title("CER per Field")
axes[0].axvline(0.05, color="green", linestyle="--", alpha=0.5)

# Confidence distribution
axes[1].hist(test_df["conf"], bins=30, color="#2196F3", edgecolor="white")
axes[1].set_title("Confidence Distribution")
axes[1].axvline(0.8, color="red", linestyle="--", label="threshold")
axes[1].legend()

# Exact match per field
exact = test_df.groupby("field")["exact"].mean() * 100
exact.sort_values().plot(kind="barh", ax=axes[2], color="#9C27B0")
axes[2].set_title("Exact Match %")
axes[2].axvline(90, color="green", linestyle="--")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. تحليل الأخطاء

# %%
errors = test_df[~test_df["exact"]].copy()
errors["char_errors"] = errors.apply(
    lambda r: editdistance.eval(r["pred"], r["gt"]), axis=1
)

print(f"❌ Total errors: {len(errors)} / {len(test_df)}")
print(f"\nTop 10 Worst Predictions:")
print("─" * 55)

for _, row in errors.nlargest(10, "char_errors").iterrows():
    print(f"  [{row['field']}] CER={row['cer']:.2f}")
    print(f"    GT  : {row['gt']}")
    print(f"    PRED: {row['pred']}")
    print()

# %% [markdown]
# ## 6. Benchmark السرعة

# %%
import statistics

# Warm-up
for _ in range(5):
    dummy = np.random.randint(0, 255, (48, 320, 3), dtype=np.uint8)
    ocr.recognize(dummy)

# Benchmark
times = []
for _ in tqdm(range(100), desc="Benchmarking"):
    dummy = np.random.randint(0, 255, (48, 320, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    ocr.recognize(dummy)
    times.append((time.perf_counter() - t0) * 1000)

print(f"\n⚡ Benchmark (100 runs):")
print(f"   Avg : {statistics.mean(times):.1f} ms")
print(f"   P50 : {statistics.median(times):.1f} ms")
print(f"   P95 : {sorted(times)[95]:.1f} ms")
print(f"   Throughput: {1000/statistics.mean(times):.0f} fields/sec")

# %% [markdown]
# ## 7. اختبار الـ API

# %%
# تشغيل السيرفر (في terminal منفصل):
#   uvicorn app.main:app --host 0.0.0.0 --port 8000

import requests

BASE = "http://localhost:8000"

# Health check
try:
    r = requests.get(f"{BASE}/health", timeout=3)
    print(f"✅ API Status: {r.json()['status']}")
except:
    print("⚠️ API not running! Start with: uvicorn app.main:app --port 8000")

# %%
# اختبار OCR
test_image = list((ROOT / "rec" / "images").glob("*.jpg"))[0]

with open(test_image, "rb") as f:
    r = requests.post(
        f"{BASE}/ocr/single-field",
        files={"file": ("test.jpg", f, "image/jpeg")},
        params={"field": test_image.stem.split("_")[-1]},
    )

if r.status_code == 200:
    data = r.json()
    print(f"📝 Text : {data['text']}")
    print(f"📊 Conf : {data['confidence']}")
    print(f"⏱️ Speed: {data['latency_ms']}ms")
else:
    print(f"❌ Error: {r.text}")

# %% [markdown]
# ---
# ## ✅ المشروع مكتمل!
# 
# | المكون | الحالة |
# |---|---|
# | Dataset (crops) | ✅ |
# | Text Labels | ✅ |
# | PaddleOCR Training | ✅ |
# | ONNX Export | ✅ |
# | Evaluation (CER/WER) | ✅ |
# | Speed Benchmark | ✅ |
# | FastAPI Service | ✅ |

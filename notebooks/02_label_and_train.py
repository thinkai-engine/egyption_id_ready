# %% [markdown]
# # 🇪🇬 Egyptian ID OCR — Notebook 2: استخراج النصوص + التدريب
#
# هذا الـ Notebook يستخرج النصوص من الحقول المقصوصة ثم يدرّب PaddleOCR.
#
# **المدخلات**: `crops_metadata.csv` + `rec/images/`  
# **المخرجات**: `crops_labeled.csv` + `rec/train.txt` + trained model

# %% [markdown]
# ## 1. إعداد البيئة

# %%
import sys, os
from pathlib import Path

ROOT = Path(os.getcwd())
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# تحميل metadata
crops_df = pd.read_csv(ROOT / "crops_metadata.csv")
print(f"📋 Total crops: {len(crops_df):,}")
print(f"   Splits: {crops_df['split'].value_counts().to_dict()}")

# %% [markdown]
# ## 2. استخراج النصوص — اختيار الـ Engine

# %%
# ═══════════════════════════════════════════════════
#  اختار واحد من الطرق الثلاثة:
# ═══════════════════════════════════════════════════

METHOD = "gemini"   # "qari" | "gemini" | "both"
GEMINI_KEY = ""     # ← ضع الـ API key هنا لو هتستخدم Gemini

# %% [markdown]
# ### Option A: Gemini API

# %%
if METHOD in ("gemini", "both"):
    from gemini_ocr import GeminiOCR
    
    gemini = GeminiOCR(api_key=GEMINI_KEY)
    
    # اختبار على عينة
    sample = crops_df[crops_df["split"] == "train"].iloc[0]
    img_path = ROOT / sample["image_path"]
    text = gemini.extract(str(img_path), sample["field"])
    print(f"🔍 Field: {sample['field']}")
    print(f"📝 Text : {text}")

# %% [markdown]
# ### Option B: QARI-OCR

# %%
if METHOD in ("qari", "both"):
    from qari_ocr import QariOCR
    
    qari = QariOCR(use_4bit=False)  # True لو VRAM أقل من 6GB
    
    # اختبار على عينة
    sample = crops_df[crops_df["split"] == "train"].iloc[0]
    img_path = ROOT / sample["image_path"]
    text = qari.extract(str(img_path), sample["field"])
    print(f"🔍 Field: {sample['field']}")
    print(f"📝 Text : {text}")

# %% [markdown]
# ## 3. تشغيل الاستخراج على كل الـ Dataset

# %%
import time

if "label_text" not in crops_df.columns:
    crops_df["label_text"] = ""

splits_to_label = ["train", "valid"]  # الـ test للتقييم فقط
subset = crops_df[
    crops_df["split"].isin(splits_to_label) &
    (crops_df["label_text"] == "")
]
print(f"📤 Pending: {len(subset):,} crops")

# %%
for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Labeling"):
    img_path = ROOT / row["image_path"]
    if not img_path.exists():
        continue
    
    try:
        if METHOD == "gemini":
            text = gemini.extract(str(img_path), row["field"])
            time.sleep(0.4)  # rate limit
        elif METHOD == "qari":
            text = qari.extract(str(img_path), row["field"])
        else:  # both
            text = qari.extract(str(img_path), row["field"])
            crops_df.at[idx, "gemini_text"] = gemini.extract(str(img_path), row["field"])
            time.sleep(0.4)
        
        crops_df.at[idx, "label_text"] = text
    except Exception as e:
        print(f"⚠️ {row['image_path']}: {e}")
    
    # حفظ كل 100 crop
    if idx % 100 == 0:
        crops_df.to_csv(ROOT / "crops_labeled.csv", index=False, encoding="utf-8-sig")

# حفظ نهائي
crops_df.to_csv(ROOT / "crops_labeled.csv", index=False, encoding="utf-8-sig")
labeled = (crops_df["label_text"].str.len() > 0).sum()
print(f"\n✅ Labeled: {labeled:,} / {len(crops_df):,}")

# %% [markdown]
# ## 4. فحص عينة من النتائج

# %%
import cv2

labeled_df = crops_df[crops_df["label_text"].str.len() > 0]
samples = labeled_df.sample(min(8, len(labeled_df)), random_state=42)

fig, axes = plt.subplots(2, 4, figsize=(18, 5))
for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    img_path = ROOT / row["image_path"]
    if img_path.exists():
        crop = cv2.imread(str(img_path))
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row['field']}\n{row['label_text'][:30]}", fontsize=7)
    ax.axis("off")

plt.suptitle("Labeled Crops — Sample", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. تجهيز ملفات PaddleOCR

# %%
from text_cleaner import prepare_paddle_label

crops_df["label_clean"] = crops_df["label_text"].apply(
    lambda x: prepare_paddle_label(str(x)) if pd.notna(x) else ""
)

# بناء القاموس
all_text = "".join(crops_df["label_clean"].dropna().tolist())
unique_chars = sorted(set(c for c in all_text if c.strip() or c == " "))
with open(ROOT / "arabic_dict.txt", "w", encoding="utf-8") as f:
    for c in unique_chars:
        f.write(c + "\n")
print(f"📖 Dictionary: {len(unique_chars)} characters")

# كتابة ملفات التدريب
(ROOT / "rec").mkdir(parents=True, exist_ok=True)
for split, fname in [("train", "train.txt"), ("valid", "val.txt"), ("test", "test.txt")]:
    sub = crops_df[(crops_df["split"] == split) & (crops_df["label_clean"].str.len() > 0)]
    with open(ROOT / "rec" / fname, "w", encoding="utf-8") as f:
        for _, row in sub.iterrows():
            f.write(f"{row['image_path']}\t{row['label_clean']}\n")
    print(f"  {split:6}: {len(sub):>6,} samples → {fname}")

# %% [markdown]
# ## 6. إحصائيات النصوص

# %%
valid = crops_df[crops_df["label_clean"].str.len() > 0]
lens = valid["label_clean"].str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# توزيع الأطوال
axes[0].hist(lens, bins=40, color="#4CAF50", edgecolor="white")
axes[0].set_title("Text Length Distribution")
axes[0].set_xlabel("Characters")
axes[0].axvline(40, color="red", linestyle="--", label="max_text_length")
axes[0].legend()

# عدد العينات لكل حقل
valid["field"].value_counts().plot(kind="barh", ax=axes[1], color="#2196F3")
axes[1].set_title("Samples per Field")

plt.tight_layout()
plt.show()

print(f"\n📊 Text Length: mean={lens.mean():.1f}, max={lens.max()}, >40={int((lens>40).sum())}")

# %% [markdown]
# ## 7. Fine-tuning PaddleOCR
#
# > ⚠️ هذه الخلية تشغّل التدريب — تأكد من إعداد كل شيء قبل التشغيل.

# %%
# ═══════════════════════════════════════════════════
#  التدريب — PaddleOCR
# ═══════════════════════════════════════════════════

import subprocess

# تنزيل النموذج المُدرَّب مسبقاً لو غير موجود
pretrained = ROOT / "arabic_PP-OCRv3_rec_train"
if not pretrained.exists():
    print("⬇️ Downloading pretrained model...")
    subprocess.run([
        "wget", "-q",
        "https://paddleocr.bj.bcebos.com/PP-OCRv3/arabic/arabic_PP-OCRv3_rec_train.tar"
    ], cwd=str(ROOT))
    subprocess.run(["tar", "-xf", "arabic_PP-OCRv3_rec_train.tar"], cwd=str(ROOT))
    print("✅ Downloaded")

# %%
# بدء التدريب
# لاستئناف التدريب من checkpoint، عدّل checkpoints في الـ config:
#   checkpoints: ./output/egyptian_id_rec/latest

print("🏋️ Starting training...")
print("   Config: configs/egyptian_id_rec.yml")
print("   GPU: ✅")
print("   Checkpoint saved every 5 epochs")
print("   Resume: set 'checkpoints' in config to resume\n")

result = subprocess.run(
    ["python", "tools/train.py", "-c", str(ROOT / "configs" / "egyptian_id_rec.yml")],
    cwd=str(ROOT / "PaddleOCR") if (ROOT / "PaddleOCR").exists() else str(ROOT),
)

if result.returncode == 0:
    print("\n✅ Training complete!")
else:
    print(f"\n❌ Training failed (exit code: {result.returncode})")

# %% [markdown]
# ## 8. استئناف التدريب من Checkpoint
#
# لو التدريب وقف، شغّل الخلية دي بعد ما تعدّل الـ config:
# ```yaml
# Global:
#   checkpoints: ./output/egyptian_id_rec/latest
# ```

# %%
# استئناف من آخر checkpoint
CHECKPOINT = ROOT / "output" / "egyptian_id_rec" / "latest"

if CHECKPOINT.exists():
    print(f"▶️ Resuming from: {CHECKPOINT}")
    result = subprocess.run([
        "python", "tools/train.py",
        "-c", str(ROOT / "configs" / "egyptian_id_rec.yml"),
        "-o", f"Global.checkpoints={CHECKPOINT}",
    ], cwd=str(ROOT / "PaddleOCR") if (ROOT / "PaddleOCR").exists() else str(ROOT))
else:
    print(f"❌ No checkpoint found at {CHECKPOINT}")
    print("   Run training first (Cell 7)")

# %% [markdown]
# ---
# ✅ **التدريب انتهى!** → شغّل الـ Notebook التالت: `03_evaluate_and_deploy.py`

# %% [markdown]
# # 🇪🇬 Egyptian ID OCR — Notebook 1: بناء الـ Dataset
# 
# هذا الـ Notebook يقص حقول البطاقة من الصور باستخدام YOLO labels.
# 
# **المدخلات**: `train/`, `valid/`, `test/` (images + labels)  
# **المخرجات**: `rec/images/` + `crops_metadata.csv`

# %% [markdown]
# ## 1. إعداد البيئة

# %%
import sys, os
from pathlib import Path

ROOT = Path(os.getcwd())
sys.path.insert(0, str(ROOT))
print(f"📂 Project root: {ROOT}")

# %%
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 4)

# %% [markdown]
# ## 2. فحص بنية المشروع

# %%
splits = {}
for split in ["train", "valid", "test"]:
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    n_imgs = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    n_lbls = len(list(lbl_dir.glob("*.txt")))
    splits[split] = {"images": n_imgs, "labels": n_lbls}
    print(f"  {split:6}: {n_imgs:>6,} images | {n_lbls:>6,} labels")

total = sum(s["images"] for s in splits.values())
print(f"\n  Total: {total:,} images ✅")

# %% [markdown]
# ## 3. فحص عينة من الـ Labels

# %%
from label_reader import parse_yolo_label, DEFAULT_FIELD_NAMES

# قراءة label واحد كمثال
sample_lbl = list((ROOT / "test" / "labels").glob("*.txt"))[0]
sample_img = ROOT / "test" / "images" / (sample_lbl.stem + ".jpg")

img = cv2.imread(str(sample_img))
h, w = img.shape[:2]
fields = parse_yolo_label(str(sample_lbl), w, h)

print(f"📄 Label: {sample_lbl.name}")
print(f"🖼️ Image: {w}x{h}\n")
print(f"{'Class ID':<10} {'Field Name':<18} {'BBox':}")
print("─" * 55)
for f in fields:
    print(f"{f['class_id']:<10} {f['class_name']:<18} {f['bbox']}")

# %% [markdown]
# ## 4. معاينة الحقول على الصورة

# %%
vis = img.copy()
colors = plt.cm.Set3(np.linspace(0, 1, 24))[:, :3] * 255

for f in fields:
    x1, y1, x2, y2 = f["bbox"]
    c = tuple(int(v) for v in colors[f["class_id"] % 24])
    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
    cv2.putText(vis, f["class_name"], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

plt.figure(figsize=(14, 10))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title("Detected Fields")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. قص الحقول من كل الـ Splits

# %%
from crop_builder import build_crops_from_split
from label_reader import OCR_FIELDS

OUTPUT_DIR = ROOT / "rec" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_dfs = []
for split in ["train", "valid", "test"]:
    split_path = ROOT / split
    if not (split_path / "images").exists():
        print(f"⚠️ Skipping {split}")
        continue
    df = build_crops_from_split(
        split=split,
        split_path=split_path,
        output_dir=OUTPUT_DIR,
    )
    all_dfs.append(df)

crops_df = pd.concat(all_dfs, ignore_index=True)
crops_df.to_csv(ROOT / "crops_metadata.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Total crops: {len(crops_df):,}")

# %% [markdown]
# ## 6. إحصائيات الـ Dataset

# %%
print(f"\n📊 Crops per split:")
display(crops_df["split"].value_counts().to_frame("count"))

print(f"\n📊 Crops per field:")
display(crops_df["field"].value_counts().to_frame("count"))

# %%
# معاينة عينة من الحقول المقصوصة
fig, axes = plt.subplots(2, 5, figsize=(16, 4))
samples = crops_df.sample(min(10, len(crops_df)), random_state=42)

for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    img_path = ROOT / row["image_path"]
    if img_path.exists():
        crop = cv2.imread(str(img_path))
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(row["field"], fontsize=8)
    ax.axis("off")

plt.suptitle("Sample Cropped Fields", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. جودة الحقول

# %%
from preprocessing import assess_field_quality

quality_records = []
for _, row in tqdm(crops_df.sample(min(500, len(crops_df))).iterrows(),
                   desc="Assessing quality"):
    img_path = ROOT / row["image_path"]
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    q = assess_field_quality(img)
    q["field"] = row["field"]
    quality_records.append(q)

qdf = pd.DataFrame(quality_records)
print("\n📊 Quality Distribution:")
display(qdf["quality"].value_counts().to_frame("count"))
print(f"\n⚠️ Issues found:")
all_issues = [i for lst in qdf["issues"] for i in lst]
display(pd.Series(all_issues).value_counts().to_frame("count"))

# %% [markdown]
# ---
# ✅ **الـ Dataset جاهز!** → شغّل الـ Notebook التاني: `02_label_and_train.py`

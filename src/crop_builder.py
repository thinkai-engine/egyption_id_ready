"""
Crop Builder
=============
Crop individual fields from ID card images using YOLO labels.
Falls back to ONNX field detector when labels are absent.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .label_reader import parse_yolo_label, OCR_FIELDS


def crop_field(
    img: np.ndarray,
    bbox: list,
    padding: int = 4,
) -> np.ndarray | None:
    """Crop a single field with padding to prevent letter clipping."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def build_crops_from_split(
    split: str,
    split_path: Path,
    output_dir: Path,
    detector=None,
    field_mapping: dict = None,
) -> pd.DataFrame:
    """
    Crop all fields from a single data split (train / valid / test).

    Parameters
    ----------
    split : str
        Split name ("train", "valid", "test").
    split_path : Path
        Directory containing images/ and labels/ subdirectories.
    output_dir : Path
        Where to save cropped field images.
    detector : YOLOFieldDetector, optional
        Fallback detector for images without labels.
    field_mapping : dict, optional
        class_id → field_name mapping override.

    Returns
    -------
    pd.DataFrame with columns:
        image_path, field, class_id, split, orig_image, label_text
    """
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )

    records = []
    stats = {"processed": 0, "crops": 0, "no_label": 0, "errors": 0}

    for img_path in tqdm(image_files, desc=f"[{split}] Cropping"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats["errors"] += 1
                continue

            h, w = img.shape[:2]
            lbl_path = labels_dir / (img_path.stem + ".txt")

            # Try labels first, fall back to detector
            if lbl_path.exists():
                fields = parse_yolo_label(
                    str(lbl_path), w, h, field_mapping
                )
            elif detector is not None:
                fields = detector.detect(img)
            else:
                stats["no_label"] += 1
                continue

            if not fields:
                stats["no_label"] += 1
                continue

            for field in fields:
                # Skip non-text fields (e.g. photo)
                if field["class_name"] not in OCR_FIELDS:
                    continue

                crop = crop_field(img, field["bbox"])
                if crop is None:
                    continue

                save_name = (
                    f"{split}_{img_path.stem}_{field['class_name']}.jpg"
                )
                save_path = output_dir / save_name
                cv2.imwrite(str(save_path), crop)

                records.append({
                    "image_path": f"rec/images/{save_name}",
                    "field": field["class_name"],
                    "class_id": field["class_id"],
                    "split": split,
                    "orig_image": img_path.name,
                    "label_text": "",   # filled later by OCR engine
                })
                stats["crops"] += 1

            stats["processed"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"⚠️  Error on {img_path.name}: {e}")

    print(
        f"  ✅ {stats['processed']} images → "
        f"{stats['crops']} crops | "
        f"⚠️ {stats['no_label']} without labels | "
        f"❌ {stats['errors']} errors"
    )
    return pd.DataFrame(records)

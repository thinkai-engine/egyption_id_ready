"""
Crop Builder
=============
Crop individual fields from ID card images using YOLO labels.
Falls back to ONNX field detector when labels are absent.

Two-Stage Detection:
1. Detect and crop ID card region (removes background noise)
2. Detect fields within the cropped card
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .label_reader import parse_yolo_label, OCR_FIELDS, detect_card_region, CARD_CLASS_NAMES


def crop_card_region(image: np.ndarray, label_path: Path) -> tuple[np.ndarray, dict]:
    """
    Crop image to ID card region using YOLO labels.
    
    Returns:
        Tuple of (card_crop, card_info) or (original_image, None) if no card found
    """
    if not label_path.exists():
        return image, None
    
    h, w = image.shape[:2]
    card_info = detect_card_region(str(label_path), w, h)
    
    if card_info is None:
        return image, None
    
    x1, y1, x2, y2 = card_info["bbox"]
    card_crop = image[y1:y2, x1:x2]
    card_info["original_size"] = (w, h)
    card_info["crop_size"] = (card_crop.shape[1], card_crop.shape[0])
    
    return card_crop, card_info


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
    use_two_stage: bool = True,  # New parameter for two-stage detection
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
    use_two_stage : bool, optional
        If True, first crop to ID card region, then detect fields (default: True).

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
    stats = {"processed": 0, "crops": 0, "no_label": 0, "errors": 0, "card_cropped": 0}

    for img_path in tqdm(image_files, desc=f"[{split}] Cropping"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats["errors"] += 1
                continue

            h, w = img.shape[:2]
            lbl_path = labels_dir / (img_path.stem + ".txt")
            
            # Initialize card variables
            card_img = None
            card_info = None

            # Two-stage detection: first crop to card, then detect fields
            if use_two_stage and lbl_path.exists():
                # Stage 1: Crop to card region
                card_img, card_info = crop_card_region(img, lbl_path)
                
                if card_info is not None:
                    stats["card_cropped"] += 1
                    # Stage 2: Parse field labels (coordinates relative to original image)
                    # Need to adjust coordinates to be relative to cropped card
                    offset_x, offset_y = 0, 0
                    if card_info:
                        offset_x = card_info["bbox"][0]
                        offset_y = card_info["bbox"][1]
                    
                    # Parse fields from labels
                    fields = parse_yolo_label(
                        str(lbl_path), w, h, field_mapping
                    )
                    
                    # Adjust field coordinates to be relative to card crop
                    for field in fields:
                        x1, y1, x2, y2 = field["bbox"]
                        field["bbox"] = [
                            max(0, x1 - offset_x),
                            max(0, y1 - offset_y),
                            max(0, x2 - offset_x),
                            max(0, y2 - offset_y),
                        ]
                else:
                    # No card detected, use full image
                    fields = parse_yolo_label(
                        str(lbl_path), w, h, field_mapping
                    )
            elif lbl_path.exists():
                # Original behavior: parse labels directly
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

            # Use card image if available, otherwise original
            process_img = card_img if (use_two_stage and card_info is not None) else img

            for field in fields:
                # Skip non-text fields (e.g. photo)
                if field["class_name"] not in OCR_FIELDS:
                    continue

                crop = crop_field(process_img, field["bbox"])
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
                    "card_cropped": card_info is not None if use_two_stage else False,
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
    if stats.get("card_cropped", 0) > 0:
        print(f"   📋 Card cropping applied: {stats['card_cropped']} images")
    return pd.DataFrame(records)

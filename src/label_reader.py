"""
YOLO Label Reader
=================
Parse YOLO-format .txt label files and convert normalised coords
to pixel bounding boxes.

Label format per line:  class_id  cx  cy  w  h   (all normalised 0-1)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# ── Default class mapping ─────────────────────────────────────
# Egyptian ID card fields — 24 classes discovered from labels.
# class_id → field_name   (update this once data.yaml is available)

# Card detection classes (for two-stage detection)
CARD_CLASS_NAMES = {
    0: "id_card",            # بطاقة الهوية (full card boundary)
}

# Field detection classes (within the card)
DEFAULT_FIELD_NAMES = {
    0:  "job_title",          # المهنة
    1:  "photo",              # الصورة الشخصية
    2:  "expiry_date",        # تاريخ الانتهاء
    3:  "birth_date",         # تاريخ الميلاد
    4:  "religion",           # الديانة
    5:  "name",               # الاسم
    6:  "address",            # العنوان
    7:  "national_id",        # الرقم القومي
    8:  "marital_status",     # الحالة الزوجية
    9:  "gender",             # الجنس
    10: "governorate",        # المحافظة
    11: "husband_name",       # اسم الزوج
    12: "issue_date",         # تاريخ الإصدار
    23: "serial_number",      # الرقم التسلسلي
}

# Fields that contain extractable Arabic/numeric text
OCR_FIELDS = {
    "name", "national_id", "birth_date", "address",
    "governorate", "gender", "expiry_date", "religion",
    "marital_status", "job_title", "husband_name",
    "issue_date", "serial_number",
}


def get_field_name(class_id: int, mapping: dict = None) -> str:
    """Return human-readable field name for a YOLO class_id."""
    m = mapping or DEFAULT_FIELD_NAMES
    return m.get(class_id, f"field_{class_id}")


def parse_yolo_label(
    label_path: str,
    img_w: int,
    img_h: int,
    field_mapping: dict = None,
) -> list[dict]:
    """
    Read a YOLO label file and return pixel-coordinate bounding boxes.

    Returns
    -------
    list of dicts with keys:
        class_id, class_name, bbox [x1,y1,x2,y2], conf, source
    """
    fields = []
    p = Path(label_path)
    if not p.exists():
        return fields

    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cid = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h

            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(img_w, int(cx + bw / 2))
            y2 = min(img_h, int(cy + bh / 2))

            if x2 > x1 and y2 > y1:
                fields.append({
                    "class_id": cid,
                    "class_name": get_field_name(cid, field_mapping),
                    "bbox": [x1, y1, x2, y2],
                    "conf": 1.0,       # manual label = full confidence
                    "source": "label",
                })
    return fields


def detect_card_region(label_path: str, img_w: int, img_h: int) -> dict:
    """
    Detect the ID card region from YOLO labels.
    Looks for class_id 0 in CARD_CLASS_NAMES (id_card).
    
    Returns:
        dict with keys: bbox [x1,y1,x2,y2], conf, or None if not found
    """
    p = Path(label_path)
    if not p.exists():
        return None
    
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cid = int(parts[0])
            
            # Check if this is the id_card class
            if cid != 0:  # id_card is class 0
                continue
            
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(img_w, int(cx + bw / 2))
            y2 = min(img_h, int(cy + bh / 2))
            
            if x2 > x1 and y2 > y1:
                return {
                    "bbox": [x1, y1, x2, y2],
                    "conf": 1.0,
                    "class_id": cid,
                    "class_name": "id_card",
                }
    
    return None


def crop_to_card(image: np.ndarray, label_path: str) -> Tuple[np.ndarray, dict]:
    """
    Crop image to ID card region using YOLO labels.
    
    Args:
        image: Full image (BGR)
        label_path: Path to YOLO label file
        
    Returns:
        Tuple of (cropped_image, card_info dict)
        If no card found, returns (original_image, None)
    """
    img_h, img_w = image.shape[:2]
    card_info = detect_card_region(label_path, img_w, img_h)
    
    if card_info is None:
        return image, None
    
    x1, y1, x2, y2 = card_info["bbox"]
    card_crop = image[y1:y2, x1:x2]
    
    # Store offset for adjusting field coordinates
    card_info["crop_offset"] = (x1, y1)
    card_info["original_size"] = (img_w, img_h)
    card_info["crop_size"] = (card_crop.shape[1], card_crop.shape[0])
    
    return card_crop, card_info


def adjust_field_bbox_to_crop(bbox: List[int], offset_x: int, offset_y: int, crop_w: int, crop_h: int) -> List[int]:
    """
    Adjust field bounding box to be relative to cropped card.
    
    Args:
        bbox: Original field bbox [x1, y1, x2, y2]
        offset_x: X offset of card crop
        offset_y: Y offset of card crop
        crop_w: Width of card crop
        crop_h: Height of card crop
        
    Returns:
        Adjusted bbox [x1, y1, x2, y2] relative to card crop
    """
    x1, y1, x2, y2 = bbox
    
    # Adjust to card crop coordinates
    adj_x1 = max(0, x1 - offset_x)
    adj_y1 = max(0, y1 - offset_y)
    adj_x2 = max(0, x2 - offset_x)
    adj_y2 = max(0, y2 - offset_y)
    
    # Clamp to crop dimensions
    adj_x1 = min(adj_x1, crop_w)
    adj_y1 = min(adj_y1, crop_h)
    adj_x2 = min(adj_x2, crop_w)
    adj_y2 = min(adj_y2, crop_h)
    
    return [adj_x1, adj_y1, adj_x2, adj_y2]

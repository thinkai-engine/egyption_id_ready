"""
YOLO Label Reader
=================
Parse YOLO-format .txt label files and convert normalised coords
to pixel bounding boxes.

Label format per line:  class_id  cx  cy  w  h   (all normalised 0-1)
"""

from pathlib import Path

# ── Default class mapping ─────────────────────────────────────
# Egyptian ID card fields — 24 classes discovered from labels.
# class_id → field_name   (update this once data.yaml is available)
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

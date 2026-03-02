# NASO7Y Model Class Mappings

## Field Detection Classes (field_detection.pt)

The NASO7Y field detection model uses **31 classes** for Egyptian ID card field detection.

### Valid Fields (for OCR pipeline)

| Class ID | Name | Arabic | Description |
|----------|------|--------|-------------|
| 0 | address | العنوان | Full address |
| 2 | dob | تاريخ الميلاد | Date of birth |
| 3 | expiry | تاريخ الانتهاء | Expiry date |
| 4 | firstName | الاسم الأول | First name |
| 22 | issue | تاريخ الإصدار | Issue date |
| 23 | job | المهنة | Job title |
| 24 | lastName | اسم العائلة | Last name |
| 25 | nid | الرقم القومي | National ID number |
| 26 | nid_back | الرقم القومي (خلف) | National ID back |
| 27 | photo | الصورة | Photo |
| 28 | poe | مكان الاستخراج | Place of extraction |
| 29 | serial | الرقم التسلسلي | Serial number |

### Invalid/Low Confidence Fields (filtered out)

| Class ID | Name | Description |
|----------|------|-------------|
| 6 | invalid_address | Low confidence address |
| 7 | invalid_barcode | Low confidence barcode |
| 8 | invalid_demo | Low confidence demo |
| 9 | invalid_dob | Low confidence DOB |
| 10 | invalid_expiry | Low confidence expiry |
| 11 | invalid_firstName | Low confidence firstName |
| 12 | invalid_logo | Low confidence logo |
| 13 | invalid_job | Low confidence job |
| 14 | invalid_lastName | Low confidence lastName |
| 15 | invalid_nid | Low confidence NID |
| 16 | invalid_nid_back | Low confidence NID back |
| 17 | invalid_photo | Low confidence photo |
| 18 | invalid_poe | Low confidence POE |
| 19 | invalid_serial | Low confidence serial |
| 20 | invalid_watermark_tut | Low confidence watermark |

### Other Fields

| Class ID | Name | Description |
|----------|------|-------------|
| 1 | demo | Demo/test field |
| 5 | front_logo | Front logo |
| 30 | watermark_tut | Watermark tutorial |

---

## Card Detection Classes (card_detection.pt)

The NASO7Y card detection model uses **8 corner/edge classes** to detect ID card boundaries.

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | back-bottom | Bottom edge of back side |
| 1 | back-left | Left edge of back side |
| 2 | back-right | Right edge of back side |
| 3 | back-up | Top edge of back side |
| 4 | front-bottom | Bottom edge of front side |
| 5 | front-left | Left edge of front side |
| 6 | front-right | Right edge of front side |
| 7 | front-up | Top edge of front side |

**Note:** The card detection model detects card corners/edges rather than the full card. The `CardDetector` class combines all detected corners to compute the full card bounding box.

---

## NID Digit Detection Classes (nid_detection.pt)

The NID detection model detects individual digits (0-9) in the national ID number.

| Class ID | Name |
|----------|------|
| 0-9 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |

---

## Usage in Code

```python
from src.card_detector import CardDetector

detector = CardDetector(
    card_model_path="weights/card_detection.pt",
    field_model_path="weights/field_detection.pt",
)

# Run detection (automatically filters invalid fields)
card_image, fields = detector.detect_full(image)

# Fields will only contain valid classes:
# address, dob, expiry, firstName, issue, job, lastName, nid, photo, serial
```

---

## Class Mapping Comparison

### Old Mapping (Your Project)
```python
{
    0: "job_title",
    1: "photo", 
    2: "expiry_date",
    3: "birth_date",
    5: "name",
    6: "address",
    7: "national_id",
    ...
}
```

### New Mapping (NASO7Y)
```python
{
    0: "address",
    2: "dob",
    3: "expiry",
    4: "firstName",
    23: "job",
    24: "lastName",
    25: "nid",
    27: "photo",
    29: "serial",
    ...
}
```

**Important:** The class indices are completely different! You cannot use your existing labels with the NASO7Y models directly. You would need to either:

1. **Retrain NASO7Y models** with your class mapping, OR
2. **Re-annotate your dataset** with NASO7Y class mapping, OR
3. **Create a class mapping translation** layer (not recommended for production)

---

## Recommended Field Mapping for Your Pipeline

For your Egyptian ID OCR pipeline, use these NASO7Y field names:

```python
FIELD_MAPPING = {
    "firstName": "name_first",      # الاسم الأول
    "lastName": "name_last",        # اسم العائلة  
    "nid": "national_id",           # الرقم القومي
    "address": "address",           # العنوان
    "dob": "birth_date",            # تاريخ الميلاد
    "expiry": "expiry_date",        # تاريخ الانتهاء
    "issue": "issue_date",          # تاريخ الإصدار
    "job": "job_title",             # المهنة
    "serial": "serial_number",      # الرقم التسلسلي
    "photo": "photo",               # الصورة
}
```

---

**Source:** https://github.com/NASO7Y/OCR_Egyptian_ID
**Date:** March 2, 2025

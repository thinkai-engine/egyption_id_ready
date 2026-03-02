# Class Mapping: NASO7Y ↔ Project Format

## Overview

This document explains how to use NASO7Y pre-trained models with your existing Egyptian ID OCR dataset that uses a different class labeling format.

## Problem

**NASO7Y Model Classes** (from `field_detection.pt`):
```
0: address, 2: dob, 3: expiry, 4: firstName, 23: job, 24: lastName, 25: nid, ...
```

**Your Project Classes**:
```
0: job_title, 1: photo, 2: expiry_date, 3: birth_date, 5: name, 6: address, 7: national_id, ...
```

The class indices and names are **completely different**!

## Solution: Class Mapping Translation

The `src/class_mapping.py` module provides translation between the two formats.

---

## Usage

### Option 1: Automatic Translation in CardDetector

```python
from src.card_detector import CardDetector

detector = CardDetector(
    card_model_path="weights/card_detection.pt",
    field_model_path="weights/field_detection.pt",
)

# Get results in NASO7Y format
card_img, fields = detector.detect_full(
    image,
    translate_to_project=False  # Keep NASO7Y names
)
# fields: {photo, firstName, dob, lastName, address, nid, serial}

# Get results in Project format (translated)
card_img, fields = detector.detect_full(
    image,
    translate_to_project=True  # Translate to project names
)
# fields: {photo, name, birth_date, address, national_id, serial_number}
# Note: firstName + lastName → merged into 'name'
```

### Option 2: Get Both Formats with Mapping

```python
card_img, fields, mapping = detector.detect_full_with_mapping(image)

print(f"Detected fields: {fields.keys()}")
print(f"Class mappings: {mapping}")

# Output:
# Detected fields: {photo, firstName, dob, lastName, address, nid, serial}
# Class mappings: {
#     'photo': 'photo',
#     'firstName': 'name',
#     'dob': 'birth_date',
#     'lastName': 'name',
#     'address': 'address',
#     'nid': 'national_id',
#     'serial': 'serial_number'
# }
```

### Option 3: Manual Translation

```python
from src.class_mapping import (
    translate_class_id,
    translate_class_name,
    get_mapper,
)

# Translate class ID
project_id = translate_class_id(25, from_format='naso7y', to_format='project')
print(f"NID (25) → National ID ({project_id})")  # 7

# Translate class name
project_name = translate_class_name('nid', from_format='naso7y', to_format='project')
print(f"nid → {project_name}")  # national_id

# Use mapper object
mapper = get_mapper()
if mapper.has_equivalent(25, 'naso7y'):
    print("Has equivalent in project format")
```

---

## Class Mapping Table

### Fields with Equivalents

| NASO7Y ID | NASO7Y Name | Project ID | Project Name | Arabic |
|-----------|-------------|------------|--------------|--------|
| 25 | nid | 7 | national_id | الرقم القومي |
| 29 | serial | 23 | serial_number | الرقم التسلسلي |
| 2 | dob | 3 | birth_date | تاريخ الميلاد |
| 3 | expiry | 2 | expiry_date | تاريخ الانتهاء |
| 22 | issue | 12 | issue_date | تاريخ الإصدار |
| 0 | address | 6 | address | العنوان |
| 23 | job | 0 | job_title | المهنة |
| 27 | photo | 1 | photo | الصورة |

### Special Case: Name Fields

| NASO7Y ID | NASO7Y Name | Project ID | Project Name | Note |
|-----------|-------------|------------|--------------|------|
| 4 | firstName | 5 | name | Merged |
| 24 | lastName | 5 | name | Merged |

**Note:** NASO7Y separates first and last names, while your project uses a single `name` field. The `CardDetector` automatically merges them with averaged confidence.

### Fields without Equivalents

| NASO7Y ID | NASO7Y Name | Reason |
|-----------|-------------|--------|
| 26 | nid_back | No equivalent in project |
| 28 | poe | No equivalent in project |

| Project ID | Project Name | Reason |
|------------|--------------|--------|
| 4 | religion | No equivalent in NASO7Y |
| 8 | marital_status | No equivalent in NASO7Y |
| 9 | gender | No equivalent in NASO7Y |
| 10 | governorate | No equivalent in NASO7Y |
| 11 | husband_name | No equivalent in NASO7Y |

---

## Migration Strategy

### Phase 1: Use NASO7Y Models as-Is

Start by using NASO7Y models with their native class format:

```python
detector = CardDetector(
    card_model_path="weights/card_detection.pt",
    field_model_path="weights/field_detection.pt",
)

card_img, fields = detector.detect_full(image, translate_to_project=False)
```

### Phase 2: Translate to Project Format

When ready to integrate with existing pipeline:

```python
card_img, fields = detector.detect_full(image, translate_to_project=True)
```

### Phase 3: Retrain Models (Optional)

For production use, consider retraining NASO7Y models with your class format:

```bash
# Export NASO7Y model to YOLO format
yolo export model=field_detection.pt format=onnx

# Prepare training data with your class mapping
python scripts/convert_labels_to_naso7y.py

# Retrain with your classes
yolo train model=field_detection.pt data=your_dataset.yaml epochs=100
```

---

## API Reference

### `translate_class_id(class_id, from_format, to_format)`

Translate a class ID between formats.

```python
# NASO7Y → Project
project_id = translate_class_id(25, 'naso7y', 'project')  # Returns 7

# Project → NASO7Y
naso7y_id = translate_class_id(7, 'project', 'naso7y')  # Returns 25
```

### `translate_class_name(class_name, from_format, to_format)`

Translate a class name between formats.

```python
# NASO7Y → Project
project_name = translate_class_name('nid', 'naso7y', 'project')  # 'national_id'

# Project → NASO7Y
naso7y_name = translate_class_name('national_id', 'project', 'naso7y')  # 'nid'
```

### `get_naso7y_valid_classes()`

Get dict of valid NASO7Y field classes (excludes invalid_* classes).

### `get_project_valid_classes()`

Get dict of valid project field classes (text fields only, excludes photo).

### `ClassMapper` class

Utility class for conversion operations:

```python
mapper = ClassMapper()

# Convert IDs
proj_id = mapper.naso7y_to_project_id(25)  # 7
naso7y_id = mapper.project_to_naso7y_id(7)  # 25

# Convert names
proj_name = mapper.naso7y_to_project_name('nid')  # 'national_id'
naso7y_name = mapper.project_to_naso7y_name('national_id')  # 'nid'

# Check equivalence
if mapper.has_equivalent(25, 'naso7y'):
    print("Has equivalent")

# Get unmapped classes
unmapped = mapper.get_unmapped_classes('naso7y')  # [26, 28]
```

---

## Examples

### Example 1: Process Image with Translation

```python
from src.card_detector import CardDetector
import cv2

detector = CardDetector(
    card_model_path="weights/card_detection.pt",
    field_model_path="weights/field_detection.pt",
)

image = cv2.imread("id_card.jpg")

# Get translated results
card_img, fields = detector.detect_full(
    image,
    translate_to_project=True,
)

# Process fields
for field_name, (crop, conf) in fields.items():
    print(f"{field_name}: {conf:.2f}")
    # Output:
    # photo: 0.92
    # name: 0.90
    # birth_date: 0.91
    # address: 0.89
    # national_id: 0.88
    # serial_number: 0.86
```

### Example 2: Merge with Existing Labels

```python
from src.card_detector import CardDetector
from src.label_reader import parse_yolo_label
from src.class_mapping import merge_detections

detector = CardDetector(...)
image = cv2.imread("id_card.jpg")

# Get NASO7Y detections
card_img, naso7y_fields = detector.detect_full(card_img, translate_to_project=False)

# Get project detections from labels
project_dets = parse_yolo_label("labels/001.txt", w, h)

# Merge (prefers project format)
merged = merge_detections(naso7y_fields, project_dets)
```

---

## Files

- `src/class_mapping.py` - Translation utilities
- `src/card_detector.py` - Updated with translation support
- `docs/NASO7Y_CLASSES.md` - Complete class reference

---

**Date:** March 2, 2025
**Status:** ✅ Complete and Tested

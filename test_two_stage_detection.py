#!/usr/bin/env python3
"""
Test Two-Stage Detection Pipeline
==================================
Test card detection → field detection pipeline.

Usage:
    python test_two_stage_detection.py <image_path> [card_model] [field_model]
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.card_detector import CardDetector, load_card_detector
from src.label_reader import detect_card_region, crop_to_card, parse_yolo_label


def test_card_detection(image_path: str, card_model: str = None, field_model: str = None):
    """Test two-stage detection on a single image."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📷 Input image: {image.shape[1]}x{image.shape[0]}")
    
    # Try CardDetector with YOLO models
    print("\n=== Testing CardDetector (YOLO models) ===")
    try:
        detector = load_card_detector(card_model, field_model)
        card_image, fields = detector.detect_full(image)
        
        print(f"✂️  Card crop: {card_image.shape[1]}x{card_image.shape[0]}")
        print(f"📋 Detected fields: {len(fields)}")
        for field_name, (crop, conf) in fields.items():
            print(f"   - {field_name}: {crop.shape[1]}x{crop.shape[0]} (conf: {conf:.2f})")
        
        # Save outputs
        output_dir = ROOT / "output" / "two_stage_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        card_output = output_dir / "card_crop.jpg"
        cv2.imwrite(str(card_output), card_image)
        print(f"💾 Card crop saved to: {card_output}")
        
        for field_name, (crop, conf) in fields.items():
            field_output = output_dir / f"field_{field_name}.jpg"
            cv2.imwrite(str(field_output), crop)
    except Exception as e:
        print(f"⚠️  CardDetector failed: {e}")
    
    # Try label-based detection (if labels exist)
    print("\n=== Testing Label-Based Detection ===")
    image_path = Path(image_path)
    label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    
    if label_path.exists():
        print(f"📋 Found label file: {label_path}")
        
        # Detect card region from labels
        h, w = image.shape[:2]
        card_info = detect_card_region(str(label_path), w, h)
        
        if card_info:
            print(f"✅ Card detected in labels:")
            print(f"   Bbox: {card_info['bbox']}")
            print(f"   Class: {card_info['class_name']} (ID: {card_info['class_id']})")
            
            # Crop to card
            card_crop, _ = crop_to_card(image, str(label_path))
            print(f"✂️  Card crop: {card_crop.shape[1]}x{card_crop.shape[0]}")
            
            # Save card crop
            output_dir = ROOT / "output" / "label_based_test"
            output_dir.mkdir(parents=True, exist_ok=True)
            card_output = output_dir / "card_crop.jpg"
            cv2.imwrite(str(card_output), card_crop)
            print(f"💾 Card crop saved to: {card_output}")
            
            # Parse and display field labels
            fields = parse_yolo_label(str(label_path), w, h)
            print(f"📋 Field labels found: {len(fields)}")
            
            # Adjust field coordinates to card crop
            if card_info:
                offset_x, offset_y = card_info['bbox'][0], card_info['bbox'][1]
                print(f"   Offset: ({offset_x}, {offset_y})")
                
                for field in fields:
                    x1, y1, x2, y2 = field['bbox']
                    adj_bbox = [
                        max(0, x1 - offset_x),
                        max(0, y1 - offset_y),
                        max(0, x2 - offset_x),
                        max(0, y2 - offset_y),
                    ]
                    print(f"   - {field['class_name']}: {field['bbox']} → {adj_bbox}")
        else:
            print("⚠️  No card region found in labels")
    else:
        print(f"⚠️  No label file found: {label_path}")
        print("   (This is expected if you haven't created labels yet)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_two_stage_detection.py <image_path> [card_model] [field_model]")
        print()
        print("Arguments:")
        print("  image_path   Path to test image")
        print("  card_model   Path to card detection YOLO model (optional)")
        print("  field_model  Path to field detection YOLO model (optional)")
        print()
        print("Example:")
        print("  python test_two_stage_detection.py train/images/001.jpg")
        print("  python test_two_stage_detection.py train/images/001.jpg weights/card.pt weights/field.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    card_model = sys.argv[2] if len(sys.argv) > 2 else None
    field_model = sys.argv[3] if len(sys.argv) > 3 else None
    
    test_card_detection(image_path, card_model, field_model)
    
    print("\n" + "="*50)
    print("✅ Test complete! Check output/ directory for results.")

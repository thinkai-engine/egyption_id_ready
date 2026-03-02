"""
Card Detector
=============
Two-stage YOLO detection for Egyptian ID cards:
1. Detect and crop the ID card from the full image
2. Detect fields within the cropped card

This improves accuracy by removing background noise and focusing on the card region.

Supports both NASO7Y model classes and project's original label format.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("⚠️  ultralytics not installed. Install with: pip install ultralytics")

from .class_mapping import (
    get_naso7y_valid_classes,
    get_project_valid_classes,
    translate_class_name,
    ClassMapper,
)


@dataclass
class Detection:
    """Represents a detected object."""
    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


class CardDetector:
    """
    Two-stage Egyptian ID card detector.
    
    Stage 1: Detect and crop the ID card from the full image
    Stage 2: Detect fields within the cropped card
    
    Usage:
        detector = CardDetector(card_model_path, field_model_path)
        card_crop, card_det = detector.detect_card(image)
        fields = detector.detect_fields(card_crop)
    """
    
    # Default class names for card detection (from NASO7Y card_detection.pt)
    # Note: This model detects card corners/edges rather than full card
    # Classes 0-3: back side corners, 4-7: front side corners
    CARD_CLASS_NAMES = {
        0: "back-bottom",     # Bottom edge of back side
        1: "back-left",       # Left edge of back side
        2: "back-right",      # Right edge of back side
        3: "back-up",         # Top edge of back side
        4: "front-bottom",    # Bottom edge of front side
        5: "front-left",      # Left edge of front side
        6: "front-right",     # Right edge of front side
        7: "front-up",        # Top edge of front side
    }
    
    # Helper mapping to group corner detections into "id_card"
    CARD_CORNER_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]  # All classes are card-related

    # Default class names for field detection (from NASO7Y field_detection.pt)
    # Source: https://github.com/NASO7Y/OCR_Egyptian_ID
    FIELD_CLASS_NAMES = {
        # Valid fields
        0:  "address",         # العنوان
        1:  "demo",            # Demo/test field
        2:  "dob",             # Date of birth (تاريخ الميلاد)
        3:  "expiry",          # Expiry date (تاريخ الانتهاء)
        4:  "firstName",       # الاسم الأول
        5:  "front_logo",      # Front logo
        22: "issue",           # Issue date (تاريخ الإصدار)
        23: "job",             # Job title (المهنة)
        24: "lastName",        # اسم العائلة
        25: "nid",             # National ID (الرقم القومي)
        26: "nid_back",        # National ID back
        27: "photo",           # Photo (الصورة)
        28: "poe",             # Place of extraction
        29: "serial",          # Serial number (الرقم التسلسلي)
        30: "watermark_tut",   # Watermark tutorial
        
        # Invalid/low confidence fields
        6:  "invalid_address",
        7:  "invalid_barcode",
        8:  "invalid_demo",
        9:  "invalid_dob",
        10: "invalid_expiry",
        11: "invalid_firstName",
        12: "invalid_logo",
        13: "invalid_job",
        14: "invalid_lastName",
        15: "invalid_nid",
        16: "invalid_nid_back",
        17: "invalid_photo",
        18: "invalid_poe",
        19: "invalid_serial",
        20: "invalid_watermark_tut",
    }
    
    def __init__(
        self,
        card_model_path: Optional[str] = None,
        field_model_path: Optional[str] = None,
        card_conf_threshold: float = 0.5,
        field_conf_threshold: float = 0.5,
    ):
        """
        Initialize the card detector.
        
        Args:
            card_model_path: Path to YOLO model for card detection
            field_model_path: Path to YOLO model for field detection (optional, uses existing if None)
            card_conf_threshold: Confidence threshold for card detection
            field_conf_threshold: Confidence threshold for field detection
        """
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")
        
        self.card_conf_threshold = card_conf_threshold
        self.field_conf_threshold = field_conf_threshold
        
        # Load card detection model
        self.card_model = None
        if card_model_path and Path(card_model_path).exists():
            self.card_model = YOLO(card_model_path)
            print(f"✅ Card detection model loaded: {card_model_path}")
        else:
            print("ℹ️  No card detection model provided. Will use full image.")
        
        # Field detection model is loaded separately if needed
        self.field_model = None
        if field_model_path and Path(field_model_path).exists():
            self.field_model = YOLO(field_model_path)
            print(f"✅ Field detection model loaded: {field_model_path}")
    
    def detect_card(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[Detection]]:
        """
        Stage 1: Detect and crop the ID card from the image.
        
        Note: The NASO7Y card detection model detects card corners/edges.
        We compute the card bounding box from all detected corners.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Tuple of (cropped_card_image, card_detection or None)
        """
        if self.card_model is None:
            # No card model - return full image
            return image, None

        # Run inference
        results = self.card_model(image, verbose=False, conf=self.card_conf_threshold)

        # Collect all card corner detections
        corner_detections = []

        for result in results:
            if result.boxes is None:
                continue

            for i in range(len(result.boxes)):
                box = result.boxes[i]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Check if this is a card corner class
                if cls_id in self.CARD_CORNER_CLASSES:
                    class_name = self.CARD_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    xyxy = box.xyxy[0].cpu().numpy()
                    corner_detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': conf,
                    })

        # Compute card bounding box from all corners
        if corner_detections:
            # Find min/max coordinates across all corners
            all_x1 = [d['bbox'][0] for d in corner_detections]
            all_y1 = [d['bbox'][1] for d in corner_detections]
            all_x2 = [d['bbox'][2] for d in corner_detections]
            all_y2 = [d['bbox'][3] for d in corner_detections]

            x1 = min(all_x1)
            y1 = min(all_y1)
            x2 = max(all_x2)
            y2 = max(all_y2)

            # Calculate average confidence
            avg_conf = sum(d['confidence'] for d in corner_detections) / len(corner_detections)

            card_det = Detection(
                bbox=[x1, y1, x2, y2],
                class_id=0,  # Unified card class
                class_name="id_card",
                confidence=avg_conf,
            )

            # Crop to card region
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                card_crop = image[y1:y2, x1:x2]
                return card_crop, card_det

        # No card detected - return full image
        return image, None

    # Valid field classes (exclude invalid/low-confidence detections)
    VALID_FIELD_CLASSES = {
        0:  "address",
        2:  "dob",
        3:  "expiry",
        4:  "firstName",
        22: "issue",
        23: "job",
        24: "lastName",
        25: "nid",
        26: "nid_back",
        27: "photo",
        28: "poe",
        29: "serial",
    }

    def detect_fields(self, card_image: np.ndarray, valid_only: bool = True) -> List[Detection]:
        """
        Stage 2: Detect fields within the cropped card image.

        Args:
            card_image: Cropped card image from detect_card()
            valid_only: If True, only return valid fields (exclude invalid_* classes)

        Returns:
            List of field detections
        """
        if self.field_model is None:
            return []

        # Run inference
        results = self.field_model(card_image, verbose=False, conf=self.field_conf_threshold)

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for i in range(len(result.boxes)):
                box = result.boxes[i]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = self.FIELD_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                
                # Skip invalid fields if valid_only is True
                if valid_only and cls_id not in self.VALID_FIELD_CLASSES:
                    continue
                
                xyxy = box.xyxy[0].cpu().numpy()

                detections.append(Detection(
                    bbox=[int(x) for x in xyxy],
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                ))

        return detections
    
    def detect_full(
        self,
        image: np.ndarray,
        output_format: str = "naso7y",
        translate_to_project: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, float]]]:
        """
        Run full two-stage detection pipeline.

        Args:
            image: Input image (BGR format)
            output_format: Output format ('naso7y' or 'project')
            translate_to_project: If True, translate NASO7Y classes to project format

        Returns:
            Tuple of (card_image, fields_dict)
            fields_dict: {field_name: (field_crop, confidence)}
        """
        # Stage 1: Detect and crop card
        card_image, card_det = self.detect_card(image)

        # Stage 2: Detect fields
        field_dets = self.detect_fields(card_image, valid_only=True)

        # Extract field crops
        fields = {}
        for det in field_dets:
            x1, y1, x2, y2 = det.bbox
            h, w = card_image.shape[:2]

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                field_crop = card_image[y1:y2, x1:x2]
                
                # Translate class name if requested
                class_name = det.class_name
                if translate_to_project:
                    translated = translate_class_name(class_name, 'naso7y', 'project')
                    if translated:
                        class_name = translated
                    else:
                        # Skip fields without project equivalent
                        continue
                
                # Handle duplicate field names (e.g., firstName + lastName → name)
                if class_name in fields:
                    # Merge with existing (average confidence)
                    existing_crop, existing_conf = fields[class_name]
                    new_conf = (existing_conf + det.confidence) / 2
                    fields[class_name] = (existing_crop, new_conf)
                else:
                    fields[class_name] = (field_crop, det.confidence)

        return card_image, fields

    def detect_full_with_mapping(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, float]], Dict[str, str]]:
        """
        Run detection and return both NASO7Y and project format mappings.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (card_image, fields_dict, mapping_dict)
            fields_dict: {field_name: (field_crop, confidence)}
            mapping_dict: {naso7y_name: project_name} for detected fields
        """
        # Stage 1: Detect and crop card
        card_image, card_det = self.detect_card(image)

        # Stage 2: Detect fields
        field_dets = self.detect_fields(card_image, valid_only=True)

        # Extract field crops with both formats
        fields = {}
        mapping = {}
        
        for det in field_dets:
            x1, y1, x2, y2 = det.bbox
            h, w = card_image.shape[:2]

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                field_crop = card_image[y1:y2, x1:x2]
                
                # Store with NASO7Y name
                fields[det.class_name] = (field_crop, det.confidence)
                
                # Get project equivalent
                project_name = translate_class_name(det.class_name, 'naso7y', 'project')
                if project_name:
                    mapping[det.class_name] = project_name

        return card_image, fields, mapping


def load_card_detector(
    card_model_path: str = None,
    field_model_path: str = None,
) -> CardDetector:
    """
    Load card detector with default model paths.
    
    Args:
        card_model_path: Override default card model path
        field_model_path: Override default field model path
        
    Returns:
        CardDetector instance
    """
    # Default paths
    if card_model_path is None:
        card_model_path = "./weights/card_detection.pt"
    
    if field_model_path is None:
        field_model_path = "./weights/field_detection.pt"
    
    return CardDetector(
        card_model_path=card_model_path,
        field_model_path=field_model_path,
    )


# Convenience function for quick testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python card_detector.py <image_path> [card_model] [field_model]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    card_model = sys.argv[2] if len(sys.argv) > 2 else None
    field_model = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load detector
    detector = load_card_detector(card_model, field_model)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        sys.exit(1)
    
    print(f"📷 Input image: {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    card_image, fields = detector.detect_full(image)
    
    print(f"✂️  Card crop: {card_image.shape[1]}x{card_image.shape[0]}")
    print(f"📋 Detected fields: {len(fields)}")
    for field_name, (crop, conf) in fields.items():
        print(f"   - {field_name}: {crop.shape[1]}x{crop.shape[0]} (conf: {conf:.2f})")
    
    # Save card crop
    card_output = Path("./output/card_crop.jpg")
    card_output.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(card_output), card_image)
    print(f"💾 Card crop saved to: {card_output}")
    
    # Save field crops
    for field_name, (crop, conf) in fields.items():
        field_output = Path(f"./output/field_{field_name}.jpg")
        cv2.imwrite(str(field_output), crop)
        print(f"   - {field_name} saved to: {field_output}")

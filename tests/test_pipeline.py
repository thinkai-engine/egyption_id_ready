"""
Egyptian ID OCR — Test Suite
==============================
12 unit tests covering core pipeline components:
  - Label parsing
  - Field cropping
  - Text cleaning
  - Preprocessing
  - Inference (CTC decode, validation)
  - API health check (requires running server)

Usage:
    python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
#  Label Reader Tests
# ═══════════════════════════════════════════════════════════════
class TestLabelReader:

    def test_parse_yolo_label_correct_bbox(self, tmp_path):
        """YOLO normalised coords convert to correct pixel coords."""
        from src.label_reader import parse_yolo_label

        # class 5 (name), centered at (0.5, 0.5), size (0.2, 0.1)
        label = tmp_path / "test.txt"
        label.write_text("5 0.5 0.5 0.2 0.1\n")

        fields = parse_yolo_label(str(label), img_w=1000, img_h=800)
        assert len(fields) == 1
        f = fields[0]
        assert f["class_name"] == "name"
        assert f["bbox"] == [400, 360, 600, 440]  # ±padding
        assert f["conf"] == 1.0
        assert f["source"] == "label"

    def test_parse_multi_fields(self, tmp_path):
        """Multiple fields parsed correctly."""
        from src.label_reader import parse_yolo_label

        label = tmp_path / "multi.txt"
        label.write_text("5 0.5 0.5 0.2 0.1\n7 0.3 0.3 0.1 0.05\n")

        fields = parse_yolo_label(str(label), img_w=640, img_h=480)
        assert len(fields) == 2
        names = {f["class_name"] for f in fields}
        assert names == {"name", "national_id"}

    def test_nonexistent_label_returns_empty(self):
        """Missing label file returns empty list."""
        from src.label_reader import parse_yolo_label
        assert parse_yolo_label("/nonexistent.txt", 100, 100) == []


# ═══════════════════════════════════════════════════════════════
#  Crop Builder Tests
# ═══════════════════════════════════════════════════════════════
class TestCropBuilder:

    def test_crop_field_correct_size(self):
        """Crop returns correct dimensions with padding."""
        from src.crop_builder import crop_field

        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        crop = crop_field(img, [50, 30, 250, 60], padding=4)
        assert crop is not None
        h, w = crop.shape[:2]
        assert h == (60 - 30 + 8)   # bbox height + 2*padding
        assert w == (250 - 50 + 8)  # bbox width + 2*padding

    def test_crop_field_edge_clamp(self):
        """Bbox near edges is clamped to image boundaries."""
        from src.crop_builder import crop_field

        img = np.ones((100, 100, 3), dtype=np.uint8)
        crop = crop_field(img, [0, 0, 100, 100], padding=10)
        assert crop is not None
        h, w = crop.shape[:2]
        assert h <= 100
        assert w <= 100

    def test_crop_field_invalid_returns_none(self):
        """Zero-area bbox returns None."""
        from src.crop_builder import crop_field

        img = np.ones((100, 100, 3), dtype=np.uint8)
        # After padding subtraction, this might still be valid
        crop = crop_field(img, [50, 50, 50, 50], padding=0)
        assert crop is None


# ═══════════════════════════════════════════════════════════════
#  Text Cleaner Tests
# ═══════════════════════════════════════════════════════════════
class TestTextCleaner:

    def test_clean_arabic_basic(self):
        """Arabic letters and digits retained, other chars removed."""
        from src.text_cleaner import clean_arabic_text

        result = clean_arabic_text("محمد  علي  $#@   123")
        assert "$" not in result
        assert "#" not in result
        assert "محمد" in result
        assert "123" in result

    def test_reverse_for_paddle(self):
        """Text is reversed for PaddleOCR left-to-right training."""
        from src.text_cleaner import reverse_for_paddle

        assert reverse_for_paddle("محمد") == "دمحم"

    def test_prepare_paddle_label_empty_on_error(self):
        """ERROR strings return empty."""
        from src.text_cleaner import prepare_paddle_label

        assert prepare_paddle_label("ERROR: API timeout") == ""
        assert prepare_paddle_label("") == ""


# ═══════════════════════════════════════════════════════════════
#  Preprocessing Tests
# ═══════════════════════════════════════════════════════════════
class TestPreprocessing:

    def test_field_preprocessor_output_shape(self):
        """Preprocessed field has target height 48px."""
        from src.preprocessing import FieldPreprocessor

        fp = FieldPreprocessor(target_height=48)
        img = np.random.randint(0, 255, (30, 200, 3), dtype=np.uint8)
        out = fp.process(img, model_type="paddleocr")
        assert out.shape[0] == 48
        assert len(out.shape) == 3  # still BGR

    def test_quality_assessment_good_image(self):
        """High-contrast clear image scores 'good'."""
        from src.preprocessing import assess_field_quality

        # Create high-contrast image
        img = np.zeros((50, 200, 3), dtype=np.uint8)
        img[:25, :] = 255  # top half white, bottom half black
        result = assess_field_quality(img)
        assert result["quality"] in ("good", "medium")
        assert result["contrast"] > 50


# ═══════════════════════════════════════════════════════════════
#  Inference Tests (without real model)
# ═══════════════════════════════════════════════════════════════
class TestInference:

    def test_validate_national_id_valid(self):
        """Valid 14-digit national ID starting with 2 or 3."""
        from src.inference import EgyptianIDOCR

        assert EgyptianIDOCR._validate("national_id", "29001011234567")
        assert EgyptianIDOCR._validate("national_id", "30001011234567")

    def test_validate_national_id_invalid(self):
        """Invalid national IDs rejected."""
        from src.inference import EgyptianIDOCR

        assert not EgyptianIDOCR._validate("national_id", "1234567890")
        assert not EgyptianIDOCR._validate("national_id", "12345678901234")

    def test_validate_gender(self):
        """Gender accepts only ذكر or أنثى."""
        from src.inference import EgyptianIDOCR

        assert EgyptianIDOCR._validate("gender", "ذكر")
        assert EgyptianIDOCR._validate("gender", "أنثى")
        assert not EgyptianIDOCR._validate("gender", "other")


# ═══════════════════════════════════════════════════════════════
#  API Test (requires running server)
# ═══════════════════════════════════════════════════════════════
class TestAPI:

    @pytest.mark.skipif(
        True,  # Set to False when server is running
        reason="API server not running"
    )
    def test_health_endpoint(self):
        import requests
        r = requests.get("http://localhost:8000/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

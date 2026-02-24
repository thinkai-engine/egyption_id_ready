"""
Egyptian ID OCR — Production Inference
========================================
ONNX-based CPU-optimised pipeline:
  YOLO Detection → PaddleOCR Recognition → CTC Decode → RTL Fix → Validation
"""

import re
import numpy as np
import cv2
import onnxruntime as ort
import arabic_reshaper
from bidi.algorithm import get_display
from dataclasses import dataclass, field as dc_field

from field_detector import YOLOFieldDetector


@dataclass
class OCRResult:
    field: str
    text: str
    confidence: float
    valid: bool
    quality: str = ""
    issues: list = dc_field(default_factory=list)


class EgyptianIDOCR:
    """
    Production pipeline — ONNX on CPU.

    Detection  : field_detector.onnx  (YOLO)
    Recognition: rec_sim.onnx         (PaddleOCR fine-tuned)
    """

    def __init__(
        self,
        det_onnx: str = "./model/field_detector.onnx",
        rec_onnx: str = "./onnx/rec_sim.onnx",
        dict_path: str = "./arabic_dict.txt",
        use_gpu: bool = False,
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        # Detection model (YOLO)
        self.detector = YOLOFieldDetector(det_onnx)

        # Recognition model
        self.rec_sess = ort.InferenceSession(rec_onnx, providers=providers)

        # Character dictionary
        with open(dict_path, encoding="utf-8") as f:
            chars = f.read().strip().split("\n")
        self.chars = ["blank"] + chars

        print(
            f"✅ EgyptianIDOCR ready | "
            f"Dict: {len(self.chars)} chars | "
            f"Provider: {self.rec_sess.get_providers()[0]}"
        )

    # ── Preprocessing for recognition ─────────────────────────
    @staticmethod
    def _preprocess_rec(
        img: np.ndarray, h: int = 48, w: int = 320
    ) -> np.ndarray:
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    # ── CTC Decode ────────────────────────────────────────────
    def _ctc_decode(self, preds: np.ndarray) -> tuple:
        indices = np.argmax(preds[0], axis=-1)
        scores = np.max(preds[0], axis=-1)

        chars, confs = [], []
        prev = -1
        for i, idx in enumerate(indices):
            if idx != prev and idx != 0:
                if idx < len(self.chars):
                    chars.append(self.chars[idx])
                    confs.append(scores[i])
            prev = idx

        raw_text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0

        # Fix RTL
        clean = get_display(arabic_reshaper.reshape(raw_text[::-1]))
        return clean, avg_conf

    # ── Recognise one field ───────────────────────────────────
    def recognize(self, crop: np.ndarray) -> tuple:
        """Recognise text from a single cropped field image."""
        inp = self._preprocess_rec(crop)
        name = self.rec_sess.get_inputs()[0].name
        pred = self.rec_sess.run(None, {name: inp})
        return self._ctc_decode(pred[0])

    # ── Validate fields ───────────────────────────────────────
    @staticmethod
    def _validate(field_name: str, text: str) -> bool:
        rules = {
            "national_id": lambda t: bool(
                re.match(r'^[23]\d{13}$', re.sub(r'\D', '', t))
            ),
            "gender": lambda t: t.strip() in ["ذكر", "أنثى"],
            "birth_date": lambda t: bool(
                re.search(r'\d{2}[/\-]\d{2}[/\-]\d{4}', t)
            ),
            "expiry_date": lambda t: bool(
                re.search(r'\d{2}[/\-]\d{2}[/\-]\d{4}', t)
            ),
        }
        if field_name in rules:
            return rules[field_name](text)
        return len(text.strip()) > 0

    # ── Extract all fields from crops ─────────────────────────
    def extract(
        self,
        id_card_path: str = "",
        crops: dict = None,
    ) -> dict:
        """
        Extract all fields from an ID card.

        Parameters
        ----------
        id_card_path : str
            Path to full ID card image (used if crops is None).
        crops : dict, optional
            Pre-cropped field images: {field_name: np.ndarray}.
        """
        if crops is None:
            img = cv2.imread(id_card_path)
            if img is None:
                raise ValueError(f"Cannot read image: {id_card_path}")
            fields = self.detector.detect(img)
            if not fields:
                raise ValueError("No fields detected in image")
            crops = {}
            for f in fields:
                x1, y1, x2, y2 = f["bbox"]
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crops[f["class_name"]] = crop

        results = {}
        for field_name, crop in crops.items():
            text, conf = self.recognize(crop)
            valid = self._validate(field_name, text)
            results[field_name] = OCRResult(
                field=field_name,
                text=text,
                confidence=round(conf, 3),
                valid=valid,
            )

        return results

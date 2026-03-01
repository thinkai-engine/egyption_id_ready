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
from typing import Optional

from .field_detector import YOLOFieldDetector
from .post_processor import OCRPostProcessor


@dataclass
class OCRResult:
    field: str
    text: str
    confidence: float
    valid: bool
    quality: str = ""
    issues: list = dc_field(default_factory=list)
    original_text: str = ""  # Before post-processing
    corrected: bool = False  # Was post-processing applied?


class EgyptianIDOCR:
    """
    Production pipeline — ONNX on CPU.

    Detection  : field_detector.onnx  (YOLO)
    Recognition: rec_sim.onnx         (PaddleOCR fine-tuned)
    
    Optional Post-Processing:
    - LLM-based error correction (AirLLM)
    - Cross-field consistency validation
    """

    def __init__(
        self,
        det_onnx: str = "./model/field_detector.onnx",
        rec_onnx: str = "./onnx/rec_sim.onnx",
        dict_path: str = "./arabic_dict.txt",
        use_gpu: bool = False,
        post_process: bool = False,
        post_process_model: str = "meta-llama/Llama-3-8B-Instruct",
        post_process_4bit: bool = False,
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

        # Optional post-processor
        self.post_processor = None
        if post_process:
            self.post_processor = OCRPostProcessor(
                model_name=post_process_model,
                use_4bit=post_process_4bit,
                enabled=True,
            )
            print(f"✅ Post-processor enabled: {post_process_model}")

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
    def recognize(
        self,
        crop: np.ndarray,
        field_name: str = None,
        apply_post_process: bool = None,
    ) -> tuple:
        """
        Recognise text from a single cropped field image.
        
        Args:
            crop: Cropped field image
            field_name: Optional field name for post-processing
            apply_post_process: Override global post_process setting
            
        Returns:
            Tuple of (text, confidence)
        """
        inp = self._preprocess_rec(crop)
        name = self.rec_sess.get_inputs()[0].name
        pred = self.rec_sess.run(None, {name: inp})
        text, conf = self._ctc_decode(pred[0])
        
        # Apply post-processing if enabled
        should_post = apply_post_process if apply_post_process is not None else (self.post_processor is not None)
        if should_post and self.post_processor is not None and field_name:
            correction = self.post_processor.correct(text, field_name, conf)
            if correction.was_corrected:
                text = correction.corrected
                conf = correction.confidence
        
        return text, conf

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
        validate_consistency: bool = True,
    ) -> dict:
        """
        Extract all fields from an ID card.

        Parameters
        ----------
        id_card_path : str
            Path to full ID card image (used if crops is None).
        crops : dict, optional
            Pre-cropped field images: {field_name: np.ndarray}.
        validate_consistency : bool, optional
            Run cross-field consistency validation (default: True).
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
            text, conf = self.recognize(crop, field_name=field_name)
            valid = self._validate(field_name, text)
            results[field_name] = OCRResult(
                field=field_name,
                text=text,
                confidence=round(conf, 3),
                valid=valid,
                original_text=text,
                corrected=False,
            )

        # Cross-field consistency validation
        if validate_consistency and self.post_processor is not None:
            validation = self.post_processor.validate_consistency(results)
            if not validation["valid"]:
                for issue in validation["issues"]:
                    # Add issue to relevant field
                    for field_name in results:
                        if field_name in issue:
                            results[field_name].issues.append(issue)
                            results[field_name].valid = False

        return results

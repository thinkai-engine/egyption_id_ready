"""
Egyptian ID Card Image Preprocessing
=====================================
Two classes:
  - IDCardPreprocessor  : Full card (deskew, glare removal, contrast, denoise, sharpen)
  - FieldPreprocessor   : Cropped field (padding, upscale, deskew, contrast, denoise, binarize, resize)
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  IDCardPreprocessor — Full card preprocessing
# ═══════════════════════════════════════════════════════════════
class IDCardPreprocessor:
    """Improve full ID card image quality before field detection."""

    def process(self, img: np.ndarray) -> np.ndarray:
        img = self._deskew(img)
        img = self._remove_glare(img)
        img = self._enhance_contrast(img)
        img = self._denoise(img)
        img = self._sharpen(img)
        return img

    # ── 1. Deskew ──────────────────────────────────────────────
    def _deskew(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        if abs(angle) < 0.5:
            return img

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ── 2. Glare removal ──────────────────────────────────────
    def _remove_glare(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = ((s < 30) & (v > 220)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # ── 3. Contrast (CLAHE) ───────────────────────────────────
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── 4. Denoise ────────────────────────────────────────────
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    # ── 5. Sharpen ────────────────────────────────────────────
    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


# ═══════════════════════════════════════════════════════════════
#  FieldPreprocessor — Cropped field preprocessing
# ═══════════════════════════════════════════════════════════════
class FieldPreprocessor:
    """
    Specialised preprocessing for cropped ID card fields.

    model_type:
      - "paddleocr" → grayscale binarise + resize to 48px
      - "vlm"       → colour-preserved enhance only (QARI / Gemini)
    """

    def __init__(
        self,
        target_height: int = 48,
        min_width: int = 100,
        padding: int = 4,
    ):
        self.target_height = target_height
        self.min_width = min_width
        self.padding = padding

    def process(
        self, img: np.ndarray, model_type: str = "paddleocr"
    ) -> np.ndarray:
        if img is None or img.size == 0:
            return img

        img = self._add_padding(img)
        img = self._upscale_if_small(img)
        img = self._deskew_field(img)
        img = self._enhance_contrast(img)
        img = self._remove_noise(img)

        if model_type == "paddleocr":
            img = self._binarize(img)
            img = self._resize_for_paddle(img)

        return img

    # ── 1. Padding ────────────────────────────────────────────
    def _add_padding(self, img: np.ndarray) -> np.ndarray:
        p = self.padding
        return cv2.copyMakeBorder(
            img, p, p, p, p,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

    # ── 2. Upscale ────────────────────────────────────────────
    def _upscale_if_small(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.target_height:
            scale = self.target_height / h
            new_w = max(self.min_width, int(w * scale))
            img = cv2.resize(
                img, (new_w, self.target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        elif w < self.min_width:
            scale = self.min_width / w
            new_h = int(h * scale)
            img = cv2.resize(
                img, (self.min_width, new_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return img

    # ── 3. Deskew inside field ────────────────────────────────
    def _deskew_field(self, img: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 3
            else img
        )
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 20:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        if abs(angle) < 0.3:
            return img

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ── 4. Contrast (CLAHE, 4×4 for small fields) ────────────
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_eq = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── 5. Denoise ────────────────────────────────────────────
    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(img, d=3, sigmaColor=25, sigmaSpace=25)

    # ── 6. Binarize (PaddleOCR only) ──────────────────────────
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 3
            else img.copy()
        )
        brightness_std = gray.std()

        if brightness_std > 40:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=8,
            )
        else:
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── 7. Resize for PaddleOCR (48px height) ────────────────
    def _resize_for_paddle(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.target_height / h
        new_w = min(1200, max(self.min_width, int(w * scale)))
        return cv2.resize(
            img, (new_w, self.target_height),
            interpolation=cv2.INTER_AREA,
        )


# ═══════════════════════════════════════════════════════════════
#  Quality assessment
# ═══════════════════════════════════════════════════════════════
def assess_field_quality(img: np.ndarray) -> dict:
    """Score image quality and detect issues."""
    gray = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 3
        else img
    )
    h, w = img.shape[:2]
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())

    issues = []
    if h < 20:
        issues.append("too_small")
    if contrast < 15:
        issues.append("low_contrast")
    if blur < 30:
        issues.append("blurry")
    if bright > 240:
        issues.append("overexposed")
    if bright < 30:
        issues.append("underexposed")

    quality = (
        "good" if not issues else
        "medium" if len(issues) == 1 else
        "poor"
    )

    return {
        "size": f"{w}x{h}",
        "contrast": round(contrast, 1),
        "sharpness": round(blur, 1),
        "brightness": round(bright, 1),
        "issues": issues,
        "quality": quality,
    }

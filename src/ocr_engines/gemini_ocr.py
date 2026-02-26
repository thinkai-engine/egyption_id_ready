"""
Gemini API OCR Engine
=====================
Extract text from cropped ID card fields using Google Gemini.
"""

import time
from pathlib import Path
from PIL import Image

# ── Field-specific prompts ────────────────────────────────────
FIELD_PROMPTS = {
    "name":           "اقرأ الاسم الرباعي بالكامل كما هو مكتوب، بدون أي تعديل.",
    "national_id":    "اقرأ الرقم القومي المكوّن من 14 رقم بدقة تامة. أرقام فقط.",
    "birth_date":     "اقرأ تاريخ الميلاد كما هو مكتوب بالضبط.",
    "address":        "اقرأ العنوان الكامل كما هو مكتوب.",
    "governorate":    "اقرأ اسم المحافظة فقط.",
    "gender":         "اقرأ كلمة الجنس فقط (ذكر أو أنثى).",
    "expiry_date":    "اقرأ تاريخ انتهاء البطاقة كما هو مكتوب.",
    "religion":       "اقرأ الديانة فقط.",
    "marital_status": "اقرأ الحالة الاجتماعية فقط.",
    "job_title":      "اقرأ المهنة فقط.",
    "husband_name":   "اقرأ اسم الزوج فقط.",
    "issue_date":     "اقرأ تاريخ الإصدار كما هو مكتوب.",
    "serial_number":  "اقرأ الرقم التسلسلي بدقة تامة.",
}


class GeminiOCR:
    """Extract text from cropped fields using Google Gemini API."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini ready | Model: {model_name}")

    def extract(self, image_path: str, field_name: str = None) -> str:
        """Extract text from a single cropped field image."""
        img = Image.open(image_path)

        if field_name and field_name in FIELD_PROMPTS:
            prompt = (
                f"أنت نظام OCR متخصص في بطاقات الهوية المصرية. "
                f"{FIELD_PROMPTS[field_name]} "
                f"أرجع النص فقط بدون أي شرح."
            )
        else:
            prompt = (
                "أنت نظام OCR. اقرأ كل النص في الصورة "
                "بدقة تامة وأرجعه فقط بدون شرح."
            )

        try:
            response = self.model.generate_content([prompt, img])
            return response.text.strip()
        except Exception as e:
            return f"ERROR: {e}"

    def label_crops(
        self,
        crops_df,
        base_dir: str,
        delay: float = 0.4,
    ):
        """
        Label all unlabeled crops using Gemini.
        Modifies DataFrame in place.
        """
        import pandas as pd
        from tqdm import tqdm

        unlabeled = crops_df[crops_df["label_text"] == ""]
        print(f"📤 Sending {len(unlabeled)} crops to Gemini...")

        for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
            img_path = Path(base_dir) / row["image_path"]
            if not img_path.exists():
                continue

            text = self.extract(str(img_path), row["field"])
            crops_df.at[idx, "label_text"] = text
            time.sleep(delay)  # rate limiting

        return crops_df

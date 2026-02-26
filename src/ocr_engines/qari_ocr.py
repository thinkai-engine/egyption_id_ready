"""
QARI-OCR Engine (HuggingFace)
==============================
Extract text from cropped ID card fields using QARI-OCR
(NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct).
"""

import os
import cv2
import numpy as np
from pathlib import Path

from .gemini_ocr import FIELD_PROMPTS

LOW_QUALITY_SUFFIX = (
    "\nملاحظة: الصورة قد تكون غير واضحة. "
    "اقرأ ما تستطيع وضع [؟] بدل الحروف غير الواضحة."
)


class QariOCR:
    """
    Extract text using QARI-OCR VLM.
    Requires GPU for reasonable speed.

    Note: Qari-OCR is a fine-tuned model (not a PEFT adapter) based on Qwen2-VL-2B-Instruct.
    """

    def __init__(
        self,
        model_name: str = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
        use_4bit: bool = False,
        device: str = "auto",
    ):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"⏳ Loading QARI model {model_name} ...")

        # Load model
        if use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map=device,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device_name = str(next(self.model.parameters()).device)
        print(f"✅ QARI ready on: {self.device_name}")

    def extract(self, image_path: str, field_name: str = None) -> str:
        """Extract text from a single cropped field image."""
        import torch
        from qwen_vl_utils import process_vision_info

        if field_name and field_name in FIELD_PROMPTS:
            prompt = (
                f"أنت نظام OCR متخصص في بطاقات الهوية المصرية.\n"
                f"{FIELD_PROMPTS[field_name]}\n"
                f"أرجع النص فقط بدون أي شرح أو تعليق."
            )
        else:
            prompt = "اقرأ كل النص في هذه الصورة بدقة تامة."

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=img_in,
            videos=vid_in,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.1,
            )

        result = self.processor.batch_decode(
            [out[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=True,
        )[0].strip()

        return result

    def fix_rtl(self, text: str) -> str:
        """Fix Arabic RTL text from model output."""
        if not text:
            return ""
        import arabic_reshaper
        from bidi.algorithm import get_display

        reversed_text = text[::-1]
        reshaped = arabic_reshaper.reshape(reversed_text)
        return get_display(reshaped)

    def label_crops(
        self,
        crops_df,
        base_dir: str,
    ):
        """
        Label all unlabeled crops using QARI-OCR.
        Modifies DataFrame in place.
        """
        from tqdm import tqdm

        unlabeled = crops_df[crops_df["label_text"] == ""]
        print(f"🤗 Processing {len(unlabeled)} crops with QARI-OCR...")

        for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
            img_path = Path(base_dir) / row["image_path"]
            if not img_path.exists():
                continue

            text = self.extract(str(img_path), row["field"])
            crops_df.at[idx, "label_text"] = text

        return crops_df

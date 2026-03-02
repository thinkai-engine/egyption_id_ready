"""
AirLLM OCR Engine (HuggingFace + AirLLM)
=========================================
Extract text from cropped ID card fields using large VLMs (up to 72B params)
on consumer GPUs via AirLLM layer-wise inference.

Supports: Qwen2-VL-72B, Llama-3-70B, and other large models.

Note: Requires airllm package: pip install airllm
If airllm fails to import, falls back to standard transformers with 4-bit quantization.
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


class AirLLMOCR:
    """
    Extract text using AirLLM with large VLMs (72B+ params).
    Uses layer-wise inference to fit on 4GB GPU.

    Note: Best for offline labeling tasks due to slower inference speed.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-72B-Instruct",
        use_4bit: bool = False,
        device: str = "auto",
        cache_dir: str = "./model/airllm_cache",
    ):
        import torch
        from transformers import BitsAndBytesConfig

        print(f"⏳ Loading model {model_name}...")

        # Create cache directory for sharded model
        os.makedirs(cache_dir, exist_ok=True)

        # Try AirLLM first, fall back to standard transformers with 4-bit
        self.use_airllm = False
        try:
            from airllm import AutoModel
            self.use_airllm = True

            # Load model with AirLLM layer-wise inference
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_4bit=use_4bit,
                device_map=device,
            )
            print("✅ Using AirLLM layer-wise inference")
        except ImportError as e:
            print(f"⚠️  AirLLM not available: {e}")
            print("   Falling back to standard transformers with 4-bit quantization")

            # Fallback: Use standard transformers with 4-bit quantization
            if use_4bit:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                bnb_cfg = None

            from transformers import Qwen2VLForConditionalGeneration

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map=device,
                dtype=torch.float16 if not use_4bit else None,
            )
            print("✅ Using standard transformers (4-bit quantization)")

        # Import processor for Qwen2-VL
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.device_name = str(next(self.model.parameters())).device
        print(f"✅ Model ready on: {self.device_name}")

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
        Label all unlabeled crops using AirLLM.
        Modifies DataFrame in place.
        
        Note: Slower than QARI-OCR but higher accuracy for large models.
        """
        from tqdm import tqdm

        unlabeled = crops_df[crops_df["label_text"] == ""]
        print(f"🤖 Processing {len(unlabeled)} crops with AirLLM (72B)...")

        for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
            img_path = Path(base_dir) / row["image_path"]
            if not img_path.exists():
                continue

            text = self.extract(str(img_path), row["field"])
            crops_df.at[idx, "label_text"] = text

        return crops_df

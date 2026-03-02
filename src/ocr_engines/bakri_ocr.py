"""
Bakri OCR Engine (HuggingFace)
==============================
Extract text from cropped ID card fields using Bakri OCR
(bakrianoo/arabic-legal-documents-ocr-1.0).

This is a LoRA adapter on top of Gemma-3-4B-IT, optimized for
Arabic legal documents OCR.
"""

import cv2
import numpy as np
from pathlib import Path

from .gemini_ocr import FIELD_PROMPTS


class BakriOCR:
    """
    Extract text using Bakri OCR VLM (Gemma-3-4B based).
    Requires GPU for reasonable speed.
    
    Note: This is a full fine-tuned model (not a LoRA adapter).
    Images are preprocessed (resize + grayscale) before inference.
    """

    def __init__(
        self,
        model_name: str = "bakrianoo/arabic-legal-documents-ocr-1.0",
        use_4bit: bool = False,
        device: str = "auto",
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"⏳ Loading Bakri OCR model {model_name} ...")

        # Load model
        if use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map=device,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map=device,
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device_name = str(next(self.model.parameters()).device)
        print(f"✅ Bakri OCR ready on: {self.device_name}")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image: resize to max 1024px width and convert to grayscale.
        This matches the training preprocessing.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize if width > 1024
        max_width = 1024
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_width = max_width
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img_gray

    def extract(self, image_path: str, field_name: str = None) -> str:
        """Extract text from a single cropped field image."""
        import torch
        from PIL import Image

        # Preprocess image
        img_gray = self.preprocess_image(image_path)
        
        # Convert to PIL Image for processor
        pil_img = Image.fromarray(img_gray)

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
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]

        # Apply chat template and process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text,
            images=pil_img,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
            )

        result = self.processor.batch_decode(
            [out[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=True,
        )[0].strip()

        return result

    def label_crops(
        self,
        crops_df,
        base_dir: str,
    ):
        """
        Label all unlabeled crops using Bakri OCR.
        Modifies DataFrame in place.
        """
        from tqdm import tqdm

        unlabeled = crops_df[crops_df["label_text"] == ""]
        print(f"🤗 Processing {len(unlabeled)} crops with Bakri OCR...")

        for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
            img_path = Path(base_dir) / row["image_path"]
            if not img_path.exists():
                continue

            text = self.extract(str(img_path), row["field"])
            crops_df.at[idx, "label_text"] = text

        return crops_df

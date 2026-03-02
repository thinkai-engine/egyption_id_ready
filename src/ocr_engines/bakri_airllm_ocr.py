"""
Bakri OCR Engine with AirLLM
=============================
Extract text from cropped ID card fields using Bakri OCR
(bakrianoo/arabic-legal-documents-ocr-1.0) with AirLLM layer-wise inference.

This enables running the Gemma-3-4B based model on GPUs with as little as 4GB VRAM
by loading layers sequentially instead of all at once.

Note: AirLLM uses optimum.bettertransformer which is deprecated in optimum >= 1.20
and transformers >= 4.49. The code automatically falls back to standard transformers
with 4-bit quantization support (via bitsandbytes) which is fully functional.

Requires: pip install airllm>=2.11.0 (optional, fallback works without it)
"""

import os
import cv2
import numpy as np
from pathlib import Path

from .gemini_ocr import FIELD_PROMPTS


class BakriAirLLMOCR:
    """
    Extract text using Bakri OCR with AirLLM layer-wise inference.
    Enables running on 4GB GPU with slower but memory-efficient inference.

    Model: bakrianoo/arabic-legal-documents-ocr-1.0 (Gemma-3-4B based)
    """

    def __init__(
        self,
        model_name: str = "bakrianoo/arabic-legal-documents-ocr-1.0",
        use_4bit: bool = False,
        device: str = "auto",
        cache_dir: str = "./model/airllm_cache_bakri",
        layers_per_batch: int = 1,
    ):
        import torch
        from transformers import BitsAndBytesConfig

        print(f"⏳ Loading Bakri OCR {model_name} with AirLLM...")

        # Create cache directory for sharded model
        os.makedirs(cache_dir, exist_ok=True)

        # Try AirLLM first, fall back to standard transformers with 4-bit
        self.use_airllm = False
        
        # Check if we have GPU available
        import torch
        has_cuda = torch.cuda.is_available()
        
        try:
            from airllm import AutoModel
            
            # Load model with AirLLM layer-wise inference (only if GPU available)
            if has_cuda:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    use_4bit=use_4bit,
                    device_map="auto",
                    layers_per_batch=layers_per_batch,
                )
                self.use_airllm = True
                print(f"✅ Using AirLLM layer-wise inference (layers_per_batch={layers_per_batch})")
                if use_4bit:
                    print("   With 4-bit quantization for reduced VRAM")
            else:
                raise ImportError("No GPU available - using standard transformers")
                
        except ImportError as e:
            # Note: BetterTransformer is deprecated in optimum >= 1.20 and transformers >= 4.49
            # This fallback to standard transformers is expected and fully functional
            print(f"ℹ️  Using standard transformers (BetterTransformer deprecated)")
            print("   Falling back to standard transformers")
            if not has_cuda:
                print("   ⚠️  No GPU detected - running on CPU (slow!)")
            print("   ✅ Standard transformers is fully functional - no action needed")

            # Fallback: Use standard transformers (try 4-bit if bitsandbytes available)
            from transformers import AutoModelForImageTextToText

            # Check if bitsandbytes is available
            try:
                import bitsandbytes
                has_bnb = True
            except ImportError:
                has_bnb = False
                if has_cuda:
                    print("   ⚠️  bitsandbytes not found - loading in full precision")
                    print("   💡 For 4-bit: pip install bitsandbytes>=0.46.1")

            if use_4bit and has_bnb:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                )
                print("✅ Using standard transformers (4-bit quantization)")
            else:
                # Full precision loading - Bakri OCR uses bfloat16
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
                    device_map="auto" if has_cuda else None,
                )
                print("✅ Using standard transformers (full precision)")
        except Exception as e:
            print(f"⚠️  AirLLM loading failed: {e}")
            print("   Falling back to standard transformers")

            # Fallback: Use standard transformers (try 4-bit if bitsandbytes available)
            from transformers import AutoModelForImageTextToText
            
            # Check if bitsandbytes is available
            try:
                import bitsandbytes
                has_bnb = True
            except ImportError:
                has_bnb = False
                print("   ⚠️  bitsandbytes not found - loading in full precision")

            if use_4bit and has_bnb:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                bnb_cfg = None

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map=device,
            )
            if bnb_cfg:
                print("✅ Using standard transformers (4-bit quantization)")
            else:
                print("✅ Using standard transformers (full precision)")

        # Import processor - try local cache first to avoid network timeouts
        from transformers import AutoProcessor
        
        try:
            # Try loading from cache first (faster, no network)
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                local_files_only=True,
            )
            print("✅ Processor loaded from cache")
        except Exception:
            # If not in cache, download from network
            print("⏳ Downloading processor (this may take a moment)...")
            self.processor = AutoProcessor.from_pretrained(model_name)

        # Get device name safely - handle different model types
        try:
            # Try to get device from model parameters
            params_iter = self.model.parameters()
            first_param = next(params_iter)
            if hasattr(first_param, 'device'):
                self.device_name = str(first_param.device).split(':')[0]
            else:
                self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        except (StopIteration, AttributeError, TypeError):
            # Fallback for models without parameters() method
            self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"✅ Bakri OCR (AirLLM) ready on: {self.device_name}")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image: resize to max 1024px width and convert to grayscale.
        This matches the training preprocessing used in Bakri OCR.
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

        # Preprocess image (resize + grayscale as per Bakri training)
        img_gray = self.preprocess_image(image_path)

        # Convert to PIL Image for processor
        pil_img = Image.fromarray(img_gray)

        # Build prompt based on field type
        if field_name and field_name in FIELD_PROMPTS:
            prompt = (
                f"أنت نظام OCR متخصص في بطاقات الهوية المصرية.\n"
                f"{FIELD_PROMPTS[field_name]}\n"
                f"أرجع النص فقط بدون أي شرح أو تعليق."
            )
            # prompt = " اقرأ كل النص في هذه الصورة بدقة تامة كما هو بدون اي اضافه او شرح او تعليق او تحسين"

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
        )

        # Store input_ids shape before moving to device
        input_ids_shape = inputs["input_ids"].shape[1]

        # Move inputs to model device
        if self.use_airllm:
            # AirLLM handles device placement internally
            pass
        else:
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
            )

        result = self.processor.batch_decode(
            [out[0][input_ids_shape:]],
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
        Label all unlabeled crops using Bakri OCR with AirLLM.
        Modifies DataFrame in place.

        Note: AirLLM inference is slower but enables low VRAM usage.
        """
        from tqdm import tqdm

        unlabeled = crops_df[crops_df["label_text"] == ""]
        print(f"🤖 Processing {len(unlabeled)} crops with Bakri OCR (AirLLM)...")

        for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
            img_path = Path(base_dir) / row["image_path"]
            if not img_path.exists():
                continue

            text = self.extract(str(img_path), row["field"])
            crops_df.at[idx, "label_text"] = text

        return crops_df

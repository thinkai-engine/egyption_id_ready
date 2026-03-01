"""
Post-Processor — LLM-based OCR Error Correction
================================================
Uses AirLLM to correct and validate OCR output using semantic context.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class CorrectionResult:
    original_text: str
    corrected_text: str
    was_corrected: bool
    confidence: float
    explanation: str = ""


class OCRPostProcessor:
    """
    Post-process OCR results using LLM-based correction.
    
    Use cases:
    - Fix ambiguous Arabic characters (ب/ت/ث, م/ن)
    - Validate field-specific constraints
    - Cross-field consistency checks
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        use_4bit: bool = False,
        cache_dir: str = "./model/airllm_cache",
        enabled: bool = False,
    ):
        """
        Initialize post-processor.
        
        Args:
            model_name: LLM model for correction (smaller models faster)
            use_4bit: Use 4-bit quantization
            cache_dir: Cache directory for model
            enabled: If False, pass-through mode (no correction)
        """
        self.enabled = enabled
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.cache_dir = cache_dir
        self._model = None
        
        # Field-specific correction prompts
        self.correction_prompts = {
            "name": (
                "صحيح النص التالي لاسم شخص مصري كما هو مكتوب في بطاقة الهوية. "
                "أرجع الاسم الصحيح فقط بدون شرح:\n'{text}'"
            ),
            "national_id": (
                "تحقق من الرقم القومي المصري (14 رقم). صحّح أي أخطاء واضحة. "
                "أرجع الرقم الصحيح فقط:\n'{text}'"
            ),
            "address": (
                "صحيح عنوان مصري كما هو مكتوب في بطاقة الهوية. "
                "أرجع العنوان الصحيح فقط:\n'{text}'"
            ),
            "governorate": (
                "صحيح اسم محافظة مصرية. المحافطات المعروفة: "
                "القاهرة، الجيزة، الإسكندرية، الدقهلية، الشرقية، الغربية، "
                "المنوفية، القليوبية، كفر الشيخ، البحيرة، مطروح، "
                "بورسعيد، الإسماعيلية، السويس، شمال سيناء، جنوب سيناء، "
                "الفيوم، بني سويف، المنيا، أسيوط، سوهاج، قنا، الأقصر، أسوان، البحر الأحمر. "
                "أرجع اسم المحافظة الصحيح فقط:\n'{text}'"
            ),
            "profession": (
                "صحيح اسم مهنة كما هو مكتوب في بطاقة الهوية المصرية. "
                "أرجع المهنة الصحيحة فقط:\n'{text}'"
            ),
        }

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is not None:
            return self._model
            
        if not self.enabled:
            return None
            
        from airllm import AutoModel
        import torch
        
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_4bit=self.use_4bit,
        )
        return self._model

    def _generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        model = self._load_model()
        if model is None:
            return ""
            
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.1,
            )
        
        result = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return result.strip()

    def correct(
        self,
        text: str,
        field_name: str,
        confidence: float = 0.0,
    ) -> CorrectionResult:
        """
        Correct OCR output for a specific field.
        
        Args:
            text: Raw OCR output
            field_name: Field type (name, national_id, etc.)
            confidence: Original OCR confidence score
            
        Returns:
            CorrectionResult with original and corrected text
        """
        if not self.enabled or not text.strip():
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                was_corrected=False,
                confidence=confidence,
            )

        # Skip if confidence is high (>0.95)
        if confidence > 0.95:
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                was_corrected=False,
                confidence=confidence,
            )

        # Get field-specific prompt
        prompt_template = self.correction_prompts.get(
            field_name,
            "صحيح النص العربي التالي:\n'{text}'"
        )
        prompt = prompt_template.format(text=text)
        
        corrected = self._generate(prompt)
        
        was_corrected = corrected.strip() != text.strip()
        
        return CorrectionResult(
            original_text=text,
            corrected_text=corrected if was_corrected else text,
            was_corrected=was_corrected,
            confidence=0.9 if was_corrected else confidence,
            explanation="LLM correction applied" if was_corrected else "",
        )

    def validate_consistency(
        self,
        fields: dict,
    ) -> dict:
        """
        Cross-field consistency validation.
        
        Args:
            fields: Dict of {field_name: OCRResult}
            
        Returns:
            Dict with validation results and issues
        """
        if not self.enabled:
            return {"valid": True, "issues": []}
        
        issues = []
        
        # Extract national ID for validation
        national_id = fields.get("national_id", {}).get("text", "")
        birth_date = fields.get("birth_date", {}).get("text", "")
        governorate = fields.get("governorate", {}).get("text", "")
        
        # Validate national ID structure
        if national_id:
            digits = re.sub(r'\D', '', national_id)
            if len(digits) == 14:
                # Century code (2=19xx, 3=20xx)
                century = digits[0]
                year = digits[1:3]
                month = digits[3:5]
                day = digits[5:7]
                gov_code = digits[7:9]
                
                # Check governorate code
                gov_codes = {
                    "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد",
                    "04": "الإسماعيلية", "11": "الجيزة", "12": "المنوفية",
                    "13": "الدقهلية", "14": "الغربية", "15": "الشرقية",
                    "16": "المنيا", "17": "أسيوط", "18": "سوهاج",
                    "19": "قنا", "20": "أسوان", "21": "الأقصر",
                    "22": "البحر الأحمر", "23": "مطروح",
                    "31": "كفر الشيخ", "32": "دمياط", "33": "البحيرة",
                    "34": "الفيوم", "35": "بني سويف", "88": "خارج الجمهورية",
                }
                
                expected_gov = gov_codes.get(gov_code, "")
                if expected_gov and governorate and expected_gov != governorate:
                    issues.append(
                        f"تعارض بين المحافظة '{governorate}' "
                        f"ورمز المحافظة في الرقم القومي '{expected_gov}'"
                    )
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

"""
Arabic Text Cleaner
====================
Clean, normalise, and reverse Arabic text for PaddleOCR training.
"""

import re


def clean_arabic_text(text: str) -> str:
    """
    Clean and normalise Arabic text extracted by OCR.

    - Strips whitespace
    - Removes non-Arabic, non-digit, non-punctuation characters
    - Collapses multiple spaces
    """
    if not text or str(text).startswith("ERROR"):
        return ""

    text = str(text).strip()

    # Keep: Arabic letters, diacritics, Arabic-Indic digits,
    #        Western digits, spaces, dash, slash
    text = re.sub(
        r'[^\u0600-\u06FF\u0660-\u0669\u0030-\u0039\s\-/.]',
        '',
        text,
    )

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def reverse_for_paddle(text: str) -> str:
    """
    Reverse Arabic text for PaddleOCR training.

    PaddleOCR reads text left-to-right internally.
    Arabic is RTL, so labels must be reversed before training.
    """
    return text[::-1]


def prepare_paddle_label(text: str) -> str:
    """Full pipeline: clean → reverse."""
    cleaned = clean_arabic_text(text)
    if not cleaned:
        return ""
    return reverse_for_paddle(cleaned)


def fix_paddle_output(raw_text: str) -> str:
    """
    Fix PaddleOCR output text (reverse + reshape for display).

    The trained model outputs reversed text,
    so we reverse it back and reshape for proper RTL rendering.
    """
    if not raw_text:
        return ""

    import arabic_reshaper
    from bidi.algorithm import get_display

    reversed_text = raw_text[::-1]
    reshaped = arabic_reshaper.reshape(reversed_text)
    return get_display(reshaped)

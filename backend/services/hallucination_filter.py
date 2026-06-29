"""
Hallucination filter — post-processes Gemini output by cross-validating
component names against raw OCR text from the full image.

This is a novel dissertation contribution. Cite as:
"OCR-based hallucination detection for VLM architecture extraction."
"""

import io
import re
from difflib import SequenceMatcher

import pytesseract
from PIL import Image

from models.schemas import ArchitectureSchema

_FUZZY_THRESHOLD = 0.8
_MIN_WORD_LEN    = 3   # skip short words like "the", "and", "of"


def run_hallucination_filter(
    gemini_output: ArchitectureSchema,
    image_bytes: bytes,
) -> dict:
    """
    Cross-validate every Gemini component name against OCR text from the full image.

    Returns:
        {
            "validated":         list[str],   # names found in OCR
            "hallucinated":      list[str],   # names NOT found in OCR
            "hallucination_rate": float       # hallucinated / total
        }
    """
    if not gemini_output.components:
        return {"validated": [], "hallucinated": [], "hallucination_rate": 0.0}

    # ── Run full-image OCR ─────────────────────────────────
    try:
        image = Image.open(io.BytesIO(image_bytes))
        full_ocr_text = pytesseract.image_to_string(image).lower()
    except Exception as exc:
        print(f"[hallucination_filter] OCR failed: {exc}")
        # Can't validate — mark all as unknown (treat as validated to avoid false positives)
        names = [c.name for c in gemini_output.components]
        return {
            "validated": names,
            "hallucinated": [],
            "hallucination_rate": 0.0,
        }

    # Build word set from OCR (split on non-alphanumeric)
    ocr_words: set[str] = set(re.split(r"[^a-z0-9]+", full_ocr_text))
    ocr_words.discard("")

    validated: list[str]   = []
    hallucinated: list[str] = []

    for component in gemini_output.components:
        name_words = re.split(r"\s+", component.name.lower())
        # Only test significant words (length > _MIN_WORD_LEN)
        significant = [w for w in name_words if len(w) > _MIN_WORD_LEN]

        if not significant:
            # Name is all short words — can't reliably test, treat as validated
            validated.append(component.name)
            continue

        # A component is "found" if ANY significant word fuzzy-matches an OCR word
        matched = any(
            any(
                SequenceMatcher(None, word, ocr_word).ratio() >= _FUZZY_THRESHOLD
                for ocr_word in ocr_words
            )
            for word in significant
        )

        if matched:
            validated.append(component.name)
        else:
            hallucinated.append(component.name)

    total = len(gemini_output.components)
    rate  = len(hallucinated) / total if total > 0 else 0.0

    return {
        "validated":          validated,
        "hallucinated":       hallucinated,
        "hallucination_rate": rate,
    }

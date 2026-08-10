"""
Stage 0 — decide which notation the diagram uses, from the IMAGE.

Replaces the keyword guess in classical_pipeline._infer_diagram_standard(),
which inferred the standard from names that had already been extracted. That
is circular: the notation should steer extraction, so it cannot be a product
of it. The keyword version survives here only as a fallback.

The result drives three things:
  1. proposer weights in hybrid_pipeline (icon bank vs shape vs text)
  2. which relationship lookup table connection_detector uses
  3. ArchitectureSchema.diagram_standard

Every signal is cheap and computed from data the pipeline already has.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from services.common import infer_diagram_standard
from services.notation_profiles import GUILLEMET_CLOSE, GUILLEMET_OPEN

# Confidence assigned when a signal fires outright vs when we fall back
STRONG, WEAK = 0.9, 0.4
# Fraction of regions matching vendor icons that settles the notation.
# Low because icon_bank now favours precision heavily — a handful of confident
# matches is strong evidence, and it rarely matches more than that.
ICON_HIT_STRONG = 0.10

# C4 tags the element kind in brackets under the name. Deployment diagrams add
# their own node kinds, which the container/component list does not cover.
_C4_TAG = re.compile(
    r"\[\s*(container|component|system|person|database|external|"
    r"deployment\s+node|infrastructure\s+node|software\s+system)\b",
    re.IGNORECASE,
)

# Same vocabulary without the brackets, for when OCR loses them. These phrases
# are C4-specific terminology and do not appear in cloud vendor diagrams.
_C4_TERMS = re.compile(
    r"\b(deployment\s+node|infrastructure\s+node|software\s+system|"
    r"container\s*:|component\s*:)", re.IGNORECASE
)
# UML marks stereotypes in guillemets. OCR seldom returns real ones — PP-OCRv5
# emits CJK 《》 — so accept every variant, else UML is never recognised and
# its rule profile never applies.
_UML_STEREOTYPE = re.compile(
    rf"(?:[{GUILLEMET_OPEN}]|<<)\s*\w[\w .\-/]*\s*(?:[{GUILLEMET_CLOSE}]|>>)"
)
_UML_WORDS = re.compile(r"\b(actor|interface|abstract|extends|implements|use\s?case)\b",
                        re.IGNORECASE)


def _hand_drawn_score(img_rgb: np.ndarray) -> float:
    """Hand-drawn strokes wobble: their width varies far more than a vector
    line's, and corners are rounded rather than sharp. Returns 0..1."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.mean()) < 110:
        gray = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    widths = dt[dt > 0.6]
    if widths.size < 50:
        return 0.0
    # Coefficient of variation of stroke width
    cv_width = float(np.std(widths) / max(1e-6, np.mean(widths)))
    return float(np.clip((cv_width - 0.35) / 0.45, 0.0, 1.0))


def classify_notation(
    img_rgb: np.ndarray,
    ocr_text: str,
    icon_hit_rate: float = 0.0,
    icon_provider: str | None = None,
    component_names: list[str] | None = None,
) -> tuple[str, float]:
    """Returns (notation, confidence).
    notation is one of: aws | azure | gcp | c4 | uml | informal."""
    try:
        # 1. Vendor icons are the strongest signal available — an icon match is
        #    an exact identity, not a resemblance.
        if icon_hit_rate >= ICON_HIT_STRONG and icon_provider:
            return icon_provider, STRONG

        # 2. UML stereotypes are unambiguous; no other notation uses guillemets.
        if _UML_STEREOTYPE.search(ocr_text):
            return "uml", STRONG

        # 3. C4 bracket tags are equally distinctive.
        #
        # Checked BEFORE any vendor-keyword fallback on purpose. A C4
        # deployment diagram of an AWS system is still a C4 diagram — the
        # vendor names are its content, not its notation. Ordering it after
        # the keyword fallback classified this whole family as "aws" and
        # applied the wrong extraction rules.
        if _C4_TAG.search(ocr_text) or _C4_TERMS.search(ocr_text):
            return "c4", STRONG

        # 4. Weaker lexical hints.
        if _UML_WORDS.search(ocr_text):
            return "uml", WEAK

        # 5. Hand-drawn geometry beats the keyword fallback: a sketch of an AWS
        #    system is still an informal diagram as far as extraction goes.
        if _hand_drawn_score(img_rgb) > 0.6:
            return "informal", WEAK

        # 6. Fallback: keyword guess. Uses the raw OCR text as well as any
        #    extracted names — an AWS diagram whose icons went unmatched still
        #    prints "CloudFront", "RDS", "S3" somewhere on the page.
        guess = infer_diagram_standard(list(component_names or []) + [ocr_text])
        return guess, WEAK

    except Exception as exc:
        print(f"[notation_classifier] failed: {exc}")
        return "informal", 0.0


def is_icon_centric(notation: str) -> bool:
    """Icon-centric notations put the label OUTSIDE the glyph; box-centric ones
    put it inside. This split drives proposer weighting, and is also the
    primary grouping for benchmark reporting (n per individual notation is too
    small for the four-way split to be the headline)."""
    return notation in {"aws", "azure", "gcp"}

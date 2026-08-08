"""
Text engine for the hybrid arm — PaddleOCR PP-OCRv5, Tesseract as fallback.

Replaces the Tesseract-per-region + TrOCR chain in hybrid_pipeline.py. That
chain had three problems this fixes:

  1. TrOCR base-printed is a SINGLE-LINE generative recogniser. Fed multi-line
     diagram crops it invented words, which is why the old code needed a
     full-page-OCR validation guard to filter its own output.
  2. Tesseract was run once per region — N passes over the same image, each on
     a small low-resolution crop, which is exactly where Tesseract is weakest.
  3. Region crops had to be expanded by guesswork (LABEL_EXPAND_DOWN/SIDE)
     because labels sit outside the shape in icon-centric diagrams.

One full-page detection+recognition pass returns every word with its own box,
so text can be assigned to regions geometrically instead of re-OCR'd per crop.

Model loads lazily and is cached at module level.
Never raises — returns [] if no engine is available.
"""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

# Detection confidence floor. PP-OCRv5 is well calibrated; below this the text
# is usually icon decoration read as glyphs.
MIN_WORD_CONF = 0.55
# Tesseract reports 0-100; its confidences run lower for the same quality.
MIN_TESS_CONF = 40.0
# PaddleOCR works best around this resolution; small diagrams get upscaled.
TARGET_LONG_SIDE = 1600
MAX_LONG_SIDE    = 2600

_engine: dict = {}
_lock = threading.Lock()

# PaddleOCR predictors are thread-affine: created on one thread and called from
# another they abort with a bare "Unknown exception". The hybrid arm runs in
# asyncio.to_thread, so every OCR call is funnelled through one dedicated
# worker that both builds and uses the predictor. Single worker also serialises
# concurrent uploads, which the predictor does not tolerate either.
_ocr_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")


class OcrWord:
    """One recognised text box, in ORIGINAL image coordinates."""

    __slots__ = ("text", "conf", "x", "y", "w", "h")

    def __init__(self, text: str, conf: float, x: int, y: int, w: int, h: int):
        self.text, self.conf = text, conf
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self) -> float: return self.x + self.w / 2

    @property
    def cy(self) -> float: return self.y + self.h / 2

    @property
    def right(self) -> int: return self.x + self.w

    @property
    def bottom(self) -> int: return self.y + self.h

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def __repr__(self) -> str:
        return f"OcrWord({self.text!r}, {self.conf:.2f}, {self.box})"


# ── Engine loading ────────────────────────────────────────────────────────────
def _load_engine() -> dict:
    """Load PaddleOCR once. Falls back to Tesseract if Paddle is unavailable so
    the arm degrades instead of failing."""
    if _engine:
        return _engine
    with _lock:
        if _engine:
            return _engine
        try:
            from paddleocr import PaddleOCR
            # Detection + recognition only. Doc orientation/unwarping modules
            # target photographed pages and only add latency on clean diagrams.
            # Mobile models, not the server default: the server pair took ~55s
            # per diagram on CPU for no measurable gain on rendered text.
            _engine["paddle"] = PaddleOCR(
                lang="en",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            _engine["kind"] = "paddle"
            print("[ocr_engine] PaddleOCR PP-OCRv5 loaded")
        except Exception as exc:
            print(f"[ocr_engine] PaddleOCR unavailable ({exc}); using Tesseract fallback")
            _engine["kind"] = "tesseract"
        return _engine


def engine_name() -> str:
    return _load_engine().get("kind", "none")


# ── Pre-processing ────────────────────────────────────────────────────────────
def _prepare(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (image_for_ocr, scale). Dark-mode diagrams are inverted so text
    is dark on light, which every OCR engine is trained for."""
    gray_mean = float(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).mean())
    work = cv2.bitwise_not(img_rgb) if gray_mean < 110 else img_rgb

    h, w = work.shape[:2]
    long_side = max(h, w)
    scale = TARGET_LONG_SIDE / long_side
    if long_side * scale > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
    if abs(scale - 1.0) < 0.05:
        return work, 1.0

    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(work, None, fx=scale, fy=scale, interpolation=interp), scale


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    # Strip wrapping punctuation but keep interior dots/dashes (Node.js, S3-logs)
    return text.strip("|/\\[](){}<>\"'`,;:")


# ── Main entry point ──────────────────────────────────────────────────────────
def read_page(img_rgb: np.ndarray) -> list[OcrWord]:
    """One full-page pass. Returns text boxes in ORIGINAL image coordinates.
    Never raises. Always executed on the dedicated OCR worker thread."""
    try:
        return _ocr_worker.submit(_read_page_sync, img_rgb).result()
    except Exception as exc:
        print(f"[ocr_engine] read_page dispatch failed: {exc}")
        return []


def _read_page_sync(img_rgb: np.ndarray) -> list[OcrWord]:
    eng = _load_engine()
    try:
        ocr_img, scale = _prepare(img_rgb)
        if eng.get("kind") == "paddle":
            words = _read_paddle(eng["paddle"], ocr_img)
        else:
            words = _read_tesseract(ocr_img)
    except Exception as exc:
        print(f"[ocr_engine] read_page failed: {exc}")
        return []

    if scale != 1.0:
        inv = 1.0 / scale
        for word in words:
            word.x = int(word.x * inv)
            word.y = int(word.y * inv)
            word.w = int(word.w * inv)
            word.h = int(word.h * inv)
    return words


def _read_paddle(ocr, img: np.ndarray) -> list[OcrWord]:
    # PaddleOCR expects BGR
    raw = ocr.predict(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    words: list[OcrWord] = []
    for page in raw or []:
        # PaddleOCR 3.x returns dict-like results; 2.x returns nested lists.
        polys  = page.get("rec_polys") if hasattr(page, "get") else None
        texts  = page.get("rec_texts") if hasattr(page, "get") else None
        scores = page.get("rec_scores") if hasattr(page, "get") else None
        if polys is None:
            words.extend(_read_paddle_legacy(page))
            continue
        for poly, text, score in zip(polys, texts, scores):
            if float(score) < MIN_WORD_CONF:
                continue
            cleaned = _clean(text)
            if not cleaned:
                continue
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            words.append(OcrWord(cleaned, float(score), x, y, w, h))
    return words


def _read_paddle_legacy(page) -> list[OcrWord]:
    """PaddleOCR 2.x shape: [[box, (text, score)], ...]."""
    words: list[OcrWord] = []
    for line in page or []:
        try:
            box, (text, score) = line[0], line[1]
        except Exception:
            continue
        if float(score) < MIN_WORD_CONF:
            continue
        cleaned = _clean(text)
        if not cleaned:
            continue
        pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        words.append(OcrWord(cleaned, float(score), x, y, w, h))
    return words


def _read_tesseract(img: np.ndarray) -> list[OcrWord]:
    import pytesseract
    from services.classical_pipeline import _TESS   # binary path already resolved there

    if _TESS:
        pytesseract.pytesseract.tesseract_cmd = _TESS

    data = pytesseract.image_to_data(
        Image.fromarray(img), output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 11",
    )
    words: list[OcrWord] = []
    for i, raw_conf in enumerate(data["conf"]):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if conf < MIN_TESS_CONF:
            continue
        text = _clean(data["text"][i])
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        words.append(OcrWord(text, conf / 100.0, int(data["left"][i]), int(data["top"][i]),
                             int(data["width"][i]), int(data["height"][i])))
    return words


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _expanded(box: tuple[int, int, int, int], expand: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    dx, dy = int(w * expand), int(h * expand)
    return (x - dx, y - dy, w + 2 * dx, h + 2 * dy)


def words_in_box(words: list[OcrWord], box: tuple[int, int, int, int],
                 expand: float = 0.0, min_overlap: float = 0.5) -> list[OcrWord]:
    """Words whose area lies at least min_overlap inside box (optionally grown
    by `expand` as a fraction of its own size)."""
    bx, by, bw, bh = _expanded(box, expand)
    hits: list[OcrWord] = []
    for word in words:
        ix = max(0, min(bx + bw, word.right) - max(bx, word.x))
        iy = max(0, min(by + bh, word.bottom) - max(by, word.y))
        area = word.w * word.h
        if area and (ix * iy) / area >= min_overlap:
            hits.append(word)
    return hits


def reading_order(words: list[OcrWord]) -> str:
    """Join words top-to-bottom, left-to-right, grouping into rows by y-centre
    with a tolerance adaptive to text height."""
    if not words:
        return ""
    th = float(np.median([w.h for w in words])) or 1.0
    rows: list[list[OcrWord]] = []
    for word in sorted(words, key=lambda w: w.cy):
        for row in rows:
            if abs(row[0].cy - word.cy) < 0.6 * th:
                row.append(word)
                break
        else:
            rows.append([word])

    parts = []
    for row in rows:
        row.sort(key=lambda w: w.x)
        parts.append(" ".join(w.text for w in row))
    return _clean(" ".join(parts))[:80]


def text_for_box(words: list[OcrWord], box: tuple[int, int, int, int],
                 expand: float = 0.0) -> str:
    return reading_order(words_in_box(words, box, expand))


def nearest_label(words: list[OcrWord], box: tuple[int, int, int, int],
                  max_gap_factor: float = 1.2) -> str:
    """Label for an icon that has no text inside it.

    Icon-centric notations (AWS, Azure, GCP) print the label BELOW the glyph,
    so an inside-only search returns nothing. Looks below first, then above,
    then either side, within max_gap_factor of the icon's own height.
    """
    x, y, w, h = box
    gap = max_gap_factor * h
    best, best_dist = None, float("inf")

    for word in words:
        # horizontal band around the icon
        if word.right < x - w * 0.5 or word.x > x + w * 1.5:
            continue
        if word.y >= y + h:                 # below
            dist = word.y - (y + h)
        elif word.bottom <= y:              # above — penalised, less conventional
            dist = (y - word.bottom) * 1.5
        else:
            continue                        # overlapping: handled by text_for_box
        if dist < gap and dist < best_dist:
            best, best_dist = word, dist

    if best is None:
        return ""
    # Pull in the rest of that label line (multi-word labels wrap under icons)
    band = [w for w in words
            if abs(w.cy - best.cy) < 0.8 * best.h and abs(w.cx - best.cx) < 3 * best.h]
    return reading_order(band or [best])

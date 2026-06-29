"""
Specialized ML Hybrid pipeline — NO LLM, NO GENERATIVE AI, NO EXTERNAL API CALLS.

Three specialized models chained, each doing exactly one job:
  1. SAM  (Meta)      — automatic mask generation → candidate component regions
  2. CLIP (OpenAI)    — zero-shot classification of each region's component type
  3. TrOCR (Microsoft)— text extraction from each region (pytesseract fallback)

Connection inference reuses the classical HoughLinesP detector on the original
image with SAM-derived bounding boxes.

All models run locally. Zero API cost. Models are loaded lazily on first call
and cached at module level (rule 9 in CLAUDE.md).
Never raises on partial results — returns what was found, even if incomplete.
"""

import asyncio
import io
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image

from models.schemas import ArchitectureSchema, ComponentSchema
from services.classical_pipeline import (
    _classify_type,
    _complexity,
    _detect_connections,
    _infer_arch_type,
    _infer_diagram_standard,
    _is_noise,
)

# ── Tuning constants ──────────────────────────────────────────────────────────
SAM_LONG_SIDE        = 1024    # downscale before SAM — CPU speed vs detail tradeoff
SAM_POINTS_PER_SIDE  = 16      # default 32 is ~4x slower on CPU
MIN_REGION_FRACTION  = 0.0005  # regions smaller than this fraction of image = noise
MAX_REGION_FRACTION  = 0.30    # regions larger than this = container/background
MAX_REGIONS          = 40      # cap candidate regions (smallest kept first)
DUPLICATE_IOU        = 0.80    # two boxes this similar = same region, keep one
CONTAINMENT_RATIO    = 0.80    # kept box lying this much inside a candidate = "contained"
MAX_NAME_CHARS       = 60      # OCR text longer than this = container noise, not a label
MAX_NAME_WORDS       = 8
CLIP_MODEL_ID        = "openai/clip-vit-base-patch32"
TROCR_MODEL_ID       = "microsoft/trocr-base-printed"

# CLIP zero-shot prompts → DiagramLens component type ("" = not a component, skip)
CLIP_PROMPTS: list[tuple[str, str]] = [
    ("a database or data storage component",       "database"),
    ("an API gateway or load balancer",            "gateway"),
    ("a backend server or microservice",           "service"),
    ("a cache like Redis or Memcached",            "cache"),
    ("a message queue or event bus",               "queue"),
    ("a CDN or content delivery network",          "cdn"),
    ("a client application or web browser",        "client"),
    ("a cloud storage service",                    "storage"),
    ("a network or security component",            "other"),
    ("a plain text label or annotation",           ""),
]

# ── Lazy model cache ──────────────────────────────────────────────────────────
_models: dict = {}
_load_lock = threading.Lock()


def _sam_checkpoint_path() -> str:
    return os.environ.get(
        "SAM_CHECKPOINT",
        os.path.join(os.path.dirname(__file__), "..", "models", "sam", "sam_vit_b_01ec64.pth"),
    )


def _load_models() -> dict:
    """Load SAM + CLIP + TrOCR once, cache at module level. Thread-safe."""
    if _models:
        return _models
    with _load_lock:
        if _models:
            return _models

        import torch  # noqa: F401 — fail early with a clear message if missing
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from transformers import (
            CLIPModel,
            CLIPProcessor,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        ckpt = _sam_checkpoint_path()
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {ckpt}. Download sam_vit_b_01ec64.pth "
                "from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        _models["mask_generator"] = SamAutomaticMaskGenerator(
            sam,
            points_per_side=SAM_POINTS_PER_SIDE,
            min_mask_region_area=200,
        )
        _models["clip_model"]     = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _models["clip_processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _models["trocr_processor"] = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        _models["trocr_model"]     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        _models["clip_model"].eval()
        _models["trocr_model"].eval()
        print("[hybrid_pipeline] SAM + CLIP + TrOCR loaded")
        return _models


# ── Stage 1: SAM segmentation ─────────────────────────────────────────────────
def _segment_regions(img_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Run SAM automatic mask generation, return filtered bounding boxes (x, y, w, h)
    in ORIGINAL image coordinates."""
    models = _load_models()
    h, w = img_rgb.shape[:2]

    scale = SAM_LONG_SIDE / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = img_rgb

    masks = models["mask_generator"].generate(small)

    def _inter(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        ax, ay, aw, ah = a
        bx, by, bw_, bh_ = b
        ix = max(0, min(ax + aw, bx + bw_) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh_) - max(ay, by))
        return ix * iy

    img_area = small.shape[0] * small.shape[1]
    boxes: list[tuple[int, int, int, int]] = []
    # Smallest first: leaf components (icons, boxes) win over enclosing containers.
    for m in sorted(masks, key=lambda m: m["area"]):
        frac = m["area"] / img_area
        if frac > MAX_REGION_FRACTION or frac < MIN_REGION_FRACTION:
            continue
        cand = tuple(int(v) for v in m["bbox"])
        cx, cy, cw, ch = cand
        cand_area = cw * ch
        if cand_area == 0:
            continue

        duplicate = False
        contains_count = 0
        for kept in boxes:
            kx, ky, kw, kh = kept
            inter = _inter(cand, kept)
            union = cand_area + kw * kh - inter
            if union > 0 and inter / union >= DUPLICATE_IOU:
                duplicate = True
                break
            # Does the candidate enclose this kept (smaller) box?
            if kw * kh > 0 and inter / (kw * kh) >= CONTAINMENT_RATIO:
                contains_count += 1
        if duplicate:
            continue
        # A region wrapping 2+ already-kept components is a group container
        # (VPC boundary, availability zone, subnet) — not a component itself.
        if contains_count >= 2:
            continue
        boxes.append(cand)
        if len(boxes) >= MAX_REGIONS:
            break

    # Map back to original coordinates
    inv = 1.0 / scale
    return [(int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)) for (x, y, bw, bh) in boxes]


# ── Stage 2: CLIP zero-shot type classification ───────────────────────────────
def _classify_regions(img_rgb: np.ndarray,
                      boxes: list[tuple[int, int, int, int]]) -> list[tuple[str, float]]:
    """Classify every region crop in ONE batched CLIP pass.
    Returns [(component_type, confidence)] aligned with boxes; type "" = skip."""
    import torch

    models = _load_models()
    crops = []
    for (x, y, bw, bh) in boxes:
        crop = img_rgb[max(0, y):y + bh, max(0, x):x + bw]
        crops.append(Image.fromarray(crop) if crop.size else Image.new("RGB", (8, 8)))

    prompts = [p for p, _ in CLIP_PROMPTS]
    inputs = models["clip_processor"](
        text=prompts, images=crops, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = models["clip_model"](**inputs).logits_per_image  # (n_crops, n_prompts)
        probs = logits.softmax(dim=1)

    results: list[tuple[str, float]] = []
    for row in probs:
        idx = int(row.argmax())
        results.append((CLIP_PROMPTS[idx][1], float(row[idx])))
    return results


# ── Stage 3: text extraction (tesseract on expanded crop + TrOCR fallback) ───
LABEL_EXPAND_DOWN  = 0.6   # diagram labels usually sit below the icon/shape
LABEL_EXPAND_SIDE  = 0.45  # generous — truncating a label's last letters is worse
TROCR_VALIDATE_RATIO = 0.8  # TrOCR word must fuzzy-match a full-page OCR word


def _page_ocr_words(img_rgb: np.ndarray) -> set[str]:
    """Full-image OCR word set — used to reject TrOCR hallucinations."""
    try:
        import pytesseract
        return set(pytesseract.image_to_string(Image.fromarray(img_rgb)).lower().split())
    except Exception:
        return set()


def _read_region_text(img_rgb: np.ndarray, box: tuple[int, int, int, int],
                      page_words: set[str]) -> str:
    """OCR the region. The crop is expanded down/sideways because diagram labels
    typically sit outside the segmented shape. pytesseract handles multi-line
    labels; TrOCR is the fallback for low-contrast single-line text, but its
    output is cross-validated against the full-page OCR because a generative
    decoder will invent words when shown pure iconography."""
    import torch

    h, w = img_rgb.shape[:2]
    x, y, bw, bh = box
    ex = max(0, x - int(bw * LABEL_EXPAND_SIDE))
    ey = max(0, y - int(bh * 0.1))
    ex2 = min(w, x + bw + int(bw * LABEL_EXPAND_SIDE))
    ey2 = min(h, y + bh + int(bh * LABEL_EXPAND_DOWN))
    crop = img_rgb[ey:ey2, ex:ex2]
    if crop.size == 0:
        return ""
    pil = Image.fromarray(crop)

    # Primary: tesseract with per-word confidence filter
    try:
        import pytesseract
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
        words = [t.strip() for t, c in zip(data["text"], data["conf"])
                 if t.strip() and float(c) > 40]
        text = " ".join(words)
        if text:
            return text
    except Exception:
        pass

    # Fallback: TrOCR on the original (unexpanded) crop, hallucination-guarded
    models = _load_models()
    tight = img_rgb[max(0, y):y + bh, max(0, x):x + bw]
    if tight.size == 0:
        return ""
    try:
        pixel_values = models["trocr_processor"](
            images=Image.fromarray(tight), return_tensors="pt"
        ).pixel_values
        with torch.no_grad():
            ids = models["trocr_model"].generate(pixel_values, max_new_tokens=32)
        raw = models["trocr_processor"].batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""

    from difflib import SequenceMatcher
    validated = [
        word for word in raw.split()
        if any(SequenceMatcher(None, word.lower(), pw).ratio() >= TROCR_VALIDATE_RATIO
               for pw in page_words)
    ]
    return " ".join(validated)


# ── Sync core (runs in a worker thread) ───────────────────────────────────────
def _extract(image_bytes: bytes, session_id: str, start: float) -> ArchitectureSchema:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    boxes = _segment_regions(img_rgb)
    if not boxes:
        raise ValueError("SAM found no candidate regions")

    classifications = _classify_regions(img_rgb, boxes)
    page_words = _page_ocr_words(img_rgb)

    components: list[ComponentSchema] = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    for box, (comp_type, clip_conf) in zip(boxes, classifications):
        if comp_type == "":          # CLIP says: text annotation, not a component
            continue
        text = _read_region_text(img_rgb, box, page_words)
        if not text or _is_noise(text):
            continue
        # Very long OCR output = region swallowed surrounding labels, not a component name
        if len(text) > MAX_NAME_CHARS or len(text.split()) > MAX_NAME_WORDS:
            continue
        # Keyword-based type beats CLIP when the label is explicit ("Redis Cache")
        keyword_type = _classify_type(text)
        final_type = keyword_type if keyword_type != "other" else comp_type
        components.append(ComponentSchema(
            id=f"h{len(components) + 1}",
            name=text,
            type=final_type,
            confidence=round(clip_conf, 3),
        ))
        kept_boxes.append(box)

    centroids = [(x + bw / 2, y + bh / 2) for (x, y, bw, bh) in kept_boxes]
    component_ids = [c.id for c in components]
    connections = _detect_connections(gray, kept_boxes, centroids, component_ids)

    names = [c.name for c in components]
    return ArchitectureSchema(
        session_id=session_id,
        pipeline="hybrid",
        diagram_standard=_infer_diagram_standard(names),
        complexity=_complexity(len(components)),
        arch_type=_infer_arch_type([c.type for c in components]),
        components=components or [ComponentSchema(id="h1", name="Unknown", type="other")],
        connections=connections,
        response_time_ms=int((time.time() - start) * 1000),
    )


# ── Entry point ───────────────────────────────────────────────────────────────
async def run_hybrid_pipeline(image_bytes: bytes, session_id: str) -> ArchitectureSchema:
    """Full SAM + CLIP + TrOCR extraction. Never raises — returns partial results."""
    start = time.time()
    try:
        return await asyncio.to_thread(_extract, image_bytes, session_id, start)
    except Exception as e:
        print(f"[hybrid_pipeline] fallback: {e}")
        return ArchitectureSchema(
            session_id=session_id,
            pipeline="hybrid",
            diagram_standard="informal",
            complexity="low",
            arch_type="other",
            components=[ComponentSchema(id="h1", name="Unknown", type="other", confidence=None)],
            connections=[],
            response_time_ms=int((time.time() - start) * 1000),
        )

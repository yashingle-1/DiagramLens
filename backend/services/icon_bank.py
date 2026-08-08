"""
Proposer P1 — icon retrieval against official vendor icon sets.

Carries the icon-centric notations (AWS, Azure, GCP), where component identity
lives in the glyph and the label sits outside it.

Why this replaces CLIP zero-shot prompting:
  The old pipeline scored crops against text prompts like "a database or data
  storage component". CLIP was trained on natural photographs, so abstract
  vector glyphs are out of distribution and the prompts landed near-randomly —
  which is why hybrid_pipeline.py already overrode CLIP's verdict whenever a
  keyword matched the label. Image-to-image retrieval is what CLIP embeddings
  are genuinely good at, and the icon vocabulary here is finite and published.

Cost: icons are embedded ONCE offline into a .npz bank. At inference a crop is
one small forward pass plus a cosine lookup.

Degrades cleanly: with no bank file present, match() returns None and the
hybrid arm falls back to shape and text proposers. Nothing breaks.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BANK_PATH = Path(os.environ.get(
    "ICON_BANK", Path(__file__).resolve().parent.parent / "data" / "icons" / "bank.npz"
))
MAPPING_PATH = BANK_PATH.parent / "mapping.json"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Acceptance is RELATIVE, not a bare cosine cutoff.
#
# Calibrated on the 1456-icon bank (see dissertation methodology):
#   random different-icon pairs   p50 0.790  p90 0.858  p99 0.912
#   each icon's nearest OTHER icon                      p50 0.953
# CLIP embeddings of flat vector graphics sit in a narrow cone, so an absolute
# threshold is meaningless: at 0.82, 29.3% of pairs of UNRELATED icons pass.
# Measured on real diagram crops it matched almost everything to one entry
# ("IAM Access Analyzer") at 0.89-0.93.
#
# Margin separates cleanly instead:
#   genuine bank icons   top1 ~1.000  margin p50 0.094  z p50 4.39
#   diagram crops        top1  0.900  margin p50 0.017  z p50 2.65
# A wrong vendor product name is worse than none — it becomes a hallucinated
# component — so these are set to favour precision over recall.
MATCH_THRESHOLD = 0.86    # absolute floor, above p90 of random pairs
MATCH_MARGIN    = 0.05    # top1 minus the mean of ranks 2..10
MATCH_ZSCORE    = 3.5     # top1 standardised against the whole bank
MIN_CROP_STD    = 8.0     # flat colour fill carries no icon to recognise
MIN_CROP_PX = 12

_state: dict = {}
# Reentrant: _load() holds this and then calls _load_clip(), which takes it
# again. A plain Lock deadlocks the first caller.
_lock = threading.RLock()


@dataclass
class IconMatch:
    name:     str      # exact vendor product name, e.g. "AWS Lambda"
    type:     str      # DiagramLens component type
    provider: str      # aws | azure | gcp
    score:    float


# ── Loading ───────────────────────────────────────────────────────────────────
def _load_clip() -> dict:
    """Load just the CLIP image encoder.

    Kept separate from bank loading because build_icon_bank.py needs the
    encoder precisely when no bank exists yet.
    """
    if "model" in _state:
        return _state
    with _lock:
        if "model" in _state:
            return _state
        import torch
        from transformers import CLIPModel, CLIPImageProcessor
        _state["model"]     = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval()
        _state["processor"] = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        _state["torch"]     = torch
        return _state


def _load() -> dict:
    """Load the embedding bank plus the CLIP encoder. Cached at module level."""
    if "ready" in _state:
        return _state
    with _lock:
        if "ready" in _state:
            return _state
        _state["ready"] = False
        if not BANK_PATH.is_file():
            print(f"[icon_bank] no bank at {BANK_PATH} — icon proposer disabled "
                  "(run backend/scripts/build_icon_bank.py to enable)")
            return _state
        try:
            # No allow_pickle: the bank holds a float32 array plus a unicode
            # array of JSON strings, so it never deserialises Python objects.
            data = np.load(BANK_PATH)
            vectors = data["vectors"].astype(np.float32)
            # Pre-normalise so matching is a single dot product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            _state["vectors"] = vectors / np.clip(norms, 1e-8, None)
            _state["meta"]    = [json.loads(s) for s in data["meta"].tolist()]
            _load_clip()
            _state["ready"]   = True
            print(f"[icon_bank] loaded {len(_state['meta'])} icons from {BANK_PATH.name}")
        except Exception as exc:
            print(f"[icon_bank] load failed ({exc}) — icon proposer disabled")
        return _state


# Default OFF, measured. On the 13-diagram ground-truth set the calibrated
# matcher accepted 1 icon across 41 candidate crops and moved component F1 by
# 0.000, while adding ~4.8s per diagram (5.7s -> 10.5s). Loosening it to fire
# more often made it match ~everything to one entry, which is worse than
# nothing. Set ICON_BANK=1 to enable it for the ablation.
ENABLED = os.environ.get("ICON_BANK", "0").lower() in ("1", "true", "yes")


def available() -> bool:
    return ENABLED and bool(_load().get("ready"))


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_images(images: list[Image.Image]) -> np.ndarray:
    """L2-normalised CLIP image embeddings, one row per image."""
    st = _load_clip()
    torch = st["torch"]
    inputs = st["processor"](images=images, return_tensors="pt")
    with torch.no_grad():
        out = st["model"].get_image_features(**inputs)

    # transformers >=5 wraps the result in a BaseModelOutputWithPooling instead
    # of returning the tensor directly. Its pooler_output is ALREADY the
    # projected 512-d image embedding — running visual_projection over it again
    # fails with "mat1 and mat2 shapes cannot be multiplied (64x512 and 768x512)".
    if torch.is_tensor(out):
        feats = out
    else:
        feats = getattr(out, "image_embeds", None)
        if feats is None:
            feats = out.pooler_output
    vecs = feats.cpu().numpy().astype(np.float32)
    return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)


def match_batch(crops: list[np.ndarray]) -> list[IconMatch | None]:
    """Nearest-neighbour lookup for many crops in one forward pass."""
    st = _load()
    if not st.get("ready") or not crops:
        return [None] * len(crops)

    usable, index = [], []
    for i, crop in enumerate(crops):
        if not crop.size or crop.shape[0] < MIN_CROP_PX or crop.shape[1] < MIN_CROP_PX:
            continue
        # A flat fill (a plain coloured box) has no glyph to recognise, and
        # CLIP maps all such crops to whichever bank entry is blandest.
        if float(crop.reshape(-1, crop.shape[-1]).std(axis=0).mean()) < MIN_CROP_STD:
            continue
        usable.append(Image.fromarray(crop).convert("RGB"))
        index.append(i)
    if not usable:
        return [None] * len(crops)

    try:
        sims = embed_images(usable) @ st["vectors"].T     # (n_crops, n_icons)
    except Exception as exc:
        print(f"[icon_bank] match failed: {exc}")
        return [None] * len(crops)

    out: list[IconMatch | None] = [None] * len(crops)
    order = np.sort(sims, axis=1)[:, ::-1]
    margins = order[:, 0] - order[:, 1:11].mean(axis=1)
    zscores = (order[:, 0] - sims.mean(axis=1)) / (sims.std(axis=1) + 1e-8)

    for row, margin, z, target in zip(sims, margins, zscores, index):
        best = int(row.argmax())
        score = float(row[best])
        # All three must hold: the bank is dense enough that any one of them
        # alone passes unrelated crops.
        if score < MATCH_THRESHOLD or margin < MATCH_MARGIN or z < MATCH_ZSCORE:
            continue
        entry = st["meta"][best]
        out[target] = IconMatch(entry["name"], entry.get("type", "other"),
                                entry.get("provider", "aws"), score)
    return out


def match(crop: np.ndarray) -> IconMatch | None:
    return match_batch([crop])[0]


def hit_rate(matches: list[IconMatch | None]) -> float:
    """Fraction of crops matched — a signal the notation classifier uses to
    decide whether this is a vendor-icon diagram at all."""
    return sum(1 for m in matches if m) / len(matches) if matches else 0.0

"""
Generates the DiagramLens system architecture figure for the dissertation.

Written as a script rather than drawn by hand so the figure can be regenerated
when the architecture changes, and so the report can state that its figures are
produced reproducibly from source.

    python evaluation/figures/make_architecture_figure.py

Outputs (next to this script):
    architecture.png   300 dpi, for Word
    architecture.pdf   vector, for LaTeX
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

# Print-safe palette: distinguishable in colour AND when printed greyscale,
# because the borders and fills differ in lightness, not just hue.
INK        = "#1a1a1a"
MUTED      = "#5c5c5c"
BAND       = "#f4f5f7"
BAND_EDGE  = "#c8ccd2"
CLASSICAL  = "#dce9f7"; CLASSICAL_E = "#3b6ea5"
HYBRID     = "#e6dcf3"; HYBRID_E    = "#6b4fa0"
GEMINI     = "#d9efe2"; GEMINI_E    = "#2f7d55"
CONTRACT   = "#fdf0d5"; CONTRACT_E  = "#b8860b"
STORE      = "#eeeeee"; STORE_E     = "#777777"

fig, ax = plt.subplots(figsize=(13.5, 10.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def band(y, h, label):
    """Horizontal layer band, labelled inside its own top-left corner.

    Rotated labels in the left margin collided once band heights varied, so
    the label sits in the band instead — it always fits, whatever the height.
    """
    ax.add_patch(FancyBboxPatch(
        (4, y), 93, h, boxstyle="round,pad=0.35,rounding_size=0.8",
        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=0.9, zorder=1))
    ax.text(5.6, y + h - 1.1, label, va="top", ha="left",
            fontsize=7.2, color=MUTED, fontweight="bold", zorder=2)


def box(x, y, w, h, title, sub=None, fc="white", ec=INK, lw=1.1,
        title_size=9.2, sub_size=7.4, mono=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.6",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ty = y + h / 2 + (1.05 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_size,
            color=INK, fontweight="bold", zorder=4,
            family="monospace" if mono else None)
    if sub:
        ax.text(x + w / 2, ty - 2.5, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=4, linespacing=1.5)


def stages(x, y, w, items, ec):
    """Numbered pipeline stages stacked inside an arm column."""
    for i, text in enumerate(items):
        yy = y - i * 3.35
        ax.add_patch(FancyBboxPatch(
            (x, yy), w, 2.75, boxstyle="round,pad=0.12,rounding_size=0.35",
            facecolor="white", edgecolor=ec, linewidth=0.75, zorder=4))
        ax.text(x + 1.4, yy + 1.38, f"{i + 1}", ha="center", va="center",
                fontsize=7.0, color=ec, fontweight="bold", zorder=5)
        ax.text(x + 2.9, yy + 1.38, text, ha="left", va="center",
                fontsize=7.3, color=INK, zorder=5)


def arrow(x1, y1, x2, y2, style="-|>", lw=1.3, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls, zorder=6,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(50, 97.6, "DiagramLens — System Architecture", ha="center",
        fontsize=15.5, fontweight="bold", color=INK)
ax.text(50, 95.2, "Three extraction paradigms converging on one schema contract",
        ha="center", fontsize=9.3, color=MUTED, style="italic")

COL_X = [6.0, 28.4, 50.8, 73.2]   # four-up row
COL_W = 20.8

# Each band reserves a 2.6-unit strip at its top for the label, so content
# never rides over it whatever the band height.
LABEL_STRIP = 2.6

# ── Presentation ──────────────────────────────────────────────────────────────
band(86.4, 7.6, "PRESENTATION")
for x, (t, s) in zip(COL_X, [
        ("Upload", "drag & drop"),
        ("DiagramCanvas", "React Flow + Dagre"),
        ("BenchmarkPanel", "scores vs ground truth"),
        ("ChatPanel", "component Q&A")]):
    box(x, 87.2, COL_W, 4.2, t, s, fc="white")

# ── API ───────────────────────────────────────────────────────────────────────
band(77.8, 7.2, "API")
for x, (t, s) in zip(COL_X, [
        ("POST /api/analyze", "multipart image"),
        ("POST /api/benchmark", "session + diagram_id"),
        ("POST /api/chat", "message + component_id"),
        ("GET /api/sessions", "history")]):
    box(x, 78.4, COL_W, 4.0, t, s, fc="white", title_size=8.3, mono=True)

arrow(50, 87.2, 50, 82.7)

# ── Orchestration ─────────────────────────────────────────────────────────────
band(70.2, 6.8, "ORCHESTRATION")
box(18, 70.5, 64, 3.9, "extraction_orchestrator.py",
    "asyncio.gather — three arms run in parallel; a failed arm returns an empty "
    "result, never a placeholder",
    fc="white", title_size=9.4, sub_size=7.0, mono=True)
arrow(50, 78.4, 50, 74.7)

# ── Extraction arms ───────────────────────────────────────────────────────────
band(33.0, 35.6, "EXTRACTION")

cols = [
    (6.0, "CLASSICAL CV", "control arm — frozen, zero ML",
     CLASSICAL, CLASSICAL_E,
     ["grayscale, auto-invert, upscale",
      "Tesseract OCR → word boxes",
      "union-find text clustering",
      "noise / acronym filtering",
      "Canny → HoughLinesP",
      "snap endpoints to centroids"]),
    (37.2, "HYBRID ML", "notation-adaptive proposal fusion",
     HYBRID, HYBRID_E,
     ["PaddleOCR PP-OCRv5 (one pass)",
      "notation_classifier → profile",
      "shape_detector + compartments",
      "icon_bank · CLIP img→img (opt.)",
      "3-proposer fusion → NMS",
      "LSD + arrowhead + line style"]),
    (68.4, "GEMINI VLM", "gemini-2.5-flash, cloud API",
     GEMINI, GEMINI_E,
     ["MIME detection from bytes",
      "v2 prompt — extraction only",
      "response_schema enforced",
      "retry ×3 + salvage counter",
      "hallucination_filter (OCR)",
      "id → name resolution"]),
]

ARM_W = 28.6
for x, title, sub, fc, ec, items in cols:
    box(x, 33.6, ARM_W, 30.4, "", fc=fc, ec=ec, lw=1.6)
    ax.text(x + ARM_W / 2, 61.5, title, ha="center", fontsize=10.4,
            fontweight="bold", color=ec, zorder=5)
    ax.text(x + ARM_W / 2, 59.3, sub, ha="center", fontsize=7.4,
            color=MUTED, style="italic", zorder=5)
    stages(x + 1.8, 54.5, 25.0, items, ec)
    arrow(50, 70.3, x + ARM_W / 2, 64.2, rad=0.0, lw=1.1, color=MUTED)
    arrow(x + ARM_W / 2, 33.6, x + ARM_W / 2, 30.7, lw=1.3, color=ec)

# Shared perception/knowledge modules, annotated against the hybrid column
ax.text(37.2 + ARM_W / 2, 35.8,
        "ocr_engine · shape_detector · connection_detector · icon_bank\n"
        "notation_profiles · relationship_table · common",
        ha="center", va="center", fontsize=6.7, color=MUTED,
        family="monospace", zorder=5, linespacing=1.6)

# ── Contract ──────────────────────────────────────────────────────────────────
band(24.0, 6.4, "SCHEMA CONTRACT")
box(13, 24.4, 74, 3.5, "models/schemas.py  ·  ArchitectureSchema",
    "components[] · connections[] with source_name/target_name · pipeline · "
    "diagram_standard · response_time_ms",
    fc=CONTRACT, ec=CONTRACT_E, lw=1.5, title_size=9.4, sub_size=6.8, mono=True)

# ── Evaluation ────────────────────────────────────────────────────────────────
band(14.6, 8.8, "EVALUATION")
for x, (t, s) in zip([6.0, 37.2, 68.4], [
        ("metrics.py",
         "normalised fuzzy match · one-to-one\nassignment · directed + undirected"),
        ("routers/benchmark.py",
         "scores 3 arms vs ground truth\nTier A universal · Tier B conditional"),
        ("evaluation/ground_truth/",
         "annotated diagrams + source_url\ncomponents and connections by name")]):
    box(x, 15.0, ARM_W, 5.2, t, s, fc="white", ec=INK,
        title_size=8.6, sub_size=6.7, mono=True)

# ── Persistence ───────────────────────────────────────────────────────────────
band(5.6, 7.6, "PERSISTENCE")
box(18, 6.0, 28.0, 4.4, "PostgreSQL", "sessions · architectures · benchmarks",
    fc=STORE, ec=STORE_E, title_size=8.8, sub_size=6.8)
box(54, 6.0, 28.0, 4.4, "Redis", "image-hash cache, 24h TTL",
    fc=STORE, ec=STORE_E, title_size=8.8, sub_size=6.8)

# Schema feeds the scorer, which then writes its rows. Both arrows stop at box
# edges rather than running through the middle column.
arrow(50, 24.4, 50, 20.5)
arrow(50, 15.0, 50, 10.7)

# Feedback edge: persisted results are re-rendered by the presentation layer
arrow(95.8, 8.2, 95.8, 87.2, style="-|>", lw=1.0, color=MUTED, ls=(0, (4, 3)),
      rad=-0.13)
ax.text(98.2, 50, "session reload", rotation=90, ha="center", va="center",
        fontsize=6.9, color=MUTED, style="italic")

# ── Footnote ──────────────────────────────────────────────────────────────────
ax.text(50, 2.4,
        "Arms share no detection code. The hybrid arm has its own connection "
        "detector rather than reusing the classical one,\nso the three-way "
        "comparison measures three distinct methods rather than two.",
        ha="center", va="center", fontsize=7.4, color=MUTED, style="italic",
        linespacing=1.7)

plt.tight_layout(pad=0.4)
for ext in ("png", "pdf"):
    path = OUT / f"architecture.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")

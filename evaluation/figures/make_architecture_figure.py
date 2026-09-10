"""
Generates the DiagramLens system architecture figure for the dissertation.

Written as a script rather than drawn by hand so the figure can be regenerated
when the architecture changes, and so the report can state that its figures are
produced reproducibly from source.

    python evaluation/figures/make_architecture_figure.py

Outputs (next to this script):
    architecture.png   300 dpi
    architecture.pdf   vector, for LaTeX

Design intent: a layered block diagram, monochrome, large type. It shows which
layers exist and how a request flows through them. The internal steps of each
extraction arm are described in Chapter 4, not repeated here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

INK       = "#111111"
MUTED     = "#555555"
BAND      = "#f2f2f2"
BAND_EDGE = "#cfcfcf"

fig, ax = plt.subplots(figsize=(10.0, 10.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

TITLE_SIZE = 19
BAND_LABEL = 11
BOX_TITLE  = 12
BOX_SUB    = 9

LABEL_STRIP = 3.4          # reserved at the top of every band for its label
GAP         = 1.6          # vertical gap between bands
_cursor = [94.0]           # running y of the top of the next band


def add_band(content_h, label):
    """Stack a band below the previous one. Returns (box_y_top, box_area_h)."""
    h = content_h + LABEL_STRIP
    top = _cursor[0]
    y = top - h
    ax.add_patch(FancyBboxPatch(
        (3, y), 94, h, boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=1.0, zorder=1))
    ax.text(5.4, top - 1.3, label, va="top", ha="left",
            fontsize=BAND_LABEL, color=MUTED, fontweight="bold", zorder=2)
    _cursor[0] = y - GAP
    return top - LABEL_STRIP, content_h    # top of usable area, its height


def box(x, y_top, w, h, title, sub=None, lw=1.3, title_size=BOX_TITLE,
        sub_size=BOX_SUB, mono=False):
    y = y_top - h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.5",
        facecolor="white", edgecolor=INK, linewidth=lw, zorder=3))
    ty = y + h / 2 + (1.4 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_size,
            color=INK, fontweight="bold", zorder=4,
            family="monospace" if mono else None)
    if sub:
        ax.text(x + w / 2, ty - 3.1, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=4, linespacing=1.5)


def arrow(x1, y1, x2, y2, lw=1.6, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17,
        linewidth=lw, color=color, linestyle=ls, zorder=6,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


# ── Title ────────────────────────────────────────────────────────────────────
ax.text(50, 99.0, "DiagramLens — System Architecture", ha="center",
        fontsize=TITLE_SIZE, fontweight="bold", color=INK)
ax.text(50, 96.4, "Three extraction arms converging on one schema contract",
        ha="center", fontsize=BOX_SUB + 1, color=MUTED, style="italic")

X4 = [5.5, 29.0, 52.5, 76.0];  W4 = 19.0
X3 = [5.5, 37.0, 68.5];        W3 = 26.0

# ── Presentation ────────────────────────────────────────────────────────────
top, _ = add_band(5.4, "PRESENTATION  (Next.js)")
for x, (t, s) in zip(X4, [("Upload", "drag & drop"), ("Canvas", "React Flow"),
                          ("Benchmark", "scores vs GT"), ("Chat", "Q&A")]):
    box(x, top, W4, 5.4, t, s)

# ── API ─────────────────────────────────────────────────────────────────────
top, _ = add_band(4.8, "API  (FastAPI)")
for x, t in zip(X4, ["/analyze", "/benchmark", "/chat", "/sessions"]):
    box(x, top, W4, 4.8, t, mono=True)
arrow(50, top + LABEL_STRIP + GAP + 0.3, 50, top - 0.2)

# ── Orchestration ───────────────────────────────────────────────────────────
top, _ = add_band(4.6, "ORCHESTRATION")
box(16, top, 68, 4.6, "extraction_orchestrator",
    "three arms run in parallel; a failed arm returns an empty result",
    mono=True, title_size=BOX_TITLE - 1)
arrow(50, top + LABEL_STRIP + GAP + 0.3, 50, top - 0.2)

# ── Extraction arms ─────────────────────────────────────────────────────────
top, _ = add_band(13.0, "EXTRACTION  (arms share no detection code)")
for x, (t, s) in zip(X3, [
        ("Classical CV", "deterministic baseline\nOpenCV + Tesseract, no ML"),
        ("Hybrid ML", "notation-adaptive fusion\nPaddleOCR + geometry + rules"),
        ("Gemini VLM", "end-to-end\ngemini-2.5-flash, cloud API")]):
    box(x, top, W3, 9.4, t, s, lw=1.6)
    arrow(50, top + LABEL_STRIP + GAP + 0.3, x + W3 / 2, top - 0.2,
          lw=1.2, color=MUTED)
ax.text(37.0 + W3 / 2, top - 10.0,
        "shared services (hybrid only): ocr_engine, shape_detector,\n"
        "connection_detector, icon_bank, notation_profiles",
        ha="center", va="top", fontsize=BOX_SUB - 2, color=MUTED,
        family="monospace", linespacing=1.5)

# ── Schema contract ─────────────────────────────────────────────────────────
top, _ = add_band(4.0, "SCHEMA CONTRACT")
box(11, top, 78, 4.0, "ArchitectureSchema",
    "components[] and connections[], by name; pipeline; diagram_standard",
    lw=2.0, mono=True, title_size=BOX_TITLE - 1)
for x in X3:
    arrow(x + W3 / 2, top + LABEL_STRIP + GAP + 4.0, x + W3 / 2, top - 0.2,
          lw=1.4)

# ── Evaluation ──────────────────────────────────────────────────────────────
top, _ = add_band(6.4, "EVALUATION  (offline; no server or database)")
for x, (t, s) in zip(X3, [
        ("metrics", "normalised fuzzy match\none-to-one assignment"),
        ("benchmark runner", "scores 3 arms vs GT\nmachine-readable JSON"),
        ("ground_truth/", "37 annotated diagrams\ncomponents + connections")]):
    box(x, top, W3, 6.4, t, s, mono=(t != "ground_truth/"),
        title_size=BOX_TITLE - 1)
arrow(50, top + LABEL_STRIP + GAP + 0.3, 50, top - 0.2, lw=1.4)

# ── Persistence ─────────────────────────────────────────────────────────────
top, _ = add_band(4.8, "PERSISTENCE")
box(16, top, 30, 4.8, "PostgreSQL", "sessions, results, benchmarks")
box(54, top, 30, 4.8, "Redis", "image-hash cache, 24 h TTL")
arrow(50, top + LABEL_STRIP + GAP + 6.4, 50, top - 0.2, lw=1.4)

bottom = _cursor[0] + GAP
ax.set_ylim(bottom - 2, 100)

# Feedback edge down the right margin
arrow(94.5, bottom + 2.0, 94.5, 91.0, lw=1.1, color=MUTED, ls=(0, (4, 3)),
      rad=-0.10)
ax.text(97.2, (bottom + 91) / 2, "session reload", rotation=90, ha="center",
        va="center", fontsize=BOX_SUB - 1, color=MUTED, style="italic")

plt.tight_layout(pad=0.5)
for ext in ("png", "pdf"):
    path = OUT / f"architecture.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")

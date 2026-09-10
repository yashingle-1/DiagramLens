"""
Figure — end-to-end data flow, drawn as a timeline.

The architecture figure shows WHAT the modules are. This one shows WHEN they
run: the three arms are concurrent, so wall-clock time is the slowest arm
rather than their sum.

Bar lengths are the median per-arm latencies from the 37-diagram benchmark
(Table 5.6): classical 2.5 s, hybrid 5.9 s, Gemini 9.3 s.

    python evaluation/figures/make_dataflow_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
INK, MUTED = "#111111", "#555555"
BAND, BAND_EDGE = "#f2f2f2", "#cfcfcf"

fig, ax = plt.subplots(figsize=(12.0, 9.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

LABEL_STRIP = 3.6
GAP = 1.6
_cur = [92.0]


def add_band(content_h, label):
    h = content_h + LABEL_STRIP
    top = _cur[0]
    y = top - h
    ax.add_patch(FancyBboxPatch((3, y), 94, h,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=1.0, zorder=1))
    ax.text(5.4, top - 1.3, label, va="top", ha="left", fontsize=11,
            color=MUTED, fontweight="bold", zorder=2)
    _cur[0] = y - GAP
    return top - LABEL_STRIP


def cell(x, y_top, w, h, title, sub=None, lw=1.4, ec=INK, mono=False):
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.22,rounding_size=0.5",
        facecolor="white", edgecolor=ec, linewidth=lw, zorder=3))
    ty = y + h / 2 + (1.5 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=4,
            family="monospace" if mono else None)
    if sub:
        ax.text(x + w / 2, ty - 3.4, sub, ha="center", va="center",
                fontsize=8.7, color=MUTED, zorder=4, linespacing=1.4)


def arrow(x1, y1, x2, y2, lw=1.5, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=16, linewidth=lw, color=color, linestyle=ls, zorder=6,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


ax.text(50, 98.0, "DiagramLens — End-to-End Data Flow", ha="center",
        fontsize=19, fontweight="bold", color=INK)
ax.text(50, 95.0, "Stages run in sequence, except extraction, where the three "
        "arms run concurrently", ha="center", fontsize=10, color=MUTED,
        style="italic")

X3 = [6, 37, 68]

# ── Client ──────────────────────────────────────────────────────────────────
top = add_band(6.2, "CLIENT")
for x, t, s in [(6, "1  Upload", "PNG / JPG / WebP"),
                (37, "8  Render", "canvas + panels"),
                (68, "9  Score", "on demand, vs GT")]:
    cell(x, top, 26, 6.2, t, s)

# ── API + orchestration ─────────────────────────────────────────────────────
top = add_band(6.2, "API  +  ORCHESTRATION")
for x, t, s in [(6, "2  save upload", "UUID filename + bytes"),
                (37, "3  hash + cache", "SHA -> Redis lookup"),
                (68, "4  dispatch", "all three arms at once")]:
    cell(x, top, 26, 6.2, t, s)
arrow(19, top + LABEL_STRIP + GAP + 0.3, 19, top - 0.2)
arrow(32, top - 3.1, 37, top - 3.1)
arrow(63, top - 3.1, 68, top - 3.1)
arrow(81, top - 6.2, 81, top - 8.0)

# ── Extraction timeline ─────────────────────────────────────────────────────
top = add_band(21.5, "EXTRACTION  (concurrent)")
T0, T_SCALE = 12.0, 7.4
def t(sec): return T0 + sec * T_SCALE
grid_lo, grid_hi = top - 20.5, top - 3.0
for s in range(0, 11, 2):
    ax.plot([t(s), t(s)], [grid_lo, grid_hi], color="#dddddd", lw=0.9, zorder=1)
    ax.text(t(s), grid_lo - 1.6, f"{s}s", ha="center", fontsize=9, color=MUTED)

bars = [
    (top - 5.0, 2.5, "Classical CV", "Tesseract, Canny, Hough"),
    (top - 10.4, 5.9, "Hybrid ML", "PaddleOCR, contours, skeleton tracing"),
    (top - 15.8, 9.3, "Gemini VLM", "network round trip + generation"),
]
for y_top, secs, label, sub in bars:
    y = y_top - 4.6
    ax.add_patch(FancyBboxPatch((t(0), y), t(secs) - t(0), 4.6,
        boxstyle="round,pad=0.15,rounding_size=0.4",
        facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3))
    ax.text(t(0) + 1.6, y + 3.0, label, va="center", fontsize=10.5,
            fontweight="bold", color=INK, zorder=5)
    ax.text(t(0) + 1.6, y + 1.3, sub, va="center", fontsize=8.2, color=MUTED,
            style="italic", zorder=5)
    ax.text(t(secs) + 1.4, y + 2.3, f"{secs:.1f} s", va="center", fontsize=9.5,
            color=MUTED, zorder=5)

ax.plot([t(9.3), t(9.3)], [grid_lo - 0.5, grid_hi + 1.5], color=INK, lw=1.8,
        ls=(0, (5, 3)), zorder=7)
ax.text(t(9.3), grid_hi + 2.6,
        "wall-clock ~9.3 s, not 17.7 s: the arms overlap",
        fontsize=9.5, color=INK, ha="center", va="bottom", fontweight="bold")

# ── Contract + persistence ──────────────────────────────────────────────────
top = add_band(13.5, "CONTRACT  +  PERSISTENCE")
cell(6, top, 88, 5.6,
     "5  unsupported-component check",
     "each Gemini component name cross-checked against full-page OCR",
     lw=1.4)
cell(6, top - 6.4, 42, 6.4, "6  ArchitectureSchema x3",
     "one Pydantic model per arm —\nthe basis for a fair comparison", lw=1.8)
cell(52, top - 6.4, 42, 6.4, "7  PostgreSQL / Redis",
     "3 architecture rows;\nimage-hash cache, 24 h TTL", ec="#666666")
arrow(50, top + LABEL_STRIP + GAP + 0.3, 50, top - 0.2)
arrow(27, top - 5.6, 27, top - 8.0)

bottom = _cur[0] + GAP
arrow(94.5, bottom + 1.5, 94.5, 88.0, lw=1.1, color=MUTED, ls=(0, (4, 3)),
      rad=-0.10)
ax.text(97.2, (bottom + 88) / 2, "session reload", rotation=90, ha="center",
        va="center", fontsize=8.5, color=MUTED, style="italic")

ax.text(50, bottom - 3.0,
        "A failed arm returns an empty component list with an error recorded, "
        "never a placeholder node:\na fabricated component would count as a "
        "false positive and distort precision.",
        ha="center", va="center", fontsize=9, color=MUTED, style="italic",
        linespacing=1.5)

ax.set_ylim(bottom - 6, 100)
plt.tight_layout(pad=0.5)
for ext in ("png", "pdf"):
    p = OUT / f"dataflow.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")

"""
Figure — the hybrid arm's three-proposer fusion.

Shows why the arm generalises across notations: three proposers with different
competences, only one of which fires on every diagram. The per-proposer
implementation detail is in Section 4.2.4, not repeated here.

    python evaluation/figures/make_fusion_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
INK, MUTED = "#111111", "#555555"
BAND, BAND_EDGE = "#f2f2f2", "#cfcfcf"

fig, ax = plt.subplots(figsize=(11.0, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def band(y, h, label):
    ax.add_patch(FancyBboxPatch((3, y), 94, h,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=1.0, zorder=1))
    ax.text(5.4, y + h - 1.3, label, va="top", ha="left", fontsize=11,
            color=MUTED, fontweight="bold", zorder=2)


def rect(x, y, w, h, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.22,rounding_size=0.5",
        facecolor="white", edgecolor=INK, linewidth=lw, zorder=3))


def htext(x, y, s, size=10, weight="normal", style="normal", color=INK,
          mono=False):
    ax.text(x, y, s, ha="center", va="center", fontsize=size,
            fontweight=weight, style=style, color=color, zorder=4,
            family="monospace" if mono else None, linespacing=1.5)


def arrow(x1, y1, x2, y2, lw=1.5, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=16, linewidth=lw, color=color, zorder=6,
        shrinkA=1, shrinkB=1))


# ── Title ────────────────────────────────────────────────────────────────────
htext(50, 97.5, "Hybrid Arm — Proposal Fusion", size=19, weight="bold")
htext(50, 94.2,
      "Three proposers of differing competence; only the text proposer fires "
      "on every notation", size=10, style="italic", color=MUTED)

# ── Input ───────────────────────────────────────────────────────────────────
rect(37, 87.5, 26, 4.4)
htext(50, 89.7, "Diagram image", size=12, weight="bold")

# ── Shared perception ───────────────────────────────────────────────────────
band(72.5, 12.0, "SHARED PERCEPTION")
for x, t, s in [
    (6,  "OCR", "PaddleOCR PP-OCRv5\nword boxes + confidence"),
    (36, "notation classifier", "stereotypes, C4 tags,\nstroke variance"),
    (66, "shape detector", "contours, compartment\nmerge, dashed boundary"),
]:
    rect(x, 73.5, 28, 8.6)
    htext(x + 14, 79.8, t, size=11, weight="bold")
    htext(x + 14, 76.2, s, size=9, color=MUTED)
arrow(50, 87.5, 50, 82.1)

# ── Profile gate ────────────────────────────────────────────────────────────
rect(26, 65.6, 48, 5.0, lw=1.8)
htext(50, 68.9, "notation rule profile", size=11, weight="bold")
htext(50, 66.5, "rules applied only above classifier confidence 0.7",
      size=9, color=MUTED)
arrow(50, 73.5, 50, 70.6)

# ── Three proposers ─────────────────────────────────────────────────────────
band(41.0, 21.0, "PROPOSERS")
props = [
    (6,  "P1 · Icon retrieval", "cloud notations only",
     "CLIP image-to-image match against\na vendor icon bank\n"
     "(disabled by default, Section 5.7.2)"),
    (36, "P2 · Shape", "diagrams with drawn boxes",
     "contour polygon classified by\nvertex count and solidity;\n"
     "compartments merged to one box"),
    (66, "P3 · Text", "every notation — the floor",
     "OCR words clustered by spatial\nproximity; always fires, because\n"
     "every component is labelled"),
]
for x, name, scope, body in props:
    rect(x, 43.0, 28, 16.4, lw=1.6)
    htext(x + 14, 56.6, name, size=11.5, weight="bold")
    htext(x + 14, 53.9, scope, size=8.5, style="italic", color=MUTED)
    htext(x + 14, 48.6, body, size=8.5, color=INK)
    arrow(x + 14, 43.0, x + 14, 37.0, lw=1.4)
arrow(50, 65.6, 20, 60.0, lw=1.1, color=MUTED)
arrow(50, 65.6, 50, 60.0, lw=1.1, color=MUTED)
arrow(50, 65.6, 80, 60.0, lw=1.1, color=MUTED)

# ── Fusion ──────────────────────────────────────────────────────────────────
band(18.5, 18.5, "FUSION")
rect(6, 28.5, 88, 6.4)
htext(50, 32.7, "Merge", size=11, weight="bold")
htext(50, 29.9,
      "text in a shape or below an icon becomes one component; orphan text is "
      "kept;\na shape with no text or icon is discarded as decoration",
      size=9, color=MUTED)
rect(6, 20.0, 88, 6.4, lw=1.8)
htext(50, 24.2, "Non-maximum suppression  (IoU 0.6)", size=11, weight="bold")
htext(50, 21.4,
      "richest evidence wins: icon+text > shape+text > icon > text;\n"
      "the winning proposer is recorded per component", size=9, color=MUTED)
arrow(50, 28.5, 50, 26.4)

# ── Output ──────────────────────────────────────────────────────────────────
rect(22, 11.0, 56, 5.4)
htext(50, 14.3, "components  ->  ArchitectureSchema", size=11, weight="bold",
      mono=True)
htext(50, 12.1, "name, type, parent_id, stereotype, icon_match, proposer",
      size=8.5, color=MUTED)
arrow(50, 20.0, 50, 16.4, lw=1.4)

htext(50, 6.0,
      "P3 is the floor, not a fallback. P1 and P2 raise precision and typing "
      "where the notation supports them,\nbut neither is required for the arm "
      "to produce output.", size=9, style="italic", color=MUTED)

ax.set_ylim(3, 100)
plt.tight_layout(pad=0.5)
for ext in ("png", "pdf"):
    p = OUT / f"fusion.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")

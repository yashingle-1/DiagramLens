"""
Figure 3 — the hybrid arm's three-proposer fusion.

This is the project's main technical contribution and warrants its own figure
rather than a single box in the architecture diagram. It shows why the arm
generalises across notations: three proposers with different competences, only
one of which fires on every diagram.

    python evaluation/figures/make_fusion_figure.py
"""

from figure_style import (ACCENT, ARM, BAND, BAND_EDGE, INK, MUTED, arrow,
                          band, box, canvas, note, save, title)

P1 = ("#e3f0fb", "#2f6ea8")   # icon bank
P2 = ("#fdeee0", "#c2701f")   # shape
P3 = ("#e8f4e4", "#3f7a34")   # text — the universal floor

fig, ax = canvas(13.5, 9.4)
title(ax, "Hybrid Arm — Notation-Adaptive Proposal Fusion",
      "Three proposers of differing competence; only the text proposer fires on "
      "every notation", y=96.8)

# ── Input ─────────────────────────────────────────────────────────────────────
box(ax, 34, 87.5, 32, 4.6, "Diagram image", "raw pixels", lw=1.4)

# ── Stage 0 — shared perception ───────────────────────────────────────────────
band(ax, 72.4, 12.8, "SHARED PERCEPTION")
box(ax, 6, 73.2, 27, 6.4, "ocr_engine.read_page()",
    "PaddleOCR PP-OCRv5, one full-page\npass → word boxes + confidence",
    title_size=8.4, sub_size=6.8, mono=True)
box(ax, 36.5, 73.2, 27, 6.4, "notation_classifier",
    "stereotypes, C4 tags, stroke variance\n→ notation + confidence",
    title_size=8.4, sub_size=6.8, mono=True)
box(ax, 67, 73.2, 26, 6.4, "shape_detector",
    "contours, compartment merging,\ndashed boundary pass",
    title_size=8.4, sub_size=6.8, mono=True)

arrow(ax, 50, 87.5, 50, 79.9)

# Profile gate
box(ax, 30, 66.2, 40, 4.4, "notation_profiles.profile_for()",
    "rules applied only above confidence 0.7",
    fc="#fdf0d5", ec=ACCENT, lw=1.5, title_size=8.6, sub_size=6.9, mono=True)
arrow(ax, 50, 73.2, 50, 70.8)

# ── Stage 1 — the three proposers ─────────────────────────────────────────────
band(ax, 36.4, 27.6, "PROPOSERS")

props = [
    (6.0, P1, "P1 · ICON BANK", "AWS · Azure · GCP",
     ["CLIP image → image retrieval",
      "1456 official vendor icons",
      "accept on margin + z-score,",
      "not absolute cosine",
      "OFF by default (measured)"]),
    (36.5, P2, "P2 · SHAPE", "C4 · UML · informal",
     ["contour polygon classification",
      "rect · cylinder · diamond · hexagon",
      "compartments fused into one box",
      "containers split from components",
      "label read from inside the shape"]),
    (67.0, P3, "P3 · TEXT", "every notation — the floor",
     ["PaddleOCR word boxes",
      "union-find spatial clustering",
      "gaps adaptive to text height",
      "always fires: every component",
      "in every notation has a label"]),
]

for x, (fc, ec), name, scope, lines in props:
    w = 27 if x < 60 else 26
    box(ax, x, 41.0, w, 18.6, "", fc=fc, ec=ec, lw=1.7)
    ax.text(x + w / 2, 57.4, name, ha="center", fontsize=9.8,
            fontweight="bold", color=ec, zorder=5)
    ax.text(x + w / 2, 55.4, scope, ha="center", fontsize=7.2, color=MUTED,
            style="italic", zorder=5)
    for i, line in enumerate(lines):
        ax.text(x + 1.6, 52.6 - i * 2.4, f"·  {line}", ha="left", fontsize=7.0,
                color=INK, zorder=5)
    arrow(ax, x + w / 2, 41.0, x + w / 2, 34.0, lw=1.3, color=ec)

arrow(ax, 50, 66.2, 20, 60.0, lw=1.0, color=MUTED)
arrow(ax, 50, 66.2, 50, 60.0, lw=1.0, color=MUTED)
arrow(ax, 50, 66.2, 79, 60.0, lw=1.0, color=MUTED)

# ── Stage 2 — merge ───────────────────────────────────────────────────────────
band(ax, 15.6, 19.6, "FUSION")

box(ax, 6, 26.0, 87, 6.6, "Merge rules",
    "text inside a shape → one component   ·   text below an icon → one component   ·   "
    "orphan text → still a component\nshape with neither text nor icon → discarded as "
    "decoration   ·   type priority: icon > label keyword > shape kind",
    title_size=9.0, sub_size=7.0)

box(ax, 6, 17.2, 87, 7.0, "Non-maximum suppression  ·  IoU ≥ 0.6",
    "richest evidence wins:   icon+text  >  shape+text  >  icon  >  text\n"
    "the winning proposer is recorded per component, so extraction provenance "
    "is reportable",
    fc="#fdf0d5", ec=ACCENT, lw=1.5, title_size=9.0, sub_size=7.0)

arrow(ax, 50, 26.0, 50, 24.4)

# ── Output ────────────────────────────────────────────────────────────────────
box(ax, 22, 9.2, 56, 4.6, "components[]  →  ArchitectureSchema",
    "name · type · parent_id · stereotype · icon_match · proposer",
    lw=1.4, title_size=9.2, sub_size=7.0, mono=True)
arrow(ax, 50, 17.2, 50, 14.0)

note(ax,
     "P3 is the floor rather than a fallback. P1 and P2 raise precision and "
     "typing where the notation supports them,\nbut neither is required for the "
     "arm to produce output.", y=4.4)

save(fig, "fusion")

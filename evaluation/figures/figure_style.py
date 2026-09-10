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
ACCENT    = "#111111"   # heavier border, not a colour

# Kept as (fill, edge) pairs for call-site compatibility. All monochrome now:
# the arms are told apart by label and border weight, not by hue.
ARM = {
    "classical": ("white", INK),
    "hybrid":    ("white", INK),
    "gemini":    ("white", INK),
}
CONTRACT = ("white", INK)
STORE    = ("white", "#666666")

PIPELINE_LABEL = {
    "classical": "Classical CV",
    "hybrid":    "Hybrid ML",
    "gemini":    "Gemini VLM",
}

TITLE_SIZE = 19
BAND_LABEL = 11
BOX_TITLE  = 12
BOX_SUB    = 9.5
NOTE_SIZE  = 9.5


def canvas(w=12.0, h=9.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def title(ax, main, sub=None, y=97.0):
    ax.text(50, y, main, ha="center", fontsize=TITLE_SIZE, fontweight="bold",
            color=INK)
    if sub:
        ax.text(50, y - 3.0, sub, ha="center", fontsize=BOX_SUB + 1, color=MUTED,
                style="italic")


def band(ax, y, h, label, x=3.0, w=94.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=1.0, zorder=1))
    ax.text(x + 2.0, y + h - 1.4, label, va="top", ha="left",
            fontsize=BAND_LABEL, color=MUTED, fontweight="bold", zorder=2)


def box(ax, x, y, w, h, title_text="", sub=None, fc="white", ec=INK, lw=1.3,
        title_size=BOX_TITLE, sub_size=BOX_SUB, mono=False, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.5",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))
    if not title_text:
        if sub:
            ax.text(x + w / 2, y + h / 2, sub, ha="center", va="center",
                    fontsize=sub_size, color=MUTED, zorder=zorder + 1,
                    linespacing=1.5)
        return
    ty = y + h / 2 + (1.3 if sub else 0)
    ax.text(x + w / 2, ty, title_text, ha="center", va="center",
            fontsize=title_size, color=INK, fontweight="bold", zorder=zorder + 1,
            family="monospace" if mono else None)
    if sub:
        ax.text(x + w / 2, ty - 3.0, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=zorder + 1,
                linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.6, color=INK, ls="-", rad=0.0,
          zorder=6):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=17, linewidth=lw,
        color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def note(ax, text, y=3.0, size=NOTE_SIZE):
    ax.text(50, y, text, ha="center", va="center", fontsize=size, color=MUTED,
            style="italic", linespacing=1.6)


def save(fig, stem):
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)

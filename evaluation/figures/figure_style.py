"""
Shared styling for every dissertation figure.

Kept in one place so the figures read as one system rather than a collection of
separately-drawn pictures, and so a change to the palette propagates.

Palette is chosen to survive greyscale printing: fills differ in lightness, not
only in hue, and every arm keeps a distinct border weight.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

INK       = "#1a1a1a"
MUTED     = "#5c5c5c"
BAND      = "#f4f5f7"
BAND_EDGE = "#c8ccd2"

ARM = {
    "classical": ("#dce9f7", "#3b6ea5"),
    "hybrid":    ("#e6dcf3", "#6b4fa0"),
    "gemini":    ("#d9efe2", "#2f7d55"),
}
CONTRACT = ("#fdf0d5", "#b8860b")
STORE    = ("#eeeeee", "#777777")
ACCENT   = "#b8860b"

PIPELINE_LABEL = {
    "classical": "Classical CV",
    "hybrid":    "Hybrid ML",
    "gemini":    "Gemini VLM",
}


def canvas(w=13.5, h=10.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def title(ax, main, sub=None, y=97.5):
    ax.text(50, y, main, ha="center", fontsize=15.5, fontweight="bold", color=INK)
    if sub:
        ax.text(50, y - 2.4, sub, ha="center", fontsize=9.3, color=MUTED,
                style="italic")


def band(ax, y, h, label, x=4.0, w=93.0):
    """Layer band, labelled inside its own top-left corner so the label always
    fits regardless of band height."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=0.9, zorder=1))
    ax.text(x + 1.6, y + h - 1.1, label, va="top", ha="left",
            fontsize=7.2, color=MUTED, fontweight="bold", zorder=2)


def box(ax, x, y, w, h, title_text="", sub=None, fc="white", ec=INK, lw=1.1,
        title_size=9.2, sub_size=7.4, mono=False, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.6",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))
    # A caption-only box (no title) centres its sub-text instead of returning
    # early, which previously drew an empty rectangle.
    if not title_text:
        if sub:
            ax.text(x + w / 2, y + h / 2, sub, ha="center", va="center",
                    fontsize=sub_size, color=MUTED, zorder=zorder + 1,
                    linespacing=1.5)
        return
    ty = y + h / 2 + (1.05 if sub else 0)
    ax.text(x + w / 2, ty, title_text, ha="center", va="center",
            fontsize=title_size, color=INK, fontweight="bold", zorder=zorder + 1,
            family="monospace" if mono else None)
    if sub:
        ax.text(x + w / 2, ty - 2.5, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=zorder + 1, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.3, color=INK, ls="-", rad=0.0,
          zorder=6):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def note(ax, text, y=2.6, size=7.4):
    ax.text(50, y, text, ha="center", va="center", fontsize=size, color=MUTED,
            style="italic", linespacing=1.7)


def save(fig, stem):
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)

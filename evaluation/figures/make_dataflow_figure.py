"""
Figure 2 — end-to-end data flow, drawn as a timeline.

The architecture figure shows WHAT the modules are. This one shows WHEN they
run, which a static block diagram cannot convey: the three arms are concurrent,
so wall-clock is the slowest arm rather than their sum.

Timings are medians measured on the 13-diagram ground-truth set.

    python evaluation/figures/make_dataflow_figure.py
"""

from figure_style import (ARM, ACCENT, CONTRACT, INK, MUTED, STORE, arrow,
                          band, box, canvas, note, save, title)

fig, ax = canvas(13.5, 9.0)
title(ax, "DiagramLens — End-to-End Data Flow",
      "Sequential stages, except extraction, where the three arms run concurrently",
      y=96.5)

# ── 1. Client ─────────────────────────────────────────────────────────────────
band(ax, 80.0, 10.0, "CLIENT")
box(ax, 6, 80.8, 26, 4.6, "1  Upload image", "PNG / JPG / WebP, multipart")
box(ax, 36, 80.8, 26, 4.6, "9  Render", "React Flow canvas + panels")
box(ax, 66, 80.8, 27, 4.6, "10  Score on demand", "BenchmarkPanel → /api/benchmark")

# ── 2. API + orchestration ────────────────────────────────────────────────────
band(ax, 66.0, 10.0, "API  +  ORCHESTRATION")
box(ax, 6, 66.8, 26, 4.6, "2  storage.save_upload()",
    "UUID filename + raw bytes", title_size=8.4, mono=True)
box(ax, 36, 66.8, 26, 4.6, "3  cache.hash_image()",
    "SHA of bytes → Redis lookup", title_size=8.4, mono=True)
box(ax, 66, 66.8, 27, 4.6, "4  asyncio.gather()",
    "dispatch all three arms at once", title_size=8.4, mono=True)

arrow(ax, 19, 80.8, 19, 71.7)
arrow(ax, 32, 69.1, 36, 69.1)
arrow(ax, 62, 69.1, 66, 69.1)
arrow(ax, 79.5, 66.8, 79.5, 61.6)

# ── 3. Extraction, drawn against a time axis ──────────────────────────────────
band(ax, 32.0, 29.0, "EXTRACTION   ( concurrent )")

T0, T_SCALE = 10.0, 8.0          # x origin, units per second
def t(seconds: float) -> float:
    return T0 + seconds * T_SCALE

GRID_LO, GRID_HI = 40.2, 56.4
for s in range(0, 11, 2):
    ax.plot([t(s), t(s)], [GRID_LO, GRID_HI], color="#dfe3e8", lw=0.8, zorder=1)
    ax.text(t(s), 38.9, f"{s}s", ha="center", fontsize=6.8, color=MUTED)

bars = [
    ("classical", 51.6, 1.6, "Classical CV", "Tesseract · Canny · Hough"),
    ("hybrid",    46.0, 8.0, "Hybrid ML",    "PaddleOCR · contours · LSD"),
    ("gemini",    40.4, 9.2, "Gemini VLM",   "network round trip + generation"),
]
for key, y, secs, label, sub in bars:
    fc, ec = ARM[key]
    box(ax, t(0), y, t(secs) - t(0), 4.2, "", fc=fc, ec=ec, lw=1.5)
    ax.text(t(0) + 1.4, y + 2.7, label, va="center", fontsize=8.6,
            fontweight="bold", color=ec, zorder=5)
    ax.text(t(0) + 1.4, y + 1.1, sub, va="center", fontsize=6.5, color=MUTED,
            style="italic", zorder=5)
    ax.text(t(secs) + 1.5, y + 2.1, f"{secs:.1f}s", va="center",
            fontsize=7.4, color=MUTED, zorder=5)

# Wall-clock marker: gather completes when the slowest arm does
ax.plot([t(9.2), t(9.2)], [39.6, 57.6], color=ACCENT, lw=1.6, ls=(0, (5, 3)),
        zorder=7)
ax.text(t(9.2) - 0.8, 58.4, "wall-clock 9.2s — not 18.8s, the arms overlap",
        fontsize=7.4, color=ACCENT, ha="right", va="bottom", fontweight="bold")

box(ax, 6, 33.2, 87, 3.2, "",
    "5  hallucination_filter — every Gemini component name cross-checked "
    "against full-page OCR of the same image", sub_size=7.2)

# ── 4. Contract + persistence ─────────────────────────────────────────────────
band(ax, 14.4, 16.2, "CONTRACT  +  PERSISTENCE")
box(ax, 6, 23.4, 87, 3.4, "6  ArchitectureSchema  ×3",
    "one Pydantic model per arm — the reason three techniques are comparable",
    fc=CONTRACT[0], ec=CONTRACT[1], lw=1.5, title_size=9.2, sub_size=7.0)
box(ax, 12, 16.6, 36, 4.4, "7  PostgreSQL",
    "1 session row + 3 architecture rows", fc=STORE[0], ec=STORE[1],
    title_size=8.6, sub_size=6.9)
box(ax, 54, 16.6, 36, 4.4, "8  Redis",
    "image-hash cache, 24h TTL", fc=STORE[0], ec=STORE[1],
    title_size=8.6, sub_size=6.9)

arrow(ax, 50, 33.2, 50, 27.0)
arrow(ax, 30, 23.4, 30, 21.2)
arrow(ax, 72, 23.4, 72, 21.2)

# Persisted results are re-read by the client on reload — routed down the right
# margin so it does not cross the timeline.
arrow(ax, 95.6, 18.8, 95.6, 83.1, style="-|>", lw=1.0, color=MUTED,
      ls=(0, (4, 3)), rad=-0.10)
ax.text(97.8, 52, "session reload", rotation=90, ha="center", va="center",
        fontsize=6.9, color=MUTED, style="italic")

note(ax,
     "A failed arm returns an empty component list with extraction_error set, "
     "never a placeholder node:\na fabricated \"Unknown\" component would count "
     "as a false positive and distort precision.", y=9.0)

save(fig, "dataflow")

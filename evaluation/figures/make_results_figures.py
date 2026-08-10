"""
Results charts, generated from the offline benchmark output.

Reads evaluation/results/offline_benchmark.json so the charts and the numbers
quoted in the dissertation come from one source and cannot drift apart.
Re-run the benchmark, re-run this, and every figure updates.

    python backend/scripts/run_offline_benchmark.py
    python evaluation/figures/make_results_figures.py

Produces:
    results_overall.{png,pdf}     mean F1 / precision / recall per arm
    results_by_notation.{png,pdf} component F1 per arm per notation, with n
    results_normalisation.{png,pdf} normalised vs raw matching, per arm
    results_speed.{png,pdf}       median latency per arm, log scale
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ARM, INK, MUTED, PIPELINE_LABEL, OUT, save

DATA = Path(__file__).resolve().parent.parent / "results" / "offline_benchmark.json"

if not DATA.exists():
    raise SystemExit(
        f"No benchmark data at {DATA}\n"
        "Run:  python backend/scripts/run_offline_benchmark.py"
    )

rows = json.loads(DATA.read_text(encoding="utf-8"))["rows"]
arms = [a for a in ("classical", "hybrid", "gemini")
        if any(r["arm"] == a for r in rows)]


def mean(sub, key):
    vals = [r[key] for r in sub]
    return sum(vals) / len(vals) if vals else 0.0


def style_axes(ax, ylabel, ymax=1.0):
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8ccd2")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color="#e8eaed", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def bar_labels(ax, bars, fmt="{:.3f}", size=7.6):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018, fmt.format(h),
                ha="center", va="bottom", fontsize=size, color=INK)


# ── 1. Overall ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.4, 5.0))
metrics = [("component_f1", "F1"), ("component_precision", "Precision"),
           ("component_recall", "Recall")]
width = 0.26
for i, arm in enumerate(arms):
    sub = [r for r in rows if r["arm"] == arm]
    xs = [j + (i - (len(arms) - 1) / 2) * width for j in range(len(metrics))]
    bars = ax.bar(xs, [mean(sub, k) for k, _ in metrics], width * 0.92,
                  label=PIPELINE_LABEL[arm], color=ARM[arm][0],
                  edgecolor=ARM[arm][1], linewidth=1.4, zorder=3)
    bar_labels(ax, bars)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=9.5, color=INK)
style_axes(ax, "Component-level score", 1.0)
ax.legend(frameon=False, fontsize=8.8, ncol=len(arms), loc="upper center",
          bbox_to_anchor=(0.5, 1.09))
ax.set_title(f"Component extraction accuracy  (n = "
             f"{len({r['diagram'] for r in rows})} diagrams)",
             fontsize=11.5, fontweight="bold", color=INK, pad=26)
fig.tight_layout()
save(fig, "results_overall")

# ── 2. By notation ────────────────────────────────────────────────────────────
notations = sorted({r["notation"] for r in rows})
counts = {n: len({r["diagram"] for r in rows if r["notation"] == n})
          for n in notations}

fig, ax = plt.subplots(figsize=(9.0, 5.0))
for i, arm in enumerate(arms):
    xs, ys = [], []
    for j, n in enumerate(notations):
        sub = [r for r in rows if r["arm"] == arm and r["notation"] == n]
        xs.append(j + (i - (len(arms) - 1) / 2) * width)
        ys.append(mean(sub, "component_f1"))
    bars = ax.bar(xs, ys, width * 0.92, label=PIPELINE_LABEL[arm],
                  color=ARM[arm][0], edgecolor=ARM[arm][1], linewidth=1.4,
                  zorder=3)
    bar_labels(ax, bars, "{:.2f}", size=7.2)
ax.set_xticks(range(len(notations)))
ax.set_xticklabels([f"{n}\n(n = {counts[n]})" for n in notations],
                   fontsize=9.0, color=INK)
style_axes(ax, "Component F1", 1.0)
ax.legend(frameon=False, fontsize=8.8, ncol=len(arms), loc="upper center",
          bbox_to_anchor=(0.5, 1.09))
ax.set_title("Component F1 by diagram notation", fontsize=11.5,
             fontweight="bold", color=INK, pad=34)
# n is printed on every group: the per-notation subsets are small and the
# figure should not imply more evidence than exists.
fig.text(0.5, 0.005,
         "Subset sizes are stated because they are small; these are indicative, "
         "not conclusive.",
         ha="center", fontsize=7.4, color=MUTED, style="italic")
fig.tight_layout(rect=(0, 0.035, 1, 1))
save(fig, "results_by_notation")

# ── 3. Effect of name normalisation ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 4.8))
xs = range(len(arms))
raw_vals  = [mean([r for r in rows if r["arm"] == a], "component_f1_raw") for a in arms]
norm_vals = [mean([r for r in rows if r["arm"] == a], "component_f1") for a in arms]
b1 = ax.bar([x - 0.18 for x in xs], raw_vals, 0.34, label="Raw character match",
            color="#e9ecef", edgecolor="#9aa0a6", linewidth=1.3, zorder=3)
b2 = ax.bar([x + 0.18 for x in xs], norm_vals, 0.34, label="Normalised match",
            color="#fdf0d5", edgecolor="#b8860b", linewidth=1.4, zorder=3)
bar_labels(ax, b1); bar_labels(ax, b2)
ax.set_xticks(list(xs))
ax.set_xticklabels([PIPELINE_LABEL[a] for a in arms], fontsize=9.5, color=INK)
style_axes(ax, "Component F1", 1.0)
ax.legend(frameon=False, fontsize=8.8)
ax.set_title("Effect of name normalisation on measured accuracy",
             fontsize=11.5, fontweight="bold", color=INK, pad=12)
fig.text(0.5, 0.005,
         "Both figures are reported so the contribution of the matching rule is "
         "visible rather than absorbed into the headline score.",
         ha="center", fontsize=7.4, color=MUTED, style="italic")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "results_normalisation")

# ── 4. Latency ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.6))
meds = [statistics.median([r["elapsed_ms"] for r in rows if r["arm"] == a])
        for a in arms]
bars = ax.bar([PIPELINE_LABEL[a] for a in arms], meds,
              color=[ARM[a][0] for a in arms],
              edgecolor=[ARM[a][1] for a in arms], linewidth=1.5, zorder=3)
for bar, m in zip(bars, meds):
    ax.text(bar.get_x() + bar.get_width() / 2, m * 1.06, f"{m/1000:.1f}s",
            ha="center", va="bottom", fontsize=8.6, color=INK)
ax.set_yscale("log")
ax.set_ylabel("Median extraction time (ms, log scale)", fontsize=9.5, color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#c8ccd2")
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.grid(axis="y", color="#e8eaed", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_title("Extraction latency per diagram", fontsize=11.5,
             fontweight="bold", color=INK, pad=12)
fig.text(0.5, 0.005,
         "Measured on CPU. The arms run concurrently in the application, so "
         "wall-clock is the slowest arm, not the sum.",
         ha="center", fontsize=7.4, color=MUTED, style="italic")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "results_speed")

print(f"\n{len({r['diagram'] for r in rows})} diagrams · arms: {', '.join(arms)}")

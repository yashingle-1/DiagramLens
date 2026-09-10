from __future__ import annotations

import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "evaluation" / "results" / "offline_benchmark.json"
OUT = ROOT / "evaluation" / "results" / "result_stability.json"

SEED = 20260830
N_FOLDS = 5
N_BOOT = 10_000

ARMS = ["classical", "hybrid", "gemini"]
METRICS = [
    ("component_f1", "Component F1"),
    ("connection_undirected_f1", "Connection F1 (undirected)"),
    ("connection_f1", "Connection F1 (directed)"),
]


def stratified_folds(rows, n_folds, rng):

    by_notation = defaultdict(list)
    for r in rows:
        by_notation[r["notation"]].append(r["diagram"])

    folds = [[] for _ in range(n_folds)]
    cursor = 0
    for notation in sorted(by_notation):
        diagrams = sorted(by_notation[notation])
        rng.shuffle(diagrams)
        for d in diagrams:
            folds[cursor % n_folds].append(d)
            cursor += 1
    return folds


def bootstrap_ci(values, n_boot, rng, alpha=0.05):
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def main() -> int:
    rows = json.loads(BENCH.read_text(encoding="utf-8"))["rows"]
    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["diagram"]] = r

    rng = random.Random(SEED)
    folds = stratified_folds([r for r in rows if r["arm"] == "classical"],
                             N_FOLDS, rng)

    print(f"{len(folds)} folds, sizes {[len(f) for f in folds]}, seed {SEED}\n")

    report = {"seed": SEED, "n_folds": N_FOLDS, "n_bootstrap": N_BOOT,
              "fold_sizes": [len(f) for f in folds], "metrics": {}}

    for key, label in METRICS:
        print(f"=== {label} ===")
        print(f"{'arm':<11}{'mean':>8}{'fold min':>10}{'fold max':>10}"
              f"{'spread':>9}   95% CI")
        report["metrics"][key] = {}
        for arm in ARMS:
            per_diagram = by_arm[arm]
            allv = [per_diagram[d][key] for d in per_diagram]
            mean = sum(allv) / len(allv)

            fold_means = []
            for f in folds:
                vals = [per_diagram[d][key] for d in f if d in per_diagram]
                if vals:
                    fold_means.append(sum(vals) / len(vals))

            lo, hi = bootstrap_ci(allv, N_BOOT, random.Random(SEED))
            spread = max(fold_means) - min(fold_means)
            print(f"{arm:<11}{mean:>8.3f}{min(fold_means):>10.3f}"
                  f"{max(fold_means):>10.3f}{spread:>9.3f}   "
                  f"[{lo:.3f}, {hi:.3f}]")
            report["metrics"][key][arm] = {
                "mean": round(mean, 4),
                "fold_means": [round(m, 4) for m in fold_means],
                "fold_min": round(min(fold_means), 4),
                "fold_max": round(max(fold_means), 4),
                "fold_spread": round(spread, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
            }
        print()

    # Does the ordering of the three arms hold inside every fold?
    print("=== ordering check (component F1) ===")
    intact = 0
    for i, f in enumerate(folds):
        m = {}
        for arm in ARMS:
            vals = [by_arm[arm][d]["component_f1"] for d in f if d in by_arm[arm]]
            m[arm] = sum(vals) / len(vals)
        ok = m["gemini"] > m["hybrid"] > m["classical"]
        intact += ok
        print(f"fold {i + 1}: classical {m['classical']:.3f}  "
              f"hybrid {m['hybrid']:.3f}  gemini {m['gemini']:.3f}   "
              f"{'ordering holds' if ok else 'ORDERING BROKEN'}")
    print(f"\nordering holds in {intact}/{len(folds)} folds")
    report["ordering_holds_in_folds"] = f"{intact}/{len(folds)}"

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

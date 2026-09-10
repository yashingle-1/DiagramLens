"""
Scores every pipeline against every annotated diagram, without the API or DB.

Runs the pipelines directly so the benchmark is reproducible from a clean
checkout and does not depend on the state of a running server. Results are
written as JSON, which the figure scripts then read so the numbers in the
dissertation and the numbers in the charts cannot drift apart.

    python backend/scripts/run_offline_benchmark.py
    python backend/scripts/run_offline_benchmark.py --arms classical hybrid

Gemini is included by default and needs GEMINI_API_KEY in backend/.env.
Exclude it with --arms classical hybrid when working offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / "backend" / ".env")
except Exception:
    pass

from services.classical_pipeline import run_classical_pipeline   # noqa: E402
from services.hybrid_pipeline import run_hybrid_pipeline         # noqa: E402
from services.metrics import (raw_ratio, score_components,       # noqa: E402
                              score_connections)

GT_DIR  = ROOT / "evaluation" / "ground_truth"
OUT     = ROOT / "evaluation" / "results" / "offline_benchmark.json"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


async def run_gemini(image_bytes: bytes, session_id: str):
    """Imported lazily so the classical/hybrid arms still run without an API key."""
    from services.extraction_orchestrator import _run_gemini
    from services.cache import cache_service
    return await _run_gemini(image_bytes, session_id, "chain_of_thought",
                             cache_service.hash_image(image_bytes))


ARMS = {
    "classical": run_classical_pipeline,
    "hybrid":    run_hybrid_pipeline,
    "gemini":    run_gemini,
}


def connections_by_name(result) -> list[dict]:
    """Ground truth stores connections by NAME; pipelines emit component ids."""
    id_to_name = {c.id: c.name for c in result.components}
    return [
        {"source": c.source_name or id_to_name.get(c.source, c.source),
         "target": c.target_name or id_to_name.get(c.target, c.target)}
        for c in result.connections
    ]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS))
    args = parser.parse_args()

    pairs = []
    for annotation in sorted(GT_DIR.glob("*.json")):
        image = next((annotation.with_suffix(e) for e in IMAGE_EXTS
                      if annotation.with_suffix(e).exists()), None)
        if image:
            pairs.append((annotation, image))
        else:
            print(f"skip {annotation.name} — no image alongside it")

    print(f"{len(pairs)} annotated diagrams · arms: {', '.join(args.arms)}\n")
    rows = []

    for annotation, image_path in pairs:
        gt = json.loads(annotation.read_text(encoding="utf-8"))
        gt_names = [c["name"] for c in gt.get("components", [])]
        gt_conns = gt.get("connections", [])
        data = image_path.read_bytes()

        for arm in args.arms:
            started = time.time()
            try:
                result = await ARMS[arm](data, "offline-benchmark")
            except Exception as exc:
                print(f"  {annotation.stem:<40} {arm:<10} FAILED: {exc}")
                continue
            elapsed_ms = int((time.time() - started) * 1000)

            names = [c.name for c in result.components]
            comp  = score_components(names, gt_names)
            raw   = score_components(names, gt_names, ratio_fn=raw_ratio)
            conn  = score_connections(connections_by_name(result), gt_conns,
                                      names, gt_names)

            rows.append({
                "diagram":  annotation.stem,
                "notation": gt.get("diagram_standard", "unknown"),
                "complexity": gt.get("complexity", "unknown"),
                "arm": arm,
                "gt_components":  len(gt_names),
                "gt_connections": len(gt_conns),
                "n_components":   len(names),
                "component_f1":        comp["f1"],
                "component_f1_raw":    raw["f1"],
                "component_precision": comp["precision"],
                "component_recall":    comp["recall"],
                "connection_f1":            conn["f1"],
                "connection_undirected_f1": conn["undirected_f1"],
                "elapsed_ms": elapsed_ms,
                "missed":       comp["missed_names"],
                "hallucinated": comp["hallucinated_names"],
            })
            print(f"  {annotation.stem:<40} {arm:<10} "
                  f"F1={comp['f1']:.3f}  P={comp['precision']:.2f}  "
                  f"R={comp['recall']:.2f}  {elapsed_ms:>6}ms")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(rows)} rows)")

    print("\nmean by arm:")
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        mean = lambda k: sum(r[k] for r in sub) / len(sub)
        print(f"  {arm:<10} compF1={mean('component_f1'):.3f}  "
              f"raw={mean('component_f1_raw'):.3f}  "
              f"P={mean('component_precision'):.3f}  "
              f"R={mean('component_recall'):.3f}  "
              f"connF1={mean('connection_f1'):.3f}  "
              f"{mean('elapsed_ms'):.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

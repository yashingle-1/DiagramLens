"""
Benchmark router — scores all three pipelines against a ground truth JSON file.

Uses fuzzy matching (SequenceMatcher >= 0.75) — never exact string matching.
Stores one Benchmark row per pipeline: classical, hybrid, gemini.
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_db
from models.database import Architecture, Benchmark
from models.schemas import BenchmarkRequest
from services.metrics import score_components, score_connections

router = APIRouter()

GROUND_TRUTH_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "ground_truth"


def _load_ground_truth(diagram_id: str) -> dict:
    path = GROUND_TRUTH_DIR / f"{diagram_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/benchmark")
async def run_benchmark(
    request: BenchmarkRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Scores all three pipelines for a session against the specified ground truth file.

    Body: { session_id: str, diagram_id: str }
    Returns: { classical: BenchmarkResult, hybrid: BenchmarkResult, gemini: BenchmarkResult }
    """

    # ── Load ground truth ──────────────────────────────────
    try:
        gt = _load_ground_truth(request.diagram_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    gt_component_names = [c["name"] for c in gt.get("components", [])]
    gt_connections     = gt.get("connections", [])
    gt_standard        = gt.get("diagram_standard", "informal")
    gt_complexity      = gt.get("complexity", "low")

    # ── Load both pipeline architectures from DB ───────────
    arch_result = await db.execute(
        select(Architecture).where(Architecture.session_id == request.session_id)
    )
    architectures = arch_result.scalars().all()

    if not architectures:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

    # Index by pipeline
    by_pipeline: dict[str, Architecture] = {}
    for arch in architectures:
        if arch.pipeline:
            by_pipeline[arch.pipeline] = arch
        elif arch.llm_provider:
            by_pipeline[arch.llm_provider] = arch

    results: dict[str, dict] = {}

    for pipeline_name in ("classical", "hybrid", "gemini"):
        arch = by_pipeline.get(pipeline_name)
        if not arch:
            results[pipeline_name] = {"error": f"No {pipeline_name} result for this session"}
            continue

        raw = arch.raw_json or {}
        extracted_names = [c["name"] for c in raw.get("components", [])]
        extracted_conns = [
            {"source": c.get("source", ""), "target": c.get("target", "")}
            for c in raw.get("connections", [])
        ]

        comp_metrics = score_components(extracted_names, gt_component_names)
        conn_metrics = score_connections(
            extracted_conns, gt_connections,
            extracted_names, gt_component_names,
        )

        # Save to DB
        bench = Benchmark(
            id=str(uuid.uuid4()),
            session_id=request.session_id,
            llm_provider=pipeline_name,
            prompt_variant=arch.prompt_variant or "n/a",
            diagram_type=gt_standard,
            component_precision=comp_metrics["precision"],
            component_recall=comp_metrics["recall"],
            component_f1=comp_metrics["f1"],
            connection_precision=conn_metrics["precision"],
            connection_recall=conn_metrics["recall"],
            connection_f1=conn_metrics["f1"],
            hallucinated_components=comp_metrics.get("hallucinated_names", []),
            missed_components=comp_metrics.get("missed_names", []),
            response_time_ms=arch.response_time_ms,
            diagram_id=request.diagram_id,
            diagram_standard=gt_standard,
            complexity=gt_complexity,
            extracted_json=raw,
            ground_truth_json=gt,
        )
        db.add(bench)

        results[pipeline_name] = {
            "pipeline":           pipeline_name,
            "diagram_id":         request.diagram_id,
            "diagram_standard":   gt_standard,
            "complexity":         gt_complexity,
            "component_precision": comp_metrics["precision"],
            "component_recall":    comp_metrics["recall"],
            "component_f1":        comp_metrics["f1"],
            "connection_precision": conn_metrics["precision"],
            "connection_recall":    conn_metrics["recall"],
            "connection_f1":        conn_metrics["f1"],
            "hallucinated_components": comp_metrics.get("hallucinated_names", []),
            "missed_components":       comp_metrics.get("missed_names", []),
            "response_time_ms":        arch.response_time_ms,
        }

    await db.commit()

    return {
        "session_id": request.session_id,
        "diagram_id": request.diagram_id,
        "classical":  results.get("classical", {}),
        "hybrid":     results.get("hybrid", {}),
        "gemini":     results.get("gemini", {}),
    }


@router.get("/benchmark/results")
async def get_benchmark_results(db: AsyncSession = Depends(get_db)):
    """Returns all stored benchmark results ordered by recency."""
    result = await db.execute(
        select(Benchmark).order_by(Benchmark.created_at.desc())
    )
    benchmarks = result.scalars().all()

    return [
        {
            "id":                b.id,
            "pipeline":          b.llm_provider,
            "diagram_id":        b.diagram_id,
            "diagram_standard":  b.diagram_standard,
            "complexity":        b.complexity,
            "component_f1":      b.component_f1,
            "connection_f1":     b.connection_f1,
            "hallucinated":      b.hallucinated_components,
            "missed":            b.missed_components,
            "response_time_ms":  b.response_time_ms,
            "created_at":        str(b.created_at),
        }
        for b in benchmarks
    ]

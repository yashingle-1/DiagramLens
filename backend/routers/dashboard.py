"""
Dashboard router — aggregates all benchmark results for the dissertation charts.

GET /api/dashboard
Returns:
  - overall F1 by pipeline
  - F1 by complexity
  - F1 by diagram standard
  - hallucination table per diagram
  - speed comparison (classical vs gemini)
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from collections import defaultdict

from db.connection import get_db
from models.database import Benchmark

router = APIRouter()


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


@router.get("/dashboard")
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    """Aggregate benchmark data for the dissertation analysis dashboard."""

    result = await db.execute(select(Benchmark))
    benchmarks = result.scalars().all()

    if not benchmarks:
        return {
            "overall": {},
            "by_complexity": {},
            "by_standard": {},
            "hallucination_table": [],
            "speed_comparison": {},
            "total_runs": 0,
        }

    # ── Overall F1 by pipeline ─────────────────────────────
    pipeline_comp_f1:   dict[str, list[float]] = defaultdict(list)
    pipeline_conn_f1:   dict[str, list[float]] = defaultdict(list)
    pipeline_speed:     dict[str, list[int]]   = defaultdict(list)

    # ── F1 by complexity ───────────────────────────────────
    complexity_comp_f1: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    complexity_conn_f1: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # ── F1 by diagram standard ─────────────────────────────
    standard_comp_f1:   dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    standard_conn_f1:   dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # ── Per-diagram hallucination ──────────────────────────
    hallucination_rows: list[dict] = []

    for b in benchmarks:
        pipeline   = b.llm_provider or "unknown"
        complexity = b.complexity or "unknown"
        standard   = b.diagram_standard or "unknown"

        if b.component_f1 is not None:
            pipeline_comp_f1[pipeline].append(b.component_f1)
            complexity_comp_f1[complexity][pipeline].append(b.component_f1)
            standard_comp_f1[standard][pipeline].append(b.component_f1)

        if b.connection_f1 is not None:
            pipeline_conn_f1[pipeline].append(b.connection_f1)
            complexity_conn_f1[complexity][pipeline].append(b.connection_f1)
            standard_conn_f1[standard][pipeline].append(b.connection_f1)

        if b.response_time_ms is not None:
            pipeline_speed[pipeline].append(b.response_time_ms)

        # Hallucination rows (gemini only — that's where hallucinations happen)
        if pipeline == "gemini" and b.diagram_id:
            hallucinated = b.hallucinated_components or []
            total_extracted = (b.extracted_json or {})
            total_components = len(total_extracted.get("components", [])) if total_extracted else 0
            rate = len(hallucinated) / total_components if total_components > 0 else 0.0
            hallucination_rows.append({
                "diagram_id":       b.diagram_id,
                "diagram_standard": standard,
                "complexity":       complexity,
                "hallucinated":     hallucinated,
                "hallucination_rate": round(rate, 4),
                "component_f1":     b.component_f1,
            })

    # ── Build response ─────────────────────────────────────
    overall: dict[str, dict] = {}
    for pipeline in set(list(pipeline_comp_f1.keys()) + list(pipeline_conn_f1.keys())):
        overall[pipeline] = {
            "avg_component_f1":  _avg(pipeline_comp_f1.get(pipeline, [])),
            "avg_connection_f1": _avg(pipeline_conn_f1.get(pipeline, [])),
            "run_count":         len(pipeline_comp_f1.get(pipeline, [])),
        }

    by_complexity: dict[str, dict] = {}
    for cx in complexity_comp_f1:
        by_complexity[cx] = {
            pipeline: {
                "avg_component_f1":  _avg(complexity_comp_f1[cx][pipeline]),
                "avg_connection_f1": _avg(complexity_conn_f1[cx].get(pipeline, [])),
            }
            for pipeline in complexity_comp_f1[cx]
        }

    by_standard: dict[str, dict] = {}
    for std in standard_comp_f1:
        by_standard[std] = {
            pipeline: {
                "avg_component_f1":  _avg(standard_comp_f1[std][pipeline]),
                "avg_connection_f1": _avg(standard_conn_f1[std].get(pipeline, [])),
            }
            for pipeline in standard_comp_f1[std]
        }

    speed_comparison: dict[str, dict] = {
        pipeline: {
            "avg_response_time_ms": _avg(times),
            "min_ms":               min(times) if times else None,
            "max_ms":               max(times) if times else None,
        }
        for pipeline, times in pipeline_speed.items()
    }

    return {
        "overall":             overall,
        "by_complexity":       by_complexity,
        "by_standard":         by_standard,
        "hallucination_table": sorted(hallucination_rows, key=lambda r: r["diagram_id"]),
        "speed_comparison":    speed_comparison,
        "total_runs":          len(benchmarks),
    }

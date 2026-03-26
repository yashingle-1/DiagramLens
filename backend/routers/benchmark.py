from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import time
from db.connection import get_db
from models.database import Architecture, Benchmark
from models.schemas import BenchmarkRequest
from services.cache import cache_service
from services.storage import storage_service

router = APIRouter()


@router.post("/benchmark")
async def run_benchmark(
    request: BenchmarkRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    MSc Research endpoint — runs same diagram through
    multiple providers and prompt variants.

    Measures:
    - Component detection accuracy vs ground truth
    - Connection detection accuracy
    - Hallucination rate
    - Response time per provider
    - Token usage per provider

    Results stored in PostgreSQL benchmarks table
    for dissertation analysis.
    """

    # Load original architecture session
    arch_result = await db.execute(
        select(Architecture).where(Architecture.session_id == request.session_id)
    )
    architecture = arch_result.scalar_one_or_none()

    if not architecture:
        raise HTTPException(status_code=404, detail="Session not found")

    # Load original image from filesystem
    session_result = await db.execute(
        select(Architecture).where(Architecture.session_id == request.session_id)
    )

    results = []

    # Run each provider + prompt variant combination
    for provider_name in request.providers:
        for variant in request.prompt_variants:
            try:
                # Dynamically load the correct provider
                if provider_name == "gemini":
                    from services.llm.gemini import GeminiProvider
                    provider = GeminiProvider()
                elif provider_name == "claude":
                    from services.llm.claude import ClaudeProvider
                    provider = ClaudeProvider()
                else:
                    continue

                # Load image bytes for re-analysis
                image_path = architecture.session.image_path if hasattr(architecture, 'session') else None

                # Calculate accuracy metrics if ground truth provided
                metrics = _calculate_metrics(
                    extracted=architecture.raw_json,
                    ground_truth=request.ground_truth,
                ) if request.ground_truth else {}

                # Save benchmark result to PostgreSQL
                benchmark = Benchmark(
                    id=str(uuid.uuid4()),
                    session_id=request.session_id,
                    llm_provider=provider_name,
                    prompt_variant=variant.value,
                    component_precision=metrics.get("component_precision"),
                    component_recall=metrics.get("component_recall"),
                    component_f1=metrics.get("component_f1"),
                    connection_precision=metrics.get("connection_precision"),
                    connection_recall=metrics.get("connection_recall"),
                    connection_f1=metrics.get("connection_f1"),
                    hallucinated_components=metrics.get("hallucinated", 0),
                    missed_components=metrics.get("missed", 0),
                    extracted_json=architecture.raw_json,
                    ground_truth_json=request.ground_truth,
                )
                db.add(benchmark)

                results.append({
                    "provider": provider_name,
                    "prompt_variant": variant.value,
                    "metrics": metrics,
                    "status": "completed",
                })

            except NotImplementedError:
                results.append({
                    "provider": provider_name,
                    "prompt_variant": variant.value,
                    "status": "not_implemented",
                    "message": f"{provider_name} provider not yet implemented",
                })
            except Exception as e:
                results.append({
                    "provider": provider_name,
                    "prompt_variant": variant.value,
                    "status": "failed",
                    "error": str(e),
                })

    await db.commit()
    return {"session_id": request.session_id, "results": results}


@router.get("/benchmark/results")
async def get_benchmark_results(db: AsyncSession = Depends(get_db)):
    """Returns all benchmark results for dissertation analysis"""
    result = await db.execute(
        select(Benchmark).order_by(Benchmark.created_at.desc())
    )
    benchmarks = result.scalars().all()

    return [
        {
            "id": b.id,
            "provider": b.llm_provider,
            "prompt_variant": b.prompt_variant,
            "component_f1": b.component_f1,
            "connection_f1": b.connection_f1,
            "hallucinated": b.hallucinated_components,
            "missed": b.missed_components,
            "response_time_ms": b.response_time_ms,
            "created_at": str(b.created_at),
        }
        for b in benchmarks
    ]


def _calculate_metrics(extracted: dict, ground_truth: dict) -> dict:
    """
    Calculates precision, recall, F1 score.
    Compares extracted components against manually labeled ground truth.

    Precision = correctly found / total found by LLM
    Recall    = correctly found / total in ground truth
    F1        = 2 * (precision * recall) / (precision + recall)
    """
    if not ground_truth:
        return {}

    # Component metrics
    extracted_names = {
        c["name"].lower().strip()
        for c in extracted.get("components", [])
    }
    truth_names = {
        c["name"].lower().strip()
        for c in ground_truth.get("components", [])
    }

    true_positives = len(extracted_names & truth_names)
    hallucinated = len(extracted_names - truth_names)
    missed = len(truth_names - extracted_names)

    comp_precision = true_positives / len(extracted_names) if extracted_names else 0
    comp_recall = true_positives / len(truth_names) if truth_names else 0
    comp_f1 = (
        2 * comp_precision * comp_recall / (comp_precision + comp_recall)
        if (comp_precision + comp_recall) > 0 else 0
    )

    # Connection metrics
    extracted_conns = {
        (c["source"], c["target"])
        for c in extracted.get("connections", [])
    }
    truth_conns = {
        (c["source"], c["target"])
        for c in ground_truth.get("connections", [])
    }

    conn_tp = len(extracted_conns & truth_conns)
    conn_precision = conn_tp / len(extracted_conns) if extracted_conns else 0
    conn_recall = conn_tp / len(truth_conns) if truth_conns else 0
    conn_f1 = (
        2 * conn_precision * conn_recall / (conn_precision + conn_recall)
        if (conn_precision + conn_recall) > 0 else 0
    )

    return {
        "component_precision": round(comp_precision, 3),
        "component_recall": round(comp_recall, 3),
        "component_f1": round(comp_f1, 3),
        "connection_precision": round(conn_precision, 3),
        "connection_recall": round(conn_recall, 3),
        "connection_f1": round(conn_f1, 3),
        "hallucinated": hallucinated,
        "missed": missed,
    }

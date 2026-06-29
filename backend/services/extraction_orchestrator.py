"""
Extraction orchestrator — runs classical CV, hybrid ML (SAM+CLIP+TrOCR), and
Gemini pipelines in parallel, applies the hallucination filter to Gemini
output, and saves all three results to the database.
"""

import asyncio
import re
import time
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Session as SessionModel, Architecture
from models.schemas import ArchitectureSchema, ComponentSchema, ConnectionSchema
from services.cache import cache_service
from services.classical_pipeline import run_classical_pipeline
from services.hallucination_filter import run_hallucination_filter
from services.hybrid_pipeline import run_hybrid_pipeline
from services.llm.factory import get_llm_provider


# ── Diagram standard keywords (same as classical_pipeline, kept local) ─────
_AWS_KEYWORDS = {"aws", "ec2", "s3", "lambda", "cloudfront", "rds", "sqs", "sns",
                 "eks", "ecs", "fargate", "dynamodb", "elasticache", "alb", "elb",
                 "cloudwatch", "route53", "kinesis", "api gateway"}
_C4_KEYWORDS  = {"person", "system", "container", "component", "bounded context", "c4"}
_UML_KEYWORDS = {"actor", "interface", "class", "abstract", "package", "module"}


def _infer_diagram_standard(names: list[str]) -> str:
    all_text = " ".join(names).lower()
    words = set(re.split(r"[\s,./]+", all_text))
    if words & _AWS_KEYWORDS:
        return "aws"
    if words & _C4_KEYWORDS:
        return "c4"
    if words & _UML_KEYWORDS:
        return "uml"
    return "informal"


def _complexity(n: int) -> str:
    if n < 8:
        return "low"
    if n <= 14:
        return "medium"
    return "high"


async def _run_gemini(
    image_bytes: bytes,
    session_id: str,
    prompt_variant: str,
    image_hash: str,
) -> ArchitectureSchema:
    """Calls Gemini and maps the result to the canonical ArchitectureSchema."""
    provider = get_llm_provider()
    provider_name = provider.get_provider_name()

    # Check cache first
    cached = await cache_service.get_architecture(image_hash, provider_name, prompt_variant)
    if cached:
        try:
            return ArchitectureSchema(**cached)
        except Exception:
            pass  # Cache has old schema shape — fall through to live call

    result = await provider.analyze_image(image_bytes, prompt_variant)
    if not result.parsed_json:
        raise ValueError(f"Gemini returned invalid JSON: {result.raw_text[:200]}")

    raw: dict = result.parsed_json
    response_ms = int(result.response_time) if result.response_time else 0

    # Map old Gemini JSON to the canonical schema
    components: list[ComponentSchema] = []
    for i, c in enumerate(raw.get("components", [])):
        # Normalise type — strip unknown values to "other"
        raw_type = str(c.get("type", "other")).lower().replace("-", "_")
        valid_types = {
            "service", "database", "gateway", "queue", "cache", "cdn",
            "load_balancer", "client", "storage", "monitoring", "notification", "other"
        }
        comp_type = raw_type if raw_type in valid_types else "other"

        pos_data  = c.get("position")
        meta_data = c.get("metadata")

        components.append(ComponentSchema(
            id=c.get("id", f"c{i + 1}"),
            name=c.get("name", f"Component {i + 1}"),
            type=comp_type,
            confidence=raw.get("confidence_score"),
            technology=c.get("technology"),
            position=pos_data,
            metadata=meta_data,
        ))

    connections: list[ConnectionSchema] = []
    for i, conn in enumerate(raw.get("connections", [])):
        connections.append(ConnectionSchema(
            id=conn.get("id", f"e{i + 1}"),
            source=conn.get("source", ""),
            target=conn.get("target", ""),
            label=conn.get("label") or "",
            directed=True,
            direction=conn.get("direction"),
            protocol=conn.get("protocol"),
            data_type=conn.get("data_type"),
        ))

    names           = [c.name for c in components]
    diagram_std     = _infer_diagram_standard(names)
    arch_type       = raw.get("arch_type") or "other"

    schema = ArchitectureSchema(
        session_id=session_id,
        pipeline="gemini",
        diagram_standard=diagram_std,
        complexity=_complexity(len(components)),
        arch_type=arch_type,
        components=components,
        connections=connections,
        response_time_ms=response_ms,
        confidence_score=raw.get("confidence_score"),
    )

    # Cache for 24 hours
    await cache_service.set_architecture(
        image_hash, provider_name, prompt_variant, schema.model_dump()
    )

    return schema


async def run_extraction(
    image_bytes: bytes,
    image_path: str,
    image_url: str,
    original_filename: str,
    db: AsyncSession,
    prompt_variant: str = "chain_of_thought",
) -> dict:
    """
    Main entry point. Runs all three pipelines concurrently.

    Returns:
        {
            "session_id": str,
            "classical":  ArchitectureSchema,
            "hybrid":     ArchitectureSchema,
            "gemini":     ArchitectureSchema,
            "hallucination_filter": dict,
        }
    """
    session_id = str(uuid.uuid4())
    image_hash = cache_service.hash_image(image_bytes)

    # ── Run all three pipelines in parallel ────────────────
    classical_result, hybrid_result, gemini_result = await asyncio.gather(
        run_classical_pipeline(image_bytes, session_id),
        run_hybrid_pipeline(image_bytes, session_id),
        _run_gemini(image_bytes, session_id, prompt_variant, image_hash),
        return_exceptions=True,
    )

    # Handle pipeline failures gracefully
    if isinstance(classical_result, Exception):
        print(f"[orchestrator] classical pipeline failed: {classical_result}")
        classical_result = ArchitectureSchema(
            session_id=session_id,
            pipeline="classical",
            diagram_standard="informal",
            complexity="low",
            arch_type="other",
            components=[ComponentSchema(id="c1", name="Unknown", type="other", confidence=None)],
            connections=[],
            response_time_ms=0,
        )

    if isinstance(hybrid_result, Exception):
        print(f"[orchestrator] hybrid pipeline failed: {hybrid_result}")
        hybrid_result = ArchitectureSchema(
            session_id=session_id,
            pipeline="hybrid",
            diagram_standard="informal",
            complexity="low",
            arch_type="other",
            components=[ComponentSchema(id="h1", name="Unknown", type="other", confidence=None)],
            connections=[],
            response_time_ms=0,
        )

    if isinstance(gemini_result, Exception):
        print(f"[orchestrator] gemini pipeline failed: {gemini_result}")
        gemini_result = ArchitectureSchema(
            session_id=session_id,
            pipeline="gemini",
            diagram_standard="informal",
            complexity="low",
            arch_type="other",
            components=[ComponentSchema(id="g1", name="Unknown", type="other", confidence=None)],
            connections=[],
            response_time_ms=0,
        )

    # ── Run hallucination filter on Gemini output ─────────
    hallucination_info = run_hallucination_filter(gemini_result, image_bytes)
    # Attach to the gemini schema so it persists and reaches the frontend
    gemini_result.hallucinated_components = hallucination_info["hallucinated"]
    gemini_result.hallucination_rate      = hallucination_info["hallucination_rate"]

    # ── Persist both results to DB ─────────────────────────
    session_record = SessionModel(
        id=session_id,
        image_path=image_path,
        image_url=image_url,
        image_hash=image_hash,
        original_filename=original_filename,
        status="done",
    )
    db.add(session_record)

    for schema in (classical_result, hybrid_result, gemini_result):
        arch_record = Architecture(
            id=str(uuid.uuid4()),
            session_id=session_id,
            raw_json=schema.model_dump(),
            component_count=len(schema.components),
            connection_count=len(schema.connections),
            arch_type=schema.arch_type,
            confidence_score=schema.confidence_score,
            llm_provider=schema.pipeline,
            prompt_variant=prompt_variant if schema.pipeline == "gemini" else f"{schema.pipeline}_pipeline",
            pipeline=schema.pipeline,
            diagram_standard=schema.diagram_standard,
            complexity=schema.complexity,
            response_time_ms=schema.response_time_ms,
        )
        db.add(arch_record)

    await db.commit()

    return {
        "session_id":           session_id,
        "classical":            classical_result,
        "hybrid":               hybrid_result,
        "gemini":               gemini_result,
        "hallucination_filter": hallucination_info,
    }

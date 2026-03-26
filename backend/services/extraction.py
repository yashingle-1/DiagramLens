import uuid
import json
from sqlalchemy.ext.asyncio import AsyncSession
from models.schemas import ArchitectureSchema
from models.database import Session as SessionModel, Architecture
from services.cache import cache_service
from services.llm.factory import get_llm_provider


class ExtractionService:

    async def extract(
        self,
        image_bytes: bytes,
        image_path: str,
        image_url: str,
        original_filename: str,
        db: AsyncSession,
        prompt_variant: str = "zero_shot",
    ) -> tuple[str, ArchitectureSchema, bool]:
        """
        Full extraction pipeline:
        1. Hash image → check Redis cache
        2. If cache hit → return cached result instantly
        3. If cache miss → call Gemini Vision
        4. Validate JSON with Pydantic
        5. Save to PostgreSQL
        6. Store in Redis cache
        7. Return session_id, architecture, was_cached

        Returns: (session_id, architecture, cached)
        """

        provider = get_llm_provider()
        image_hash = cache_service.hash_image(image_bytes)
        provider_name = provider.get_provider_name()

        # ── Step 1: Check Redis Cache ─────────────────────
        cached_data = await cache_service.get_architecture(
            image_hash, provider_name, prompt_variant
        )

        if cached_data:
            # Cache hit — validate and return immediately
            architecture = ArchitectureSchema(**cached_data)
            session_id = await self._save_session(
                db, image_path, image_url, image_hash,
                original_filename, architecture,
                provider_name, prompt_variant, from_cache=True
            )
            return session_id, architecture, True

        # ── Step 2: Call Gemini Vision ────────────────────
        result = await provider.analyze_image(image_bytes, prompt_variant)

        if not result.parsed_json:
            raise ValueError(
                f"LLM returned invalid JSON after retries. "
                f"Raw response: {result.raw_text[:200]}"
            )

        # ── Step 3: Validate with Pydantic ────────────────
        # This will raise ValidationError if JSON doesn't match schema
        architecture = ArchitectureSchema(**result.parsed_json)

        # ── Step 4: Save to Redis Cache ───────────────────
        await cache_service.set_architecture(
            image_hash,
            provider_name,
            prompt_variant,
            result.parsed_json,
        )

        # ── Step 5: Save to PostgreSQL ────────────────────
        session_id = await self._save_session(
            db, image_path, image_url, image_hash,
            original_filename, architecture,
            provider_name, prompt_variant,
            from_cache=False,
            tokens_used=result.tokens_used,
            response_time=result.response_time,
        )

        return session_id, architecture, False

    async def _save_session(
        self,
        db: AsyncSession,
        image_path: str,
        image_url: str,
        image_hash: str,
        original_filename: str,
        architecture: ArchitectureSchema,
        provider_name: str,
        prompt_variant: str,
        from_cache: bool = False,
        tokens_used: int = None,
        response_time: float = None,
    ) -> str:
        """Saves session and architecture to PostgreSQL"""

        session_id = str(uuid.uuid4())
        arch_id = str(uuid.uuid4())

        # Create session record
        session = SessionModel(
            id=session_id,
            image_path=image_path,
            image_url=image_url,
            image_hash=image_hash,
            original_filename=original_filename,
            status="done",
        )
        db.add(session)

        # Create architecture record
        arch_data = architecture.model_dump()
        arch_record = Architecture(
            id=arch_id,
            session_id=session_id,
            raw_json=arch_data,
            component_count=len(architecture.components),
            connection_count=len(architecture.connections),
            arch_type=architecture.arch_type,
            confidence_score=architecture.confidence_score,
            llm_provider=provider_name,
            prompt_variant=prompt_variant,
        )
        db.add(arch_record)
        await db.commit()

        return session_id


# Singleton instance
extraction_service = ExtractionService()

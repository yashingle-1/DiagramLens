import redis.asyncio as redis
import json
import hashlib
from typing import Optional
from config import settings

# TTL = how long cached results live in Redis
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


class CacheService:
    def __init__(self):
        self.client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    # ── Image Hashing ─────────────────────────────────────
    # Converts image bytes into a unique fingerprint string
    # Same image = same hash = cache hit
    def hash_image(self, image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    # ── Cache Key Builder ─────────────────────────────────
    # Includes provider and prompt variant so different
    # LLM providers don't share cached results
    def build_key(self, image_hash: str, provider: str, prompt_variant: str) -> str:
        return f"arch:{image_hash}:{provider}:{prompt_variant}"

    # ── Get From Cache ────────────────────────────────────
    async def get_architecture(
        self,
        image_hash: str,
        provider: str,
        prompt_variant: str,
    ) -> Optional[dict]:
        key = self.build_key(image_hash, provider, prompt_variant)
        cached = await self.client.get(key)
        if cached:
            return json.loads(cached)
        return None

    # ── Set In Cache ──────────────────────────────────────
    async def set_architecture(
        self,
        image_hash: str,
        provider: str,
        prompt_variant: str,
        architecture: dict,
    ) -> None:
        key = self.build_key(image_hash, provider, prompt_variant)
        await self.client.setex(
            key,
            CACHE_TTL_SECONDS,
            json.dumps(architecture),
        )

    # ── Cache Chat Context ────────────────────────────────
    # Stores conversation history per session
    async def get_chat_history(self, session_id: str) -> list:
        cached = await self.client.get(f"chat:{session_id}")
        if cached:
            return json.loads(cached)
        return []

    async def set_chat_history(self, session_id: str, history: list) -> None:
        await self.client.setex(
            f"chat:{session_id}",
            CACHE_TTL_SECONDS,
            json.dumps(history),
        )

    # ── Health Check ──────────────────────────────────────
    async def ping(self) -> bool:
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    # ── Cleanup ───────────────────────────────────────────
    async def close(self):
        await self.client.aclose()


# Singleton instance — imported everywhere
cache_service = CacheService()

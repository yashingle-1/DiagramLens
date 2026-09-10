"""
Gemini provider uses the google-genai SDK (google.generativeai is EOL).

Robustness notes (these caused real "zero components" bugs):
- response.text raises when the model returns no text part (e.g. finish_reason
  MAX_TOKENS with only thinking tokens). We read candidate parts manually.
- Large diagrams can truncate JSON mid-object at max_output_tokens. The parser
  salvages truncated JSON by trimming to the last complete object and closing
  open brackets, instead of returning None.
- Extraction retries up to 3 times with backoff on transport errors AND on
  unparseable output (except quota errors, which fail fast).
"""

import io
import json
import re
import asyncio
import time
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types

from config import settings
from models.schemas import GeminiExtraction
from services.llm.base import LLMProvider, LLMResult
from services.llm.prompts import (
    EXTRACTION_PROMPTS,
    EXTRACTION_PROMPTS_V2,
    CHAT_SYSTEM_PROMPT,
    CHAT_INTERVIEW_PROMPT,
    COMPONENT_EXPLAIN_PROMPT,
)

MODEL_ID = "gemini-2.5-flash"
MAX_ATTEMPTS = 3

_MIME_BY_FORMAT = {
    "jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png",
    "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp",
}


def _detect_mime(image_bytes: bytes) -> str:
    """Declaring a JPEG as image/png misrepresents the upload to the API."""
    try:
        fmt = (Image.open(io.BytesIO(image_bytes)).format or "").lower()
        return _MIME_BY_FORMAT.get(fmt, "image/png")
    except Exception:
        return "image/png"


class GeminiProvider(LLMProvider):

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.salvage_count = 0
        self.extraction_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=settings.gemini_max_output_tokens,
            response_mime_type="application/json",
            response_schema=GeminiExtraction,
            thinking_config=types.ThinkingConfig(
                thinking_budget=settings.gemini_thinking_budget
            ),
        )

        self.explain_config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=settings.gemini_max_output_tokens,
            response_mime_type="application/json",
        )

        self.legacy_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=32768,
        )

    def get_provider_name(self) -> str:
        return "gemini"

    # ── Safe text extraction ──────────────────────────────
    # response.text raises "requires the response to contain a valid Part"
    # when the candidate has no text parts. Read parts manually instead.
    @staticmethod
    def _extract_text(response) -> tuple[str, Optional[str]]:
        """Returns (text, finish_reason). Never raises."""
        finish_reason = None
        if not response.candidates:
            return "", None
        candidate = response.candidates[0]
        if candidate.finish_reason is not None:
            finish_reason = str(candidate.finish_reason)
        if candidate.content is None or not candidate.content.parts:
            return "", finish_reason
        text = "".join(p.text for p in candidate.content.parts if p.text)
        return text, finish_reason

    # ── Image Analysis ────────────────────────────────────
    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt_variant: str = "zero_shot",
    ) -> LLMResult:

        start_time = time.time()
        legacy  = settings.prompt_version == "v1"
        prompts = EXTRACTION_PROMPTS if legacy else EXTRACTION_PROMPTS_V2
        prompt  = prompts.get(prompt_variant, prompts["zero_shot"])
        config  = self.legacy_config if legacy else self.extraction_config
        mime    = _detect_mime(image_bytes)

        last_error: Optional[str] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self.client.aio.models.generate_content(
                    model=MODEL_ID,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        prompt,
                    ],
                    config=config,
                )
            except Exception as e:
                error_str = str(e).lower()
                # Quota / rate-limit errors: fail fast, retrying wastes the quota window
                if "quota" in error_str or "exhausted" in error_str or "429" in error_str:
                    raise Exception(f"Gemini quota exceeded. Please wait and try again: {e}")
                last_error = repr(e)
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            raw_text, finish_reason = self._extract_text(response)
            parsed = self._parse_json(raw_text)

            if parsed is not None:
                tokens = None
                try:
                    tokens = response.usage_metadata.total_token_count
                except Exception:
                    pass
                return LLMResult(
                    raw_text=raw_text,
                    parsed_json=parsed,
                    tokens_used=tokens,
                    response_time=(time.time() - start_time) * 1000,
                    provider=self.get_provider_name(),
                    prompt_variant=prompt_variant,
                )

            # Unparseable output log and retry (model output is stochastic)
            last_error = (
                f"unparseable output (finish_reason={finish_reason}, "
                f"text_len={len(raw_text)}): {raw_text[:200]!r}"
            )
            print(f"[gemini] attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {last_error}")
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt)

        raise Exception(f"Gemini extraction failed after {MAX_ATTEMPTS} attempts: {last_error}")

    # ── Chat ──────────────────────────────────────────────
    async def chat(
        self,
        message: str,
        architecture_context: dict,
        conversation_history: list,
        interview_mode: bool = False,
        focus_component: dict | None = None,
    ) -> str:

        system = CHAT_INTERVIEW_PROMPT if interview_mode else CHAT_SYSTEM_PROMPT

        # A selected component resolves deictic questions "what does this do?"
        # has no referent otherwise, and the model can only ask which one.
        focus_block = ""
        if focus_component:
            focus_block = f"""
SELECTED COMPONENT the user has this one selected on the canvas. Unless they
name a different component, "this", "it" and "this component" all refer to it.
Answer about this component specifically; do not ask which one they mean.
{json.dumps(focus_component, indent=2)}
"""

        full_prompt = f"""
{system}

ARCHITECTURE CONTEXT:
{json.dumps(architecture_context, indent=2)}
{focus_block}
CONVERSATION HISTORY:
{self._format_history(conversation_history)}

USER MESSAGE: {message}

Respond helpfully based on the architecture context above.
"""

        response = await self.client.aio.models.generate_content(
            model=MODEL_ID,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,        # higher temp for more natural chat
                max_output_tokens=2048,
            ),
        )
        text, _ = self._extract_text(response)
        return text

    # ── Component Explanation ─────────────────────────────
    async def explain_component(
        self,
        component: dict,
        full_architecture: dict,
    ) -> dict:

        prompt = f"""
{COMPONENT_EXPLAIN_PROMPT}

FULL ARCHITECTURE CONTEXT:
{json.dumps(full_architecture, indent=2)}

COMPONENT TO ANALYZE:
{json.dumps(component, indent=2)}
"""

        response = await self.client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=self.explain_config,
        )
        text, _ = self._extract_text(response)
        return self._parse_json(text)

    # ── JSON Parser ───────────────────────────────────────
    # LLMs wrap JSON in markdown fences and sometimes truncate mid-object.
    def _parse_json(self, text: str) -> Optional[dict]:
        if not text:
            return None

        cleaned = text.strip()

        # Remove markdown fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # 1. Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Outermost braces (prose before/after the JSON)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

        # 3. Salvage truncated JSON (output hit max_output_tokens mid-object)
        if start != -1:
            salvaged = self._salvage_truncated_json(cleaned[start:])
            if salvaged is not None:
                # Salvage trims to the last complete object, so components are
                # lost here. Counted so the rate is reportable, not invisible.
                self.salvage_count += 1
                print(f"[gemini] salvaged truncated JSON output "
                      f"(salvage #{self.salvage_count} components may be missing)")
                return salvaged

        return None

    @staticmethod
    def _salvage_truncated_json(fragment: str) -> Optional[dict]:
        """Repairs JSON cut off mid-stream: trims back to the last complete
        object/array boundary, then closes any still-open brackets.
        Best-effort returns None if nothing parseable can be recovered."""
        # Single forward pass: record bracket nesting at every "}" / "]" that
        # sits outside a string. Each such index is a candidate cut point.
        stack: list[str] = []
        in_string = False
        escape = False
        # (index_after_closer, snapshot of open brackets at that point)
        cut_points: list[tuple[int, tuple[str, ...]]] = []
        for i, ch in enumerate(fragment):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = in_string
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack or {"}": "{", "]": "["}[ch] != stack[-1]:
                    break  # malformed beyond repair from here on
                stack.pop()
                cut_points.append((i + 1, tuple(stack)))

        # Try the latest complete boundaries first (keeps the most data).
        for cut, open_at_cut in reversed(cut_points[-50:]):
            chunk = fragment[:cut].rstrip().rstrip(",")
            repaired = chunk + "".join(
                "}" if o == "{" else "]" for o in reversed(open_at_cut)
            )
            try:
                result = json.loads(repaired)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
        return None

    def _format_history(self, history: list) -> str:
        if not history:
            return "No previous messages."
        formatted = []
        for msg in history[-10:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

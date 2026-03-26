import google.generativeai as genai
import json
import time
import re
from typing import Optional
from config import settings
from services.llm.base import LLMProvider, LLMResult
from services.llm.prompts import (
    EXTRACTION_PROMPTS,
    CHAT_SYSTEM_PROMPT,
    CHAT_INTERVIEW_PROMPT,
    COMPONENT_EXPLAIN_PROMPT,
)


class GeminiProvider(LLMProvider):

    def __init__(self):
        # Configure Gemini with API key from settings
        genai.configure(api_key=settings.gemini_api_key)

        # gemini-1.5-flash is free tier and fast
        # good balance of speed and accuracy for extraction
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

        # Generation config — low temperature means more consistent
        # structured output (less creative, more reliable JSON)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,        # low = more consistent output
            max_output_tokens=4096, # enough for complex architectures
        )

    def get_provider_name(self) -> str:
        return "gemini"

    # ── Image Analysis ────────────────────────────────────
    # Core method — sends image to Gemini, gets architecture JSON back
    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt_variant: str = "zero_shot",
    ) -> LLMResult:

        start_time = time.time()

        # Get the correct prompt for this variant
        prompt = EXTRACTION_PROMPTS.get(
            prompt_variant,
            EXTRACTION_PROMPTS["zero_shot"]
        )

        # Try up to 3 times — Gemini sometimes returns malformed JSON
        last_error = None
        for attempt in range(1):
            try:
                # Build the message with image and prompt
                response = self.model.generate_content(
                    contents=[
                        {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": image_bytes,
                                    }
                                },
                                {"text": prompt}
                            ]
                        }
                    ],
                    generation_config=self.generation_config,
                )

                raw_text = response.text
                response_time = (time.time() - start_time) * 1000

                # Parse the JSON from the response
                parsed = self._parse_json(raw_text)

                # Count tokens if available
                tokens = None
                try:
                    tokens = response.usage_metadata.total_token_count
                except Exception:
                    pass

                return LLMResult(
                    raw_text=raw_text,
                    parsed_json=parsed,
                    tokens_used=tokens,
                    response_time=response_time,
                    provider=self.get_provider_name(),
                    prompt_variant=prompt_variant,
                )

            except Exception as e:
                last_error = e
                # Don't retry on rate limit or quota errors
                error_str = str(e).lower()
                if "quota" in error_str or "exhausted" in error_str or "429" in error_str:
                    raise Exception(f"Gemini quota exceeded. Please wait and try again: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue

        # All 3 attempts failed
        raise Exception(f"Gemini extraction failed after 3 attempts: {last_error}")

    # ── Chat ──────────────────────────────────────────────
    # Sends message with full architecture context to Gemini
    async def chat(
        self,
        message: str,
        architecture_context: dict,
        conversation_history: list,
        interview_mode: bool = False,
    ) -> str:

        # Pick system prompt based on mode
        system = CHAT_INTERVIEW_PROMPT if interview_mode else CHAT_SYSTEM_PROMPT

        # Build the full prompt with architecture context embedded
        full_prompt = f"""
{system}

ARCHITECTURE CONTEXT:
{json.dumps(architecture_context, indent=2)}

CONVERSATION HISTORY:
{self._format_history(conversation_history)}

USER MESSAGE: {message}

Respond helpfully based on the architecture context above.
"""

        response = self.model.generate_content(
            contents=full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,        # higher temp for more natural chat
                max_output_tokens=2048,
            ),
        )

        return response.text

    # ── Component Explanation ─────────────────────────────
    # Deep analysis of a single component within its architecture context
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

        response = self.model.generate_content(
            contents=prompt,
            generation_config=self.generation_config,
        )

        return self._parse_json(response.text)

    # ── JSON Parser ───────────────────────────────────────
    # Cleans and parses JSON from LLM response
    # LLMs sometimes wrap JSON in markdown code blocks
    def _parse_json(self, text: str) -> Optional[dict]:
        if not text:
            return None

        # Remove markdown code blocks if present
        # e.g. ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.replace("```", "").strip()

        # Try to find JSON object in the response
        # Sometimes LLMs add text before or after the JSON
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    # ── History Formatter ─────────────────────────────────
    # Formats conversation history for the prompt context
    def _format_history(self, history: list) -> str:
        if not history:
            return "No previous messages."
        formatted = []
        for msg in history[-10:]:  # last 10 messages only
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ── Result dataclass ──────────────────────────────────────
# Every provider returns this same structure
# Makes it easy to compare results across providers
@dataclass
class LLMResult:
    raw_text:       str             # raw response from LLM
    parsed_json:    Optional[dict]  # extracted architecture JSON
    tokens_used:    Optional[int]   # for benchmarking cost tracking
    response_time:  Optional[float] # milliseconds — for benchmarking
    provider:       str             # which provider returned this
    prompt_variant: str             # which prompt was used


# ── Abstract Base Class ───────────────────────────────────
# All LLM providers MUST implement these methods
# This enforces a consistent interface across Gemini, Claude, GPT4V
class LLMProvider(ABC):

    @abstractmethod
    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt_variant: str = "zero_shot",
    ) -> LLMResult:
        """
        Takes raw image bytes, sends to Vision LLM,
        returns structured architecture JSON
        """
        pass

    @abstractmethod
    async def chat(
        self,
        message: str,
        architecture_context: dict,
        conversation_history: list,
        interview_mode: bool = False,
    ) -> str:
        """
        Takes user message + full architecture as context,
        returns AI response string
        """
        pass

    @abstractmethod
    async def explain_component(
        self,
        component: dict,
        full_architecture: dict,
    ) -> dict:
        """
        Takes one component + full architecture context,
        returns detailed analysis of that component
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns provider name string e.g. 'gemini'"""
        pass

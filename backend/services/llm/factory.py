from functools import lru_cache
from config import settings
from services.llm.base import LLMProvider


@lru_cache()
def get_llm_provider() -> LLMProvider:
    """
    Returns the correct LLM provider based on LLM_PROVIDER in .env
    
    Usage in .env:
        LLM_PROVIDER=gemini   → uses GeminiProvider (free, default)
        LLM_PROVIDER=claude   → uses ClaudeProvider (benchmarking)
    
    lru_cache means only one instance is created — singleton pattern
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        from services.llm.gemini import GeminiProvider
        return GeminiProvider()

    elif provider == "claude":
        from services.llm.claude import ClaudeProvider
        return ClaudeProvider()

    else:
        # Default to Gemini if unknown provider specified
        print(f"Warning: Unknown provider '{provider}', defaulting to Gemini")
        from services.llm.gemini import GeminiProvider
        return GeminiProvider()

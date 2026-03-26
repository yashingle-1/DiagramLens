from services.llm.base import LLMProvider, LLMResult


# Stub implementation for benchmarking research
# Swap LLM_PROVIDER=claude in .env to use this
# Fill in real API calls when doing comparison study
class ClaudeProvider(LLMProvider):

    def __init__(self):
        try:
            import anthropic
            from config import settings
            self.client = anthropic.Anthropic(api_key=settings.claude_api_key)
        except ImportError:
            raise Exception("anthropic package not installed. Run: pip install anthropic")

    def get_provider_name(self) -> str:
        return "claude"

    async def analyze_image(self, image_bytes: bytes, prompt_variant: str = "zero_shot") -> LLMResult:
        # TODO: Implement for Phase 8 benchmarking
        # Structure mirrors GeminiProvider.analyze_image exactly
        raise NotImplementedError("Claude provider coming in Phase 8 benchmarking module")

    async def chat(self, message: str, architecture_context: dict, conversation_history: list, interview_mode: bool = False) -> str:
        raise NotImplementedError("Claude provider coming in Phase 8 benchmarking module")

    async def explain_component(self, component: dict, full_architecture: dict) -> dict:
        raise NotImplementedError("Claude provider coming in Phase 8 benchmarking module")

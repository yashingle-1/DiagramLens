from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    claude_api_key: str = ""
    openai_api_key: str = ""

    # Database
    database_url: str = "postgresql://admin:password@localhost:5432/archexplain"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Gemini extraction budget. v2 prompts emit ids/names/types/connections
    # only — roughly 30 tokens per component — so a 25-component diagram needs
    # well under 4k. The old 32768 ceiling existed because v1 prompts also
    # generated per-component analysis. thinking_budget is counted against
    # max_output_tokens, so it is capped explicitly rather than left dynamic.
    gemini_max_output_tokens: int = 8192
    gemini_thinking_budget:   int = 1024
    prompt_version: str = "v2"        # "v1" reproduces the metadata-heavy prompts

    # App
    environment: str = "development"
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 10

    # CORS — frontend origin
    frontend_url: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        case_sensitive = False


# lru_cache means this is only created once — singleton pattern
@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
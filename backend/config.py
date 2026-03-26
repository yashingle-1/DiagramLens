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
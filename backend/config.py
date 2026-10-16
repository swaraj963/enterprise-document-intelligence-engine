from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- AI (free tier) ---
    groq_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage ---
    chroma_dir: Path = ROOT / "data" / "vault"
    upload_tmp_dir: Path = ROOT / "data" / "tmp"

    # --- Retrieval ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_relevance: float = 0.20  # below this, we decline without calling the LLM

    # --- Security ---
    max_upload_mb: int = 15
    max_question_chars: int = 1000
    rate_limit_chat: str = "15/minute"
    rate_limit_upload: str = "6/minute"
    rate_limit_default: str = "60/minute"
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    debug: bool = False

    # --- Sessions / private vaults ---
    session_secret: SecretStr = Field(default=SecretStr(""))
    cookie_secure: bool = False  # set true when served over HTTPS

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    s.upload_tmp_dir.mkdir(parents=True, exist_ok=True)
    return s
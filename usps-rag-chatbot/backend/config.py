"""Environment-driven configuration.

All runtime settings live here. The app never reads os.environ directly
outside this module, which keeps tests hermetic and deployments auditable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: str = Field(default="echo", alias="LLM_PROVIDER")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # Embeddings / vector store
    embeddings_provider: str = Field(default="sentence-transformers", alias="EMBEDDINGS_PROVIDER")
    embeddings_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDINGS_MODEL"
    )
    vector_store: str = Field(default="faiss", alias="VECTOR_STORE")
    index_path: Path = Field(default=Path("./backend/rag/_index"), alias="INDEX_PATH")

    # RAG tuning
    top_k: int = Field(default=5, alias="TOP_K")
    min_score: float = Field(default=0.25, alias="MIN_SCORE")
    max_context_chars: int = Field(default=8000, alias="MAX_CONTEXT_CHARS")

    # API
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:8000,http://127.0.0.1:5500",
        alias="ALLOWED_ORIGINS",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Data
    data_root: Path = Field(default=Path("./data"))

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

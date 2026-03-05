"""
Application configuration loaded from environment variables.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Database ---
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/bns_rag"
    database_url_sync: str = "postgresql://user:password@localhost:5432/bns_rag"

    # --- LLM Providers ---
    groq_api_key: str = ""
    gemini_api_key: str = ""

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- Reranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma_db"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8501"]

    # --- Retrieval ---
    hybrid_search_semantic_weight: float = 0.6
    hybrid_search_keyword_weight: float = 0.4
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # --- LLM ---
    primary_llm: str = "groq"
    fallback_llm: str = "gemini"
    max_output_tokens: int = 2000
    llm_timeout_seconds: int = 10

    # --- Rate Limiting ---
    rate_limit: str = "30/minute"

    # --- Backend URL ---
    backend_url: str = "http://localhost:8000"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()

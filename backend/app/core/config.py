"""Application configuration.

All runtime settings are read from environment variables (12-factor style) so the
same image can run in development, CI and the competition demo environment.
``.env`` files are *never* committed; ``.env.example`` documents every knob.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (backend/app/core/config.py -> backend/app/core -> backend -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings, populated from ``.env`` / process environment."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "SciAgent"
    app_version: str = "0.1.0"
    environment: Literal["dev", "prod", "test"] = "dev"
    debug: bool = True
    # Comma separated list of CORS origins ("*" allows all).
    cors_origins: str = "*"

    # ------------------------------------------------------------------ storage
    # Directory that holds uploaded PDFs, rendered page images and parse artefacts.
    data_dir: Path = Field(default=REPO_ROOT / "data")
    # SQLite URL for metadata / chunk / graph storage.
    database_url: str = ""
    # Directory used for vector indexes (npz + json sidecar files).
    index_dir: Path = Field(default=REPO_ROOT / "data" / "index")

    # ------------------------------------------------------------------ llm
    # OpenAI-compatible endpoint exposed by vLLM on the GPU host.
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "Qwen3.8-27B"
    llm_api_key: str = ""  # optional; vLLM usually runs without auth
    llm_timeout: int = 300
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.2
    # Upper bound on how many documents may participate in one multi-paper reasoning call.
    llm_max_context_papers: int = 12

    # ------------------------------------------------------------------ embedding
    # Dense retriever. bge-m3 is multilingual (1024-dim) which matters for the
    # Chinese <-> English side of the product.
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    embedding_max_length: int = 512
    # Number of CPU threads used by torch when encoding.
    #
    # This knob matters a lot: on a 192-core host torch defaults to one
    # thread per core, which oversubscribes the machine and drops the
    # measured throughput from ~100 passages/s down to ~7 passages/s.
    # 64 threads is the sweet spot for bge-m3 on this class of hardware.
    embedding_threads: int = 64
    # Cross-encoder reranker used on top of hybrid retrieval.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True
    reranker_max_passages: int = 24

    # ------------------------------------------------------------------ retrieval
    # Number of candidates recalled per retrieval channel before fusion.
    retrieve_top_k_sparse: int = 40
    retrieve_top_k_dense: int = 40
    retrieve_top_k_final: int = 8
    # Minimum fused score required before a passage may be used as evidence.
    retrieve_min_score: float = 0.0

    # ------------------------------------------------------------------ parsing
    # DPI used when rasterising pages for OCR / VLM reading.
    render_dpi: int = 150
    # Pages parsed in parallel by the process pool.
    parse_workers: int = 8
    # Hard cap on pages per document (0 = unlimited). Used for smoke tests.
    parse_max_pages: int = 0

    # ------------------------------------------------------------------ agent
    agent_max_steps: int = 8
    agent_max_parallel_tools: int = 4
    # Directory where the prompt audit trail is persisted.
    audit_dir: Path = Field(default=REPO_ROOT / "data" / "audit")

    # ------------------------------------------------------------------ security
    # Optional reviewer account. When set, the API requires `X-API-Key`.
    api_key: str = ""
    reviewer_user: str = "reviewer"
    reviewer_password: str = ""

    # ------------------------------------------------------------------ misc
    log_level: str = "INFO"
    log_json: bool = False
    # Optional directory with the built frontend.  When it is set (and exists)
    # the same process serves the SPA *and* the API, which keeps the public
    # demo behind a single URL and removes the need for a separate reverse
    # proxy.  Empty disables the static hosting.
    static_dir: str = ""

    @field_validator("data_dir", "index_dir", "audit_dir", mode="before")
    @classmethod
    def _expand(cls, v: str | Path) -> Path:
        return Path(os.path.expandvars(str(v))).expanduser().resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlite_url(self) -> str:
        """Resolve the SQLite URL, defaulting to ``data/sciagent.db``."""
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{self.data_dir / 'sciagent.db'}"

    def ensure_dirs(self) -> None:
        """Create every runtime directory (idempotent)."""
        for d in (self.data_dir, self.index_dir, self.audit_dir,
                  self.data_dir / "uploads", self.data_dir / "pages", self.data_dir / "artifacts"):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    s = Settings()
    s.ensure_dirs()
    return s

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator  
from pydantic_settings import BaseSettings, SettingsConfigDict 

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings (env prefix ``RAG_``)."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Models (Ollama)
    # ------------------------------------------------------------------ #
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"
    #: Expected embedding dimensionality (nomic-embed-text = 768). Verified at runtime.
    embed_dim: int = Field(default=768, gt=0)
    #: Context window passed to Ollama as ``options.num_ctx``.
    num_ctx: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    #: nomic-embed-text was trained with task prefixes; omitting them hurts quality.
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    embed_batch_size: int = Field(default=16, gt=0)
    ollama_timeout_s: float = Field(default=300.0, gt=0)

    # ------------------------------------------------------------------ #
    # Storage paths
    # ------------------------------------------------------------------ #
    qdrant_path: Path = Path("./qdrant_local")
    qdrant_collection: str = "code_chunks"
    index_store_path: Path = Path("./index_store")
    models_cache_dir: Path = Path("./models")
    #: Where remote repositories are cloned.
    repos_dir: Path = Path("./repos")

    # ------------------------------------------------------------------ #
    # Ingestion limits
    # ------------------------------------------------------------------ #
    max_file_bytes: int = Field(default=512_000, gt=0)
    max_source_files: int = Field(default=2000, gt=0)
    #: Chunks above this (estimated) token count are split by the fallback splitter.
    max_chunk_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=32, ge=0)
    #: Bytes inspected for NUL characters when detecting binary files.
    binary_sniff_bytes: int = Field(default=8192, gt=0)
    ignored_dirs: frozenset[str] = frozenset(
        {
            ".git",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".tox",
            ".idea",
            ".vscode",
        }
    )
    #: Max chars of a class/function signature kept in a context header or member list.
    header_signature_max_chars: int = Field(default=200, gt=0)
    #: Lower bound for the per-chunk character budget (guards against tiny budgets).
    min_chunk_chars: int = Field(default=200, gt=0)
    #: ``--depth`` used when cloning a remote repository (shallow clone).
    git_clone_depth: int = Field(default=1, ge=1)

    # ------------------------------------------------------------------ #
    # Token estimation and context budget
    # ------------------------------------------------------------------ #
    chars_per_token: float = Field(default=3.2, gt=0)
    generation_reserve_tokens: int = Field(default=1200, gt=0)
    system_prompt_reserve_tokens: int = Field(default=500, gt=0)
    #: Fraction of the remaining prompt budget given to graph paths (rest = snippets).
    graph_budget_fraction: float = Field(default=0.25, ge=0.0, le=0.9)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    top_k: int = Field(default=5, ge=1, le=20)
    dense_top_n: int = Field(default=30, ge=1)
    bm25_top_n: int = Field(default=30, ge=1)
    #: RRF constant: score(d) = sum 1 / (rrf_k + rank_i(d)).
    rrf_k: int = Field(default=60, ge=1)
    use_reranker: bool = True
    #: Must be a key of flashrank's model_file_map (verified for 0.2.10).
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_candidates: int = Field(default=20, ge=1)

    # ------------------------------------------------------------------ #
    # Graph expansion
    # ------------------------------------------------------------------ #
    default_hops: int = Field(default=1, ge=1, le=2)
    graph_max_nodes: int = Field(default=40, ge=1)
    pagerank_damping: float = Field(default=0.85, gt=0.0, lt=1.0)

    # ------------------------------------------------------------------ #
    # Repo map / router
    # ------------------------------------------------------------------ #
    repo_map_tree_depth: int = Field(default=3, ge=1)
    readme_excerpt_chars: int = Field(default=1500, ge=0)
    repo_map_top_k_nodes: int = Field(default=15, ge=1)
    repo_map_max_files_in_summary: int = Field(default=60, ge=1)
    #: Use the LLM to classify ambiguous queries (heuristics always run first).
    use_llm_router_fallback: bool = False

    # ------------------------------------------------------------------ #
    # API / UI
    # ------------------------------------------------------------------ #
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    #: Qdrant local mode locks its folder, so exactly one worker is allowed.
    api_workers: int = 1
    api_base_url: str = "http://127.0.0.1:8000"
    api_request_timeout_s: float = Field(default=600.0, gt=0)
    job_poll_interval_s: float = Field(default=1.0, gt=0)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = "INFO"
    log_format: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("api_workers")
    @classmethod
    def _single_worker_only(cls, value: int) -> int:
        """Qdrant local mode is single-process; refuse multi-worker setups."""
        if value != 1:
            raise ValueError(
                "api_workers must be 1: Qdrant local mode locks its storage "
                "folder and cannot be shared between processes."
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown log level: {value!r}")
        return level

    @model_validator(mode="after")
    def _check_budget(self) -> "Settings":
        """Ensure the fixed reservations leave room for context."""
        if self.prompt_budget_tokens < 512:
            raise ValueError(
                "num_ctx is too small: after reserving generation and system "
                f"prompt tokens only {self.prompt_budget_tokens} remain for context."
            )
        if self.chunk_overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than max_chunk_tokens")
        return self

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #
    @property
    def prompt_budget_tokens(self) -> int:
        """Tokens available for snippets + graph paths + the user question."""
        return self.num_ctx - self.generation_reserve_tokens - self.system_prompt_reserve_tokens

    def estimate_tokens(self, text: str) -> int:
        """Conservative offline token estimate (``len(text) / chars_per_token``, rounded up).

        A heuristic, not Qwen's or nomic's tokenizer; deliberately over-counts.
        """
        return math.ceil(len(text) / self.chars_per_token) if text else 0

    @property
    def max_chunk_chars(self) -> int:
        """Character form of ``max_chunk_tokens`` under the same heuristic."""
        return int(self.max_chunk_tokens * self.chars_per_token)

    @property
    def manifest_path(self) -> Path:
        """Index manifest: file hashes, chunk ids, and BM25/Qdrant sync state."""
        return self.index_store_path / "manifest.json"

    @property
    def bm25_path(self) -> Path:
        """Persisted BM25 corpus (tokenised chunks + chunk ids)."""
        return self.index_store_path / "bm25.pkl"

    @property
    def graph_path(self) -> Path:
        """Persisted code graph (node-link JSON)."""
        return self.index_store_path / "graph.json"

    @property
    def repo_map_path(self) -> Path:
        """Persisted repo map (JSON)."""
        return self.index_store_path / "repo_map.json"

    def ensure_dirs(self) -> None:
        """Create all persistent directories (idempotent)."""
        for path in (
            self.qdrant_path,
            self.index_store_path,
            self.models_cache_dir,
            self.repos_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    cfg = settings or get_settings()
    logging.basicConfig(level=cfg.log_level, format=cfg.log_format)

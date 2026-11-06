"""Shared pydantic data models: chunks, graph, retrieval results, API schemas.

Conventions
-----------
* Line numbers are **1-based and inclusive** everywhere (``start_line`` to
  ``end_line``), matching what editors and ``git blame`` show.
* ``node_id`` is a qualified name such as ``pkg.module.Class.method``. The
  ingestor produces these; this module only defines the placeholder prefixes
  and does not interpret ids beyond that.
* Placeholder graph nodes (``unresolved::foo``, ``external::requests``)
  exist only so that an unresolved edge has a target. They are never
  guessed to be a real definition.
"""

from __future__ import annotations

import hashlib
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import get_settings

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Prefix for placeholder nodes created for names that could not be resolved.
UNRESOLVED_PREFIX = "unresolved::"
#: Prefix for placeholder nodes standing for modules outside the indexed repo.
EXTERNAL_PREFIX = "external::"
#: Fixed namespace so chunk ids map to stable Qdrant UUID point ids.
_CHUNK_UUID_NAMESPACE = uuid.UUID("8f1d3c0e-5b7a-4c52-9a41-2f6f0d9c7e11")


def hash_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` (UTF-8, errors replaced)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def make_chunk_id(file_path: str, node_id: str, part: int, content_hash: str) -> str:
    """Build a deterministic chunk id.

    Includes ``content_hash`` so an edited chunk gets a new id, and ``part`` so
    pieces produced by the oversize-function fallback stay distinct.
    """
    raw = f"{file_path}|{node_id}|{part}|{content_hash}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:24]  # noqa: S324 - id, not security


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class ChunkKind(str, Enum):
    """What a chunk represents."""

    MODULE = "module"  # top-level statements + imports
    CLASS = "class"  # signature, docstring, method signature list
    FUNCTION = "function"
    METHOD = "method"


class GraphNodeKind(str, Enum):
    """Node types in the code graph."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"  # methods included
    EXTERNAL = "external"  # placeholder: module outside the repo
    UNRESOLVED = "unresolved"  # placeholder: name we could not resolve


class EdgeKind(str, Enum):
    """Edge types in the code graph."""

    DEFINES = "defines"
    IMPORTS = "imports"
    CALLS = "calls"


class Resolution(str, Enum):
    """Whether an edge target was resolved to exactly one node."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class Route(str, Enum):
    """Query route chosen by the router."""

    MACRO = "macro"
    MICRO = "micro"


class ContextItemKind(str, Enum):
    """Kinds of items the ContextBuilder may drop."""

    SNIPPET = "snippet"
    PATH_1HOP = "path_1hop"
    PATH_2HOP = "path_2hop"
    REPO_MAP = "repo_map"


class JobStatus(str, Enum):
    """Lifecycle of a background indexing job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """One retrievable unit of code (function, class, module, or a fragment)."""

    model_config = ConfigDict(frozen=False)

    chunk_id: str
    node_id: str
    file_path: str = Field(description="Path relative to the repo root, POSIX separators.")
    language: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    kind: ChunkKind
    parent_node_id: Optional[str] = None
    content_hash: str
    text: str
    #: Index of this piece when an oversize body was split (0 when not split).
    part: int = Field(default=0, ge=0)
    #: Total number of pieces for this node (1 when not split).
    num_parts: int = Field(default=1, ge=1)
    #: Short header (file path + enclosing class signature) prepended for embedding only.
    context_header: str = ""

    @model_validator(mode="after")
    def _check_lines_and_parts(self) -> "Chunk":
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) < start_line ({self.start_line}) for {self.node_id}"
            )
        if self.part >= self.num_parts:
            raise ValueError(f"part ({self.part}) must be < num_parts ({self.num_parts})")
        return self

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder / BM25 (header + body, no task prefix)."""
        return f"{self.context_header}\n{self.text}" if self.context_header else self.text

    @property
    def point_id(self) -> str:
        """Stable UUID usable as a Qdrant point id (Qdrant requires int or UUID)."""
        return str(uuid.uuid5(_CHUNK_UUID_NAMESPACE, self.chunk_id))

    def to_payload(self) -> dict[str, Any]:
        """Serialise to a Qdrant payload (JSON-safe)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        """Rebuild a chunk from a Qdrant payload."""
        return cls.model_validate(payload)


class ScoredChunk(BaseModel):
    """A chunk plus the scores/ranks that put it in a result list."""

    chunk: Chunk
    score: float
    dense_rank: Optional[int] = Field(default=None, description="1-based rank in dense list, if present.")
    bm25_rank: Optional[int] = Field(default=None, description="1-based rank in BM25 list, if present.")
    rerank_score: Optional[float] = None


# --------------------------------------------------------------------------- #
# Code graph
# --------------------------------------------------------------------------- #


class GraphNode(BaseModel):
    """A node in the code graph, keyed by ``node_id``."""

    node_id: str
    kind: GraphNodeKind
    name: str = Field(description="Short display name (last segment of node_id).")
    file_path: Optional[str] = None
    start_line: Optional[int] = Field(default=None, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    language: Optional[str] = None

    @property
    def is_placeholder(self) -> bool:
        """True for external/unresolved placeholder nodes (no source location)."""
        return self.kind in (GraphNodeKind.EXTERNAL, GraphNodeKind.UNRESOLVED)


class GraphEdge(BaseModel):
    """A directed edge. ``resolution`` is only meaningful for imports/calls."""

    source: str
    target: str
    kind: EdgeKind
    resolution: Resolution = Resolution.RESOLVED
    #: Line of the call/import site in the source file, if known.
    line: Optional[int] = Field(default=None, ge=1)


class GraphPath(BaseModel):
    """A short path (1-2 hops) injected into the LLM context."""

    nodes: list[str] = Field(min_length=2)
    edges: list[GraphEdge] = Field(min_length=1)

    @property
    def hops(self) -> int:
        """Number of edges in the path."""
        return len(self.edges)

    @property
    def is_fully_resolved(self) -> bool:
        """True only if every edge on the path is resolved."""
        return all(e.resolution is Resolution.RESOLVED for e in self.edges)

    def describe(self) -> str:
        """Render for the prompt, labelling uncertain edges explicitly."""
        parts: list[str] = [self.nodes[0]]
        for edge in self.edges:
            label = edge.kind.value
            if edge.resolution is Resolution.UNRESOLVED:
                label = f"possible {label}, unresolved"
            parts.append(f" --{label}--> {edge.target}")
        return "".join(parts)


# --------------------------------------------------------------------------- #
# Query response
# --------------------------------------------------------------------------- #


class Citation(BaseModel):
    """A citation built from chunk metadata, never from model-generated text."""

    index: int = Field(ge=1, description="The [n] number shown to the model.")
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    node_id: str
    kind: ChunkKind
    cited: bool = Field(description="True if the model's answer actually used [index].")
    snippet: Optional[str] = Field(default=None, description="Preview of the chunk text for the UI.")


class DroppedItem(BaseModel):
    """Something the ContextBuilder removed to fit the token budget."""

    kind: ContextItemKind
    identifier: str = Field(description="chunk_id, node_id, or a path description.")
    est_tokens: int = Field(ge=0)
    reason: str


class QueryResponse(BaseModel):
    """Full, structured result of ``POST /query``."""

    answer: str
    route: Route
    citations: list[Citation] = Field(default_factory=list)
    #: ``[n]`` markers in the answer that do not map to a provided chunk.
    invalid_citations: list[int] = Field(default_factory=list)
    graph_paths: list[GraphPath] = Field(default_factory=list)
    dropped_context: list[DroppedItem] = Field(default_factory=list)
    #: Stage timings in milliseconds (e.g. retrieve_ms, rerank_ms, llm_ms, total_ms).
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# API request / job schemas
# --------------------------------------------------------------------------- #


class IndexRequest(BaseModel):
    """Body of ``POST /index``."""

    source: str = Field(min_length=1, description="Git URL or local directory path.")
    force: bool = False


class IndexJobResponse(BaseModel):
    """Returned immediately by ``POST /index``."""

    job_id: str


class IndexCounts(BaseModel):
    """Incremental-index counters plus index size stats."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0
    files: int = 0
    chunks: int = 0
    nodes: int = 0
    edges: int = 0


class IndexStatus(BaseModel):
    """Body of ``GET /index/{job_id}``."""

    status: JobStatus
    stage: str = ""
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    counts: IndexCounts = Field(default_factory=IndexCounts)
    error: Optional[str] = None


class QueryRequest(BaseModel):
    """Body of ``POST /query`` and ``POST /query/stream``."""

    question: str = Field(min_length=1)
    top_k: int = Field(default_factory=lambda: get_settings().top_k, ge=1, le=20)
    hops: int = Field(default_factory=lambda: get_settings().default_hops, ge=1, le=2)
    use_reranker: bool = Field(default_factory=lambda: get_settings().use_reranker)


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    ollama_reachable: bool
    llm_model_available: bool
    embed_model_available: bool
    index_ready: bool
    reranker_available: bool = False
    details: dict[str, str] = Field(default_factory=dict)
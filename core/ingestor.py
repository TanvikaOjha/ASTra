from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from langchain_text_splitters import Language as SplitterLanguage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, configure_logging, get_settings
from core.languages import (
    Definition,
    LanguageSpec,
    extract_definitions,
    get_spec_for_path,
    module_name_for,
    parse_source,
)
from core.models import Chunk, ChunkKind, hash_text, make_chunk_id

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when a source cannot be resolved, cloned, or read."""


# Hashing

_HASH_BLOCK_BYTES = 1 << 20


def hash_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


# Source resolution (git URL or local directory)

# https/http/git/ssh/file URLs and scp-like ``user@host:path``. Anything that
# starts with ``-`` or contains whitespace cannot match (no option injection).
_REMOTE_URL_RE = re.compile(r"^(?:(?:https?|git|ssh|file)://\S+|[\w.\-]+@[\w.\-]+:\S+)$")
_SAFE_NAME_RE = re.compile(r"[^\w.\-]+")


@dataclass(frozen=True)
class ResolvedSource:
    """A repository on local disk, ready to be walked."""

    root: Path
    name: str
    is_remote: bool
    #: HEAD commit sha when the directory is a git checkout, else ``None``.
    commit: Optional[str]


def _repo_name_from_url(url: str) -> str:
    tail = re.sub(r"\.git$", "", url.rstrip("/")).rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return _SAFE_NAME_RE.sub("_", tail) or "repo"


def _head_commit(root: Path) -> Optional[str]:
    """Best-effort HEAD sha; ``None`` when ``root`` is not a git checkout."""
    if not (root / ".git").exists():
        return None
    try:
        from git import Repo

        return Repo(root).head.commit.hexsha
    except Exception as exc:  # noqa: BLE001 - informational only
        logger.debug("Could not read HEAD of %s: %s", root, exc)
        return None


def _clone_or_update(url: str, settings: Settings) -> ResolvedSource:
    """Shallow-clone ``url`` into ``repos_dir`` or fast-forward an existing clone."""
    try:
        from git import GitCommandError, GitCommandNotFound, Repo
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IngestError("GitPython is required to clone remote repositories") from exc

    name = _repo_name_from_url(url)
    dest = settings.repos_dir / f"{name}-{hashlib.sha1(url.encode()).hexdigest()[:8]}"  # noqa: S324 - dir name
    settings.repos_dir.mkdir(parents=True, exist_ok=True)
    try:
        if (dest / ".git").is_dir():
            try:
                logger.info("Updating existing clone %s", dest)
                Repo(dest).remotes.origin.pull(ff_only=True)
            except GitCommandError as exc:
                logger.warning("Could not update %s (%s); using the existing clone as is.", dest, exc)
        else:
            logger.info("Cloning %s into %s (depth=%d)", url, dest, settings.git_clone_depth)
            Repo.clone_from(url, dest, depth=settings.git_clone_depth)
    except GitCommandNotFound as exc:
        raise IngestError("The 'git' executable was not found; install git or pass a local path.") from exc
    except GitCommandError as exc:
        if not (dest / ".git").is_dir():
            shutil.rmtree(dest, ignore_errors=True)
        raise IngestError(f"git clone failed for {url!r}: {exc}") from exc
    return ResolvedSource(root=dest.resolve(), name=name, is_remote=True, commit=_head_commit(dest))


def resolve_source(source: str, settings: Optional[Settings] = None) -> ResolvedSource:
    """Turn a git URL or local directory path into a ``ResolvedSource``."""
    cfg = settings or get_settings()
    source = source.strip()
    if not source:
        raise IngestError("Source is empty.")
    if _REMOTE_URL_RE.match(source):
        return _clone_or_update(source, cfg)
    path = Path(source).expanduser()
    if not path.is_dir():
        raise IngestError(f"Source is neither a git URL nor an existing directory: {source!r}")
    root = path.resolve()
    return ResolvedSource(root=root, name=root.name, is_remote=False, commit=_head_commit(root))


# File discovery


@dataclass(frozen=True)
class SourceFile:
    """A supported source file selected for indexing."""

    rel_path: str  # POSIX separators, relative to the repo root
    abs_path: Path
    language: str  # key of ``core.languages.LANGUAGES``
    size_bytes: int


@dataclass(frozen=True)
class SkippedFile:
    """A supported-extension file that was not indexed, and why."""

    rel_path: str
    reason: str


@dataclass
class DiscoveryResult:
    """Outcome of walking a repository."""

    root: Path
    files: list[SourceFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    #: True if more than ``max_source_files`` were found and the deepest were dropped.
    truncated: bool = False


def _looks_binary(path: Path, sniff_bytes: int) -> bool:
    """True if the first ``sniff_bytes`` contain a NUL byte."""
    with path.open("rb") as handle:
        return b"\x00" in handle.read(sniff_bytes)


def discover_files(root: Path, settings: Optional[Settings] = None) -> DiscoveryResult:
    """Walk ``root`` and select supported, readable, reasonably sized source files.

    Ignored directories are pruned, symlinks are never followed, and every
    skipped supported-extension file is logged with its reason. If there are
    more than ``max_source_files`` candidates, the shallowest are kept and
    ``truncated`` is set.
    """
    cfg = settings or get_settings()
    result = DiscoveryResult(root=root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in cfg.ignored_dirs)
        for filename in sorted(filenames):
            spec = get_spec_for_path(filename)
            if spec is None:
                continue
            abs_path = Path(dirpath) / filename
            rel_path = abs_path.relative_to(root).as_posix()
            reason = _skip_reason(abs_path, cfg)
            if reason is not None:
                result.skipped.append(SkippedFile(rel_path, reason))
                logger.info("Skipping %s: %s", rel_path, reason)
                continue
            result.files.append(
                SourceFile(rel_path, abs_path, spec.name, abs_path.stat().st_size)
            )

    if len(result.files) > cfg.max_source_files:
        result.truncated = True
        ranked = sorted(result.files, key=lambda f: (f.rel_path.count("/"), f.rel_path))
        for dropped in ranked[cfg.max_source_files :]:
            result.skipped.append(SkippedFile(dropped.rel_path, "over max_source_files"))
        kept = {f.rel_path for f in ranked[: cfg.max_source_files]}
        result.files = [f for f in result.files if f.rel_path in kept]
        logger.warning(
            "Found %d source files; indexing only the %d shallowest (max_source_files). "
            "Raise RAG_MAX_SOURCE_FILES to include more.",
            len(ranked),
            cfg.max_source_files,
        )
    logger.info("Discovered %d source files (%d skipped) under %s", len(result.files), len(result.skipped), root)
    return result


def _skip_reason(path: Path, cfg: Settings) -> Optional[str]:
    """Return why ``path`` must be skipped, or ``None`` to keep it."""
    try:
        if path.is_symlink():
            return "symlink"
        size = path.stat().st_size
        if size == 0:
            return "empty file"
        if size > cfg.max_file_bytes:
            return f"larger than max_file_bytes ({size} > {cfg.max_file_bytes})"
        if _looks_binary(path, cfg.binary_sniff_bytes):
            return "binary content"
    except OSError as exc:
        return f"unreadable ({exc.__class__.__name__}: {exc})"
    return None


# Chunking


@dataclass
class ChunkedFile:
    """All chunks of one file plus provenance."""

    rel_path: str
    language: str
    sha256: str  # over the original bytes; authoritative for the manifest
    line_count: int
    chunks: list[Chunk] = field(default_factory=list)
    has_syntax_errors: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """A maximal block of consecutive, not-excluded source lines."""

    start: int
    end: int
    text: str


@dataclass
class _Piece:
    """One future chunk body with its exact line range."""

    start: int
    end: int
    text: str


def _decode(data: bytes) -> tuple[str, bool]:
    """Decode to normalised text. Returns ``(text, had_undecodable_bytes)``."""
    try:
        text, lossy = data.decode("utf-8"), False
    except UnicodeDecodeError:
        text, lossy = data.decode("utf-8", errors="replace"), True
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n"), lossy


def _clip(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` chars with an ellipsis."""
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _collect_runs(lines: list[str], lo: int, hi: int, excluded: set[int]) -> list[_Run]:
    """Maximal runs of lines in ``[lo, hi]`` not in ``excluded``; blank edges trimmed."""
    runs: list[_Run] = []
    i = lo
    while i <= hi:
        if i in excluded:
            i += 1
            continue
        j = i
        while j + 1 <= hi and (j + 1) not in excluded:
            j += 1
        start, end = i, j
        while start <= end and not lines[start - 1].strip():
            start += 1
        while end >= start and not lines[end - 1].strip():
            end -= 1
        if start <= end:
            runs.append(_Run(start, end, "\n".join(lines[start - 1 : end])))
        i = j + 1
    return runs


def _marker(prefix: str, first: int, last: int) -> str:
    """Comment line that states which lines were left out of a chunk."""
    return f"{prefix} ... [lines {first}-{last} omitted: indexed as separate chunks] ..."


class CodeChunker:
    """Tree-sitter based chunker (see the module docstring for the chunk model)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or get_settings()
        self._splitters: dict[tuple[str, int, int], RecursiveCharacterTextSplitter] = {}

    # Public API

    def chunk_file(self, source_file: SourceFile) -> ChunkedFile:
        """Read ``source_file`` from disk and chunk it. Raises ``OSError`` on read failure."""
        return self.chunk_bytes(source_file.rel_path, source_file.abs_path.read_bytes())

    def chunk_bytes(self, rel_path: str, data: bytes) -> ChunkedFile:
        """Chunk raw file content. Pure (no disk access); this is what tests call."""
        spec = get_spec_for_path(rel_path)
        if spec is None:
            raise IngestError(f"Unsupported or skipped file type: {rel_path}")

        text, lossy = _decode(data)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        result = ChunkedFile(rel_path=rel_path, language=spec.name, sha256=hash_bytes(data), line_count=len(lines))
        if lossy:
            result.warnings.append("contains undecodable bytes (replaced with U+FFFD)")
        if not any(line.strip() for line in lines):
            return result

        defs: list[Definition] = []
        try:
            tree = parse_source(spec, text.encode("utf-8"))
            result.has_syntax_errors = tree.root_node.has_error
            defs = extract_definitions(spec, tree.root_node, text.encode("utf-8"))
            defs = [d for d in defs if 1 <= d.start_line <= d.end_line <= len(lines)]
        except Exception as exc:  # noqa: BLE001 - never lose a file because the parser failed
            logger.exception("Parsing failed for %s; indexing it as plain module text", rel_path)
            result.warnings.append(f"parse failed ({exc.__class__.__name__}); whole file indexed as one module")
            result.has_syntax_errors = True
            defs = []
        if result.has_syntax_errors:
            result.warnings.append("syntax errors: chunk boundaries may be inaccurate")
            logger.warning("%s has syntax errors; chunk boundaries may be inaccurate", rel_path)

        result.chunks = self._build_chunks(spec, rel_path, lines, defs)
        return result

    # Chunk assembly

    def _build_chunks(self, spec: LanguageSpec, rel_path: str, lines: list[str], defs: list[Definition]) -> list[Chunk]:
        module = module_name_for(spec, rel_path)
        node_ids = {id(d): f"{module}.{d.qualified_name}" for d in defs}
        chunks: list[Chunk] = []

        # 1. module-level chunk: everything outside definitions
        covered: set[int] = set()
        for d in defs:
            covered.update(range(d.start_line, d.end_line + 1))
        module_runs = _collect_runs(lines, 1, len(lines), covered)
        if module_runs:
            header_len = len(self._header(spec, rel_path, None, None, 0, 2))
            pieces = self._pack(module_runs, spec, self._budget(header_len))
            chunks += self._emit(pieces, spec, rel_path, rel_path, ChunkKind.MODULE, None, None, None)

        # 2. one chunk (or part-list) per definition
        for d in defs:
            node_id = node_ids[id(d)]
            parent_id = node_ids[id(d.parent)] if d.parent is not None else rel_path
            parent_sig = d.parent.signature if d.parent is not None else None
            receiver = d.receiver_type if d.parent is None else None
            header_len = len(self._header(spec, rel_path, parent_sig, receiver, 0, 2))
            budget = self._budget(header_len)

            if d.kind is ChunkKind.CLASS:
                pieces = self._class_pieces(spec, lines, defs, d, budget)
            else:
                body = _Run(d.start_line, d.end_line, "\n".join(lines[d.start_line - 1 : d.end_line]))
                pieces = self._pack([body], spec, budget)
            chunks += self._emit(pieces, spec, rel_path, node_id, d.kind, parent_id, parent_sig, receiver)
        return chunks

    def _class_pieces(
        self, spec: LanguageSpec, lines: list[str], defs: list[Definition], cdef: Definition, budget: int
    ) -> list[_Piece]:
        """Class summary: own lines + omission markers + member signature list."""
        children = [d for d in defs if d.parent is cdef]
        excluded: set[int] = set()
        for child in children:
            excluded.update(range(max(child.start_line, cdef.start_line), min(child.end_line, cdef.end_line) + 1))
        runs = _collect_runs(lines, cdef.start_line, cdef.end_line, excluded)
        pieces = self._pack(runs, spec, budget) if runs else []
        if not children:
            return pieces or [_Piece(cdef.start_line, cdef.end_line, _clip(cdef.signature, budget))]

        listing = self._member_list(spec, children, budget)
        if pieces and len(pieces[-1].text) + 1 + len(listing) <= budget:
            pieces[-1].text += "\n" + listing
        else:
            pieces.append(_Piece(cdef.start_line, cdef.end_line, listing))
        return pieces

    def _member_list(self, spec: LanguageSpec, members: list[Definition], budget: int) -> str:
        """``members:`` comment block of signatures, truncated to ``budget`` chars."""
        prefix = spec.line_comment_prefix
        out = [f"{prefix} members:"]
        used = len(out[0])
        for shown, member in enumerate(members):
            line = f"{prefix}   - {_clip(member.signature, self._cfg.header_signature_max_chars)}"
            if used + len(line) + 1 > budget:
                out.append(f"{prefix}   ... (+{len(members) - shown} more)")
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out)

    def _budget(self, header_len: int) -> int:
        """Character budget for a chunk body once the header is accounted for."""
        return max(self._cfg.min_chunk_chars, self._cfg.max_chunk_chars - header_len - 1)

    def _header(
        self,
        spec: LanguageSpec,
        rel_path: str,
        parent_signature: Optional[str],
        receiver_type: Optional[str],
        part: int,
        num_parts: int,
    ) -> str:
        """Short context header: file path, enclosing signature, and part marker."""
        prefix = spec.line_comment_prefix
        parts = [f"{prefix} file: {rel_path}"]
        if parent_signature:
            parts.append(f"{prefix} in: {_clip(parent_signature, self._cfg.header_signature_max_chars)}")
        elif receiver_type:
            parts.append(f"{prefix} receiver type: {receiver_type}")
        if num_parts > 1:
            parts.append(f"{prefix} part {part + 1}/{num_parts}")
        return "\n".join(parts)

    def _emit(
        self,
        pieces: list[_Piece],
        spec: LanguageSpec,
        rel_path: str,
        node_id: str,
        kind: ChunkKind,
        parent_node_id: Optional[str],
        parent_signature: Optional[str],
        receiver_type: Optional[str],
    ) -> list[Chunk]:
        """Turn pieces into ``Chunk`` objects (ids, hashes, headers, part numbers)."""
        pieces = [p for p in pieces if p.text.strip()]
        total = len(pieces)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            content_hash = hash_text(piece.text)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(rel_path, node_id, index, content_hash),
                    node_id=node_id,
                    file_path=rel_path,
                    language=spec.name,
                    start_line=piece.start,
                    end_line=piece.end,
                    kind=kind,
                    parent_node_id=parent_node_id,
                    content_hash=content_hash,
                    text=piece.text,
                    part=index,
                    num_parts=total,
                    context_header=self._header(spec, rel_path, parent_signature, receiver_type, index, total),
                )
            )
        return chunks

    # Packing and oversize fallback

    def _pack(self, runs: list[_Run], spec: LanguageSpec, budget: int) -> list[_Piece]:
        """Group runs into pieces that fit ``budget``; split any single oversize run."""
        prefix = spec.line_comment_prefix
        pieces: list[_Piece] = []
        current: list[_Run] = []
        used = 0

        def flush() -> None:
            nonlocal current, used
            if current:
                pieces.append(_Piece(current[0].start, current[-1].end, self._join_runs(current, prefix)))
            current, used = [], 0

        for run in runs:
            if len(run.text) > budget:
                flush()
                pieces.extend(self._split_run(run, spec, budget))
                continue
            extra = len(run.text)
            if current:
                extra += len(_marker(prefix, current[-1].end + 1, run.start - 1)) + 2
                if used + extra > budget:
                    flush()
                    extra = len(run.text)
            current.append(run)
            used += extra
        flush()
        return pieces

    @staticmethod
    def _join_runs(runs: list[_Run], prefix: str) -> str:
        """Join runs, inserting an omission marker for each gap between them."""
        parts = [runs[0].text]
        for previous, run in zip(runs, runs[1:]):
            parts.append(_marker(prefix, previous.end + 1, run.start - 1))
            parts.append(run.text)
        return "\n".join(parts)

    def _splitter(self, spec: LanguageSpec, size: int) -> RecursiveCharacterTextSplitter:
        overlap = min(int(self._cfg.chunk_overlap_tokens * self._cfg.chars_per_token), size // 4)
        key = (spec.name, size, overlap)
        if key not in self._splitters:
            self._splitters[key] = RecursiveCharacterTextSplitter.from_language(
                SplitterLanguage[spec.splitter_language], chunk_size=size, chunk_overlap=overlap
            )
        return self._splitters[key]

    def _split_run(self, run: _Run, spec: LanguageSpec, budget: int) -> list[_Piece]:
        """Split an oversize contiguous run; each piece gets exact line numbers."""
        splitter = self._splitter(spec, budget)
        texts = [t for t in splitter.split_text(run.text) if t.strip()]
        if not texts:
            return [_Piece(run.start, run.end, run.text)]
        pieces: list[_Piece] = []
        cursor = 0
        for piece_text in texts:
            position = run.text.find(piece_text, cursor)
            if position < 0:
                position = run.text.find(piece_text)
            if position < 0:
                logger.warning("Fallback splitter piece not found in source near line %d; using cursor", run.start)
                position = cursor
            start_line = run.start + run.text.count("\n", 0, position)
            end_line = start_line + piece_text.count("\n")
            pieces.append(_Piece(start_line, min(end_line, run.end), piece_text))
            cursor = position + 1
        return pieces


# Convenience iterator


def iter_chunked_files(files: Iterable[SourceFile], chunker: Optional[CodeChunker] = None) -> Iterator[ChunkedFile]:
    """Chunk files one by one; unreadable files are logged and skipped."""
    worker = chunker or CodeChunker()
    for source_file in files:
        try:
            yield worker.chunk_file(source_file)
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", source_file.rel_path, exc)


# CLI: ``python -m core.ingestor PATH [--show N]``


def _main() -> None:
    parser = argparse.ArgumentParser(description="Discover and chunk a repository (no indexing).")
    parser.add_argument("source", help="Local directory or git URL")
    parser.add_argument("--show", type=int, default=3, help="Number of sample chunks to print")
    args = parser.parse_args()

    cfg = get_settings()
    configure_logging(cfg)
    resolved = resolve_source(args.source, cfg)
    discovery = discover_files(resolved.root, cfg)
    chunker = CodeChunker(cfg)

    kinds: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    all_chunks: list[Chunk] = []
    for chunked in iter_chunked_files(discovery.files, chunker):
        languages[chunked.language] += 1
        for chunk in chunked.chunks:
            kinds[chunk.kind.value] += 1
            all_chunks.append(chunk)
    split = sum(1 for c in all_chunks if c.num_parts > 1)
    biggest = max((cfg.estimate_tokens(c.embedding_text) for c in all_chunks), default=0)
    logger.info("Files by language: %s", dict(languages))
    logger.info("Chunks by kind: %s (parts of split nodes: %d)", dict(kinds), split)
    logger.info("Largest chunk ~%d tokens (limit %d)", biggest, cfg.max_chunk_tokens)
    for chunk in all_chunks[: args.show]:
        preview = "\n    ".join(chunk.text.splitlines()[:6])
        logger.info(
            "[%s] %s L%d-%d part %d/%d\n  %s\n    %s",
            chunk.kind.value,
            chunk.node_id,
            chunk.start_line,
            chunk.end_line,
            chunk.part + 1,
            chunk.num_parts,
            chunk.context_header.replace("\n", "\n  "),
            preview,
        )


if __name__ == "__main__":
    _main()
"""Deterministic incremental repository and mixed-HTML component index."""

from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fnmatch
import hashlib
from html.parser import HTMLParser
import json
import math
import os
import posixpath
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
from typing import Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class IndexEntry:
    path: str
    kind: str
    name: str
    start: int
    end: int
    file_hash: str
    text: str
    confidence: float = 1.0
    provenance: str = "parser"


@dataclass(frozen=True, slots=True)
class CodeRelation:
    path: str
    kind: str
    source: str
    target: str
    line: int
    text: str
    confidence: float = 1.0
    provenance: str = "parser"


@dataclass(frozen=True, slots=True)
class SearchHit:
    entry: IndexEntry
    score: float
    channels: tuple[str, ...]
    confidence: float = 1.0
    reason: str = ""
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: str
    size: int
    mtime_ns: int
    file_hash: str


@dataclass(frozen=True, slots=True)
class RepositoryContextSlice:
    query: str
    entries: tuple[IndexEntry, ...]
    relations: tuple[CodeRelation, ...]
    callers: Mapping[str, tuple[str, ...]]
    callees: Mapping[str, tuple[str, ...]]
    dependencies: Mapping[str, tuple[str, ...]]
    omitted_entries: int = 0
    size_chars: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "entries": [
                {
                    "path": item.path,
                    "kind": item.kind,
                    "name": item.name,
                    "start": item.start,
                    "end": item.end,
                    "file_hash": item.file_hash,
                    "text": item.text,
                    "confidence": item.confidence,
                    "provenance": item.provenance,
                }
                for item in self.entries
            ],
            "relations": [
                {
                    "path": item.path,
                    "kind": item.kind,
                    "source": item.source,
                    "target": item.target,
                    "line": item.line,
                    "text": item.text,
                    "confidence": item.confidence,
                    "provenance": item.provenance,
                }
                for item in self.relations
            ],
            "callers": {key: list(value) for key, value in self.callers.items()},
            "callees": {key: list(value) for key, value in self.callees.items()},
            "dependencies": {key: list(value) for key, value in self.dependencies.items()},
            "omitted_entries": self.omitted_entries,
            "size_chars": self.size_chars,
        }


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class HashingEmbeddingProvider:
    """Deterministic local embedding fallback.

    This is intentionally dependency-free.  It provides an explicit embedding
    seam that can later be replaced by a GPU/local model while still giving the
    harness stable retrieval behavior in offline tests.
    """

    dimensions: int = 128

    def embed(self, text: str) -> tuple[float, ...]:
        dimensions = max(8, int(self.dimensions))
        values = [0.0] * dimensions
        for token in RepositoryIndex._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            values[bucket] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return tuple(value / norm for value in values)

    def embed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed(text) for text in texts)


class OllamaEmbeddingProvider:
    """Real local semantic embeddings with a deterministic offline fallback.

    The provider uses Ollama's batched ``/api/embed`` endpoint.  A daemon or
    model outage must not make repository inspection unavailable, so failures
    fall back to hashing embeddings and are exposed through ``last_error``.
    Enable it with ``AGENT_EMBEDDING_MODEL`` (for example
    ``nomic-embed-text:latest``).
    """

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        timeout_seconds: float = 30.0,
        fallback: EmbeddingProvider | None = None,
    ) -> None:
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("Ollama embedding model must be non-empty")
        self.host = str(host or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.fallback = fallback or HashingEmbeddingProvider()
        self.last_error = ""

    @classmethod
    def from_environment(cls) -> EmbeddingProvider:
        model = str(os.getenv("AGENT_EMBEDDING_MODEL") or "").strip()
        return cls(model) if model else HashingEmbeddingProvider()

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many((text,))[0]

    def embed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        values = tuple(str(text) for text in texts)
        if not values:
            return ()
        request = Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model, "input": list(values)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            embeddings = payload.get("embeddings") if isinstance(payload, Mapping) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(values):
                raise ValueError("Ollama embed response did not contain one vector per input")
            result = tuple(tuple(float(item) for item in vector) for vector in embeddings)
            if any(not vector for vector in result):
                raise ValueError("Ollama embed response contained an empty vector")
            self.last_error = ""
            return result
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            embed_many = getattr(self.fallback, "embed_many", None)
            if callable(embed_many):
                return tuple(tuple(float(item) for item in vector) for vector in embed_many(values))
            return tuple(tuple(float(item) for item in self.fallback.embed(text)) for text in values)


class HnswEmbeddingIndex:
    """Optional in-memory HNSW accelerator with deterministic brute-force fallback."""

    def __init__(self) -> None:
        try:
            import hnswlib
        except ImportError:
            hnswlib = None
        self._library = hnswlib
        self._index = None
        self._keys: tuple[tuple[str, str, str, int], ...] = ()
        self.available = hnswlib is not None
        self.last_error = "" if self.available else "hnswlib is not installed"

    def build(self, vectors: Mapping[tuple[str, str, str, int], Sequence[float]]) -> bool:
        self._index = None
        self._keys = ()
        if not self.available or not vectors:
            return False
        items = [(key, tuple(float(value) for value in vector)) for key, vector in vectors.items() if vector]
        if not items:
            return False
        dimension = len(items[0][1])
        items = [item for item in items if len(item[1]) == dimension]
        if not items:
            return False
        try:
            index = self._library.Index(space="cosine", dim=dimension)
            index.init_index(max_elements=len(items), ef_construction=160, M=24)
            index.add_items([vector for _key, vector in items], list(range(len(items))))
            index.set_ef(min(max(64, int(math.sqrt(len(items))) * 4), max(64, len(items))))
            self._index = index
            self._keys = tuple(key for key, _vector in items)
            self.last_error = ""
            return True
        except (RuntimeError, TypeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
    ) -> dict[tuple[str, str, str, int], float]:
        if self._index is None or not self._keys or not vector:
            return {}
        count = min(max(1, int(limit)), len(self._keys))
        try:
            labels, distances = self._index.knn_query([list(vector)], k=count)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {}
        return {
            self._keys[int(label)]: max(-1.0, min(1.0, 1.0 - float(distance)))
            for label, distance in zip(labels[0], distances[0])
        }


class _DOMParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, int]] = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        name = values.get("id") or values.get("class") or values.get("aria-label") or tag
        self.entries.append((str(name), self.getpos()[0]))


class RepositoryIndex:
    _CACHE_VERSION = 3
    _CHUNK_LINES = 160
    _CHUNK_OVERLAP = 20
    _MAX_ENTRY_CHARS = 24_000
    _MAX_EMBED_CHARS = 8_000
    _INDEXABLE_SUFFIXES = {
        ".py", ".html", ".htm", ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx", ".css", ".json",
    }
    _SKIP_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".coding-agent",
    }

    def __init__(
        self,
        workspace: str | Path,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        cache_path: str | Path | None = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.entries: dict[str, tuple[IndexEntry, ...]] = {}
        self.relations: dict[str, tuple[CodeRelation, ...]] = {}
        self._vectors: dict[tuple[str, str, str, int], dict[str, float]] = {}
        self.embedding_provider = embedding_provider or OllamaEmbeddingProvider.from_environment()
        self._embeddings: dict[tuple[str, str, str, int], tuple[float, ...]] = {}
        self._hnsw = HnswEmbeddingIndex()
        self._hnsw_dirty = True
        self._file_snapshots: dict[str, FileSnapshot] = {}
        self.cache_path = Path(cache_path).resolve(strict=False) if cache_path is not None else None
        self._cache_save_suspended = False
        self.last_update_stats: dict[str, int] = {
            "seen": 0,
            "updated": 0,
            "reused": 0,
            "removed": 0,
            "loaded": 0,
        }
        if self.cache_path is not None:
            self.load_cache()

    def update(self, relative_path: str) -> tuple[IndexEntry, ...]:
        analysis = self._analyze_file(relative_path)
        self._commit_analysis(*analysis)
        if self.cache_path is not None and not self._cache_save_suspended:
            self.save_cache()
        return self.entries[relative_path]

    def _analyze_file(
        self,
        relative_path: str,
    ) -> tuple[str, tuple[IndexEntry, ...], tuple[CodeRelation, ...], FileSnapshot]:
        """Parse one file without mutating shared index state."""

        path = (self.workspace / relative_path).resolve()
        path.relative_to(self.workspace)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8", errors="replace")
        stat = path.stat()
        lines = text.splitlines()
        line_count = max(1, len(lines))
        file_text = text if len(text) <= self._MAX_ENTRY_CHARS else self._bounded_text(text)
        found: list[IndexEntry] = [IndexEntry(relative_path, "file", path.name, 1, line_count, digest, file_text)]
        found.extend(self._chunk_entries(relative_path, lines, digest))
        relations: list[CodeRelation] = []
        if path.suffix.lower() == ".py":
            py_entries, py_relations = self._python_entries(relative_path, text, digest)
            found.extend(py_entries)
            relations.extend(py_relations)
        if path.suffix.lower() in {".html", ".htm"}:
            found.extend(self._html_entries(relative_path, text, digest))
        if path.suffix.lower() in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
            js_entries, js_relations = self._javascript_entries(relative_path, text, digest)
            found.extend(js_entries)
            relations.extend(js_relations)
        if path.suffix.lower() == ".css":
            found.extend(self._css_entries(relative_path, text, digest))
        snapshot = FileSnapshot(
            relative_path,
            int(stat.st_size),
            int(stat.st_mtime_ns),
            digest,
        )
        return relative_path, tuple(found), tuple(relations), snapshot

    def _commit_analysis(
        self,
        relative_path: str,
        found: tuple[IndexEntry, ...],
        relations: tuple[CodeRelation, ...],
        snapshot: FileSnapshot,
    ) -> None:
        # Remove only the previous entries owned by this file. Scanning the
        # complete repository vector map once per changed file made a 50-file
        # refresh grow with total repository size instead of changed size.
        previous_keys = tuple(
            self._entry_key(item) for item in self.entries.get(relative_path, ())
        )
        for key in previous_keys:
            self._vectors.pop(key, None)
            self._embeddings.pop(key, None)
        self.entries[relative_path] = found
        self.relations[relative_path] = relations
        embedding_texts = [self._embedding_text(item) for item in found]
        embed_many = getattr(self.embedding_provider, "embed_many", None)
        if callable(embed_many):
            dense_vectors = tuple(embed_many(embedding_texts))
        else:
            dense_vectors = tuple(self.embedding_provider.embed(value) for value in embedding_texts)
        if len(dense_vectors) != len(found):
            raise ValueError("embedding provider returned an unexpected vector count")
        for item, embedding_text, dense_vector in zip(found, embedding_texts, dense_vectors):
            self._vectors[self._entry_key(item)] = self._vector(
                embedding_text
            )
            self._embeddings[self._entry_key(item)] = tuple(float(value) for value in dense_vector)
        self._file_snapshots[relative_path] = snapshot
        self._hnsw_dirty = True

    @staticmethod
    def _entry_payload(item: IndexEntry) -> dict[str, object]:
        return {
            "path": item.path,
            "kind": item.kind,
            "name": item.name,
            "start": item.start,
            "end": item.end,
            "file_hash": item.file_hash,
            "text": item.text,
            "confidence": item.confidence,
            "provenance": item.provenance,
        }

    @staticmethod
    def _entry_from_payload(value: Mapping[str, object]) -> IndexEntry:
        return IndexEntry(
            str(value.get("path") or ""),
            str(value.get("kind") or ""),
            str(value.get("name") or ""),
            int(value.get("start") or 1),
            int(value.get("end") or 1),
            str(value.get("file_hash") or ""),
            str(value.get("text") or ""),
            float(value.get("confidence") or 1.0),
            str(value.get("provenance") or "parser"),
        )

    @staticmethod
    def _relation_payload(item: CodeRelation) -> dict[str, object]:
        return {
            "path": item.path,
            "kind": item.kind,
            "source": item.source,
            "target": item.target,
            "line": item.line,
            "text": item.text,
            "confidence": item.confidence,
            "provenance": item.provenance,
        }

    @staticmethod
    def _relation_from_payload(value: Mapping[str, object]) -> CodeRelation:
        return CodeRelation(
            str(value.get("path") or ""),
            str(value.get("kind") or ""),
            str(value.get("source") or ""),
            str(value.get("target") or ""),
            int(value.get("line") or 1),
            str(value.get("text") or ""),
            float(value.get("confidence") or 1.0),
            str(value.get("provenance") or "parser"),
        )

    @staticmethod
    def _snapshot_payload(item: FileSnapshot) -> dict[str, object]:
        return {
            "path": item.path,
            "size": item.size,
            "mtime_ns": item.mtime_ns,
            "file_hash": item.file_hash,
        }

    @staticmethod
    def _snapshot_from_payload(value: Mapping[str, object]) -> FileSnapshot:
        return FileSnapshot(
            str(value.get("path") or ""),
            int(value.get("size") or 0),
            int(value.get("mtime_ns") or 0),
            str(value.get("file_hash") or ""),
        )

    @staticmethod
    def _cache_key_payload(key: tuple[str, str, str, int]) -> list[object]:
        return [key[0], key[1], key[2], key[3]]

    @staticmethod
    def _cache_key_from_payload(value: Sequence[object]) -> tuple[str, str, str, int]:
        return (str(value[0]), str(value[1]), str(value[2]), int(value[3]))

    def _embedding_provider_signature(self) -> str:
        dimensions = getattr(self.embedding_provider, "dimensions", None)
        model = getattr(self.embedding_provider, "model", None)
        host = getattr(self.embedding_provider, "host", None)
        return (
            f"{self.embedding_provider.__class__.__module__}."
            f"{self.embedding_provider.__class__.__qualname__}:"
            f"dimensions={dimensions}:model={model}:host={host}"
        )

    def save_cache(self) -> bool:
        if self.cache_path is None:
            return False
        payload = {
            "version": self._CACHE_VERSION,
            "workspace": str(self.workspace),
            "embedding_provider": self._embedding_provider_signature(),
            "snapshots": {
                path: self._snapshot_payload(snapshot)
                for path, snapshot in sorted(self._file_snapshots.items())
            },
            "entries": {
                path: [self._entry_payload(item) for item in items]
                for path, items in sorted(self.entries.items())
            },
            "relations": {
                path: [self._relation_payload(item) for item in items]
                for path, items in sorted(self.relations.items())
            },
            "vectors": [
                {"key": self._cache_key_payload(key), "value": value}
                for key, value in self._vectors.items()
            ],
            "embeddings": [
                {"key": self._cache_key_payload(key), "value": list(value)}
                for key, value in self._embeddings.items()
            ],
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_name(f"{self.cache_path.name}.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            temporary.replace(self.cache_path)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def load_cache(self) -> bool:
        if self.cache_path is None or not self.cache_path.is_file():
            return False
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return False
        if not isinstance(payload, Mapping) or payload.get("version") != self._CACHE_VERSION:
            return False
        if str(payload.get("workspace") or "") != str(self.workspace):
            return False
        try:
            snapshots = {
                str(path): self._snapshot_from_payload(value)
                for path, value in dict(payload.get("snapshots") or {}).items()
                if isinstance(value, Mapping)
            }
            entries = {
                str(path): tuple(
                    self._entry_from_payload(item)
                    for item in items
                    if isinstance(item, Mapping)
                )
                for path, items in dict(payload.get("entries") or {}).items()
                if isinstance(items, list)
            }
            relations = {
                str(path): tuple(
                    self._relation_from_payload(item)
                    for item in items
                    if isinstance(item, Mapping)
                )
                for path, items in dict(payload.get("relations") or {}).items()
                if isinstance(items, list)
            }
            vectors = {
                self._cache_key_from_payload(item["key"]): {
                    str(key): float(value)
                    for key, value in dict(item.get("value") or {}).items()
                }
                for item in payload.get("vectors") or ()
                if isinstance(item, Mapping) and isinstance(item.get("key"), list)
            }
            embeddings = {
                self._cache_key_from_payload(item["key"]): tuple(
                    float(value) for value in (item.get("value") or ())
                )
                for item in payload.get("embeddings") or ()
                if isinstance(item, Mapping) and isinstance(item.get("key"), list)
            }
        except (TypeError, ValueError, IndexError):
            return False
        self._file_snapshots = snapshots
        self.entries = entries
        self.relations = relations
        if str(payload.get("embedding_provider") or "") == self._embedding_provider_signature():
            self._vectors = vectors
            self._embeddings = embeddings
        else:
            self._vectors = {}
            self._embeddings = {}
            for items in self.entries.values():
                for item in items:
                    embedding_text = f"{item.path} {self._module_name(item.path)} {item.kind} {item.name} {item.text}"
                    self._vectors[self._entry_key(item)] = self._vector(embedding_text)
                    self._embeddings[self._entry_key(item)] = tuple(
                        float(value) for value in self.embedding_provider.embed(embedding_text)
                    )
        self._hnsw_dirty = True
        self.last_update_stats = {
            "seen": 0,
            "updated": 0,
            "reused": 0,
            "removed": 0,
            "loaded": len(self.entries),
        }
        return True

    def _remove_path(self, relative_path: str) -> None:
        previous_keys = tuple(
            self._entry_key(item) for item in self.entries.get(relative_path, ())
        )
        self.entries.pop(relative_path, None)
        self.relations.pop(relative_path, None)
        self._file_snapshots.pop(relative_path, None)
        for key in previous_keys:
            self._vectors.pop(key, None)
            self._embeddings.pop(key, None)
        self._hnsw_dirty = True

    def _can_reuse_snapshot(self, relative_path: str, path: Path) -> bool:
        snapshot = self._file_snapshots.get(relative_path)
        if snapshot is None or relative_path not in self.entries:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        return snapshot.size == int(stat.st_size) and snapshot.mtime_ns == int(stat.st_mtime_ns)

    def update_all(
        self,
        *,
        suffixes: Iterable[str] | None = None,
        max_files: int | None = None,
        force: bool = False,
    ) -> tuple[IndexEntry, ...]:
        allowed = {item.casefold() for item in (suffixes or self._INDEXABLE_SUFFIXES)}
        collected: list[IndexEntry] = []
        indexed = 0
        seen_paths: set[str] = set()
        changed_paths: list[str] = []
        stats = {"seen": 0, "updated": 0, "reused": 0, "removed": 0, "loaded": len(self.entries)}
        self._cache_save_suspended = True
        try:
            paths = sorted(self.workspace.rglob("*"))
            for path in paths:
                if not path.is_file():
                    continue
                relative_parts = path.relative_to(self.workspace).parts
                if any(part in self._SKIP_DIRS for part in relative_parts[:-1]):
                    continue
                if path.suffix.casefold() not in allowed:
                    continue
                relative = path.relative_to(self.workspace).as_posix()
                seen_paths.add(relative)
                stats["seen"] += 1
                if not force and self._can_reuse_snapshot(relative, path):
                    collected.extend(self.entries.get(relative, ()))
                    stats["reused"] += 1
                else:
                    changed_paths.append(relative)
                indexed += 1
                if max_files is not None and indexed >= max(1, int(max_files)):
                    break
            if changed_paths:
                worker_count = min(
                    len(changed_paths),
                    max(1, min(32, (os.cpu_count() or 2) + 4)),
                )
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="repository-ast",
                ) as pool:
                    analyses = tuple(pool.map(self._analyze_file, changed_paths))
                for analysis in analyses:
                    self._commit_analysis(*analysis)
                    collected.extend(analysis[1])
                    stats["updated"] += 1
        finally:
            self._cache_save_suspended = False
        if max_files is None:
            for relative in tuple(self.entries):
                suffix = Path(relative).suffix.casefold()
                if suffix in allowed and relative not in seen_paths:
                    self._remove_path(relative)
                    stats["removed"] += 1
        self.last_update_stats = stats
        if self.cache_path is not None:
            self.save_cache()
        # HNSW rebuild is intentionally lazy. Rebuilding the entire graph on a
        # 50-file incremental batch makes update latency proportional to the
        # whole repository. The next semantic query rebuilds once; lexical and
        # graph retrieval remain immediately available meanwhile.
        return tuple(collected)

    def _ensure_hnsw(self) -> bool:
        if not self._hnsw_dirty:
            return self._hnsw._index is not None
        self._hnsw_dirty = False
        return self._hnsw.build(self._embeddings)

    @staticmethod
    def _entry_key(item: IndexEntry) -> tuple[str, str, str, int]:
        return (item.path, item.kind, item.name, item.start)

    @classmethod
    def _bounded_text(cls, value: str) -> str:
        if len(value) <= cls._MAX_ENTRY_CHARS:
            return value
        half = cls._MAX_ENTRY_CHARS // 2
        return value[:half] + "\n... [bounded repository index entry] ...\n" + value[-half:]

    @classmethod
    def _embedding_text(cls, item: IndexEntry) -> str:
        prefix = f"{item.path} {cls._module_name(item.path)} {item.kind} {item.name} "
        available = max(256, cls._MAX_EMBED_CHARS - len(prefix))
        body = item.text
        if len(body) > available:
            half = available // 2
            body = body[:half] + "\n...\n" + body[-half:]
        return prefix + body

    @classmethod
    def _chunk_entries(
        cls,
        path: str,
        lines: Sequence[str],
        digest: str,
    ) -> list[IndexEntry]:
        if len(lines) <= cls._CHUNK_LINES:
            return []
        chunks: list[IndexEntry] = []
        step = max(1, cls._CHUNK_LINES - cls._CHUNK_OVERLAP)
        for start_index in range(0, len(lines), step):
            chunk_lines = lines[start_index : start_index + cls._CHUNK_LINES]
            if not chunk_lines:
                break
            start = start_index + 1
            end = start_index + len(chunk_lines)
            chunks.append(
                IndexEntry(
                    path,
                    "code_chunk",
                    f"lines-{start}-{end}",
                    start,
                    end,
                    digest,
                    cls._bounded_text("\n".join(chunk_lines)),
                )
            )
            if end >= len(lines):
                break
        return chunks

    @staticmethod
    def _module_name(path: str) -> str:
        value = str(path).replace("\\", "/")
        if value.endswith("/__init__.py"):
            value = value[: -len("/__init__.py")]
        elif value.endswith(".py"):
            value = value[:-3]
        else:
            value = value.rsplit(".", 1)[0]
        return ".".join(part for part in value.split("/") if part)

    @staticmethod
    def _identifier_words(value: str) -> tuple[str, ...]:
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        return tuple(
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " ").replace("-", " "))
            if token
        )

    @classmethod
    def _tokens(cls, value: str) -> tuple[str, ...]:
        base: list[str] = []
        for raw in re.findall(r"[\w.$:-]+", value):
            lowered = raw.casefold().strip(".$:-")
            if lowered:
                base.append(lowered)
            base.extend(cls._identifier_words(raw))
            if "." in raw:
                base.extend(part for part in raw.casefold().split(".") if part)
        expanded: list[str] = []
        synonyms = {
            "deserialize": ("parse", "load", "loads", "json"),
            "deserialise": ("parse", "load", "loads", "json"),
            "serialize": ("dump", "dumps", "json"),
            "serialise": ("dump", "dumps", "json"),
            "parser": ("parse", "load", "loads"),
            "parse": ("parser", "load", "loads"),
            "caller": ("call", "calls", "invoke", "invokes"),
            "callee": ("call", "called", "target"),
            "invoke": ("call", "calls", "caller"),
            "invokes": ("call", "calls", "caller"),
            "render": ("draw", "paint", "canvas", "visual"),
            "visual": ("render", "draw", "paint", "style"),
            "graphics": ("render", "visual", "canvas", "shader"),
            "animation": ("animate", "frame", "keyframe", "requestanimationframe"),
            "responsive": ("media", "viewport", "layout"),
            "collision": ("intersect", "hit", "impact"),
            "projectile": ("arrow", "bullet", "launch"),
            "auth": ("authentication", "authorization", "token", "login"),
            "config": ("configuration", "settings", "options"),
        }
        for token in base:
            expanded.append(token)
            expanded.extend(synonyms.get(token, ()))
        return tuple(token for token in expanded if len(token) > 1)

    @classmethod
    def _vector(cls, value: str) -> dict[str, float]:
        counts = Counter(cls._tokens(value))
        if not counts:
            return {}
        norm = math.sqrt(sum(weight * weight for weight in counts.values())) or 1.0
        return {token: weight / norm for token, weight in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(token, 0.0) for token, weight in left.items())

    @staticmethod
    def _dense_cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        if size <= 0:
            return 0.0
        numerator = sum(float(left[index]) * float(right[index]) for index in range(size))
        left_norm = math.sqrt(sum(float(value) * float(value) for value in left[:size]))
        right_norm = math.sqrt(sum(float(value) * float(value) for value in right[:size]))
        denominator = left_norm * right_norm
        return (numerator / denominator) if denominator else 0.0

    @staticmethod
    def _line_text(lines: list[str], line: int) -> str:
        return lines[line - 1].strip() if 1 <= line <= len(lines) else ""

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = RepositoryIndex._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return RepositoryIndex._call_name(node.func)
        if isinstance(node, ast.Subscript):
            return RepositoryIndex._call_name(node.value)
        return ""

    def _python_entries(
        self,
        path: str,
        text: str,
        digest: str,
    ) -> tuple[list[IndexEntry], list[CodeRelation]]:
        lines = text.splitlines()
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError as exc:
            line = int(exc.lineno or 1)
            return (
                [
                    IndexEntry(
                        path,
                        "python_syntax_error",
                        str(exc.msg),
                        line,
                        line,
                        digest,
                        self._line_text(lines, line),
                    )
                ],
                [],
            )

        entries: list[IndexEntry] = []
        relations: list[CodeRelation] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            @property
            def owner(self) -> str:
                return ".".join(self.stack) if self.stack else "<module>"

            def _entry(self, kind: str, name: str, node: ast.AST) -> None:
                start = int(getattr(node, "lineno", 1) or 1)
                end = int(getattr(node, "end_lineno", start) or start)
                entries.append(
                    IndexEntry(
                        path,
                        kind,
                        name,
                        start,
                        end,
                        digest,
                        RepositoryIndex._bounded_text("\n".join(lines[start - 1:end])) if start <= len(lines) else "",
                    )
                )

            def _relation(self, kind: str, source: str, target: str, node: ast.AST) -> None:
                if not target:
                    return
                line = int(getattr(node, "lineno", 1) or 1)
                relations.append(
                    CodeRelation(
                        path,
                        kind,
                        source,
                        target,
                        line,
                        RepositoryIndex._line_text(lines, line),
                    )
                )

            def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
                for alias in node.names:
                    self._entry("py_import", alias.name, node)
                    self._relation("import", path, alias.name, node)
                    self._relation("import_alias", alias.asname or alias.name.split(".", 1)[0], alias.name, node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
                module = "." * int(node.level or 0) + (node.module or "")
                for alias in node.names:
                    target = f"{module}.{alias.name}".strip(".")
                    self._entry("py_import", target, node)
                    self._relation("import", path, target, node)
                    self._relation("import_alias", alias.asname or alias.name, target, node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                qualified = ".".join((*self.stack, node.name)) if self.stack else node.name
                self._entry("py_class", qualified, node)
                self._relation("owns", self.owner, qualified, node)
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self._function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                self._function(node)

            def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                qualified = ".".join((*self.stack, node.name)) if self.stack else node.name
                kind = "py_method" if self.stack else "py_function"
                self._entry(kind, qualified, node)
                self._relation("owns", self.owner, qualified, node)
                for decorator in node.decorator_list:
                    route_name = RepositoryIndex._call_name(decorator)
                    if any(route_name.casefold().endswith(f".{verb}") for verb in ("get", "post", "put", "patch", "delete", "route")):
                        self._entry("py_route", route_name, decorator)
                        self._relation("route", route_name, qualified, decorator)
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                target = RepositoryIndex._call_name(node.func)
                self._entry("py_call", target or "<dynamic>", node)
                self._relation("call", self.owner, target, node)
                self.generic_visit(node)

        Visitor().visit(tree)
        return entries, relations

    def _html_entries(self, path: str, text: str, digest: str) -> list[IndexEntry]:
        result: list[IndexEntry] = []
        parser = _DOMParser(); parser.feed(text)
        lines = text.splitlines()
        for name, line in parser.entries:
            result.append(IndexEntry(path, "dom", name, line, line, digest, lines[line-1] if line <= len(lines) else ""))
        patterns = {
            "css_rule": r"(?m)([^{}]+)\{([^{}]*)\}",
            "css_variable": r"(?m)(--[\w-]+)\s*:\s*([^;}{]+)",
            "keyframe": r"(?is)@keyframes\s+([\w-]+)\s*\{.*?\}\s*\}",
            "js_function": r"(?m)(?:function\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
            "event_handler": r"(?m)addEventListener\s*\(\s*['\"]([^'\"]+)",
            "responsive": r"(?is)@media\s*([^\{]+)\{",
            "svg_element": r"(?is)<(?:g|path|rect|circle|polygon|line|text)\b[^>]*(?:id|class)=['\"]([^'\"]+)['\"][^>]*>",
            "canvas_setup": r"(?m)(getContext|transferControlToOffscreen|OffscreenCanvas)\s*\(",
            "state_variable": r"(?m)(?:const|let|var)\s+([\w$]*(?:state|speed|velocity|timer|projectile|arrow|catapult|collision)[\w$]*)\s*=",
            "animation_loop": r"(?m)(requestAnimationFrame|setInterval|setTimeout)\s*\(",
            "projectile_system": r"(?im)(function\s+|(?:const|let|var)\s+)([\w$]*(?:arrow|projectile|catapult|fire|launch)[\w$]*)",
            "collision_logic": r"(?im)(function\s+|(?:const|let|var)\s+)([\w$]*(?:collid|intersect|hit|impact)[\w$]*)",
            "asset": r"(?im)(?:src|href)\s*=\s*['\"]([^'\"]+)|url\(\s*['\"]?([^)'\"]+)",
            "accessibility": r"(?im)(aria-[\w-]+|role|tabindex)\s*=\s*['\"]([^'\"]+)",
        }
        for kind, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                name = next((value for value in match.groups() if value), match.group(0)[:60]).strip()
                start = text.count("\n", 0, match.start()) + 1
                end = text.count("\n", 0, match.end()) + 1
                result.append(IndexEntry(path, kind, name, start, end, digest, match.group(0)))
        return result

    @staticmethod
    def _tree_sitter_parser(language: str):
        try:
            from tree_sitter_language_pack import get_parser

            return get_parser(language)
        except (ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def _javascript_entries(
        self,
        path: str,
        text: str,
        digest: str,
    ) -> tuple[list[IndexEntry], list[CodeRelation]]:
        language = "typescript" if Path(path).suffix.casefold() in {".ts", ".tsx"} else "javascript"
        parser = self._tree_sitter_parser(language)
        source = text.encode("utf-8", errors="replace")
        lines = text.splitlines()
        entries: list[IndexEntry] = []
        relations: list[CodeRelation] = []

        def line_text(line: int) -> str:
            return lines[line - 1].strip() if 1 <= line <= len(lines) else ""

        if parser is not None:
            try:
                tree = parser.parse(source)

                def node_text(node: object) -> str:
                    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")

                def name_of(node: object) -> str:
                    name = node.child_by_field_name("name")
                    if name is None:
                        name = node.child_by_field_name("property")
                    return node_text(name).strip() if name is not None else ""

                def add_entry(kind: str, name: str, node: object, confidence: float = 0.95) -> None:
                    start = int(node.start_point[0]) + 1
                    end = int(node.end_point[0]) + 1
                    entries.append(
                        IndexEntry(
                            path, kind, name or "<anonymous>", start, end, digest,
                            self._bounded_text(node_text(node)), confidence, f"tree_sitter:{language}",
                        )
                    )

                def add_relation(kind: str, source_name: str, target: str, node: object, confidence: float = 0.9) -> None:
                    if not target:
                        return
                    line = int(node.start_point[0]) + 1
                    relations.append(
                        CodeRelation(
                            path, kind, source_name, target, line, line_text(line),
                            confidence, f"tree_sitter:{language}",
                        )
                    )

                def walk(node: object, owner: str = "<module>") -> None:
                    node_type = str(node.type)
                    next_owner = owner
                    if node_type in {"class_declaration", "abstract_class_declaration"}:
                        name = name_of(node)
                        add_entry("js_class", name, node)
                        add_relation("owns", owner, name, node)
                        next_owner = name or owner
                    elif node_type in {"function_declaration", "generator_function_declaration"}:
                        name = name_of(node)
                        add_entry("js_function", name, node)
                        add_relation("owns", owner, name, node)
                        next_owner = name or owner
                    elif node_type in {"method_definition", "method_signature"}:
                        name = name_of(node)
                        qualified = f"{owner}.{name}" if owner not in {"", "<module>"} else name
                        add_entry("js_method", qualified, node)
                        add_relation("owns", owner, qualified, node)
                        next_owner = qualified or owner
                    elif node_type in {"import_statement", "export_statement"}:
                        raw = node_text(node)
                        match = re.search(r"\bfrom\s+['\"]([^'\"]+)|\bimport\s*['\"]([^'\"]+)", raw)
                        target = next((item for item in (match.groups() if match else ()) if item), "")
                        if target:
                            add_entry("js_import", target, node)
                            add_relation("import", path, target, node, 0.98)
                    elif node_type in {"call_expression", "new_expression"}:
                        function = node.child_by_field_name("function") or node.child_by_field_name("constructor")
                        target = node_text(function).strip() if function is not None else "<dynamic>"
                        add_entry("js_call", target, node, 0.85 if target != "<dynamic>" else 0.45)
                        add_relation("call", owner, target, node, 0.80 if target != "<dynamic>" else 0.35)
                    elif node_type == "variable_declarator":
                        value = node.child_by_field_name("value")
                        if value is not None and str(value.type) in {"arrow_function", "function_expression"}:
                            name = name_of(node)
                            add_entry("js_function", name, node, 0.92)
                            add_relation("owns", owner, name, node, 0.92)
                            next_owner = name or owner
                    for child in node.children:
                        walk(child, next_owner)

                walk(tree.root_node)
                for match in re.finditer(
                    r"(?m)\b(?:app|router)\.(get|post|put|patch|delete|use)\s*\(\s*['\"]([^'\"]+)",
                    text,
                ):
                    line = text.count("\n", 0, match.start()) + 1
                    target = f"{match.group(1).upper()} {match.group(2)}"
                    entries.append(IndexEntry(path, "js_route", target, line, line, digest, match.group(0), 0.9, f"tree_sitter:{language}+route_adapter"))
                    relations.append(CodeRelation(path, "route", path, target, line, match.group(0), 0.9, f"tree_sitter:{language}+route_adapter"))
                return entries, relations
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                entries.clear()
                relations.clear()

        # Explicitly low-confidence fallback for environments without a parser.
        patterns = {
            "js_class": r"(?m)\bclass\s+([A-Za-z_$][\w$]*)",
            "js_function": r"(?m)(?:\bfunction\s+([A-Za-z_$][\w$]*)|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)",
            "js_import": r"(?m)\bfrom\s+['\"]([^'\"]+)|\bimport\s*['\"]([^'\"]+)",
            "js_call": r"(?m)\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(",
        }
        for kind, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                name = next((item for item in match.groups() if item), match.group(0)).strip()
                start = text.count("\n", 0, match.start()) + 1
                end = text.count("\n", 0, match.end()) + 1
                entries.append(
                    IndexEntry(
                        path, kind, name, start, end, digest, match.group(0),
                        0.45, "regex_fallback:javascript",
                    )
                )
                if kind == "js_import":
                    relations.append(CodeRelation(path, "import", path, name, start, match.group(0), 0.45, "regex_fallback:javascript"))
                elif kind == "js_call":
                    relations.append(CodeRelation(path, "call", "<module>", name, start, match.group(0), 0.35, "regex_fallback:javascript"))
                elif kind in {"js_class", "js_function"}:
                    relations.append(CodeRelation(path, "owns", "<module>", name, start, match.group(0), 0.45, "regex_fallback:javascript"))
        return entries, relations

    def _css_entries(self, path: str, text: str, digest: str) -> list[IndexEntry]:
        result: list[IndexEntry] = []
        for kind, pattern in {
            "css_rule": r"(?m)([^@{}][^{}]+)\{([^{}]*)\}",
            "css_variable": r"(?m)(--[\w-]+)\s*:\s*([^;}{]+)",
            "keyframe": r"(?is)@keyframes\s+([\w-]+)\s*\{",
            "responsive": r"(?is)@media\s*([^\{]+)\{",
        }.items():
            for match in re.finditer(pattern, text):
                name = next((item for item in match.groups() if item), match.group(0)).strip()[:200]
                start = text.count("\n", 0, match.start()) + 1
                end = text.count("\n", 0, match.end()) + 1
                result.append(IndexEntry(path, kind, name, start, end, digest, match.group(0), 0.8, "css_parser"))
        return result

    def search(self, query: str, *, kinds: tuple[str, ...] = ()) -> tuple[IndexEntry, ...]:
        terms = [term.casefold() for term in re.findall(r"[\w-]+", query)]
        matches = []
        for entries in self.entries.values():
            for item in entries:
                if kinds and item.kind not in kinds:
                    continue
                haystack = f"{item.name} {item.text}".casefold()
                score = sum(term in haystack for term in terms)
                if score:
                    matches.append((score, item.start, item))
        return tuple(item for _, _, item in sorted(matches, key=lambda value: (-value[0], value[1])))

    def semantic_search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (),
        limit: int = 50,
    ) -> tuple[IndexEntry, ...]:
        return tuple(hit.entry for hit in self.search_with_scores(query, kinds=kinds, limit=limit) if "semantic" in hit.channels)

    def embedding_search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (),
        limit: int = 50,
    ) -> tuple[IndexEntry, ...]:
        return tuple(hit.entry for hit in self.search_with_scores(query, kinds=kinds, limit=limit) if "embedding" in hit.channels)

    def search_with_scores(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (),
        relation_kinds: tuple[str, ...] = (),
        limit: int = 50,
    ) -> tuple[SearchHit, ...]:
        terms = [term.casefold() for term in re.findall(r"[\w.-]+", query)]
        query_vector = self._vector(query)
        query_embedding = tuple(float(value) for value in self.embedding_provider.embed(query))
        hnsw_scores = (
            self._hnsw.search(query_embedding, limit=max(100, int(limit) * 8))
            if self._ensure_hnsw()
            else {}
        )
        scored: dict[tuple[str, str, str, int], tuple[float, IndexEntry, set[str]]] = {}
        channel_scores: dict[str, dict[tuple[str, str, str, int], float]] = {}

        def add(item: IndexEntry, score: float, channel: str) -> None:
            if kinds and item.kind not in kinds:
                return
            key = self._entry_key(item)
            previous_score, previous_item, previous_channels = scored.get(key, (0.0, item, set()))
            previous_channels.add(channel)
            scored[key] = (previous_score + score, previous_item, previous_channels)
            channel_scores.setdefault(channel, {})[key] = max(
                score, channel_scores.get(channel, {}).get(key, 0.0)
            )

        for entries in self.entries.values():
            for item in entries:
                if kinds and item.kind not in kinds:
                    continue
                haystack = f"{item.kind} {item.name} {item.text}".casefold()
                lexical = sum(1.0 for term in terms if term and term in haystack)
                if lexical:
                    add(item, lexical * 4.0, "lexical")
                semantic = self._cosine(query_vector, self._vectors.get(self._entry_key(item), {}))
                if semantic >= 0.08:
                    add(item, semantic * 12.0, "semantic")
                key = self._entry_key(item)
                embedding = (
                    hnsw_scores.get(key, 0.0)
                    if hnsw_scores
                    else self._dense_cosine(query_embedding, self._embeddings.get(key, ()))
                )
                if embedding >= 0.08:
                    add(item, embedding * 10.0, "embedding")

        for relation in self.search_graph(query, relation_kinds=relation_kinds):
            names = {relation.source, relation.target}
            for entry in self.entries.get(relation.path, ()):
                if entry.name in names or any(part and part in entry.name for name in names for part in name.split(".")):
                    add(entry, 16.0, f"graph:{relation.kind}")

        reciprocal_scores: dict[tuple[str, str, str, int], float] = {
            key: 0.0 for key in scored
        }
        for _channel, values in channel_scores.items():
            ranked = sorted(values, key=lambda key: (-values[key], key))
            for rank, key in enumerate(ranked, 1):
                reciprocal_scores[key] += 1.0 / (60.0 + rank)
        ranked_hits: list[SearchHit] = []
        for key, (raw_score, item, channels) in scored.items():
            fused = reciprocal_scores.get(key, 0.0) * 100.0 + min(raw_score, 100.0) * 0.001
            confidence = min(
                1.0,
                item.confidence
                * (0.70 + 0.10 * min(3, len(channels))),
            )
            ranked_hits.append(
                SearchHit(
                    entry=item,
                    score=fused,
                    channels=tuple(sorted(channels)),
                    confidence=confidence,
                    reason="hybrid RRF: " + ", ".join(sorted(channels)),
                    provenance=tuple(sorted({item.provenance, *(f"retrieval:{channel}" for channel in channels)})),
                )
            )
        return tuple(
            sorted(
                ranked_hits,
                key=lambda hit: (-hit.score, -hit.confidence, hit.entry.path, hit.entry.start),
            )[: max(1, min(limit, 1_000))]
        )

    def search_graph(
        self,
        query: str,
        *,
        relation_kinds: tuple[str, ...] = (),
    ) -> tuple[CodeRelation, ...]:
        terms = [term.casefold() for term in re.findall(r"[\w.-]+", query)]
        matches: list[tuple[int, int, CodeRelation]] = []
        for relations in self.relations.values():
            for item in relations:
                if relation_kinds and item.kind not in relation_kinds:
                    continue
                haystack = f"{item.kind} {item.source} {item.target} {item.text}".casefold()
                score = sum(term in haystack for term in terms)
                if score:
                    matches.append((score, item.line, item))
        return tuple(item for _, _, item in sorted(matches, key=lambda value: (-value[0], value[1])))

    def _all_entries(self) -> tuple[IndexEntry, ...]:
        return tuple(item for entries in self.entries.values() for item in entries)

    def _definition_entries(self) -> tuple[IndexEntry, ...]:
        return tuple(
            item
            for item in self._all_entries()
            if item.kind
            in {
                "py_class",
                "py_method",
                "py_function",
                "js_function",
                "js_class",
                "js_method",
                "dom",
                "css_rule",
                "keyframe",
            }
        )

    def symbol_index(self) -> dict[str, tuple[IndexEntry, ...]]:
        """Return a multi-key symbol index with local and module-qualified names."""

        result: dict[str, list[IndexEntry]] = {}
        for entry in self._definition_entries():
            module = self._module_name(entry.path)
            keys = {
                entry.name,
                f"{entry.path}:{entry.name}",
                f"{module}.{entry.name}" if module else entry.name,
            }
            if "." in entry.name:
                keys.add(entry.name.rsplit(".", 1)[-1])
            for key in keys:
                result.setdefault(key, [])
                if entry not in result[key]:
                    result[key].append(entry)
        return {key: tuple(value) for key, value in result.items()}

    def _module_to_path(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in self.entries:
            suffix = Path(path).suffix.casefold()
            if suffix not in {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
                continue
            module = self._module_name(path)
            result[module] = path
            result[path.rsplit(".", 1)[0]] = path
            result["./" + path.rsplit(".", 1)[0]] = path
        return result

    def _import_aliases(self, path: str) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for relation in self.relations.get(path, ()):
            if relation.kind == "import_alias":
                aliases[relation.source] = relation.target
        return aliases

    def _qualify_local_symbol(self, path: str, name: str) -> str:
        if name in {"<module>", path}:
            return self._module_name(path)
        module = self._module_name(path)
        return f"{module}.{name}" if module and not name.startswith(f"{module}.") else name

    def resolve_symbol(self, path: str, symbol: str) -> tuple[IndexEntry, ...]:
        """Resolve a local/imported symbol reference to indexed definitions when possible."""

        symbol = str(symbol or "").strip()
        if not symbol:
            return ()
        aliases = self._import_aliases(path)
        parts = symbol.split(".")
        base = parts[0]
        candidates: list[str] = []
        if base in aliases:
            alias_target = aliases[base]
            candidates.append(".".join((alias_target, *parts[1:])))
            if len(parts) > 2 and parts[1] == parts[0]:
                candidates.append(".".join((alias_target, *parts[2:])))
        candidates.append(self._qualify_local_symbol(path, symbol))
        candidates.append(symbol)
        symbol_index = self.symbol_index()
        matches: list[IndexEntry] = []
        for candidate in candidates:
            direct = symbol_index.get(candidate, ())
            for item in direct:
                if item not in matches:
                    matches.append(item)
            suffix = f".{candidate}"
            for key, entries in symbol_index.items():
                if key.endswith(suffix) or candidate.endswith(f".{key}"):
                    for item in entries:
                        if item not in matches:
                            matches.append(item)
        return tuple(matches)

    def resolved_dependency_graph(self) -> dict[str, tuple[str, ...]]:
        modules = self._module_to_path()
        graph: dict[str, list[str]] = {}
        for path, relations in self.relations.items():
            graph.setdefault(path, [])
            for relation in relations:
                if relation.kind != "import":
                    continue
                target = relation.target
                if target.startswith("."):
                    base = PurePosixPath(path).parent
                    target = posixpath.normpath((base / target).as_posix())
                prefixes = [target]
                while "." in target:
                    target = target.rsplit(".", 1)[0]
                    prefixes.append(target)
                for module in prefixes:
                    target_path = modules.get(module)
                    if target_path and target_path not in graph[path]:
                        graph[path].append(target_path)
                        break
        return {key: tuple(value) for key, value in graph.items()}

    def resolved_call_graph(self) -> dict[str, tuple[str, ...]]:
        graph: dict[str, list[str]] = {}
        for path, relations in self.relations.items():
            for relation in relations:
                if relation.kind != "call":
                    continue
                source = self._qualify_local_symbol(path, relation.source)
                graph.setdefault(source, [])
                resolved = self.resolve_symbol(path, relation.target)
                targets = [
                    self._qualify_local_symbol(item.path, item.name)
                    for item in resolved
                ] or [relation.target]
                for target in targets:
                    if target and target not in graph[source]:
                        graph[source].append(target)
        return {key: tuple(value) for key, value in graph.items()}

    def callers_of(self, symbol: str) -> tuple[str, ...]:
        wanted = str(symbol or "").casefold()
        callers: list[str] = []
        for source, targets in self.resolved_call_graph().items():
            if any(target.casefold() == wanted or target.casefold().endswith(f".{wanted}") for target in targets):
                callers.append(source)
        return tuple(dict.fromkeys(callers))

    def callees_of(self, symbol: str) -> tuple[str, ...]:
        wanted = str(symbol or "").casefold()
        for source, targets in self.resolved_call_graph().items():
            if source.casefold() == wanted or source.casefold().endswith(f".{wanted}"):
                return targets
        return ()

    def _entry_for_qualified_symbol(self, symbol: str) -> IndexEntry | None:
        wanted = str(symbol or "")
        for entries in self.symbol_index().values():
            for item in entries:
                qualified = self._qualify_local_symbol(item.path, item.name)
                if qualified == wanted or qualified.endswith(f".{wanted}") or wanted.endswith(f".{qualified}"):
                    return item
        return None

    @staticmethod
    def _entry_size(item: IndexEntry) -> int:
        return len(item.path) + len(item.kind) + len(item.name) + len(item.text) + 64

    def context_slice(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (),
        relation_kinds: tuple[str, ...] = (),
        max_entries: int = 20,
        budget_chars: int = 20_000,
        include_graph_neighborhood: bool = True,
    ) -> RepositoryContextSlice:
        hits = self.search_with_scores(
            query,
            kinds=kinds,
            relation_kinds=relation_kinds,
            limit=max(1, max_entries),
        )
        selected: list[IndexEntry] = []
        selected_keys: set[tuple[str, str, str, int]] = set()
        used = 0
        budget = max(500, int(budget_chars))

        def add_entry(entry: IndexEntry) -> bool:
            nonlocal used
            key = self._entry_key(entry)
            if key in selected_keys:
                return True
            cost = self._entry_size(entry)
            if selected and (used + cost > budget or len(selected) >= max(1, int(max_entries))):
                return False
            selected_keys.add(key)
            selected.append(entry)
            used += cost
            return True

        for hit in hits:
            if set(hit.channels) == {"embedding"} and hit.score < 5.0:
                continue
            add_entry(hit.entry)
        callers: dict[str, tuple[str, ...]] = {}
        callees: dict[str, tuple[str, ...]] = {}
        if include_graph_neighborhood:
            for entry in tuple(selected):
                qualified = self._qualify_local_symbol(entry.path, entry.name)
                caller_values = self.callers_of(qualified)
                callee_values = self.callees_of(qualified)
                if caller_values:
                    callers[qualified] = caller_values
                if callee_values:
                    callees[qualified] = callee_values
                for symbol in (*caller_values, *callee_values):
                    target_entry = self._entry_for_qualified_symbol(symbol)
                    if target_entry is not None:
                        add_entry(target_entry)
        selected_paths = {entry.path for entry in selected}
        dependency_graph = self.resolved_dependency_graph()
        dependencies = {
            path: targets
            for path, targets in dependency_graph.items()
            if path in selected_paths or any(target in selected_paths for target in targets)
        }
        relations = tuple(
            relation
            for path, items in self.relations.items()
            if path in selected_paths
            for relation in items
            if not relation_kinds or relation.kind in relation_kinds
        )
        return RepositoryContextSlice(
            query=query,
            entries=tuple(selected),
            relations=relations,
            callers=callers,
            callees=callees,
            dependencies=dependencies,
            omitted_entries=max(0, len(hits) - len(selected)),
            size_chars=used,
        )

    def hybrid_search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (),
        relation_kinds: tuple[str, ...] = (),
    ) -> tuple[IndexEntry, ...]:
        return tuple(
            hit.entry
            for hit in self.search_with_scores(
                query,
                kinds=kinds,
                relation_kinds=relation_kinds,
                limit=1_000,
            )
        )

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        graph: dict[str, list[str]] = {}
        for path, relations in self.relations.items():
            imports = [item.target for item in relations if item.kind == "import"]
            graph[path] = tuple(dict.fromkeys(imports))
        return graph

    def call_graph(self) -> dict[str, tuple[str, ...]]:
        graph: dict[str, list[str]] = {}
        for relations in self.relations.values():
            for item in relations:
                if item.kind != "call":
                    continue
                graph.setdefault(item.source, [])
                if item.target not in graph[item.source]:
                    graph[item.source].append(item.target)
        return {key: tuple(value) for key, value in graph.items()}

    def ownership_graph(self) -> dict[str, tuple[str, ...]]:
        graph: dict[str, list[str]] = {}
        for relations in self.relations.values():
            for item in relations:
                if item.kind != "owns":
                    continue
                graph.setdefault(item.source, [])
                if item.target not in graph[item.source]:
                    graph[item.source].append(item.target)
        return {key: tuple(value) for key, value in graph.items()}

    def ownership_records(self) -> dict[str, tuple[tuple[str, str, float], ...]]:
        """Resolve declared owners and bounded Git-history contributors."""

        candidates = (
            self.workspace / ".github" / "CODEOWNERS",
            self.workspace / "CODEOWNERS",
            self.workspace / "docs" / "CODEOWNERS",
        )
        rules: list[tuple[str, tuple[str, ...]]] = []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for raw in lines:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                pattern = parts[0].lstrip("/")
                rules.append((pattern, tuple(parts[1:])))
            break
        records: dict[str, list[tuple[str, str, float]]] = {
            path: [] for path in self.entries
        }
        for path in self.entries:
            owners: tuple[str, ...] = ()
            for pattern, values in rules:
                normalized = pattern
                if normalized.endswith("/"):
                    normalized += "**"
                if fnmatch.fnmatchcase(path, normalized) or PurePosixPath(path).match(normalized):
                    owners = values  # Last matching CODEOWNERS rule wins.
            records[path].extend((owner, "CODEOWNERS", 1.0) for owner in owners)
        try:
            completed = subprocess.run(
                ["git", "log", "-n", "500", "--format=@@%ae", "--name-only", "--no-renames"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            current_author = ""
            counts: dict[str, Counter[str]] = {}
            for raw in completed.stdout.splitlines():
                line = raw.strip()
                if line.startswith("@@"):
                    current_author = line[2:].strip()
                    continue
                path = line.replace("\\", "/")
                if current_author and path in self.entries:
                    counts.setdefault(path, Counter())[current_author] += 1
            for path, authors in counts.items():
                total = sum(authors.values()) or 1
                declared = {owner for owner, _source, _confidence in records[path]}
                for author, count in authors.most_common(3):
                    if author in declared:
                        continue
                    records[path].append((author, "git_history", min(0.85, 0.4 + 0.45 * count / total)))
        return {path: tuple(values) for path, values in records.items()}

    def file_ownership_graph(self) -> dict[str, tuple[str, ...]]:
        return {
            path: tuple(owner for owner, _source, _confidence in values)
            for path, values in self.ownership_records().items()
        }

    def semantic_map(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        ownership = self.ownership_records()
        dependencies = self.resolved_dependency_graph()
        for path, entries in self.entries.items():
            by_kind: dict[str, list[str]] = {}
            for entry in entries:
                if entry.kind == "file":
                    continue
                by_kind.setdefault(entry.kind, [])
                if entry.name not in by_kind[entry.kind]:
                    by_kind[entry.kind].append(entry.name)
            result[path] = {
                "symbols": {kind: tuple(names) for kind, names in by_kind.items()},
                "dependencies": dependencies.get(path, ()),
                "relations": tuple(self.relations.get(path, ())),
                "owners": tuple(owner for owner, _source, _confidence in ownership.get(path, ())),
                "ownership": tuple(
                    {"owner": owner, "source": source, "confidence": confidence}
                    for owner, source, confidence in ownership.get(path, ())
                ),
                "file_hash": entries[0].file_hash if entries else "",
                "parser_confidence": min((entry.confidence for entry in entries), default=0.0),
                "provenance": tuple(sorted({entry.provenance for entry in entries})),
            }
        return result

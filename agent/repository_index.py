"""Deterministic incremental repository and mixed-HTML component index."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class IndexEntry:
    path: str
    kind: str
    name: str
    start: int
    end: int
    file_hash: str
    text: str


@dataclass(frozen=True, slots=True)
class CodeRelation:
    path: str
    kind: str
    source: str
    target: str
    line: int
    text: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    entry: IndexEntry
    score: float
    channels: tuple[str, ...]


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


class _DOMParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, int]] = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        name = values.get("id") or values.get("class") or values.get("aria-label") or tag
        self.entries.append((str(name), self.getpos()[0]))


class RepositoryIndex:
    _INDEXABLE_SUFFIXES = {".py", ".html", ".htm"}
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
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self._embeddings: dict[tuple[str, str, str, int], tuple[float, ...]] = {}
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
        path = (self.workspace / relative_path).resolve()
        path.relative_to(self.workspace)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8", errors="replace")
        stat = path.stat()
        found: list[IndexEntry] = [IndexEntry(relative_path, "file", path.name, 1, text.count("\n") + 1, digest, text)]
        relations: list[CodeRelation] = []
        if path.suffix.lower() == ".py":
            py_entries, py_relations = self._python_entries(relative_path, text, digest)
            found.extend(py_entries)
            relations.extend(py_relations)
        if path.suffix.lower() in {".html", ".htm"}:
            found.extend(self._html_entries(relative_path, text, digest))
        for key in tuple(key for key in self._vectors if key[0] == relative_path):
            self._vectors.pop(key, None)
            self._embeddings.pop(key, None)
        self.entries[relative_path] = tuple(found)
        self.relations[relative_path] = tuple(relations)
        for item in found:
            embedding_text = f"{item.path} {self._module_name(item.path)} {item.kind} {item.name} {item.text}"
            self._vectors[self._entry_key(item)] = self._vector(
                embedding_text
            )
            self._embeddings[self._entry_key(item)] = tuple(float(value) for value in self.embedding_provider.embed(embedding_text))
        self._file_snapshots[relative_path] = FileSnapshot(
            relative_path,
            int(stat.st_size),
            int(stat.st_mtime_ns),
            digest,
        )
        if self.cache_path is not None and not self._cache_save_suspended:
            self.save_cache()
        return self.entries[relative_path]

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
        return f"{self.embedding_provider.__class__.__module__}.{self.embedding_provider.__class__.__qualname__}:{dimensions}"

    def save_cache(self) -> bool:
        if self.cache_path is None:
            return False
        payload = {
            "version": 1,
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
        if not isinstance(payload, Mapping) or payload.get("version") != 1:
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
        self.last_update_stats = {
            "seen": 0,
            "updated": 0,
            "reused": 0,
            "removed": 0,
            "loaded": len(self.entries),
        }
        return True

    def _remove_path(self, relative_path: str) -> None:
        self.entries.pop(relative_path, None)
        self.relations.pop(relative_path, None)
        self._file_snapshots.pop(relative_path, None)
        for key in tuple(key for key in self._vectors if key[0] == relative_path):
            self._vectors.pop(key, None)
            self._embeddings.pop(key, None)

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
                    collected.extend(self.update(relative))
                    stats["updated"] += 1
                indexed += 1
                if max_files is not None and indexed >= max(1, int(max_files)):
                    break
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
        return tuple(collected)

    @staticmethod
    def _entry_key(item: IndexEntry) -> tuple[str, str, str, int]:
        return (item.path, item.kind, item.name, item.start)

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
                        "\n".join(lines[start - 1:end]) if start <= len(lines) else "",
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
        scored: dict[tuple[str, str, str, int], tuple[float, IndexEntry, set[str]]] = {}

        def add(item: IndexEntry, score: float, channel: str) -> None:
            if kinds and item.kind not in kinds:
                return
            key = self._entry_key(item)
            previous_score, previous_item, previous_channels = scored.get(key, (0.0, item, set()))
            previous_channels.add(channel)
            scored[key] = (previous_score + score, previous_item, previous_channels)

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
                embedding = self._dense_cosine(query_embedding, self._embeddings.get(self._entry_key(item), ()))
                if embedding >= 0.08:
                    add(item, embedding * 10.0, "embedding")

        for relation in self.search_graph(query, relation_kinds=relation_kinds):
            names = {relation.source, relation.target}
            for entry in self.entries.get(relation.path, ()):
                if entry.name in names or any(part and part in entry.name for name in names for part in name.split(".")):
                    add(entry, 16.0, f"graph:{relation.kind}")

        return tuple(
            SearchHit(entry=item, score=score, channels=tuple(sorted(channels)))
            for score, item, channels in sorted(
                scored.values(),
                key=lambda value: (-value[0], value[1].path, value[1].start),
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
        return {
            self._module_name(path): path
            for path in self.entries
            if path.casefold().endswith(".py")
        }

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

    def semantic_map(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
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
                "dependencies": self.dependency_graph().get(path, ()),
                "relations": tuple(self.relations.get(path, ())),
            }
        return result

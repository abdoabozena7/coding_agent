"""Harness-owned materialized component packages for Ultra specialists.

Specialists may propose files, but only this store validates and writes them.
The shared final workspace remains owned by parent/final assemblers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Iterable, Mapping, Sequence

from .tools.web_preview import _verify


MAX_COMPONENT_FILES = 48
MAX_COMPONENT_FILE_BYTES = 512_000
MAX_COMPONENT_TOTAL_BYTES = 2_000_000


class ComponentArtifactError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return normalized[:96] or "component"


def _relative_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or ":" in pure.parts[0]
    ):
        raise ComponentArtifactError(f"invalid component-relative path: {value!r}")
    return pure.as_posix()


def _role_for(path: str, supplied: str = "") -> str:
    role = str(supplied or "").strip().casefold()
    if role in {"implementation", "preview", "test", "asset", "manifest"}:
        return role
    lowered = path.casefold()
    if lowered.endswith((".test.js", ".spec.js", ".test.ts", ".spec.ts", "_test.py")):
        return "test"
    if lowered.endswith((".html", ".htm")) and "preview" in lowered:
        return "preview"
    return "implementation"


@dataclass(frozen=True, slots=True)
class ComponentFileV2:
    path: str
    content_hash: str
    size: int
    media_type: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InterfaceContractV1:
    node_id: str
    exports: tuple[str, ...]
    imports: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    integration_points: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.exports:
            raise ComponentArtifactError("interface contract requires node_id and exports")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MaterializedComponentPackageV2:
    id: str
    run_id: str
    node_id: str
    revision: int
    root: str
    files: tuple[ComponentFileV2, ...]
    interface: InterfaceContractV1
    preview_entrypoint: str
    dependencies: tuple[str, ...]
    evidence: tuple[Mapping[str, Any], ...]
    quality: Mapping[str, Any]
    parent_package_ids: tuple[str, ...] = ()
    schema_name: str = "MaterializedComponentPackageV2"
    version: int = 2

    @property
    def content_hash(self) -> str:
        payload = {
            "node_id": self.node_id,
            "revision": self.revision,
            "files": [item.to_dict() for item in self.files],
            "interface": self.interface.to_dict(),
            "dependencies": list(self.dependencies),
            "parents": list(self.parent_package_ids),
        }
        return _sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        value = {
            **asdict(self),
            "content_hash": self.content_hash,
            "files": [item.to_dict() for item in self.files],
            "interface": self.interface.to_dict(),
        }
        if include_content:
            root = Path(self.root)
            value["file_contents"] = {
                item.path: (root / item.path).read_text(encoding="utf-8")
                for item in self.files
                if item.role in {"implementation", "preview", "test"}
                and (root / item.path).is_file()
                and item.size <= MAX_COMPONENT_FILE_BYTES
            }
        return value


@dataclass(frozen=True, slots=True)
class PackageConsumptionEvidenceV1:
    assembler_node_id: str
    package_id: str
    consumed_file_hashes: tuple[str, ...]
    target_paths: tuple[str, ...]
    passed: bool
    findings: tuple[str, ...] = ()
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return


class ComponentArtifactStore:
    """Materialize and verify component-only output outside final output paths."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.staging_root = self.workspace / "run-artifacts" / ".ultra-staging"

    def package_root(self, run_id: str, node_id: str, revision: int) -> Path:
        return (
            self.staging_root
            / _safe_segment(run_id)
            / _safe_segment(node_id)
            / f"r{max(1, int(revision))}"
        )

    def draft_root(self, run_id: str, node_id: str) -> Path:
        return (
            self.staging_root
            / _safe_segment(run_id)
            / _safe_segment(node_id)
            / "drafts"
        )

    def stage_draft_file(
        self,
        *,
        run_id: str,
        node_id: str,
        path: str,
        content: str,
        role: str = "",
    ) -> Mapping[str, Any]:
        relative = _relative_path(path)
        data = str(content).encode("utf-8")
        if len(data) > MAX_COMPONENT_FILE_BYTES:
            raise ComponentArtifactError(f"component file {relative} is too large")
        root = self.draft_root(run_id, node_id)
        target = (root / relative).resolve()
        target.relative_to(root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        return {
            "path": relative,
            "content_hash": _sha256(data),
            "size": len(data),
            "role": _role_for(relative, role),
        }

    def draft_files(self, *, run_id: str, node_id: str) -> tuple[Mapping[str, Any], ...]:
        root = self.draft_root(run_id, node_id)
        if not root.is_dir():
            return ()
        values: list[Mapping[str, Any]] = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            values.append(
                {
                    "path": relative,
                    "content": path.read_text(encoding="utf-8"),
                    "role": _role_for(relative),
                }
            )
        return tuple(values)

    @staticmethod
    def _raw_files(component: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        implementation = component.get("implementation")
        implementation = implementation if isinstance(implementation, Mapping) else {}
        raw = implementation.get("files", component.get("files", ()))
        values: list[Mapping[str, Any]] = []
        if isinstance(raw, Mapping):
            values.extend(
                {"path": str(path), "content": content, "role": "implementation"}
                for path, content in raw.items()
            )
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(dict(item) for item in raw if isinstance(item, Mapping))
        preview = component.get("preview")
        if isinstance(preview, Mapping):
            preview_files = preview.get("files", ())
            if isinstance(preview_files, Mapping):
                values.extend(
                    {"path": str(path), "content": content, "role": "preview"}
                    for path, content in preview_files.items()
                )
            elif isinstance(preview_files, Sequence) and not isinstance(
                preview_files, (str, bytes)
            ):
                values.extend(
                    {**dict(item), "role": str(item.get("role") or "preview")}
                    for item in preview_files
                    if isinstance(item, Mapping)
                )
        tests = component.get("tests", ())
        if isinstance(tests, Sequence) and not isinstance(tests, (str, bytes)):
            values.extend(
                {**dict(item), "role": str(item.get("role") or "test")}
                for item in tests
                if isinstance(item, Mapping) and "content" in item
            )
        deduplicated: dict[str, Mapping[str, Any]] = {}
        for item in values:
            path = _relative_path(str(item.get("path") or ""))
            if path in deduplicated:
                raise ComponentArtifactError(f"duplicate component file: {path}")
            deduplicated[path] = {**item, "path": path}
        return tuple(deduplicated.values())

    @staticmethod
    def _interface(node_id: str, component: Mapping[str, Any]) -> InterfaceContractV1:
        raw = component.get("interface")
        raw = raw if isinstance(raw, Mapping) else {}

        def strings(value: Any) -> tuple[str, ...]:
            if isinstance(value, str):
                value = (value,)
            if not isinstance(value, Iterable):
                return ()
            return tuple(
                dict.fromkeys(
                    str(item).strip() for item in value if str(item).strip()
                )
            )

        exports = strings(raw.get("exports", raw.get("owned_interfaces")))
        if not exports:
            raise ComponentArtifactError(
                f"component {node_id} must declare at least one concrete export"
            )
        return InterfaceContractV1(
            node_id=node_id,
            exports=exports,
            imports=strings(raw.get("imports")),
            invariants=strings(raw.get("invariants")),
            integration_points=strings(
                raw.get("integration_points", raw.get("integration_guidance"))
            ),
        )

    def materialize(
        self,
        *,
        run_id: str,
        node_id: str,
        component: Mapping[str, Any],
        revision: int = 1,
        dependencies: Iterable[str] = (),
        evidence: Iterable[Mapping[str, Any]] = (),
        quality: Mapping[str, Any] | None = None,
        parent_package_ids: Iterable[str] = (),
    ) -> MaterializedComponentPackageV2:
        raw_files = self._raw_files(component)
        if not raw_files:
            raise ComponentArtifactError(
                f"component {node_id} returned no materialized files"
            )
        if len(raw_files) > MAX_COMPONENT_FILES:
            raise ComponentArtifactError(
                f"component {node_id} exceeds {MAX_COMPONENT_FILES} files"
            )
        prepared: list[tuple[str, bytes, str]] = []
        total = 0
        for item in raw_files:
            path = _relative_path(str(item["path"]))
            content = item.get("content")
            if not isinstance(content, str):
                raise ComponentArtifactError(f"component file {path} requires text content")
            data = content.encode("utf-8")
            if len(data) > MAX_COMPONENT_FILE_BYTES:
                raise ComponentArtifactError(f"component file {path} is too large")
            total += len(data)
            prepared.append((path, data, _role_for(path, str(item.get("role") or ""))))
        if total > MAX_COMPONENT_TOTAL_BYTES:
            raise ComponentArtifactError(
                f"component {node_id} exceeds the materialized package byte budget"
            )
        preview = component.get("preview")
        preview = preview if isinstance(preview, Mapping) else {}
        preview_entrypoint = _relative_path(
            str(preview.get("entrypoint") or "preview.html")
        )
        available = {path for path, _data, _role in prepared}
        if preview_entrypoint not in available or not preview_entrypoint.casefold().endswith(
            (".html", ".htm")
        ):
            raise ComponentArtifactError(
                f"component {node_id} requires a materialized HTML preview entrypoint"
            )
        interface = self._interface(node_id, component)
        root = self.package_root(run_id, node_id, revision)
        root.mkdir(parents=True, exist_ok=True)
        records: list[ComponentFileV2] = []
        for relative, data, role in prepared:
            target = (root / relative).resolve()
            target.relative_to(root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
            temporary.write_bytes(data)
            os.replace(temporary, target)
            records.append(
                ComponentFileV2(
                    path=relative,
                    content_hash=_sha256(data),
                    size=len(data),
                    media_type=mimetypes.guess_type(relative)[0] or "text/plain",
                    role=role,
                )
            )
        package = MaterializedComponentPackageV2(
            id=f"mpkg-{_sha256(f'{run_id}:{node_id}:{revision}'.encode())[:20]}",
            run_id=run_id,
            node_id=node_id,
            revision=max(1, int(revision)),
            root=str(root),
            files=tuple(records),
            interface=interface,
            preview_entrypoint=preview_entrypoint,
            dependencies=tuple(dict.fromkeys(str(item) for item in dependencies)),
            evidence=tuple(dict(item) for item in evidence),
            quality=dict(quality or {}),
            parent_package_ids=tuple(
                dict.fromkeys(str(item) for item in parent_package_ids if str(item))
            ),
        )
        (root / "component-package.json").write_text(
            json.dumps(package.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return package

    def verify_preview(
        self,
        package: MaterializedComponentPackageV2,
        *,
        settle_ms: int = 1_200,
    ) -> dict[str, Any]:
        root = Path(package.root)
        handler = partial(_QuietHandler, directory=str(root))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = int(server.server_address[1])
        screenshot = root / "evidence" / "preview.png"
        try:
            result = _verify(
                f"http://127.0.0.1:{port}/{package.preview_entrypoint}",
                screenshot,
                settle_ms,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        return {
            **result,
            "package_id": package.id,
            "node_id": package.node_id,
            "preview_entrypoint": package.preview_entrypoint,
        }

    @staticmethod
    def _search_files(paths: Iterable[str | Path]) -> tuple[Path, ...]:
        files: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(item for item in path.rglob("*") if item.is_file())
        return tuple(files)

    def verify_consumption(
        self,
        *,
        assembler_node_id: str,
        packages: Iterable[MaterializedComponentPackageV2],
        target_paths: Iterable[str | Path],
    ) -> tuple[PackageConsumptionEvidenceV1, ...]:
        targets = self._search_files(target_paths)
        target_blobs: list[tuple[Path, bytes]] = []
        for target in targets:
            try:
                target_blobs.append((target, target.read_bytes()))
            except OSError:
                continue
        evidence: list[PackageConsumptionEvidenceV1] = []
        for package in packages:
            consumed: list[str] = []
            findings: list[str] = []
            implementation_files = [
                item for item in package.files if item.role == "implementation"
            ]
            for item in implementation_files:
                source = Path(package.root) / item.path
                try:
                    blob = source.read_bytes()
                except OSError:
                    findings.append(f"missing staged component file {item.path}")
                    continue
                exact_file = any(_sha256(candidate) == item.content_hash for _path, candidate in target_blobs)
                inlined = bool(blob.strip()) and any(blob.strip() in candidate for _path, candidate in target_blobs)
                if exact_file or inlined:
                    consumed.append(item.content_hash)
                else:
                    findings.append(
                        f"assembler output did not consume materialized file {item.path}"
                    )
            passed = bool(implementation_files) and len(consumed) == len(
                implementation_files
            )
            evidence.append(
                PackageConsumptionEvidenceV1(
                    assembler_node_id=assembler_node_id,
                    package_id=package.id,
                    consumed_file_hashes=tuple(consumed),
                    target_paths=tuple(str(item) for item in targets),
                    passed=passed,
                    findings=tuple(findings),
                )
            )
        return tuple(evidence)


__all__ = [
    "ComponentArtifactError",
    "ComponentArtifactStore",
    "ComponentFileV2",
    "InterfaceContractV1",
    "MaterializedComponentPackageV2",
    "PackageConsumptionEvidenceV1",
]

"""Repository-evidenced verification mechanics.

Discovery is intentionally mechanical.  These plugins report commands that the
repository itself advertises; they never decide what the product is supposed
to do or whether a command is sufficient for a particular acceptance
criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VerifierPluginV1:
    name: str
    command: tuple[str, ...]
    evidence_path: str
    authority: str
    kind: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value


def _relative(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def discover_verifier_plugins(workspace: Path) -> tuple[VerifierPluginV1, ...]:
    """Return only verifier commands evidenced by repository configuration."""

    root = workspace.resolve()
    discovered: list[VerifierPluginV1] = []

    def add(
        name: str,
        command: tuple[str, ...],
        evidence: Path,
        authority: str,
        kind: str,
    ) -> None:
        if evidence.is_file():
            discovered.append(
                VerifierPluginV1(
                    name=name,
                    command=command,
                    evidence_path=_relative(root, evidence),
                    authority=authority,
                    kind=kind,
                )
            )

    for config in ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
        path = root / config
        if path.is_file():
            add("pytest", ("python", "-m", "pytest"), path, "configured Python test runner", "test")
            break
    for config, name, command, authority, kind in (
        ("mypy.ini", "mypy", ("python", "-m", "mypy", "."), "configured Python type checker", "typecheck"),
        ("pyrightconfig.json", "pyright", ("pyright",), "configured Python type checker", "typecheck"),
        ("ruff.toml", "ruff", ("python", "-m", "ruff", "check", "."), "configured Python linter", "lint"),
        ("Cargo.toml", "cargo-test", ("cargo", "test"), "Cargo project manifest", "test"),
        ("go.mod", "go-test", ("go", "test", "./..."), "Go module manifest", "test"),
        ("pom.xml", "maven-test", ("mvn", "test"), "Maven project manifest", "test"),
        ("build.gradle", "gradle-test", ("gradle", "test"), "Gradle build configuration", "test"),
        ("tsconfig.json", "typescript", ("npx", "tsc", "--noEmit"), "TypeScript project configuration", "typecheck"),
        ("playwright.config.ts", "playwright", ("npx", "playwright", "test"), "Playwright test configuration", "browser"),
        ("playwright.config.js", "playwright", ("npx", "playwright", "test"), "Playwright test configuration", "browser"),
        ("cypress.config.ts", "cypress", ("npx", "cypress", "run"), "Cypress test configuration", "browser"),
        ("cypress.config.js", "cypress", ("npx", "cypress", "run"), "Cypress test configuration", "browser"),
    ):
        add(name, command, root / config, authority, kind)

    package_json = root / "package.json"
    if package_json.is_file():
        try:
            parsed = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            parsed = {}
        scripts = parsed.get("scripts", {}) if isinstance(parsed, dict) else {}
        if isinstance(scripts, dict):
            for script in ("test", "typecheck", "lint", "build", "check", "e2e"):
                if str(scripts.get(script) or "").strip():
                    add(
                        f"package-{script}",
                        ("npm", "run", script),
                        package_json,
                        f"package.json scripts.{script}",
                        script,
                    )

    unique: dict[tuple[str, tuple[str, ...], str], VerifierPluginV1] = {}
    for item in discovered:
        unique[(item.name, item.command, item.evidence_path)] = item
    return tuple(unique.values())


__all__ = ["VerifierPluginV1", "discover_verifier_plugins"]

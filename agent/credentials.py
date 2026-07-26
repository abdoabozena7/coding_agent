"""Small, explicit credential settings for provider discovery.

Secrets are written only after the user invokes the dedicated settings command.
Callers receive the environment-variable name, never the secret value.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping


PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


def normalize_api_key_provider(provider: str) -> str:
    value = str(provider or "").strip().casefold()
    value = {"google-ai": "gemini", "google_ai": "gemini"}.get(value, value)
    if value not in PROVIDER_API_KEY_ENV:
        supported = ", ".join(("openai", "gemini", "ollama"))
        raise ValueError(f"provider must be one of: {supported}")
    return value


def api_key_status(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, bool]:
    source = os.environ if environ is None else environ
    return {
        provider: bool(str(source.get(variable) or "").strip())
        for provider, variable in PROVIDER_API_KEY_ENV.items()
        if provider != "google"
    }


def save_provider_api_key(
    env_path: str | os.PathLike[str],
    provider: str,
    secret: str,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Atomically save one explicitly supplied provider key in the app .env."""

    normalized = normalize_api_key_provider(provider)
    variable = PROVIDER_API_KEY_ENV[normalized]
    value = str(secret or "").strip()
    if not value:
        raise ValueError("API key cannot be empty")
    if any(character in value for character in ("\r", "\n", "\0")):
        raise ValueError("API key must be a single line")

    target = Path(env_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError as exc:
        raise ValueError(f"could not read credential settings: {exc}") from exc

    pattern = re.compile(rf"^\s*{re.escape(variable)}\s*=.*$", re.MULTILINE)
    replacement = f"{variable}={value}"
    updated = (
        pattern.sub(replacement, existing, count=1)
        if pattern.search(existing)
        else existing.rstrip("\r\n") + ("\n" if existing else "") + replacement + "\n"
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="\n")
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"could not save credential settings: {exc}") from exc

    destination = os.environ if environ is None else environ
    destination[variable] = value
    return variable

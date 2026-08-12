"""Open a visible, persistent Playwright browser session."""

import socket
import time
from urllib.parse import urlsplit

from . import browser_session

REQUIRES_APPROVAL = True
SCHEMA = {
    "type": "function",
    "function": {
        "name": "browser_open",
        "description": (
            "Open a Playwright-controlled browser that remains available for later "
            "inspection, interaction, and screenshots. Supply exactly one URL or "
            "workspace-relative HTML path. The browser is visible by default."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "maxLength": 4096},
                "path": {"type": "string", "maxLength": 4096},
                "visible": {"type": "boolean", "default": True},
                "width": {"type": "integer", "minimum": 320, "maximum": 3840, "default": 1440},
                "height": {"type": "integer", "minimum": 240, "maximum": 2160, "default": 900},
                "settle_ms": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 700},
            },
            "additionalProperties": False,
        },
    },
}
RESULT_CONTRACT = {
    "format": "json",
    "fields": [
        "browser_session_id", "status", "browser_opened", "url", "title",
        "http_status", "viewport", "browser_engine", "browser_launch_fallbacks",
        "text", "interaction_targets",
        "console_errors", "page_errors", "network_errors",
    ],
}


def _loopback_preflight(url: str, *, timeout_seconds: float = 2.0) -> str:
    """Avoid opening Chromium against a local port that is not listening."""

    parsed = urlsplit(str(url or "").strip())
    host = (parsed.hostname or "").casefold()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            with socket.create_connection((host, port), timeout=0.35):
                return ""
        except OSError:
            if time.monotonic() >= deadline:
                return (
                    f"loopback URL {url!r} is not accepting connections; "
                    "start the correct project component and verify its port or URL readiness first"
                )
            time.sleep(0.10)


def run(
    url: str = "",
    path: str = "",
    visible: bool = True,
    width: int = 1440,
    height: int = 900,
    settle_ms: int = 700,
) -> str:
    try:
        if url:
            preflight = _loopback_preflight(url)
            if preflight:
                return "Error: browser_open preflight failed: " + preflight
        capability = browser_session.web_preview.browser_capability()
        if not capability.get("playwright"):
            return "Error: Playwright is not installed in the active Python environment; install the declared project dependencies or launch GA3BAD from its virtual environment"
        if not capability.get("available"):
            return "Error: Chrome, Edge, or Chromium is not available for Playwright browser control"
        return browser_session.open_browser(
            url=url,
            path=path,
            visible=visible,
            width=width,
            height=height,
            settle_ms=settle_ms,
        )
    except (RuntimeError, ValueError) as exc:
        return f"Error: browser_open failed: {exc}"

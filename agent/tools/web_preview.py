"""Secure loopback HTML previews with optional Playwright verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import atexit
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import subprocess
import tempfile
from threading import RLock, Thread
import time
from typing import Any
from urllib.parse import quote, unquote, urlsplit
from urllib.request import urlopen

from ._security import (
    get_workspace,
    is_sensitive_path,
    resolve_workspace_path,
    safe_os_error,
    sensitive_content_reason,
)
from .run_bash import _terminate


@dataclass
class Preview:
    id: str
    token: str
    entry_path: str
    server: ThreadingHTTPServer
    thread: Thread
    url: str
    browser_process: subprocess.Popen[bytes] | None = None
    profile_path: Path | None = None
    verification: dict[str, Any] = field(default_factory=dict)


_LOCK = RLock()
_PREVIEWS: dict[tuple[str, str], Preview] = {}


class _Handler(BaseHTTPRequestHandler):
    server_version = "GA3BADPreview/1"

    def do_GET(self) -> None:  # noqa: N802
        owner: Path = self.server.workspace  # type: ignore[attr-defined]
        token: str = self.server.preview_token  # type: ignore[attr-defined]
        parsed = urlsplit(self.path)
        raw = unquote(parsed.path)
        if raw == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        prefix = f"/{token}/"
        if raw.startswith(prefix):
            relative = raw[len(prefix):]
        elif prefix in str(self.headers.get("Referer") or ""):
            # Preserve root-relative assets without making the token optional.
            relative = raw.lstrip("/")
        else:
            self.send_error(404)
            return
        if not relative or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts):
            self.send_error(404)
            return
        try:
            with __import__("contextlib").nullcontext():
                candidate = (owner / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
            candidate.relative_to(owner)
            if not candidate.is_file() or is_sensitive_path(candidate):
                raise ValueError("unavailable")
            data = candidate.read_bytes()
            if sensitive_content_reason(data.decode("utf-8", errors="ignore")) is not None:
                raise ValueError("unavailable")
        except (OSError, RuntimeError, ValueError):
            self.send_error(404)
            return
        media = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", media + ("; charset=utf-8" if media.startswith(("text/", "application/javascript")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _key(preview_id: str) -> tuple[str, str]:
    return str(get_workspace()), preview_id


def _browser_executable() -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str | None]] = [
        ("chrome", shutil.which("google-chrome") or shutil.which("chrome")),
        ("msedge", shutil.which("msedge")),
        ("chromium", shutil.which("chromium") or shutil.which("chromium-browser")),
    ]
    if os.name == "nt":
        program_files = [os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")]
        for root in filter(None, program_files):
            candidates.extend([
                ("chrome", str(Path(root) / "Google/Chrome/Application/chrome.exe")),
                ("msedge", str(Path(root) / "Microsoft/Edge/Application/msedge.exe")),
            ])
    for channel, value in candidates:
        if value and Path(value).is_file():
            return channel, value
    return None, None


def browser_capability() -> dict[str, Any]:
    channel, executable = _browser_executable()
    try:
        import playwright.sync_api  # noqa: F401
        playwright_available = True
    except ImportError:
        playwright_available = False
    return {"playwright": playwright_available, "channel": channel, "executable": executable, "available": bool(executable)}


def _verify(url: str, screenshot_path: Path, settle_ms: int) -> dict[str, Any]:
    capability = browser_capability()
    result: dict[str, Any] = {
        "status": "unavailable",
        "console_errors": [],
        "page_errors": [],
        "network_errors": [],
        "screenshot_path": None,
    }
    if not capability["playwright"]:
        result["reason"] = "Python Playwright is not installed"
        return result
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            launch: dict[str, Any] = {"headless": True}
            if capability["channel"] in {"chrome", "msedge"}:
                launch["channel"] = capability["channel"]
            elif capability["executable"]:
                launch["executable_path"] = capability["executable"]
            browser = playwright.chromium.launch(**launch)
            context = browser.new_context(viewport={"width": 1280, "height": 720})
            page = context.new_page()
            page.on("console", lambda message: result["console_errors"].append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
            page.on("requestfailed", lambda request: result["network_errors"].append(f"{request.method} {request.url}: {request.failure}"))
            page.on("response", lambda response: result["network_errors"].append(f"HTTP {response.status} {response.url}") if response.status >= 400 else None)
            response = page.goto(url, wait_until="load", timeout=30_000)
            page.wait_for_timeout(max(0, min(int(settle_ms), 10_000)))
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)
            result.update({
                "status": (
                    "passed"
                    if response and response.ok
                    and not result["console_errors"]
                    and not result["page_errors"]
                    and not result["network_errors"]
                    else "failed"
                ),
                "http_status": response.status if response else None,
                "title": page.title(),
                "screenshot_path": str(screenshot_path),
            })
            context.close()
            browser.close()
    except Exception as exc:
        result.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    return result


def _open_visible(preview: Preview) -> tuple[bool, str | None]:
    _channel, executable = _browser_executable()
    if not executable:
        return False, "Chrome, Edge, or Chromium was not found"
    profile = get_workspace() / ".coding-agent" / "previews" / f"{preview.id}-profile"
    profile.mkdir(parents=True, exist_ok=True)
    try:
        args = [executable, f"--user-data-dir={profile}", "--no-first-run", "--new-window", preview.url]
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        preview.browser_process = process
        preview.profile_path = profile
        return True, None
    except OSError as exc:
        return False, safe_os_error(exc)


def create(path: str, open_browser: bool = True, verify: bool = True, settle_ms: int = 1500) -> str:
    try:
        entry = resolve_workspace_path(path, must_exist=True)
        if entry.suffix.casefold() not in {".html", ".htm"} or not entry.is_file():
            return "Error: preview_html requires an existing .html or .htm file"
        workspace = get_workspace()
        relative = entry.relative_to(workspace).as_posix()
        token = secrets.token_urlsafe(24)
        preview_id = "preview-" + secrets.token_hex(8)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.workspace = workspace  # type: ignore[attr-defined]
        server.preview_token = token  # type: ignore[attr-defined]
        thread = Thread(target=server.serve_forever, name=preview_id, daemon=True)
        port = int(server.server_address[1])
        url = f"http://127.0.0.1:{port}/{token}/{quote(relative)}"
        preview = Preview(preview_id, token, relative, server, thread, url)
        thread.start()
        try:
            with urlopen(url, timeout=3) as response:
                http_status = int(response.status)
        except OSError as exc:
            server.shutdown(); server.server_close()
            return f"Error: preview server health check failed: {safe_os_error(exc)}"
        screenshot = workspace / ".coding-agent" / "previews" / f"{preview_id}.png"
        preview.verification = _verify(url, screenshot, settle_ms) if verify else {"status": "not_requested"}
        opened, open_error = _open_visible(preview) if open_browser else (False, None)
        with _LOCK:
            _PREVIEWS[_key(preview_id)] = preview
        payload = {
            "status": "running",
            "preview_id": preview_id,
            "url": url,
            "http_status": http_status,
            "browser_opened": opened,
            "browser_error": open_error,
            "verification": preview.verification.get("status"),
            "console_errors": preview.verification.get("console_errors", []),
            "page_errors": preview.verification.get("page_errors", []),
            "network_errors": preview.verification.get("network_errors", []),
            "screenshot_path": preview.verification.get("screenshot_path"),
        }
        return json.dumps(payload, ensure_ascii=False)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"Error: HTML preview could not start: {safe_os_error(exc) if isinstance(exc, OSError) else exc}"


def inspect(preview_id: str, settle_ms: int = 500) -> str:
    with _LOCK:
        preview = _PREVIEWS.get(_key(preview_id))
    if preview is None:
        return f"Error: unknown preview {preview_id!r}"
    screenshot = get_workspace() / ".coding-agent" / "previews" / f"{preview.id}-latest.png"
    preview.verification = _verify(preview.url, screenshot, settle_ms)
    return json.dumps({"preview_id": preview_id, "url": preview.url, **preview.verification}, ensure_ascii=False)


def stop(preview_id: str) -> str:
    with _LOCK:
        preview = _PREVIEWS.pop(_key(preview_id), None)
    if preview is None:
        return f"Error: unknown preview {preview_id!r}"
    preview.server.shutdown(); preview.server.server_close(); preview.thread.join(timeout=3)
    if preview.browser_process and preview.browser_process.poll() is None:
        _terminate(preview.browser_process)
    if preview.profile_path:
        shutil.rmtree(preview.profile_path, ignore_errors=True)
    return json.dumps({"preview_id": preview_id, "status": "stopped"})


def list_previews() -> tuple[dict[str, Any], ...]:
    root = str(get_workspace())
    with _LOCK:
        return tuple({"preview_id": item.id, "url": item.url, "entry_path": item.entry_path} for (owner, _), item in _PREVIEWS.items() if owner == root)


def shutdown_workspace(workspace: str | Path) -> None:
    root = str(Path(workspace).resolve())
    with _LOCK:
        items = [item for (owner, _), item in _PREVIEWS.items() if owner == root]
        for item in items:
            _PREVIEWS.pop((root, item.id), None)
    for preview in items:
        preview.server.shutdown(); preview.server.server_close()
        if preview.browser_process and preview.browser_process.poll() is None:
            _terminate(preview.browser_process)
        if preview.profile_path:
            shutil.rmtree(preview.profile_path, ignore_errors=True)


def _shutdown_all() -> None:
    with _LOCK:
        workspaces = {owner for owner, _ in _PREVIEWS}
    for workspace in workspaces:
        shutdown_workspace(workspace)


atexit.register(_shutdown_all)

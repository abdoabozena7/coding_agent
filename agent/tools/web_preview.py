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
import re
import secrets
import shutil
import subprocess
import tempfile
from threading import RLock, Thread
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit
from urllib.request import urlopen

from ._security import (
    get_tool_context,
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
    artifact_dir: Path
    browser_process: subprocess.Popen[bytes] | None = None
    profile_path: Path | None = None
    verification: dict[str, Any] = field(default_factory=dict)


_LOCK = RLock()
_PREVIEWS: dict[tuple[str, str], Preview] = {}


def _safe_segment(value: str, fallback: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value).strip()
    ).strip("-")
    return cleaned[:80] or fallback


def _artifact_directory() -> Path:
    context = get_tool_context()
    return (
        context.workspace
        / "output"
        / "playwright"
        / _safe_segment(context.session_id, "workspace-session")
        / _safe_segment(context.goal_id, "goal")
        / _safe_segment(context.task_id, "task")
    )


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


def _interaction_locator(page: Any, item: Mapping[str, Any]) -> Any:
    selector = str(item.get("selector") or "").strip()
    if selector:
        return page.locator(selector)
    role = str(item.get("role") or "").strip()
    name = str(item.get("name") or "").strip()
    if not role:
        raise ValueError("interaction target requires role or selector")
    return page.get_by_role(role, name=name or None, exact=bool(item.get("exact", True)))


def _require_unique_interaction_locator(page: Any, item: Mapping[str, Any]) -> Any:
    """Resolve one target without Playwright's long actionability timeout.

    A model-authored selector that matches nothing is a verification-contract
    error, not evidence that the application failed.  Counting first makes that
    distinction immediately instead of waiting 30 seconds per invented target.
    """

    try:
        locator = _interaction_locator(page, item)
        count = locator.count()
    except Exception:
        locator = None
        count = 0
    # Small structured-output models sometimes put a DOM id in the accessible
    # ``name`` slot. Resolve that transport alias only when the live DOM proves
    # it names exactly one element.
    name = str(item.get("name") or "").strip()
    if count == 0 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", name):
        id_locator = page.locator(f"#{name}")
        if id_locator.count() == 1:
            return id_locator
    # For text assertions, the expected visible value itself is an
    # authoritative locator fallback. This repairs invented labels without
    # weakening the assertion or changing the application.
    prop = str(item.get("property") or "")
    expected_text = item.get("equals")
    contains_text = item.get("contains")
    visible_text = expected_text if expected_text is not None else contains_text
    if count == 0 and prop in {"text", "textContent"} and visible_text is not None:
        text_locator = page.get_by_text(
            str(visible_text),
            exact=expected_text is not None,
        )
        if text_locator.count() == 1:
            return text_locator
    if count == 0:
        target = str(item.get("selector") or "").strip() or (
            f"role={str(item.get('role') or '').strip()!r} "
            f"name={str(item.get('name') or '').strip()!r}"
        )
        raise ValueError(f"interaction target matched no elements: {target}")
    if count > 1:
        raise ValueError(f"interaction target matched {count} elements; target must be unique")
    return locator


def _interaction_target_inventory(page: Any) -> list[dict[str, str]]:
    """Return bounded, authoritative targets for a subsequent repair request."""

    values = page.locator(
        "button,input,select,textarea,a,[role],[data-value]"
    ).evaluate_all(
        """elements => elements.slice(0, 100).map(element => ({
            tag: (element.tagName || '').toLowerCase(),
            id: element.id || '',
            role: element.getAttribute('role') || '',
            name: element.getAttribute('aria-label') || element.innerText || element.value || '',
            data_value: element.getAttribute('data-value') || '',
            type: element.getAttribute('type') || ''
        }))"""
    )
    inventory: list[dict[str, str]] = []
    for raw in values if isinstance(values, list) else ():
        if not isinstance(raw, Mapping):
            continue
        item = {
            key: str(raw.get(key) or "").strip()[:240]
            for key in ("tag", "id", "role", "name", "data_value", "type")
        }
        if any(item.values()):
            inventory.append(item)
    return inventory


def _run_interaction_scenarios(
    page: Any,
    url: str,
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw in scenarios[:20]:
        scenario = dict(raw)
        name = str(scenario.get("name") or "browser interaction")[:160]
        receipt: dict[str, Any] = {"name": name, "passed": False, "assertions": []}
        try:
            if bool(scenario.get("reload", True)):
                page.goto(url, wait_until="load", timeout=30_000)
            for raw_step in list(scenario.get("steps") or ())[:40]:
                step = dict(raw_step)
                action = str(step.get("action") or "")
                if action == "press":
                    key = str(step.get("key") or "")
                    if not key:
                        raise ValueError("press interaction requires key")
                    page.keyboard.press(key)
                    continue
                locator = _require_unique_interaction_locator(page, step)
                if action == "click":
                    locator.click(timeout=5_000)
                elif action == "fill":
                    locator.fill(str(step.get("value") or ""), timeout=5_000)
                else:
                    raise ValueError(f"unsupported interaction action {action!r}")
            assertions: list[dict[str, Any]] = []
            for raw_assertion in list(scenario.get("assertions") or ())[:20]:
                assertion = dict(raw_assertion)
                locator = _require_unique_interaction_locator(page, assertion)
                prop = str(assertion.get("property") or "")
                observed = locator.input_value() if prop == "value" else locator.inner_text()
                expected = assertion.get("equals")
                contains = assertion.get("contains")
                passed = (
                    observed == str(expected)
                    if expected is not None
                    else str(contains) in observed
                    if contains is not None
                    else False
                )
                assertions.append(
                    {
                        "property": prop,
                        "observed": observed,
                        "expected": expected,
                        "contains": contains,
                        "passed": passed,
                    }
                )
            receipt["assertions"] = assertions
            receipt["passed"] = bool(assertions) and all(item["passed"] for item in assertions)
            if not receipt["passed"]:
                receipt["error"] = "one or more interaction assertions failed"
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            receipt["failure_kind"] = (
                "contract"
                if "unknown key" in str(exc).casefold()
                or isinstance(exc, ValueError)
                else "application"
            )
        results.append(receipt)
    return results


def _verify(url: str, screenshot_path: Path, settle_ms: int, interactions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    capability = browser_capability()
    result: dict[str, Any] = {
        "status": "unavailable",
        "failure_kind": "",
        "interaction_targets": [],
        "console_errors": [],
        "page_errors": [],
        "network_errors": [],
        "screenshot_path": None,
        "interaction_results": [],
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
            result["interaction_targets"] = _interaction_target_inventory(page)
            interaction_results = _run_interaction_scenarios(page, url, interactions)
            result["interaction_results"] = interaction_results
            for item in interaction_results:
                if not item.get("passed"):
                    result["page_errors"].append(
                        f"Interaction {item.get('name')}: {item.get('error') or 'assertion failed'}"
                    )
            failed_interactions = [
                item for item in interaction_results if not item.get("passed")
            ]
            if failed_interactions:
                result["failure_kind"] = (
                    "contract"
                    if all(item.get("failure_kind") == "contract" for item in failed_interactions)
                    else "application"
                )
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


def create(path: str, open_browser: bool = True, verify: bool = True, settle_ms: int = 1500, interactions: Sequence[Mapping[str, Any]] = ()) -> str:
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
        artifact_dir = _artifact_directory()
        preview = Preview(preview_id, token, relative, server, thread, url, artifact_dir)
        thread.start()
        try:
            with urlopen(url, timeout=3) as response:
                http_status = int(response.status)
        except OSError as exc:
            server.shutdown(); server.server_close()
            return f"Error: preview server health check failed: {safe_os_error(exc)}"
        screenshot = artifact_dir / f"{preview_id}.png"
        preview.verification = _verify(url, screenshot, settle_ms, interactions) if verify else {"status": "not_requested"}
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
            "failure_kind": preview.verification.get("failure_kind", ""),
            "interaction_targets": preview.verification.get("interaction_targets", []),
            "console_errors": preview.verification.get("console_errors", []),
            "page_errors": preview.verification.get("page_errors", []),
            "network_errors": preview.verification.get("network_errors", []),
            "screenshot_path": preview.verification.get("screenshot_path"),
            "interaction_results": preview.verification.get("interaction_results", []),
        }
        return json.dumps(payload, ensure_ascii=False)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"Error: HTML preview could not start: {safe_os_error(exc) if isinstance(exc, OSError) else exc}"


def inspect(preview_id: str, settle_ms: int = 500) -> str:
    with _LOCK:
        preview = _PREVIEWS.get(_key(preview_id))
    if preview is None:
        return f"Error: unknown preview {preview_id!r}"
    screenshot = preview.artifact_dir / f"{preview.id}-latest.png"
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

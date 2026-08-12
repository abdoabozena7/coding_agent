"""Persistent Playwright browser sessions for general workspace automation.

Unlike the legacy preview helper, this module keeps one Playwright-owned page
alive across tool calls.  The model can therefore inspect the page it actually
opened, interact with it, and capture the resulting state without launching a
second unrelated browser process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import atexit
import hashlib
import json
from pathlib import Path
from queue import Queue
import re
import secrets
import struct
from threading import Event, RLock, Thread
from typing import Any, Mapping
from urllib.parse import urlsplit

from . import web_preview
from ._security import (
    ToolSecurityError,
    display_path,
    get_tool_context,
    get_workspace,
    resolve_workspace_path,
)


def _safe_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip()).strip("-")
    return cleaned[:80] or fallback


def _normalise_url(value: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ToolSecurityError(
            "browser URLs must be credential-free http(s) addresses"
        )
    return parsed._replace(fragment="").geturl()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    return struct.unpack(">II", header[16:24])


def perceptual_hash(path: Path) -> str:
    """Return a compact dHash for deterministic near-duplicate rejection."""

    try:
        from PIL import Image

        with Image.open(path) as source:
            resized = source.convert("L").resize(
                (17, 16),
                Image.Resampling.LANCZOS,
            )
            pixels = list(
                resized.get_flattened_data()
                if hasattr(resized, "get_flattened_data")
                else resized.getdata()
            )
        bits = 0
        count = 0
        for row in range(16):
            offset = row * 17
            for column in range(16):
                bits = (bits << 1) | int(
                    pixels[offset + column] > pixels[offset + column + 1]
                )
                count += 1
        return f"{bits:0{count // 4}x}"
    except (ImportError, OSError, ValueError):
        return ""


@dataclass(slots=True)
class _Request:
    operation: str
    payload: dict[str, Any]
    done: Event = field(default_factory=Event)
    result: Any = None
    error: BaseException | None = None


def _launch_browser_with_fallback(
    chromium: Any,
    capability: Mapping[str, Any],
    *,
    visible: bool,
) -> tuple[Any, str, list[str]]:
    """Launch the first working installed browser without changing app state."""

    raw_candidates = capability.get("candidates")
    candidates = (
        list(raw_candidates)
        if isinstance(raw_candidates, list) and raw_candidates
        else [{
            "channel": capability.get("channel"),
            "executable": capability.get("executable"),
        }]
    )
    launch_attempts: list[tuple[str, dict[str, Any]]] = []
    seen_launches: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        channel = str(candidate.get("channel") or "").strip()
        executable = str(candidate.get("executable") or "").strip()
        launch: dict[str, Any] = {"headless": not visible}
        if channel in {"chrome", "msedge"}:
            launch["channel"] = channel
            key = ("channel", channel)
            label = channel
        elif executable:
            launch["executable_path"] = executable
            key = ("executable", executable.casefold())
            label = channel or Path(executable).stem
        else:
            continue
        if key in seen_launches:
            continue
        seen_launches.add(key)
        launch_attempts.append((label, launch))

    failures: list[str] = []
    for label, launch in launch_attempts:
        try:
            return chromium.launch(**launch), label, failures
        except Exception as exc:
            summary = " ".join(str(exc).split())
            failures.append(f"{label}: {summary[:500]}")
    detail = " | ".join(failures) or "no installed browser launch candidate"
    raise RuntimeError(f"all Playwright browser launch candidates failed: {detail}")


class _BrowserWorker:
    def __init__(
        self,
        *,
        session_id: str,
        workspace: Path,
        artifact_dir: Path,
        visible: bool,
        width: int,
        height: int,
    ) -> None:
        self.session_id = session_id
        self.workspace = workspace
        self.artifact_dir = artifact_dir
        self.visible = visible
        self.width = width
        self.height = height
        self.queue: Queue[_Request | None] = Queue()
        self.ready = Event()
        self.failure: BaseException | None = None
        self.thread = Thread(
            target=self._run,
            name=f"ga3bad-browser-{session_id}",
            daemon=True,
        )
        self.console_errors: list[str] = []
        self.page_errors: list[str] = []
        self.network_errors: list[str] = []
        self.backing_preview_id = ""
        self.last_http_status: int | None = None
        self.browser_engine = ""
        self.browser_launch_failures: list[str] = []

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(30):
            raise RuntimeError("Playwright browser launch timed out")
        if self.failure is not None:
            raise RuntimeError(f"Playwright browser launch failed: {self.failure}")

    def invoke(self, operation: str, *, timeout: float = 45, **payload: Any) -> Any:
        request = _Request(operation=operation, payload=dict(payload))
        self.queue.put(request)
        if not request.done.wait(timeout):
            raise RuntimeError(f"browser {operation} timed out")
        if request.error is not None:
            raise RuntimeError(f"browser {operation} failed: {request.error}") from request.error
        return request.result

    def stop(self) -> None:
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=8)
        if self.backing_preview_id:
            try:
                web_preview.stop(self.backing_preview_id)
            except Exception:
                pass

    def _page_snapshot(self, page: Any) -> dict[str, Any]:
        body_text = ""
        try:
            body_text = page.locator("body").inner_text(timeout=2_000)[:12_000]
        except Exception:
            pass
        return {
            "browser_session_id": self.session_id,
            "status": "running",
            "browser_opened": self.visible,
            "url": page.url,
            "title": page.title(),
            "http_status": self.last_http_status,
            "viewport": {"width": self.width, "height": self.height},
            "browser_engine": self.browser_engine,
            "browser_launch_fallbacks": list(self.browser_launch_failures),
            "text": body_text,
            "interaction_targets": web_preview._interaction_target_inventory(page),
            "console_errors": list(self.console_errors[-50:]),
            "page_errors": list(self.page_errors[-50:]),
            "network_errors": list(self.network_errors[-50:]),
        }

    @staticmethod
    def _locator(page: Any, target: Mapping[str, Any]) -> Any:
        return web_preview._require_unique_interaction_locator(page, target)

    def _dispatch(self, page: Any, operation: str, payload: Mapping[str, Any]) -> Any:
        if operation == "navigate":
            url = _normalise_url(str(payload.get("url") or ""))
            # A navigation starts a fresh verification attempt. Retaining
            # errors from the previous load makes a repaired companion service
            # look broken forever and prevents evidence-based recovery.
            self.console_errors.clear()
            self.page_errors.clear()
            self.network_errors.clear()
            response = page.goto(
                url,
                wait_until=str(payload.get("wait_until") or "load"),
                timeout=min(60_000, max(1_000, int(payload.get("timeout_ms") or 30_000))),
            )
            self.last_http_status = int(response.status) if response is not None else None
            page.wait_for_timeout(min(10_000, max(0, int(payload.get("settle_ms") or 500))))
            return self._page_snapshot(page)
        if operation == "inspect":
            page.wait_for_timeout(min(10_000, max(0, int(payload.get("settle_ms") or 0))))
            return self._page_snapshot(page)
        if operation == "act":
            receipts: list[dict[str, Any]] = []
            for index, raw in enumerate(list(payload.get("actions") or ())[:40], start=1):
                action = dict(raw)
                kind = str(action.get("action") or "")
                receipt: dict[str, Any] = {"index": index, "action": kind, "passed": False}
                try:
                    if kind == "press" and not any(action.get(key) for key in ("selector", "role", "name")):
                        page.keyboard.press(str(action.get("key") or ""))
                    else:
                        locator = self._locator(page, action)
                        if kind == "click":
                            locator.click(timeout=5_000)
                        elif kind == "fill":
                            locator.fill(str(action.get("value") or ""), timeout=5_000)
                        elif kind == "press":
                            locator.press(str(action.get("key") or ""), timeout=5_000)
                        elif kind == "select":
                            locator.select_option(str(action.get("value") or ""), timeout=5_000)
                        elif kind == "check":
                            locator.check(timeout=5_000)
                        elif kind == "uncheck":
                            locator.uncheck(timeout=5_000)
                        elif kind == "hover":
                            locator.hover(timeout=5_000)
                        elif kind == "upload":
                            resolved_files: list[str] = []
                            for raw_path in list(action.get("files") or ()):
                                candidate = Path(str(raw_path))
                                candidate = (
                                    candidate.resolve(strict=False)
                                    if candidate.is_absolute()
                                    else (self.workspace / candidate).resolve(strict=False)
                                )
                                if not candidate.is_relative_to(self.workspace):
                                    raise ValueError(
                                        "browser uploads must be existing files inside the active workspace"
                                    )
                                if not candidate.is_file():
                                    raise ValueError(
                                        "browser upload file does not exist inside the active workspace"
                                    )
                                resolved_files.append(str(candidate))
                            if not resolved_files:
                                raise ValueError("upload requires at least one workspace file")
                            locator.set_input_files(resolved_files, timeout=5_000)
                        else:
                            raise ValueError(f"unsupported browser action {kind!r}")
                    receipt["passed"] = True
                except Exception as exc:
                    receipt["error"] = f"{type(exc).__name__}: {exc}"
                    receipts.append(receipt)
                    raise ValueError(receipt["error"]) from exc
                receipts.append(receipt)
            page.wait_for_timeout(min(10_000, max(0, int(payload.get("settle_ms") or 300))))
            return {**self._page_snapshot(page), "actions": receipts}
        if operation == "screenshot":
            name = _safe_segment(str(payload.get("name") or "screenshot"), "screenshot")
            suffix = secrets.token_hex(4)
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            target = self.artifact_dir / f"{name}-{suffix}.png"
            page.screenshot(
                path=str(target),
                full_page=bool(payload.get("full_page", True)),
            )
            width, height = _png_dimensions(target)
            relative = target.relative_to(self.workspace).as_posix()
            snapshot = self._page_snapshot(page)
            return {
                # Screenshot receipts are durable evidence. Keep them compact
                # so screenshot_path/sha256 can never be truncated behind a
                # large page text or 100-item interaction inventory. DOM
                # targets belong to browser_inspect, not the image receipt.
                "browser_session_id": self.session_id,
                "status": "running",
                "browser_opened": self.visible,
                "url": page.url,
                "title": snapshot.get("title"),
                "http_status": self.last_http_status,
                "viewport": {"width": self.width, "height": self.height},
                "browser_engine": self.browser_engine,
                "browser_launch_fallbacks": list(self.browser_launch_failures),
                "console_errors": list(self.console_errors[-20:]),
                "page_errors": list(self.page_errors[-20:]),
                "network_errors": list(self.network_errors[-20:]),
                "screenshot_path": str(target),
                "workspace_path": relative,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "perceptual_hash": perceptual_hash(target),
                "byte_size": target.stat().st_size,
                "image_width": width,
                "image_height": height,
                "full_page": bool(payload.get("full_page", True)),
            }
        raise ValueError(f"unknown browser operation {operation!r}")

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                capability = web_preview.browser_capability()
                browser, self.browser_engine, failures = _launch_browser_with_fallback(
                    playwright.chromium,
                    capability,
                    visible=self.visible,
                )
                self.browser_launch_failures = failures
                context = browser.new_context(viewport={"width": self.width, "height": self.height})
                page = context.new_page()
                page.on(
                    "console",
                    lambda message: self.console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.on("pageerror", lambda error: self.page_errors.append(str(error)))
                page.on(
                    "requestfailed",
                    lambda request: self.network_errors.append(
                        f"{request.method} {request.url}: {request.failure}"
                    ),
                )
                page.on(
                    "response",
                    lambda response: self.network_errors.append(
                        f"HTTP {response.status} {response.url}"
                    )
                    if response.status >= 400
                    else None,
                )
                self.ready.set()
                while True:
                    request = self.queue.get()
                    if request is None:
                        break
                    try:
                        request.result = self._dispatch(page, request.operation, request.payload)
                    except BaseException as exc:  # propagate safely to the invoking tool
                        request.error = exc
                    finally:
                        request.done.set()
                context.close()
                browser.close()
        except BaseException as exc:
            self.failure = exc
            self.ready.set()


_LOCK = RLock()
_SESSIONS: dict[tuple[str, str], _BrowserWorker] = {}


def _key(session_id: str) -> tuple[str, str]:
    return str(get_workspace()), str(session_id)


def _worker(session_id: str) -> _BrowserWorker:
    with _LOCK:
        result = _SESSIONS.get(_key(session_id))
    if result is None:
        raise ToolSecurityError(f"unknown browser session {session_id!r}")
    return result


def open_browser(
    *,
    url: str = "",
    path: str = "",
    visible: bool = True,
    width: int = 1440,
    height: int = 900,
    settle_ms: int = 700,
) -> str:
    context = get_tool_context()
    if bool(str(url).strip()) == bool(str(path).strip()):
        return "Error: browser_open requires exactly one of url or path"
    backing_preview_id = ""
    target_url = ""
    if path:
        entry = resolve_workspace_path(path, must_exist=True)
        if entry.suffix.casefold() not in {".html", ".htm"} or not entry.is_file():
            return "Error: browser_open path must be an existing HTML file"
        served = json.loads(
            web_preview.create(display_path(entry), open_browser=False, verify=False)
        )
        target_url = str(served.get("url") or "")
        backing_preview_id = str(served.get("preview_id") or "")
    else:
        target_url = _normalise_url(url)
    session_id = "browser-" + secrets.token_hex(8)
    artifact_dir = (
        context.workspace
        / "output"
        / "browser"
        / _safe_segment(context.session_id, "workspace-session")
        / _safe_segment(context.goal_id, "goal")
        / _safe_segment(context.task_id, "task")
    )
    worker = _BrowserWorker(
        session_id=session_id,
        workspace=context.workspace,
        artifact_dir=artifact_dir,
        visible=bool(visible),
        width=max(320, min(int(width), 3840)),
        height=max(240, min(int(height), 2160)),
    )
    worker.backing_preview_id = backing_preview_id
    try:
        worker.start()
        with _LOCK:
            _SESSIONS[(str(context.workspace), session_id)] = worker
        snapshot = worker.invoke("navigate", url=target_url, settle_ms=settle_ms)
        snapshot["source"] = "workspace_html" if path else "url"
        snapshot["source_path"] = display_path(resolve_workspace_path(path, must_exist=True)) if path else ""
        return json.dumps(snapshot, ensure_ascii=False)
    except Exception:
        worker.stop()
        raise


def navigate(session_id: str, url: str, settle_ms: int = 500) -> str:
    return json.dumps(
        _worker(session_id).invoke("navigate", url=url, settle_ms=settle_ms),
        ensure_ascii=False,
    )


def inspect(session_id: str, settle_ms: int = 0) -> str:
    return json.dumps(
        _worker(session_id).invoke("inspect", settle_ms=settle_ms),
        ensure_ascii=False,
    )


def act(session_id: str, actions: list[dict[str, Any]], settle_ms: int = 300) -> str:
    return json.dumps(
        _worker(session_id).invoke("act", actions=actions, settle_ms=settle_ms),
        ensure_ascii=False,
    )


def screenshot(session_id: str, name: str = "screenshot", full_page: bool = True) -> str:
    return json.dumps(
        _worker(session_id).invoke(
            "screenshot", name=name, full_page=full_page
        ),
        ensure_ascii=False,
    )


def close(session_id: str) -> str:
    with _LOCK:
        worker = _SESSIONS.pop(_key(session_id), None)
    if worker is None:
        return f"Error: unknown browser session {session_id!r}"
    worker.stop()
    return json.dumps({"browser_session_id": session_id, "status": "closed"})


def list_sessions() -> tuple[dict[str, Any], ...]:
    root = str(get_workspace())
    with _LOCK:
        return tuple(
            {
                "browser_session_id": session_id,
                "visible": worker.visible,
                "status": "running" if worker.thread.is_alive() else "closed",
            }
            for (owner, session_id), worker in _SESSIONS.items()
            if owner == root
        )


def shutdown_workspace(workspace: str | Path) -> None:
    root = str(Path(workspace).resolve())
    with _LOCK:
        entries = [
            (key, worker)
            for key, worker in _SESSIONS.items()
            if key[0] == root
        ]
        for key, _worker_value in entries:
            _SESSIONS.pop(key, None)
    for _key_value, worker in entries:
        worker.stop()


def _shutdown_all() -> None:
    with _LOCK:
        roots = {owner for owner, _session_id in _SESSIONS}
    for root in roots:
        shutdown_workspace(root)


atexit.register(_shutdown_all)

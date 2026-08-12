"""FastAPI shell and lifecycle for GA3BAD's temporary local workspaces."""

from __future__ import annotations

import asyncio
import json
from queue import Empty
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..store import NotFoundError, StalePlanError, StateStoreError
from ..runtime import ProviderUnavailableError
from ..events import LiveWorkflowEventV1
from .schemas import (
    ExplanationRequestPayload,
    PlanApprovalPayload,
    PlanDocumentPayload,
    PlanPayload,
    PlanRequestPayload,
    QueuePromptPayload,
    QueueReorderPayload,
    ReviewSubmissionPayload,
    ToolApprovalPayload,
    TraceRevealPayload,
    WorkspaceActionRequest,
)
from .security import SessionSecurity
from .service import CoreWebAdapter


VIEWS = frozenset({
    "plan", "live", "thread", "review", "agents", "execution", "history",
    "tree", "diff", "show-diff", "advanced-tracing", "output",
})


def _set_web_control_connected(runtime: Any, connected: bool) -> None:
    """Publish actual browser ownership to both runtime and terminal UI."""

    value = bool(connected)
    setattr(runtime, "web_control_connected", value)
    sink = getattr(runtime, "web_control_state_sink", None)
    if callable(sink):
        sink(value)


def _error_code(message: str, status: int) -> str:
    value = str(message or "").casefold()
    if status == 401:
        return "session_expired"
    if status == 403:
        return "permission_denied"
    if status == 404:
        return "not_found"
    if status == 409:
        return "stale_state"
    if any(token in value for token in ("quota", "usage limit", "limit exhausted")):
        return "quota_exhausted"
    if any(
        token in value
        for token in ("local model runner", "ollama", "local runner")
    ) and any(token in value for token in ("unavailable", "unreachable", "offline", "connection")):
        return "local_runner_unreachable"
    if status == 429 or "rate limit" in value or "too many requests" in value:
        return "rate_limited"
    if any(token in value for token in ("network", "unreachable", "connection", "timed out")):
        return "runtime_unreachable"
    if status == 422:
        return "invalid_request"
    if status >= 500:
        return "runtime_error"
    return "request_failed"


def _error(message: str, status: int, **extra: Any) -> JSONResponse:
    return JSONResponse(
        {"error": message, "code": _error_code(message, status), **extra},
        status_code=status,
    )


def create_app(adapter: CoreWebAdapter, security: SessionSecurity) -> FastAPI:
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(
        title="GA3BAD Local Web Views",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.port = 0
    app.state.web_connections = 0
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.middleware("http")
    async def loopback_security(request: Request, call_next: Any):
        host = request.headers.get("host", "").split(":", 1)[0].strip("[]").casefold()
        if host not in {"127.0.0.1", "localhost"}:
            return _error("invalid host", 400)
        if request.url.path.startswith("/api/"):
            if not security.validate_cookie(request.cookies.get("ga3bad_session")):
                return _error("session authentication required", 401)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                if not security.validate_csrf(
                    request.cookies.get("ga3bad_csrf"),
                    request.headers.get("x-ga3bad-csrf"),
                ):
                    return _error("CSRF validation failed", 403)
                origin = request.headers.get("origin")
                allowed = {
                    f"http://127.0.0.1:{app.state.port}",
                    f"http://localhost:{app.state.port}",
                }
                if origin and origin not in allowed:
                    return _error("origin validation failed", 403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return _error("Request validation failed.", 422, details=exc.errors())

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        return _error(str(exc.detail), int(exc.status_code), details=exc.detail)

    @app.exception_handler(NotFoundError)
    async def not_found(_request: Request, exc: NotFoundError):
        return _error(str(exc), 404)

    @app.exception_handler(StalePlanError)
    async def stale_plan(_request: Request, exc: StalePlanError):
        try:
            current = adapter.plan_snapshot()["revision"]
        except Exception:
            current = None
        return _error(str(exc), 409, current_revision=current)

    @app.exception_handler(StateStoreError)
    async def state_conflict(_request: Request, exc: StateStoreError):
        return _error(str(exc), 409)

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable(_request: Request, exc: ProviderUnavailableError):
        # Provider failures are expected workflow boundaries, not ASGI faults.
        # Returning a named, retryable response lets the client refresh the
        # durable checkpoint and expose Retry/local/Stop without a raw 500.
        return _error(
            str(exc),
            503,
            retryable=True,
            saved_stage=True,
        )

    @app.exception_handler(ValueError)
    async def bad_value(_request: Request, exc: ValueError):
        return _error(str(exc), 422)

    @app.get("/")
    async def root():
        return RedirectResponse(
            f"/sessions/{adapter.session_id}/plan",
            status_code=307,
        )

    @app.get("/sessions/{session_id}/{view_name}")
    async def shell(request: Request, session_id: str, view_name: str):
        if session_id != adapter.session_id or view_name not in VIEWS:
            raise HTTPException(status_code=404, detail="view not found")
        standalone_trace = view_name == "advanced-tracing"
        standalone_diff = view_name == "show-diff"
        standalone_output = view_name == "output"
        public_view = (
            "advanced-tracing"
            if standalone_trace
            else (
                "show-diff"
                if standalone_diff
                else (
                    "output"
                    if standalone_output
                    else ("plan" if view_name == "plan" else "live")
                )
            )
        )
        if not standalone_trace and not standalone_diff and not standalone_output:
            adapter.request_view(public_view)
        token = request.query_params.get("token", "")
        if token:
            if not security.validate_handshake(token):
                raise HTTPException(status_code=401, detail="invalid or expired session token")
            response = RedirectResponse(
                f"/sessions/{session_id}/{public_view}",
                status_code=303,
            )
            response.set_cookie(
                "ga3bad_session",
                security.cookie_value,
                httponly=True,
                secure=False,
                samesite="strict",
                path="/",
                max_age=security.lifetime_seconds,
            )
            response.set_cookie(
                "ga3bad_csrf",
                security.csrf_token,
                httponly=False,
                secure=False,
                samesite="strict",
                path="/",
                max_age=security.lifetime_seconds,
            )
            return response
        if not security.validate_cookie(request.cookies.get("ga3bad_session")):
            raise HTTPException(status_code=401, detail="open this view from GA3BAD")
        if view_name != public_view:
            return RedirectResponse(
                f"/sessions/{session_id}/{public_view}",
                status_code=303,
            )
        return FileResponse(
            static_dir / (
                "advanced-tracing.html"
                if standalone_trace
                else (
                    "show-diff.html"
                    if standalone_diff
                    else "output.html"
                    if standalone_output
                    else "index.html"
                )
            ),
            media_type="text/html",
        )

    def check_session(session_id: str) -> None:
        if session_id != adapter.session_id:
            raise HTTPException(status_code=404, detail="session not found")

    @app.get("/api/sessions")
    async def get_sessions():
        """Left-rail project/task projection for the current local owner."""

        return adapter.sessions_index_snapshot()

    @app.get("/api/sessions/{session_id}/plan")
    async def get_plan(session_id: str):
        check_session(session_id)
        return adapter.plan_snapshot()

    @app.get("/api/sessions/{session_id}/workspace")
    async def get_workspace(session_id: str):
        check_session(session_id)
        return adapter.workspace_context()

    @app.get("/api/sessions/{session_id}/output")
    async def get_output(session_id: str):
        check_session(session_id)
        return adapter.output_snapshot()

    @app.get("/api/sessions/{session_id}/output/assets/{asset_id}")
    async def get_output_asset(
        session_id: str,
        asset_id: str,
        download: bool = False,
    ):
        check_session(session_id)
        path = adapter.output_asset_path(asset_id)
        return FileResponse(
            path,
            filename=path.name if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    @app.get("/api/sessions/{session_id}/thread")
    async def get_thread(
        session_id: str,
        after_sequence: int = 0,
        limit: int = 200,
    ):
        check_session(session_id)
        return adapter.thread_snapshot(after_sequence=after_sequence, limit=limit)

    @app.get("/api/sessions/{session_id}/inspector")
    async def get_inspector(session_id: str, section: str | None = None):
        check_session(session_id)
        return adapter.inspector_snapshot(section)

    @app.get("/api/sessions/{session_id}/models")
    async def get_models(session_id: str):
        check_session(session_id)
        return await asyncio.to_thread(adapter.model_catalog_snapshot)

    @app.get("/api/sessions/{session_id}/project-settings")
    async def get_project_settings(session_id: str):
        check_session(session_id)
        return await asyncio.to_thread(adapter.project_settings_snapshot)

    @app.get("/api/sessions/{session_id}/events")
    async def live_events(request: Request, session_id: str):
        """Stream safe workflow activity; durable snapshots repair reconnects."""

        check_session(session_id)
        observer_only = str(request.query_params.get("observer") or "").lower() in {
            "1",
            "true",
            "yes",
        }
        raw_after = request.headers.get("last-event-id") or request.query_params.get("after") or "0"
        try:
            after_sequence = max(0, int(raw_after))
        except (TypeError, ValueError):
            after_sequence = 0

        async def stream():
            queue = adapter.events.open_queue()
            cursor = after_sequence
            if not observer_only:
                app.state.web_connections = int(app.state.web_connections) + 1
                _set_web_control_connected(adapter.runtime, True)
            try:
                snapshot = adapter.workspace_context()
                latest = adapter.events.latest_sequence
                cursor = max(cursor, latest)
                snapshot_identity = dict(snapshot.get("workflow_identity") or {})
                snapshot["activity_sequence"] = latest
                snapshot["content_revision"] = snapshot_identity.get(
                    "content_revision", 0
                )
                yield "retry: 1000\n"
                yield (
                    f"id: {latest}\n"
                    "event: snapshot\n"
                    f"data: {json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
                )
                # Replay the bounded in-process history requested by a client
                # that reconnects without forcing a full page rebuild.
                for item in adapter.events.list_live_events(
                    after_sequence=after_sequence,
                    limit=256,
                ):
                    if item.sequence <= cursor:
                        continue
                    cursor = item.sequence
                    payload = item.to_dict()
                    payload["activity_sequence"] = item.sequence
                    try:
                        identity = dict(
                            adapter.workspace_context().get("workflow_identity") or {}
                        )
                        payload["content_revision"] = identity.get(
                            "content_revision", 0
                        )
                    except Exception:
                        payload["content_revision"] = snapshot_identity.get(
                            "content_revision", 0
                        )
                    yield (
                        f"id: {item.sequence}\n"
                        "event: activity\n"
                        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
                    )
                while not security.expired:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(queue.get, True, 1.0)
                    except Empty:
                        yield ": keepalive\n\n"
                        continue
                    item = LiveWorkflowEventV1.from_ui_event(event)
                    if item.sequence <= cursor:
                        continue
                    cursor = item.sequence
                    payload = item.to_dict()
                    payload["activity_sequence"] = item.sequence
                    try:
                        identity = dict(
                            adapter.workspace_context().get("workflow_identity") or {}
                        )
                        payload["content_revision"] = identity.get(
                            "content_revision", 0
                        )
                    except Exception:
                        payload["content_revision"] = snapshot_identity.get(
                            "content_revision", 0
                        )
                    yield (
                        f"id: {item.sequence}\n"
                        "event: activity\n"
                        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
                    )
            finally:
                adapter.events.close_queue(queue)
                if not observer_only:
                    app.state.web_connections = max(0, int(app.state.web_connections) - 1)
                    _set_web_control_connected(
                        adapter.runtime, bool(app.state.web_connections)
                    )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/sessions/{session_id}/plan/draft")
    async def save_draft(session_id: str, payload: PlanPayload):
        check_session(session_id)
        return adapter.save_plan_draft(payload)

    @app.delete("/api/sessions/{session_id}/plan/draft")
    async def discard_draft(session_id: str):
        check_session(session_id)
        return adapter.discard_plan_draft()

    @app.post("/api/sessions/{session_id}/plan/apply")
    async def apply_plan(session_id: str, payload: PlanPayload):
        check_session(session_id)
        return adapter.apply_plan(payload)

    @app.post("/api/sessions/{session_id}/plan/revision")
    async def save_revision(session_id: str, payload: PlanPayload):
        check_session(session_id)
        return adapter.save_plan_revision(payload)

    @app.post("/api/sessions/{session_id}/plan/team-preview")
    async def prepare_team_preview(session_id: str, payload: PlanDocumentPayload):
        check_session(session_id)
        return adapter.prepare_team_preview(payload)

    @app.post("/api/sessions/{session_id}/plan/approve")
    async def approve_plan(session_id: str, payload: PlanApprovalPayload):
        check_session(session_id)
        return adapter.approve_plan(
            payload.revision,
            plan_fingerprint=payload.plan_fingerprint,
            team_fingerprint=payload.team_fingerprint,
        )

    @app.post("/api/sessions/{session_id}/tool-approval")
    async def resolve_tool_approval(session_id: str, payload: ToolApprovalPayload):
        check_session(session_id)
        return adapter.resolve_tool_approval(payload.action_fingerprint, payload.decision)

    @app.post("/api/sessions/{session_id}/plan/request")
    async def submit_plan_request(session_id: str, payload: PlanRequestPayload):
        check_session(session_id)
        return adapter.submit_plan_request(payload.request)

    @app.post("/api/sessions/{session_id}/plan/depth")
    async def increase_plan_depth(session_id: str):
        check_session(session_id)
        return adapter.increase_execution_depth()

    @app.get("/api/sessions/{session_id}/queue")
    async def get_queue(session_id: str):
        check_session(session_id)
        return adapter.queue_snapshot()

    @app.post("/api/sessions/{session_id}/queue")
    async def enqueue_prompt(session_id: str, payload: QueuePromptPayload):
        check_session(session_id)
        return adapter.enqueue_queue_prompt(payload.text, payload.mode)

    @app.patch("/api/sessions/{session_id}/queue/order")
    async def reorder_queue(session_id: str, payload: QueueReorderPayload):
        check_session(session_id)
        return adapter.reorder_queue(payload.ordered_ids)

    @app.get("/api/sessions/{session_id}/review")
    async def get_review(session_id: str, checkpoint: str | None = None):
        check_session(session_id)
        return adapter.review_snapshot(checkpoint)

    @app.post("/api/sessions/{session_id}/review/submit")
    async def submit_review(session_id: str, payload: ReviewSubmissionPayload):
        check_session(session_id)
        return adapter.submit_review(payload)

    @app.get("/api/sessions/{session_id}/agents")
    async def get_agents(session_id: str):
        check_session(session_id)
        return adapter.agents_snapshot()

    @app.get("/api/sessions/{session_id}/execution")
    async def get_execution(session_id: str):
        check_session(session_id)
        return adapter.execution_snapshot()

    @app.get("/api/sessions/{session_id}/tree")
    async def get_tree(session_id: str):
        check_session(session_id)
        return adapter.execution_snapshot()

    @app.get("/api/sessions/{session_id}/diff")
    async def get_diff(session_id: str, checkpoint: str | None = None):
        check_session(session_id)
        return adapter.review_snapshot(checkpoint)

    @app.get("/api/sessions/{session_id}/show-diff")
    async def get_workflow_diff(session_id: str):
        check_session(session_id)
        return await asyncio.to_thread(adapter.diff_workflow_snapshot)

    @app.get("/api/sessions/{session_id}/show-diff/{change_id}")
    async def get_workflow_diff_detail(session_id: str, change_id: str):
        check_session(session_id)
        return await asyncio.to_thread(adapter.diff_workflow_detail, change_id)

    @app.get("/api/sessions/{session_id}/history")
    async def get_history(
        session_id: str,
        goal_id: str | None = None,
        after: int = 0,
        limit: int = 100,
        phase: str | None = None,
        actor: str | None = None,
        entity_id: str | None = None,
        failures_only: bool = False,
    ):
        check_session(session_id)
        return adapter.history_snapshot(
            goal_id=goal_id,
            after_sequence=after,
            limit=limit,
            phase=phase,
            actor=actor,
            entity_id=entity_id,
            failures_only=failures_only,
        )

    @app.get("/api/sessions/{session_id}/history/{sequence}")
    async def get_history_event(session_id: str, sequence: int):
        check_session(session_id)
        history = adapter.history_snapshot(after_sequence=max(0, sequence - 1), limit=1)
        items = [item for item in history.get("items", []) if item.get("sequence") == sequence]
        if not items:
            raise HTTPException(status_code=404, detail="history event not found")
        return items[0]

    @app.get("/api/sessions/{session_id}/plan/revisions")
    async def get_plan_revisions(session_id: str, goal_id: str | None = None):
        check_session(session_id)
        return adapter.plan_revisions_snapshot(goal_id)

    @app.post("/api/sessions/{session_id}/actions")
    async def workspace_action(session_id: str, payload: WorkspaceActionRequest):
        check_session(session_id)
        return adapter.apply_workspace_action(payload)

    @app.post("/api/sessions/{session_id}/agents/explain")
    async def request_explanation(session_id: str, payload: ExplanationRequestPayload):
        check_session(session_id)
        return adapter.request_agent_explanation(payload.agent_id, payload.question)

    @app.get("/api/sessions/{session_id}/connection")
    async def connection(session_id: str):
        check_session(session_id)
        return {
            "connected": not security.expired,
            "session_id": adapter.session_id,
            "server_time": time.time(),
        }

    @app.get("/api/sessions/{session_id}/advanced-tracing/overview")
    async def advanced_trace_overview(
        session_id: str,
        goal_id: str | None = None,
        run_id: str | None = None,
    ):
        check_session(session_id)
        return adapter.advanced_trace_overview(goal_id=goal_id, run_id=run_id)

    @app.get("/api/sessions/{session_id}/advanced-tracing/timeline")
    async def advanced_trace_timeline(
        session_id: str,
        goal_id: str | None = None,
        run_id: str | None = None,
        after: int = 0,
        limit: int = 250,
        category: str = "",
        query: str = "",
    ):
        check_session(session_id)
        return adapter.advanced_trace_timeline(
            goal_id=goal_id,
            run_id=run_id,
            after=after,
            limit=limit,
            category=category,
            query=query,
        )

    @app.get("/api/sessions/{session_id}/advanced-tracing/sections/{section}")
    async def advanced_trace_section(
        session_id: str,
        section: str,
        goal_id: str | None = None,
        run_id: str | None = None,
    ):
        check_session(session_id)
        return adapter.advanced_trace_section(
            section,
            goal_id=goal_id,
            run_id=run_id,
        )

    @app.get("/api/sessions/{session_id}/advanced-tracing/inspector/{entity_type}/{entity_id}")
    async def advanced_trace_inspector(
        session_id: str,
        entity_type: str,
        entity_id: str,
        goal_id: str | None = None,
        run_id: str | None = None,
    ):
        check_session(session_id)
        return adapter.advanced_trace_inspector(
            entity_type,
            entity_id,
            goal_id=goal_id,
            run_id=run_id,
        )

    @app.post("/api/sessions/{session_id}/advanced-tracing/reveal")
    async def advanced_trace_reveal(
        session_id: str,
        payload: TraceRevealPayload,
    ):
        check_session(session_id)
        return adapter.advanced_trace_reveal_prompt(
            payload.trace_id,
            goal_id=payload.goal_id,
            run_id=payload.run_id,
        )

    @app.get("/api/sessions/{session_id}/advanced-tracing/export")
    async def advanced_trace_export(
        session_id: str,
        goal_id: str | None = None,
        run_id: str | None = None,
        include_stored_text: bool = False,
    ):
        check_session(session_id)
        payload = adapter.advanced_trace_export(
            goal_id=goal_id,
            run_id=run_id,
            include_stored_text=include_stored_text,
        )
        response = JSONResponse(payload)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="ga3bad-trace-{session_id[:8]}.json"'
        )
        return response

    return app


class LocalWebServer:
    """Own one loopback Uvicorn thread for the lifetime of one CLI runtime."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._execution_requested = threading.Event()
        self.adapter = CoreWebAdapter(
            runtime,
            on_execution_requested=self.request_execution,
        )
        self.security = SessionSecurity(runtime.session_id)
        self.app = create_app(self.adapter, self.security)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._opened_artifacts: set[str] = set()
        self._unsubscribe = runtime.events.subscribe(self._on_runtime_event)
        self.port = 0

    def request_execution(self) -> bool:
        """Wake the owning terminal controller after a web approval."""

        self._execution_requested.set()
        return True

    def take_execution_request(self) -> bool:
        """Consume one coalesced execution wake-up without replaying approval."""

        if not self._execution_requested.is_set():
            return False
        self._execution_requested.clear()
        return True

    def _on_runtime_event(self, event: Any) -> None:
        """Keep monitoring opt-in; Ultra Plan is the only automatic web launch."""

        del event

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._server and self._server.started)

    def start(self, timeout: float = 5.0) -> "LocalWebServer":
        if self.running:
            return self
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.set_inheritable(False)
        self.port = int(sock.getsockname()[1])
        self.app.state.port = self.port
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._socket = sock
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [sock]},
            name="ga3bad-local-web",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if self._server.started:
                # A running loopback server is not the same as an attached Web
                # control surface. The SSE stream owns this flag so terminal
                # fallback remains available until a real browser connects.
                _set_web_control_connected(self.runtime, False)
                return self
            if not self._thread.is_alive():
                break
            time.sleep(0.01)
        self.stop()
        raise RuntimeError("GA3BAD local web server did not start")

    def url_for(self, view_name: str, *, include_token: bool = True) -> str:
        if view_name not in VIEWS:
            raise ValueError(f"unknown local web view: {view_name}")
        if not self.running:
            raise RuntimeError("GA3BAD local web server is not running")
        base = f"http://127.0.0.1:{self.port}/sessions/{self.adapter.session_id}/{view_name}"
        return base + ("?" + urlencode({"token": self.security.token}) if include_token else "")

    def open_view(self, view_name: str) -> dict[str, Any]:
        if view_name not in {"advanced-tracing", "show-diff", "output"}:
            self.adapter.request_view(view_name)
        url = self.url_for(view_name)
        opened = bool(webbrowser.open(url, new=2))
        self.adapter.events.publish(
            "web.view.opened",
            f"{view_name.title()} view opened in your browser." if opened else (
                f"Could not open {view_name.title()} automatically."
            ),
            session_id=self.adapter.session_id,
            view=view_name,
            source="terminal",
            browser_opened=opened,
        )
        return {
            "view": view_name,
            "browser_opened": opened,
            "url": None if opened else url,
            "manual_url": url if not opened else None,
        }

    def stop(self, timeout: float = 5.0) -> None:
        _set_web_control_connected(self.runtime, False)
        self.security.invalidate()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, timeout))
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None
        self._thread = None
        self._server = None

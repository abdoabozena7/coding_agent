"""FastAPI shell and lifecycle for GA3BAD's temporary local workspaces."""

from __future__ import annotations

import json
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..store import NotFoundError, StalePlanError
from .schemas import (
    ExplanationRequestPayload,
    PlanPayload,
    ReviewSubmissionPayload,
)
from .security import SessionSecurity
from .service import CoreWebAdapter


VIEWS = frozenset({"plan", "review", "agents"})


def _error(message: str, status: int, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def create_app(adapter: CoreWebAdapter, security: SessionSecurity) -> FastAPI:
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(
        title="GA3BAD Local Web Views",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.port = 0
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
        token = request.query_params.get("token", "")
        if token:
            if not security.validate_handshake(token):
                raise HTTPException(status_code=401, detail="invalid or expired session token")
            response = RedirectResponse(
                f"/sessions/{session_id}/{view_name}",
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
        return FileResponse(static_dir / "index.html", media_type="text/html")

    def check_session(session_id: str) -> None:
        if session_id != adapter.session_id:
            raise HTTPException(status_code=404, detail="session not found")

    @app.get("/api/sessions/{session_id}/plan")
    async def get_plan(session_id: str):
        check_session(session_id)
        return adapter.plan_snapshot()

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

    return app


class LocalWebServer:
    """Own one loopback Uvicorn thread for the lifetime of one CLI runtime."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.adapter = CoreWebAdapter(runtime)
        self.security = SessionSecurity(runtime.session_id)
        self.app = create_app(self.adapter, self.security)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._opened_artifacts: set[str] = set()
        self._unsubscribe = runtime.events.subscribe(self._on_runtime_event)
        self.port = 0

    def _on_runtime_event(self, event: Any) -> None:
        """Auto-open only mandatory artifact gates; monitoring stays opt-in."""

        view_name = ""
        artifact_key = ""
        if event.kind in {"checkpoint.review_ready", "review.required"}:
            view_name = "review"
            artifact_key = f"review:{event.data.get('checkpoint_id') or event.message}"
        if not view_name or artifact_key in self._opened_artifacts or not self.running:
            return
        self._opened_artifacts.add(artifact_key)
        threading.Thread(
            target=self.open_view,
            args=(view_name,),
            name=f"ga3bad-open-{view_name}",
            daemon=True,
        ).start()

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

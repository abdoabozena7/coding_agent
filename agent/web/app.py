"""FastAPI routes and localhost security for the GA3BAD web client."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import ipaddress
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .registry import WebRegistry
from .schemas import (
    AccessChangeV1,
    AttentionResolutionV1,
    DraftPatchV1,
    InferenceProfileV1,
    MessageDirectionPatchV1,
    ModelRolePatchV1,
    ModelValidationV1,
    PlanDecisionV1,
    ProjectCreateV1,
    SettingsPatchV1,
    ThreadCreateV1,
    ThreadInputV1,
    ThreadPatchV1,
    TerminalCommandV1,
    WorkspaceViewPatchV1,
    WorkflowModePatchV1,
)
from .service import EventHub, JobScheduler, RuntimeManager, jsonable
from .terminal import TerminalManager, TerminalPolicyError
from ..sandbox import DockerSandbox


SESSION_COOKIE = "ga3bad_session"


def _problem(status: int, code: str, message: str, **details: Any) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"code": code, "message": message, "details": details},
    )


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in {"testclient", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _origin_is_allowed(origin: str | None, host_header: str, *, dev: bool) -> bool:
    """Allow the exact loopback page origin plus the configured Vite origin in dev."""
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if (
        parsed.scheme == "http"
        and parsed.username is None
        and parsed.password is None
        and _is_loopback(parsed.hostname)
        and parsed.netloc.lower() == host_header.strip().lower()
    ):
        return True
    if not dev:
        return False
    allowed = {"http://127.0.0.1:5173", "http://localhost:5173"}
    configured = os.getenv("GA3BAD_WEB_DEV_ORIGIN", "").strip().rstrip("/")
    if configured:
        allowed.add(configured)
    return origin.rstrip("/") in allowed


class LocalSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, *, testing: bool = False, dev: bool = False) -> None:
        super().__init__(app)
        self.testing = testing
        self.dev = dev

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self.testing and not _is_loopback(request.client.host if request.client else None):
            return Response("Local access only", status_code=403)
        origin = request.headers.get("origin")
        if not _origin_is_allowed(origin, request.headers.get("host", ""), dev=self.dev):
            return Response("Origin is not allowed", status_code=403)
        public = request.url.path in {"/health", "/"} or request.url.path.startswith("/assets/")
        session_secret = str(request.app.state.session_secret)
        supplied = request.cookies.get(SESSION_COOKIE) or request.headers.get("x-ga3bad-token", "")
        if not public and not hmac.compare_digest(str(supplied), session_secret):
            return Response("Local session is required", status_code=401)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            csrf = request.headers.get("x-ga3bad-csrf", "")
            if not hmac.compare_digest(csrf, str(request.app.state.csrf_token)):
                return Response("CSRF token is required", status_code=403)
        return await call_next(request)


def create_app(
    *,
    registry_path: str | os.PathLike[str] | None = None,
    launch_token: str | None = None,
    testing: bool = False,
    dev: bool = False,
) -> FastAPI:
    token = launch_token or secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = WebRegistry(registry_path)
        hub = EventHub()
        runtimes = RuntimeManager(registry, hub)
        scheduler = JobScheduler(registry, runtimes, hub)
        terminals = TerminalManager(registry, runtimes.effective_access)
        app.state.registry = registry
        app.state.hub = hub
        app.state.runtimes = runtimes
        app.state.scheduler = scheduler
        app.state.terminals = terminals
        try:
            yield
        finally:
            scheduler.close()
            runtimes.close()
            registry.close()

    app = FastAPI(
        title="GA3BAD Local Web API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs" if dev else None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.launch_token = token
    app.state.session_secret = secrets.token_urlsafe(32)
    app.state.csrf_token = secrets.token_urlsafe(24)
    app.state.testing = testing
    app.state.dev = dev
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver", "testclient"],
    )
    app.add_middleware(LocalSecurityMiddleware, testing=testing, dev=dev)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "local_only": True}

    @app.get("/")
    async def root(request: Request, token: str | None = None) -> Response:
        if token:
            expected = str(request.app.state.launch_token or "")
            if not expected or not hmac.compare_digest(token, expected):
                raise _problem(401, "invalid_launch_token", "The launch token is invalid or already used.")
            request.app.state.launch_token = None
            response = RedirectResponse(url="/", status_code=303)
            response.set_cookie(
                SESSION_COOKIE,
                str(request.app.state.session_secret),
                httponly=True,
                samesite="strict",
                secure=False,
                path="/",
            )
            return response
        supplied = request.cookies.get(SESSION_COOKIE) or request.headers.get("x-ga3bad-token", "")
        if not hmac.compare_digest(str(supplied), str(request.app.state.session_secret)):
            return HTMLResponse(
                "<main style='font:16px system-ui;padding:48px;background:#151515;color:#eee;min-height:100vh'>"
                "<h1>GA3BAD</h1><p>Open the one-time URL printed by <code>python -m agent.web</code>.</p></main>",
                status_code=401,
            )
        index = Path(__file__).resolve().parents[2] / "web" / "dist" / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(
            "<main style='font:16px system-ui;padding:48px;background:#151515;color:#eee;min-height:100vh'>"
            "<h1>GA3BAD web build is missing</h1><p>Run <code>npm install</code> and <code>npm run build</code> in the web directory.</p></main>",
            status_code=503,
        )

    @app.get("/api/v1/bootstrap")
    async def bootstrap(request: Request) -> dict[str, Any]:
        payload = request.app.state.scheduler.bootstrap()
        payload["csrf_token"] = str(request.app.state.csrf_token)
        return payload

    @app.get("/api/v1/projects")
    async def list_projects(request: Request) -> list[dict[str, Any]]:
        return request.app.state.registry.list_projects()

    @app.post("/api/v1/projects", status_code=201)
    async def add_project(body: ProjectCreateV1, request: Request) -> dict[str, Any]:
        try:
            project, created = request.app.state.registry.add_project(body.path)
        except ValueError as exc:
            raise _problem(422, "invalid_project", str(exc)) from exc
        request.app.state.hub.publish("project.updated", project, project_id=project["id"])
        return {"project": project, "created": created}

    @app.post("/api/v1/projects/pick-folder")
    async def pick_project_folder() -> dict[str, Any]:
        if os.name != "nt":
            raise _problem(409, "folder_picker_unavailable", "The native folder picker is available on Windows; enter an absolute path on this platform.")

        def choose() -> str:
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog=New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$dialog.Description='Choose a GA3BAD project folder'; "
                "$dialog.ShowNewFolderButton=$false; "
                "if($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){$dialog.SelectedPath}"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", script],
                capture_output=True, text=True, timeout=300, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return completed.stdout.strip()

        try:
            selected = await asyncio.to_thread(choose)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _problem(409, "folder_picker_failed", str(exc)) from exc
        return {"path": selected, "cancelled": not bool(selected)}

    @app.get("/api/v1/projects/{project_id}/threads")
    async def list_threads(project_id: str, request: Request) -> list[dict[str, Any]]:
        if request.app.state.registry.get_project(project_id) is None:
            raise _problem(404, "project_not_found", "Project was not found.")
        return request.app.state.registry.list_threads(project_id)

    @app.post("/api/v1/projects/{project_id}/threads", status_code=201)
    async def create_thread(project_id: str, body: ThreadCreateV1, request: Request) -> dict[str, Any]:
        try:
            thread = request.app.state.registry.create_thread(project_id, body.title)
        except KeyError as exc:
            raise _problem(404, "project_not_found", "Project was not found.") from exc
        request.app.state.hub.publish(
            "thread.updated", thread, project_id=project_id, thread_id=thread["id"]
        )
        return thread

    @app.get("/api/v1/threads/{thread_id}")
    async def get_thread(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.thread_snapshot(thread_id)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc

    @app.patch("/api/v1/threads/{thread_id}")
    async def patch_thread(thread_id: str, body: ThreadPatchV1, request: Request) -> dict[str, Any]:
        try:
            thread = request.app.state.registry.update_thread(
                thread_id, **body.model_dump(exclude_none=True)
            )
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        request.app.state.hub.publish(
            "thread.updated", thread,
            project_id=thread["project_id"], thread_id=thread_id,
        )
        return thread

    @app.patch("/api/v1/threads/{thread_id}/view")
    async def set_thread_view(thread_id: str, body: WorkspaceViewPatchV1, request: Request) -> dict[str, Any]:
        thread = request.app.state.registry.get_thread(thread_id)
        if thread is None:
            raise _problem(404, "thread_not_found", "Task thread was not found.")
        if body.expected_revision is not None and int(thread.get("state_revision", 1)) != body.expected_revision:
            raise _problem(409, "stale_thread", "The task state changed; refresh and try again.")
        return request.app.state.registry.update_thread(thread_id, view_mode=body.view_mode)

    @app.get("/api/v1/threads/{thread_id}/draft")
    async def get_draft(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.registry.get_draft(thread_id)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc

    @app.patch("/api/v1/threads/{thread_id}/draft")
    async def save_draft(thread_id: str, body: DraftPatchV1, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.registry.save_draft(thread_id, body.text, body.expected_revision)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except ValueError as exc:
            raise _problem(409, "stale_draft", "This draft changed in another window; refresh before overwriting it.") from exc

    @app.post("/api/v1/threads/{thread_id}/mode")
    async def set_thread_mode(thread_id: str, body: WorkflowModePatchV1, request: Request) -> dict[str, Any]:
        active = request.app.state.scheduler.active_job_id
        active_job = request.app.state.registry.get_job(active) if active else None
        if active_job and str(active_job["thread_id"]) == thread_id:
            raise _problem(409, "unsafe_reconfiguration", "Pause active work before changing mode.")
        try:
            return request.app.state.scheduler.set_mode(thread_id, body.mode, body.expected_revision)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except ValueError as exc:
            raise _problem(409, "stale_thread", "The task state changed; refresh and try again.") from exc
        except RuntimeError as exc:
            raise _problem(409, "mode_unavailable", str(exc)) from exc

    @app.patch("/api/v1/threads/{thread_id}/messages/{message_id}")
    async def set_message_direction(
        thread_id: str, message_id: int, body: MessageDirectionPatchV1, request: Request,
    ) -> dict[str, Any]:
        try:
            message = request.app.state.registry.update_message_direction(
                thread_id, message_id, body.direction
            )
        except KeyError as exc:
            raise _problem(404, "message_not_found", "Message was not found.") from exc
        except ValueError as exc:
            raise _problem(409, "invalid_message_direction", str(exc)) from exc
        thread = request.app.state.registry.get_thread(thread_id) or {}
        request.app.state.hub.publish(
            "message.updated", message,
            project_id=str(thread.get("project_id") or ""), thread_id=thread_id,
        )
        return message

    @app.post("/api/v1/threads/{thread_id}/plan/decision", status_code=202)
    async def decide_plan(thread_id: str, body: PlanDecisionV1, request: Request) -> dict[str, Any]:
        try:
            job, created = request.app.state.scheduler.plan_decision(
                thread_id,
                action=body.action,
                revision=body.revision,
                fingerprint=body.fingerprint,
                feedback=body.feedback,
                client_request_id=body.client_request_id,
            )
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except ValueError as exc:
            code = "stale_plan" if "stale_plan" in str(exc) else "invalid_plan_feedback"
            raise _problem(409, code, str(exc)) from exc
        return {"job": job, "created": created}

    @app.post("/api/v1/threads/{thread_id}/access")
    async def change_access(thread_id: str, body: AccessChangeV1, request: Request) -> dict[str, Any]:
        thread = request.app.state.registry.get_thread(thread_id)
        if thread is None:
            raise _problem(404, "thread_not_found", "Task thread was not found.")
        if body.expected_revision is not None and int(thread.get("state_revision", 1)) != body.expected_revision:
            raise _problem(409, "stale_thread", "The task state changed; refresh and try again.")
        active = request.app.state.scheduler.active_job_id
        active_job = request.app.state.registry.get_job(active) if active else None
        if active_job and str(active_job["thread_id"]) == thread_id:
            raise _problem(409, "unsafe_reconfiguration", "Pause active work before changing access.")
        if body.policy in {"full", "host"} and not body.confirmation_token:
            return request.app.state.runtimes.prepare_access(thread_id, body.policy)
        try:
            updated = request.app.state.runtimes.apply_access(
                thread_id, body.policy, body.confirmation_token
            )
        except ValueError as exc:
            raise _problem(409, "confirmation_required", str(exc)) from exc
        except RuntimeError as exc:
            raise _problem(409, "docker_unavailable", str(exc)) from exc
        request.app.state.hub.publish(
            "thread.access", updated,
            project_id=str(thread["project_id"]), thread_id=thread_id,
        )
        return {"requires_confirmation": False, "thread": updated}

    @app.post("/api/v1/threads/{thread_id}/inputs", status_code=202)
    async def submit_input(thread_id: str, body: ThreadInputV1, request: Request) -> dict[str, Any]:
        try:
            job, created = request.app.state.scheduler.submit(
                thread_id, body.kind, body.text, body.client_request_id, body.delivery
            )
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except OverflowError as exc:
            raise _problem(409, "queue_full", "This task already has 10 queued messages. The draft was kept.") from exc
        except RuntimeError as exc:
            code = str(exc)
            message = {
                "another_thread_is_running": "Another task is using the global execution slot. Your draft was kept.",
                "guidance_requires_active_thread": "Guidance can only target the currently running task.",
                "workspace_goal_owned_by_other_thread": "Another task owns this workspace's unfinished goal. Finish or cancel it before sending from this task; your draft was kept.",
                "task_killed_create_new": "This task was killed and is preserved as history. Create a new task to continue.",
            }.get(code, code.replace("_", " "))
            raise _problem(409, code, message) from exc
        queued = request.app.state.registry.thread_queue(thread_id)
        position = next((index for index, item in enumerate(queued, 1) if item["id"] == job.get("id")), 0)
        return {"job": job, "created": created, "state": "queued" if position else "executing", "queue_position": position}

    @app.get("/api/v1/threads/{thread_id}/queue")
    async def thread_queue(thread_id: str, request: Request) -> dict[str, Any]:
        if request.app.state.registry.get_thread(thread_id) is None:
            raise _problem(404, "thread_not_found", "Task thread was not found.")
        return {"items": request.app.state.registry.thread_queue(thread_id), "limit": 10}

    @app.delete("/api/v1/threads/{thread_id}/queue/{job_id}")
    async def cancel_queued(thread_id: str, job_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.cancel_queued(thread_id, job_id)
        except KeyError as exc:
            raise _problem(409, "queue_item_started", "This queued message no longer exists or already started.") from exc

    @app.post("/api/v1/threads/{thread_id}/queue/continue")
    async def continue_queue(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            jobs = request.app.state.scheduler.wake_thread_queue(thread_id)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except RuntimeError as exc:
            detail = str(exc)
            if detail.startswith("queue_blocked:"):
                status = detail.partition(":")[2].replace("_", " ")
                raise _problem(409, "queue_blocked", f"The task is {status}; resolve or resume it before continuing the queue.") from exc
            raise _problem(409, "execution_slot_busy", "The execution slot is still busy.") from exc
        return {"items": jobs}

    @app.post("/api/v1/threads/{thread_id}/resume", status_code=202)
    async def resume_thread(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            job, created = request.app.state.scheduler.resume(thread_id, secrets.token_urlsafe(18))
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except RuntimeError as exc:
            if str(exc) == "no_recoverable_checkpoint":
                raise _problem(409, "no_recoverable_checkpoint", "No safe durable input or paused goal is available to resume.") from exc
            raise _problem(409, "execution_slot_busy", "Pause the running task before resuming this one.") from exc
        return {"job": job, "created": created}

    @app.get("/api/v1/threads/{thread_id}/visualization")
    async def visualization(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.visualization(thread_id)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc

    @app.post("/api/v1/threads/{thread_id}/attention/{request_id}/resolve")
    async def resolve_attention(
        thread_id: str,
        request_id: str,
        body: AttentionResolutionV1,
        request: Request,
    ) -> dict[str, Any]:
        resolved = request.app.state.scheduler.resolve_attention(
            thread_id, request_id, body.option_key, body.text
        )
        if not resolved:
            raise _problem(409, "stale_attention", "This question or approval is no longer active.")
        return {"resolved": True}

    @app.post("/api/v1/jobs/{job_id}/checkpoint")
    async def checkpoint(job_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.checkpoint(job_id)
        except KeyError as exc:
            raise _problem(404, "job_not_found", "Job was not found.") from exc

    @app.post("/api/v1/threads/{thread_id}/kill")
    async def kill_task(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.kill_task(thread_id)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except RuntimeError as exc:
            if str(exc) == "task_not_killable":
                raise _problem(409, "task_not_killable", "This task is already finished and has no running or paused work.") from exc
            raise

    @app.get("/api/v1/settings")
    async def get_settings(request: Request) -> dict[str, Any]:
        return request.app.state.registry.settings()

    @app.patch("/api/v1/settings")
    async def patch_settings(body: SettingsPatchV1, request: Request) -> dict[str, Any]:
        changes = body.model_dump(exclude_none=True)
        for key, value in changes.items():
            request.app.state.registry.set_setting(key, value)
        sessions = tuple(request.app.state.runtimes._sessions.values())
        for session in sessions:
            if "experience" in changes:
                session.presentation.set_mode(changes["experience"])
            if "locale" in changes:
                session.presentation.set_locale(changes["locale"])
        settings = request.app.state.registry.settings()
        request.app.state.hub.publish("settings.updated", settings)
        return settings

    @app.get("/api/v1/models")
    async def models(request: Request) -> list[dict[str, Any]]:
        return request.app.state.runtimes.models()

    @app.get("/api/v1/model-settings")
    async def model_settings(request: Request, thread_id: str | None = None) -> dict[str, Any]:
        if thread_id and request.app.state.registry.get_thread(thread_id) is None:
            raise _problem(404, "thread_not_found", "Task thread was not found.")
        return request.app.state.runtimes.model_settings(thread_id)

    @app.post("/api/v1/models/refresh")
    async def refresh_models(request: Request) -> dict[str, Any]:
        return request.app.state.runtimes.model_settings()

    @app.get("/api/v1/settings/advanced")
    async def advanced_settings(request: Request) -> dict[str, Any]:
        return request.app.state.runtimes.inference_profile()

    @app.patch("/api/v1/settings/advanced")
    async def patch_advanced_settings(body: InferenceProfileV1, request: Request) -> dict[str, Any]:
        if request.app.state.scheduler.active_job_id:
            raise _problem(
                409,
                "unsafe_reconfiguration",
                "Checkpoint the active task before changing inference resources.",
            )
        profile = body.model_dump()
        request.app.state.registry.set_setting("inference_profile", profile)
        cached = tuple(request.app.state.runtimes._sessions)
        for thread_id in cached:
            request.app.state.runtimes.drop(thread_id)
        request.app.state.hub.publish("settings.advanced.updated", {"inference_profile": profile})
        return profile

    @app.post("/api/v1/models/validate")
    async def validate_model(body: ModelValidationV1, request: Request) -> dict[str, Any]:
        result = request.app.state.runtimes.validate_model(body.descriptor_id)
        if not result.get("valid"):
            raise _problem(409, "model_unavailable", str(result.get("message") or "Model unavailable"), result=result)
        return result

    @app.patch("/api/v1/model-settings/{role}")
    async def set_default_model_role(role: str, body: ModelRolePatchV1, request: Request) -> dict[str, Any]:
        if role not in {"main", "router", "verifier", "embedding"}:
            raise _problem(404, "model_role_not_found", "Model role was not found.")
        if role == "main" and not body.descriptor_id:
            raise _problem(409, "main_model_required", "The Main model cannot be cleared.")
        if body.descriptor_id and role != "embedding":
            result = request.app.state.runtimes.validate_model(body.descriptor_id)
            if not result.get("valid"):
                raise _problem(409, "model_unavailable", str(result.get("message")))
        profile = dict(request.app.state.registry.settings().get("model_profile") or {})
        profile[role] = body.descriptor_id
        request.app.state.registry.set_setting("model_profile", profile)
        request.app.state.hub.publish("models.updated", {"defaults": profile})
        return request.app.state.runtimes.model_settings()

    @app.patch("/api/v1/threads/{thread_id}/models/{role}")
    async def set_thread_model_role(
        thread_id: str, role: str, body: ModelRolePatchV1, request: Request,
    ) -> dict[str, Any]:
        if role not in {"main", "router", "verifier", "embedding"}:
            raise _problem(404, "model_role_not_found", "Model role was not found.")
        thread = request.app.state.registry.get_thread(thread_id)
        if thread is None:
            raise _problem(404, "thread_not_found", "Task thread was not found.")
        if body.descriptor_id and role != "embedding":
            result = request.app.state.runtimes.validate_model(body.descriptor_id)
            if not result.get("valid"):
                raise _problem(409, "model_unavailable", str(result.get("message")))
        updated = request.app.state.scheduler.queue_model_change(thread_id, role, body.descriptor_id)
        active = request.app.state.scheduler.active_job_id
        active_job = request.app.state.registry.get_job(active) if active else None
        return {
            "thread": updated,
            "state": "queued_for_checkpoint" if active_job and str(active_job["thread_id"]) == thread_id else "applied",
            **request.app.state.runtimes.model_settings(thread_id),
        }

    @app.post("/api/v1/threads/{thread_id}/terminal")
    async def open_terminal(thread_id: str, request: Request) -> dict[str, Any]:
        thread = request.app.state.registry.get_thread(thread_id)
        if thread is None:
            raise _problem(404, "thread_not_found", "Task thread was not found.")
        project = request.app.state.registry.get_project(str(thread["project_id"]))
        assert project is not None
        return request.app.state.terminals.open(thread_id, Path(str(project["path"])))

    @app.post("/api/v1/terminal/{session_id}/command")
    async def terminal_command(session_id: str, body: TerminalCommandV1, request: Request) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(request.app.state.terminals.execute, session_id, body.command)
        except KeyError as exc:
            raise _problem(404, "terminal_not_found", "Terminal session was not found.") from exc
        except TerminalPolicyError as exc:
            raise _problem(409, "terminal_policy", str(exc), remediation="Use a read-only command or explicitly enable the appropriate Full Access boundary.") from exc
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise _problem(409, "terminal_failed", str(exc)) from exc

    @app.delete("/api/v1/terminal/{session_id}")
    async def close_terminal(session_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.terminals.close(session_id)
        except KeyError as exc:
            raise _problem(404, "terminal_not_found", "Terminal session was not found.") from exc

    @app.get("/api/v1/execution/docker")
    async def docker_status() -> dict[str, Any]:
        return jsonable(DockerSandbox().status())

    @app.post("/api/v1/execution/docker/setup")
    async def docker_setup() -> dict[str, Any]:
        try:
            initial = DockerSandbox().status()
            if not initial.docker_available and os.name == "nt":
                desktop = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe"
                if not desktop.is_file():
                    raise RuntimeError("Docker Desktop is not installed in the expected location.")
                subprocess.Popen(
                    [str(desktop)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                for _ in range(60):
                    await asyncio.sleep(1)
                    if DockerSandbox().status().docker_available:
                        break
                else:
                    raise RuntimeError("Docker Desktop started but its Linux daemon did not become ready within 60 seconds.")
            config = await asyncio.to_thread(DockerSandbox().setup)
        except Exception as exc:
            raise _problem(409, "docker_setup_failed", str(exc)) from exc
        return {"configured": True, "config": jsonable(config), "status": jsonable(DockerSandbox().status())}

    @app.get("/api/v1/threads/{thread_id}/telemetry")
    async def task_telemetry(thread_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.telemetry(thread_id)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc

    @app.get("/api/v1/threads/{thread_id}/files/preview")
    async def file_preview(thread_id: str, path: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.scheduler.file_preview(thread_id, path)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc
        except ValueError as exc:
            code = str(exc)
            message = {
                "file_not_previewable": "This file is outside the bounded workspace index or is sensitive.",
                "file_too_large": "Preview is limited to text files smaller than 512 KB.",
                "binary_file": "Binary files cannot be previewed as text.",
            }.get(code, "The file cannot be previewed safely.")
            raise _problem(409, code, message) from exc

    @app.get("/api/v1/threads/{thread_id}/{name}")
    async def inspector(thread_id: str, name: str, request: Request) -> dict[str, Any]:
        if name not in {
            "plan", "changes", "versions", "history", "evidence", "agents",
            "memory", "traces", "metrics", "artifacts", "resources", "files",
        }:
            raise _problem(404, "inspector_not_found", "Inspector was not found.")
        try:
            return request.app.state.scheduler.inspector(thread_id, name)
        except KeyError as exc:
            raise _problem(404, "thread_not_found", "Task thread was not found.") from exc

    @app.websocket("/api/v1/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if not testing and not _is_loopback(websocket.client.host if websocket.client else None):
            await websocket.close(code=1008)
            return
        origin = websocket.headers.get("origin")
        supplied = websocket.cookies.get(SESSION_COOKIE) or websocket.headers.get("x-ga3bad-token", "")
        if (
            not _origin_is_allowed(origin, websocket.headers.get("host", ""), dev=dev)
            or not hmac.compare_digest(str(supplied), str(app.state.session_secret))
        ):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        token_id, queue = app.state.hub.subscribe()
        try:
            await websocket.send_json(
                app.state.hub.canonical_snapshot(app.state.scheduler.bootstrap())
            )
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                except asyncio.TimeoutError:
                    event = app.state.hub.publish("app.ping", {})
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            app.state.hub.unsubscribe(token_id)

    dist = Path(__file__).resolve().parents[2] / "web" / "dist"

    @app.get("/assets/{asset_path:path}")
    async def assets(asset_path: str) -> Response:
        root = (dist / "assets").resolve(strict=False)
        target = (root / asset_path).resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise _problem(404, "asset_not_found", "Asset was not found.") from exc
        if not target.is_file():
            raise _problem(404, "asset_not_found", "Asset was not found.")
        return FileResponse(target)

    @app.get("/{client_path:path}")
    async def spa_fallback(client_path: str, request: Request) -> Response:
        supplied = request.cookies.get(SESSION_COOKIE) or request.headers.get("x-ga3bad-token", "")
        if not hmac.compare_digest(str(supplied), str(request.app.state.session_secret)):
            return RedirectResponse("/")
        index = dist / "index.html"
        return FileResponse(index) if index.is_file() else HTMLResponse("Web build missing", status_code=503)

    return app


__all__ = ["SESSION_COOKIE", "create_app"]

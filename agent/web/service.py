"""Runtime/session orchestration for the local web application."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
import io
import json
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Event, RLock, Thread
import secrets
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from dotenv import load_dotenv

from ..cli import APP_ROOT, _descriptor_for_explicit_model
from ..config import InteractionMode, RuntimeConfig, SessionPreferences
from ..controller import AgentController
from ..events import EventBus, UIEvent
from ..model_catalog import ModelCatalog, ModelDescriptor
from ..providers import get_provider
from ..runtime import AgentRuntime
from ..repository_index import OllamaEmbeddingProvider
from ..sandbox import AccessLevel, DockerSandbox, PermissionAdapter
from ..store import StateStore
from .. import tools as agent_tools
from ..ui import ConsoleUI
from ..ui_state import (
    ActivityStage, AttentionKind, AttentionOption, AttentionRequest, WorkspaceUIStore,
    answer_question, answer_recommended_remaining, question_attention, question_session,
)
from ..version_control import GitProtectionManager, VersionControlError
from .registry import WebRegistry
from .schemas import WebEventV1
from .telemetry import (
    DEFAULT_INFERENCE_PROFILE,
    latest_context_usage,
    progress_estimate,
    resource_snapshot,
)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        # ``dataclasses.asdict`` deep-copies every value and fails on immutable
        # capability mappings (mappingproxy). Serialization only needs a
        # read-only field walk.
        payload = {field.name: getattr(value, field.name) for field in fields(value)}
        if isinstance(value, ModelDescriptor):
            payload["id"] = value.id
        return jsonable(payload)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_PUBLIC_TOOL_ACTIVITY: dict[str, tuple[str, str, str]] = {
    "list_files": ("search", "Inspecting workspace files", "Workspace files inspected"),
    "read_file": ("search", "Reading a project file", "Project file inspected"),
    "grep": ("search", "Searching the workspace", "Workspace search completed"),
    "propose_plan": ("planning", "Validating a plan candidate", "Plan candidate validated"),
    "submit_plan_review": ("validation", "Reviewing the plan independently", "Plan review completed"),
    "request_plan_input": ("question", "Preparing a product decision", "Product decision prepared"),
    "write_file": ("edit", "Updating a project file", "Project file updated"),
    "apply_patch": ("edit", "Applying a bounded project change", "Project change applied"),
    "run_bash": ("command", "Running a project command", "Project command completed"),
    "run_powershell": ("command", "Running a project command", "Project command completed"),
}


def public_runtime_event(event: UIEvent) -> tuple[str, str, dict[str, Any]] | None:
    """Reduce runtime events to safe, useful harness activity.

    Provider thoughts and narrated protocol text are deliberately excluded.
    They are neither product output nor suitable WebSocket payloads.
    """

    kind = str(event.kind or "activity")
    data = dict(event.data or {})
    if kind in {"model_thought", "model_text"}:
        return None
    if kind == "intake.question_answered":
        # The scheduler records one calm decision milestone. Provider/runtime
        # events would otherwise add one noisy row per saved field.
        return None
    if kind == "tool_call":
        tool = str(event.message or data.get("tool") or "tool")
        public_kind, active, _done = _PUBLIC_TOOL_ACTIVITY.get(
            tool, ("tool", "Running a bounded tool", "Tool step completed")
        )
        return public_kind, active, {"tool": tool, "actor": str(data.get("actor") or "agent")}
    if kind == "tool_result":
        tool = str(data.get("tool") or "tool")
        public_kind, _active, done = _PUBLIC_TOOL_ACTIVITY.get(
            tool, ("tool", "Running a bounded tool", "Tool step completed")
        )
        raw = str(event.message or "")
        if raw.startswith(("Error:", "Permission denied")):
            return "problem", f"{tool.replace('_', ' ').title()} stopped at a safe boundary", {"tool": tool}
        if tool == "list_files" and "(no files under" in raw.casefold():
            done = "Workspace inspected · empty greenfield project"
        return public_kind, done, {"tool": tool, "actor": str(data.get("actor") or "agent")}
    fixed = {
        "step": "Model pass started",
        "planning.inspection_recorded": "Workspace inspection recorded",
        "goal_contract.projected": "Execution brief prepared",
        "provider.capability_selected": "Model capabilities verified",
        "provider.request_adapter_selected": "Model adapter selected",
        "action_started": "Safe action started",
        "action_completed": "Safe action completed",
        "usage": "Context usage updated",
        "intake.analyzed": "Requirements analyzed",
    }
    message = fixed.get(kind)
    if message is None:
        if kind.startswith("ultra.") or kind in {
            "phase", "plan", "warning", "error", "validation", "delegation",
            "intake.analyzed", "intake.question_answered", "workflow.retry",
        }:
            message = " ".join(str(event.message or kind.replace(".", " ")).split())[:500]
        else:
            message = kind.replace(".", " ").replace("_", " ").strip().capitalize()
    safe_data = {
        key: jsonable(data[key])
        for key in ("status", "phase", "role", "tool", "actor", "step", "completed", "total")
        if key in data
    }
    return kind, message, safe_data


def workspace_files(workspace: Path, *, limit: int = 400) -> list[dict[str, Any]]:
    """Return a bounded, symlink-safe file index for the read-only web explorer."""
    root = workspace.resolve(strict=True)
    ignored = {
        ".git", ".coding-agent", ".playwright-cli", ".pytest_cache",
        "node_modules", "__pycache__", ".venv", "venv", "run-artifacts", "output",
    }
    result: list[dict[str, Any]] = []
    visited_directories = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        visited_directories += 1
        if visited_directories > 1_000:
            break
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if name not in ignored and not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            normalized = filename.casefold()
            if (
                normalized == ".env"
                or normalized.endswith((".pem", ".key", ".p12", ".pfx"))
                or "credential" in normalized
                or "secret" in normalized
            ):
                continue
            target = current_path / filename
            if target.is_symlink():
                continue
            try:
                relative = target.resolve(strict=True).relative_to(root).as_posix()
                size = target.stat().st_size
            except (OSError, ValueError):
                continue
            result.append({"path": relative, "size": size})
            if len(result) >= max(1, min(limit, 1_000)):
                return result
    return result


def presentation_payload(store: WorkspaceUIStore) -> dict[str, Any]:
    snapshot = store.snapshot()
    return jsonable(snapshot)


class EventHub:
    """Thread-safe broadcast hub for one-process localhost WebSockets."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._sequence = 0
        self._subscribers: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]] = {}

    def publish(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = WebEventV1(
                sequence=self._sequence,
                project_id=project_id,
                thread_id=thread_id,
                type=str(event_type),
                payload=jsonable(dict(payload or {})),
            ).model_dump()
            subscribers = tuple(self._subscribers.values())

        def offer(queue: asyncio.Queue[dict[str, Any]]) -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(offer, queue)
            except RuntimeError:
                continue
        return event

    def subscribe(self) -> tuple[str, asyncio.Queue[dict[str, Any]]]:
        token = uuid4().hex
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers[token] = (asyncio.get_running_loop(), queue)
        return token, queue

    def canonical_snapshot(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Create a connection-local snapshot without consuming a broadcast sequence."""

        with self._lock:
            sequence = self._sequence
        return WebEventV1(
            sequence=sequence,
            type="app.snapshot",
            payload=jsonable(dict(payload)),
        ).model_dump()

    def unsubscribe(self, token: str) -> None:
        with self._lock:
            self._subscribers.pop(str(token), None)


@dataclass(slots=True)
class WebRuntimeSession:
    project_id: str
    thread_id: str
    runtime: AgentRuntime
    store: StateStore
    presentation: WorkspaceUIStore
    console: ConsoleUI
    controller: AgentController
    unsubscribe: Callable[[], None]
    current_turn_id: str | None = None


class RuntimeManager:
    def __init__(self, registry: WebRegistry, hub: EventHub) -> None:
        self.registry = registry
        self.hub = hub
        self._lock = RLock()
        self._sessions: dict[str, WebRuntimeSession] = {}
        self._stores: dict[str, StateStore] = {}
        self._host_full_threads: set[str] = set()
        self._access_challenges: dict[str, tuple[str, str, float]] = {}
        load_dotenv(APP_ROOT / ".env", override=False)

    @staticmethod
    def _environment_descriptor(catalog: ModelCatalog | None = None) -> ModelDescriptor:
        provider_name = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        environment_provider = get_provider(provider_name)
        configured_model = str(getattr(environment_provider, "model", "")).strip()
        catalog = catalog or ModelCatalog()
        try:
            discovered = catalog.discover()
        except Exception:
            discovered = ()
        match = next(
            (
                item
                for item in discovered
                if item.provider == provider_name and item.model == configured_model
            ),
            None,
        )
        if match is not None:
            return match
        return _descriptor_for_explicit_model(provider_name, configured_model, catalog=catalog)

    def _catalog(self) -> tuple[ModelCatalog, tuple[ModelDescriptor, ...]]:
        catalog = ModelCatalog()
        try:
            values = tuple(catalog.discover())
        except Exception:
            values = ()
        return catalog, values

    def _descriptor(self, thread: Mapping[str, Any]) -> ModelDescriptor:
        catalog, values = self._catalog()
        profile = dict(self.registry.settings().get("model_profile") or {})
        overrides = dict(thread.get("model_overrides") or {})
        selected = str(overrides.get("main") or profile.get("main") or "")
        match = next((item for item in values if item.id == selected), None)
        return match or self._environment_descriptor(catalog)

    def inference_profile(self) -> dict[str, Any]:
        return {
            **DEFAULT_INFERENCE_PROFILE,
            **dict(self.registry.settings().get("inference_profile") or {}),
        }

    def _apply_inference_profile(self, provider: Any, descriptor: ModelDescriptor) -> None:
        """Apply supported local-runner controls without pretending cloud APIs obey them."""

        if descriptor.provider != "ollama" or not hasattr(provider, "context_size"):
            return
        profile = self.inference_profile()
        provider.context_size = int(profile["context_window"])
        provider.max_output_tokens = int(profile["max_output_tokens"])
        provider.temperature = float(profile["temperature"])
        provider.top_p = float(profile["top_p"])
        provider.top_k = int(profile["top_k"])
        provider.num_thread = int(profile["cpu_threads"])
        provider.num_batch = {"eco": 128, "balanced": 512, "performance": 1_024}[str(profile["performance"])]
        device = str(profile["device"])
        layers = int(profile["gpu_layers"])
        provider.num_gpu = 0 if device == "cpu" else (999 if device == "gpu" and layers < 0 else layers if layers >= 0 else None)
        provider.require_gpu = device == "gpu"

    def runtime_config(self) -> RuntimeConfig:
        profile = self.inference_profile()
        return replace(
            RuntimeConfig.from_env(),
            planning_steps=int(profile["planning_steps"]),
            work_quantum_steps=int(profile["work_quantum_steps"]),
            review_steps=int(profile["review_steps"]),
            max_provider_retries=int(profile["max_provider_retries"]),
            ultra_cloud_concurrency=int(profile["ultra_cloud_concurrency"]),
            ultra_max_depth=int(profile["ultra_max_depth"]),
        )

    def effective_access(self, thread_id: str) -> str:
        if str(thread_id) in self._host_full_threads:
            return "host"
        thread = self.registry.get_thread(thread_id)
        policy = str((thread or {}).get("access_policy") or "default")
        return {"default": "normal", "bounded": "bounded", "full": "full"}.get(policy, "normal")

    def get(self, thread_id: str) -> WebRuntimeSession:
        with self._lock:
            cached = self._sessions.get(str(thread_id))
            if cached is not None:
                return cached
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        if str(thread.get("status") or "") == "killed":
            raise RuntimeError("task_killed_create_new")
        project = self.registry.get_project(str(thread["project_id"]))
        if project is None:
            raise KeyError("project not found")
        workspace = Path(str(project["path"])).resolve(strict=True)
        with self._lock:
            state = self._stores.get(str(project["id"]))
            if state is None:
                state = StateStore(workspace)
                self._stores[str(project["id"])] = state
        descriptor = self._descriptor(thread)
        provider = descriptor.create_provider()
        self._apply_inference_profile(provider, descriptor)
        settings = self.registry.settings()
        workflow_mode = str(thread.get("workflow_mode") or "normal")
        profile = dict(settings.get("model_profile") or {})
        overrides = dict(thread.get("model_overrides") or {})
        _role_catalog, role_models = self._catalog()
        role_ids = {
            role: str(overrides.get(role) or profile.get(role) or "")
            for role in ("main", "router", "verifier", "embedding")
        }
        role_providers: dict[str, Any] = {}
        for role in ("router", "verifier"):
            role_descriptor = next((item for item in role_models if item.id == role_ids[role]), None)
            if role_descriptor is not None:
                role_providers[role] = role_descriptor.create_provider()
                self._apply_inference_profile(role_providers[role], role_descriptor)
        access = AccessLevel.parse(self.effective_access(thread_id))
        permission_adapter = PermissionAdapter(access, DockerSandbox())
        presentation = WorkspaceUIStore()
        presentation.set_mode(str(settings.get("experience", "simple")))
        presentation.set_locale(str(settings.get("locale", "auto")))
        console = ConsoleUI(
            stream=io.StringIO(),
            color=False,
            input_func=lambda _prompt="": "",
            interaction_mode=str(thread.get("workflow_mode", "normal")),
            plain=True,
            reduced_motion=True,
        )
        console.set_runtime_identity(
            access_level=permission_adapter.access_level.value,
            execution_class=descriptor.execution_class.value,
            model=descriptor.model,
            workspace=str(workspace),
        )
        console.bind_workspace_store(presentation)
        bus = EventBus()
        bus.subscribe(console.on_event)
        holder: dict[str, WebRuntimeSession] = {}

        def relay(event: UIEvent) -> None:
            public = public_runtime_event(event)
            if public is None:
                return
            public_kind, public_message, public_data = public
            active = holder.get("session")
            turn_id = active.current_turn_id if active is not None else None
            if turn_id:
                self.registry.append_activity(
                    str(thread["id"]), turn_id, public_kind, public_message,
                    details=json.dumps(public_data, ensure_ascii=False) if public_data else "",
                )
            self.hub.publish(
                "runtime.event",
                {"kind": public_kind, "message": public_message, "data": public_data},
                project_id=str(project["id"]),
                thread_id=str(thread["id"]),
            )

        unsubscribe = bus.subscribe(relay)
        runtime = AgentRuntime(
            provider,
            state,
            workspace,
            session_id=str(thread["session_id"]),
            events=bus,
            approval=console.confirm_action_decision,
            config=self.runtime_config(),
            model_descriptor=descriptor,
            permission_adapter=permission_adapter,
            auto_promote_ultra=False,
            role_providers=role_providers,
            workflow_mode=workflow_mode,
            direct_normal_execution=True,
        )
        embedding_name = role_ids.get("embedding", "").strip()
        if embedding_name:
            if embedding_name.startswith("ollama:"):
                embedding_name = embedding_name.split(":", 1)[1].split("@", 1)[0]
            runtime.repository_index.embedding_provider = OllamaEmbeddingProvider(embedding_name)
        preferences = SessionPreferences(
            mode=InteractionMode.parse(str(thread.get("workflow_mode", "normal")))
        )
        controller = AgentController(runtime, console, presentation, preferences)
        session = WebRuntimeSession(
            project_id=str(project["id"]),
            thread_id=str(thread["id"]),
            runtime=runtime,
            store=state,
            presentation=presentation,
            console=console,
            controller=controller,
            unsubscribe=unsubscribe,
        )
        holder["session"] = session
        runtime.model_roles = role_ids

        def changed() -> None:
            self.hub.publish(
                "presentation.updated",
                presentation_payload(presentation),
                project_id=session.project_id,
                thread_id=session.thread_id,
            )

        presentation.subscribe(changed)
        with self._lock:
            existing = self._sessions.get(str(thread_id))
            if existing is not None:
                runtime.close()
                unsubscribe()
                return existing
            self._sessions[str(thread_id)] = session
        return session

    def cached(self, thread_id: str) -> WebRuntimeSession | None:
        with self._lock:
            return self._sessions.get(str(thread_id))

    def models(self) -> list[dict[str, Any]]:
        _catalog, values = self._catalog()
        return [jsonable(value) for value in values]

    def model_settings(self, thread_id: str | None = None) -> dict[str, Any]:
        catalog, values = self._catalog()
        thread = self.registry.get_thread(thread_id) if thread_id else None
        return {
            "models": [jsonable(value) for value in values],
            "defaults": dict(self.registry.settings().get("model_profile") or {}),
            "overrides": dict((thread or {}).get("model_overrides") or {}),
            "diagnostics": [jsonable(item) for item in catalog.diagnostics],
            "roles": ["main", "router", "verifier", "embedding"],
            "embedding_default": str(os.getenv("AGENT_EMBEDDING_MODEL") or ""),
        }

    def validate_model(self, descriptor_id: str) -> dict[str, Any]:
        catalog, values = self._catalog()
        descriptor = next((item for item in values if item.id == str(descriptor_id)), None)
        if descriptor is None:
            return {
                "valid": False,
                "message": "The model is unavailable or does not advertise required capabilities.",
                "diagnostics": [jsonable(item) for item in catalog.diagnostics],
            }
        return {"valid": True, "model": jsonable(descriptor), "message": "Model is available."}

    def drop(self, thread_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(str(thread_id), None)
        if session is None:
            return
        session.presentation.mark_exit()
        session.runtime.close()
        session.console.close()
        session.unsubscribe()

    def prepare_access(self, thread_id: str, policy: str) -> dict[str, Any]:
        if self.registry.get_thread(thread_id) is None:
            raise KeyError("thread not found")
        if policy not in {"full", "host"}:
            return {"requires_confirmation": False, "policy": policy}
        token = secrets.token_urlsafe(24)
        self._access_challenges[str(thread_id)] = (token, policy, time.monotonic() + 120)
        docker = jsonable(DockerSandbox().status()) if policy == "full" else None
        return {
            "requires_confirmation": True,
            "confirmation_token": token,
            "policy": policy,
            "docker": docker,
            "warning": (
                "Docker Full skips repeated tool approvals inside the isolated container."
                if policy == "full"
                else "Host Full runs commands directly with your Windows user permissions for this task only."
            ),
        }

    def apply_access(self, thread_id: str, policy: str, confirmation_token: str = "") -> dict[str, Any]:
        if policy in {"full", "host"}:
            expected = self._access_challenges.pop(str(thread_id), None)
            if (
                expected is None or expected[0] != str(confirmation_token)
                or expected[1] != policy or expected[2] < time.monotonic()
            ):
                raise ValueError("a fresh Full Access confirmation is required")
        if policy == "full":
            status = DockerSandbox().status()
            if not status.ready:
                raise RuntimeError(status.reason or "Docker Full is not ready")
        if policy == "host":
            self._host_full_threads.add(str(thread_id))
        else:
            self._host_full_threads.discard(str(thread_id))
            self.registry.update_thread(thread_id, access_policy=policy)
        session = self.cached(thread_id)
        if session is not None:
            session.runtime.replace_permission_adapter(
                PermissionAdapter(AccessLevel.parse(self.effective_access(thread_id)), session.runtime.permission_adapter.sandbox)
            )
        thread = self.registry.get_thread(thread_id) or {}
        return {**thread, "effective_access": self.effective_access(thread_id)}

    def close(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            stores = tuple(self._stores.values())
            self._sessions.clear()
            self._stores.clear()
            self._host_full_threads.clear()
            self._access_challenges.clear()
        for session in sessions:
            session.presentation.mark_exit()
            try:
                session.runtime.checkpoint_interrupt()
            except Exception:
                pass
            session.runtime.close()
            session.console.close()
            session.unsubscribe()
        for store in stores:
            store.close()


class _ForceKilled(RuntimeError):
    """Internal control signal raised when the user abandons unsafe in-flight work."""


class JobScheduler:
    """One durable FIFO execution slot shared by every registered project."""

    def __init__(self, registry: WebRegistry, runtimes: RuntimeManager, hub: EventHub) -> None:
        self.registry = registry
        self.runtimes = runtimes
        self.hub = hub
        self._queue: Queue[str] = Queue()
        self._stop = Event()
        self._lock = RLock()
        self._slot = Condition(self._lock)
        self._active_job_id: str | None = None
        self._job_threads: dict[str, Thread] = {}
        self._force_killed: set[str] = set()
        self.registry.recover_jobs()
        for job in self.registry.list_jobs(statuses=("queued",)):
            thread = self.registry.get_thread(str(job["thread_id"])) or {}
            if str(thread.get("status")) != "recovery_required":
                self._queue.put(str(job["id"]))
        self._worker = Thread(target=self._run, name="ga3bad-web-scheduler", daemon=True)
        self._worker.start()

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

    def _workspace_goal_owner(self, thread: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the task that owns the workspace's one unfinished goal."""

        return next(
            (
                candidate
                for candidate in self.registry.list_threads(str(thread["project_id"]))
                if str(candidate.get("goal_id") or "")
                and str(candidate.get("status") or "") not in {
                    "completed", "cancelled", "failed", "paused",
                    "pause_requested", "waiting_for_input",
                    "recovery_required", "problem", "killed",
                }
            ),
            None,
        )

    def submit(
        self, thread_id: str, kind: str, text: str, client_request_id: str,
        delivery: str = "queue",
    ) -> tuple[dict[str, Any], bool]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        if str(thread.get("status") or "") == "killed":
            raise RuntimeError("task_killed_create_new")
        active_id = self.active_job_id
        active = self.registry.get_job(active_id) if active_id else None
        if active is not None and str(active.get("status") or "") not in {"running", "pause_requested"}:
            active = None
        if delivery == "guidance":
            if active is None or str(active["thread_id"]) != str(thread_id):
                raise RuntimeError("guidance_requires_active_thread")
            session = self.runtimes.cached(thread_id)
            if session is None:
                raise RuntimeError("active_runtime_unavailable")
            session.runtime.add_guidance(str(text))
            turn_id = str(active["id"])
            self.registry.append_message(thread_id, "user", text, turn_id=turn_id)
            self.registry.append_activity(thread_id, turn_id, "guidance", "Guidance saved for the next safe checkpoint")
            self.hub.publish("guidance.queued", {"text": str(text)}, project_id=str(thread["project_id"]), thread_id=thread_id)
            return {**active, "delivery": "guidance", "guidance_queued": True}, True
        existing = self.registry.get_job_by_request(thread_id, client_request_id)
        if existing is not None:
            return existing, False
        owner = self._workspace_goal_owner(thread)
        if owner is not None and str(owner["id"]) != str(thread_id):
            raise RuntimeError("workspace_goal_owned_by_other_thread")
        if active is not None and str(active["thread_id"]) != str(thread_id):
            raise RuntimeError("another_thread_is_running")
        pending = self.registry.thread_queue(thread_id)
        if len(pending) >= 10:
            raise OverflowError("queue_full")
        job, created = self.registry.create_job(
            str(thread["project_id"]),
            str(thread["id"]),
            kind=kind,
            text=text,
            client_request_id=client_request_id,
            delivery=delivery,
        )
        if created:
            prior_status = str(thread.get("status") or "")
            if prior_status in {
                "paused", "pause_requested", "waiting_for_input",
                "recovery_required", "problem",
            }:
                # Queueing guidance behind a blocked task must not erase the
                # reason that currently prevents its queue from running.
                self.registry.update_thread(thread_id, status=prior_status)
            self.registry.append_message(thread_id, "user", text, turn_id=str(job["id"]))
            position = len(self.registry.thread_queue(thread_id))
            self.registry.append_activity(
                thread_id, str(job["id"]), "received",
                "Request queued" if active is not None or position > 1 else "Request received",
                details=json.dumps({"queue_position": position}),
            )
            if str(thread["title"]) == "New task" and not text.lstrip().startswith(("/", ":")):
                self.registry.update_thread(
                    thread_id,
                    title=" ".join(text.split())[:72] or "New task",
                )
            self._queue.put(str(job["id"]))
            self.hub.publish(
                "job.updated", job,
                project_id=str(thread["project_id"]), thread_id=thread_id,
            )
        return job, created

    def cancel_queued(self, thread_id: str, job_id: str) -> dict[str, Any]:
        result = self.registry.cancel_queued_job(thread_id, job_id)
        thread = self.registry.get_thread(thread_id) or {}
        self.hub.publish("queue.updated", {"cancelled": job_id}, project_id=str(thread.get("project_id") or ""), thread_id=thread_id)
        return result

    def wake_thread_queue(self, thread_id: str) -> list[dict[str, Any]]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        if self.active_job_id is not None:
            raise RuntimeError("execution_slot_busy")
        blocked_status = str(thread.get("status") or "")
        if blocked_status in {
            "paused", "pause_requested", "recovery_required", "problem",
            "failed", "waiting_for_input",
        }:
            raise RuntimeError(f"queue_blocked:{blocked_status}")
        jobs = self.registry.thread_queue(thread_id)
        for job in jobs:
            self._queue.put(str(job["id"]))
        return jobs

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                job = self.registry.get_job(job_id)
                thread = self.registry.get_thread(str(job["thread_id"])) if job else None
                thread_status = str((thread or {}).get("status") or "")
                blocked = thread_status in {
                    "paused", "pause_requested", "recovery_required", "problem",
                    "failed", "waiting_for_input",
                } or (
                    thread_status == "awaiting_plan_approval"
                    and str((thread or {}).get("workflow_mode") or "normal") != "plan"
                )
                if blocked or job is None or str(job.get("status") or "") != "queued":
                    continue
                with self._slot:
                    while self._active_job_id is not None and not self._stop.is_set():
                        self._slot.wait(timeout=0.25)
                    if self._stop.is_set():
                        continue
                    # Reserve before starting the thread so the dispatcher can
                    # never launch two mutating jobs in the same slot.
                    self._active_job_id = job_id
                    worker = Thread(
                        target=self._execute,
                        args=(job_id,),
                        name=f"ga3bad-job-{job_id[-8:]}",
                        daemon=True,
                    )
                    self._job_threads[job_id] = worker
                    worker.start()
            finally:
                self._queue.task_done()

    def _is_force_killed(self, job_id: str) -> bool:
        with self._lock:
            return str(job_id) in self._force_killed

    def _raise_if_force_killed(self, job_id: str) -> None:
        if self._is_force_killed(job_id):
            raise _ForceKilled("task was killed by the user")

    def _yield_slot_for_attention(self, job_id: str, request: AttentionRequest) -> None:
        """Pause one controller without blocking unrelated queued work."""

        self._raise_if_force_killed(job_id)
        job = self.registry.get_job(job_id)
        if job is None:
            raise _ForceKilled("task no longer exists")
        thread_id = str(job["thread_id"])
        self.registry.update_job(
            job_id, "paused", blocked_reason=f"attention:{request.id}"
        )
        self.registry.update_thread(thread_id, status="waiting_for_input")
        with self._slot:
            if self._active_job_id == job_id:
                self._active_job_id = None
            self._slot.notify_all()

    def _reacquire_slot_after_attention(self, job_id: str, request: AttentionRequest) -> None:
        """Resume the exact controller only after the global slot is free."""

        with self._slot:
            while (
                self._active_job_id is not None
                and not self._stop.is_set()
                and job_id not in self._force_killed
            ):
                self._slot.wait(timeout=0.25)
            if self._stop.is_set() or job_id in self._force_killed:
                raise _ForceKilled("task was killed while waiting for input")
            self._active_job_id = job_id
        job = self.registry.get_job(job_id)
        if job is None or str(job.get("status") or "") == "cancelled":
            raise _ForceKilled("task was killed while waiting for input")
        self.registry.update_job(job_id, "running", blocked_reason="")
        self.registry.update_thread(str(job["thread_id"]), status="running")

    def _execute(self, job_id: str) -> None:
        job = self.registry.get_job(job_id)
        if job is None or job["status"] != "queued":
            return
        if job.get("cancel_requested"):
            self.registry.update_job(job_id, "cancelled", completed_at=datetime.now().astimezone().isoformat())
            return
        thread_id = str(job["thread_id"])
        project_id = str(job["project_id"])
        with self._slot:
            if self._active_job_id is None:
                self._active_job_id = job_id
        job = self.registry.update_job(
            job_id, "running", started_at=datetime.now().astimezone().isoformat()
        )
        self.registry.update_thread(thread_id, status="running")
        self.hub.publish("job.updated", job, project_id=project_id, thread_id=thread_id)
        try:
            session = self.runtimes.get(thread_id)
            session.current_turn_id = str(job_id)
            session.presentation.set_attention_wait_hooks(
                lambda request: self._yield_slot_for_attention(job_id, request),
                lambda request: self._reacquire_slot_after_attention(job_id, request),
            )
            self.registry.append_activity(thread_id, str(job_id), "running", "Work started")
            before = session.presentation.snapshot()
            last_id = before.transcript[-1].id if before.transcript else 0
            try:
                prior_versions = tuple(session.runtime.version_history(1))
                prior_checkpoint = str(getattr(prior_versions[0], "commit", getattr(prior_versions[0], "id", ""))) if prior_versions else ""
            except (AttributeError, IndexError, OSError, RuntimeError):
                prior_checkpoint = ""
            session.presentation.observe_user_text(str(job["input_text"]))
            session.presentation.append_transcript("user", str(job["input_text"]))
            session.presentation.set_activity(
                ActivityStage.UNDERSTANDING,
                "Understanding your request",
                running=True,
            )
            thread = self.registry.get_thread(thread_id) or {}
            workflow_mode = str(thread.get("workflow_mode") or "normal")
            text = str(job["input_text"])
            if workflow_mode == "normal" and job["kind"] == "message" and session.runtime.active_goal() is None:
                assessment = session.runtime.intent_architect.assess_complexity(text)
                if assessment.ultra_required:
                    resolution = session.presentation.request_attention(
                        AttentionRequest(
                            id=f"mode-suggestion:{job_id}",
                            kind=AttentionKind.QUESTION,
                            title="Ultra is recommended",
                            message="This request has deeper integration or quality requirements. Choose the orchestration before planning.",
                            options=(
                                AttentionOption("ultra", "Use Ultra", "ultra", description="Use recursive specialists and deeper verification.", primary=True),
                                AttentionOption("normal", "Stay Normal", "normal", description="Keep one durable goal and the standard review loop."),
                            ),
                            source="intent_architect",
                        )
                    )
                    if resolution.value == "ultra":
                        self.registry.update_thread(thread_id, workflow_mode="ultra")
                        session.controller.execute("/mode ultra")
                        workflow_mode = "ultra"
                    self.registry.append_activity(
                        thread_id, str(job_id), "mode",
                        f"Mode decision: {resolution.value or 'normal'}",
                    )
            if job["kind"] == "plan_implement":
                revision_text, fingerprint = text.split("|", 1)
                plan = session.runtime.latest_plan()
                if plan is None or plan.revision != int(revision_text) or plan.fingerprint != fingerprint:
                    raise RuntimeError("stale_plan: the plan changed before implementation")
                if workflow_mode == "plan":
                    self.registry.update_thread(thread_id, workflow_mode="normal")
                result = session.controller.execute(f"/approve {plan.revision}")
            elif session.runtime.latest_plan() is not None and session.runtime.dashboard().status == "awaiting_plan_approval" and (
                job["kind"] == "plan_feedback" or (workflow_mode == "plan" and job["kind"] == "message")
            ):
                result = session.controller.execute(f"/replan {text}")
            else:
                result = session.controller.execute(text)
            self._raise_if_force_killed(job_id)

            # Rich web clients receive typed questions.  Keep the global slot
            # at the decision boundary and never leak the CLI's numbered menu,
            # keyboard hints, or slash-command advice into assistant messages.
            while True:
                interview = question_session(session.runtime)
                if interview is None or interview.current is None:
                    break
                decision_number = interview.completed + 1
                session.presentation.set_activity(
                    ActivityStage.PAUSED,
                    f"Waiting for decision {decision_number} of {interview.total}",
                    running=False,
                )
                self.registry.update_thread(thread_id, status="waiting_for_input")
                self.registry.append_activity(
                    thread_id,
                    str(job_id),
                    "question",
                    f"Waiting for decision {decision_number} of {interview.total}",
                )
                resolution = session.presentation.request_attention(
                    question_attention(interview)
                )
                self._raise_if_force_killed(job_id)
                if resolution.value == "__recommended_all__":
                    answer_recommended_remaining(session.runtime)
                    self.registry.append_activity(
                        thread_id,
                        str(job_id),
                        "question_answered",
                        "Recommended answers applied to all remaining decisions",
                    )
                    self.registry.update_thread(thread_id, status="running")
                    session.presentation.set_activity(
                        ActivityStage.UNDERSTANDING,
                        "Applying the recommended decisions",
                        running=True,
                    )
                    continue
                answer = resolution.text if resolution.key == "custom" else resolution.value
                answer_question(
                    session.runtime,
                    interview,
                    str(interview.current.get("id") or ""),
                    answer or "1",
                )
                self.registry.append_activity(
                    thread_id,
                    str(job_id),
                    "question_answered",
                    f"Decision {decision_number} of {interview.total} answered",
                )
                self.registry.update_thread(thread_id, status="running")
                session.presentation.set_activity(
                    ActivityStage.UNDERSTANDING,
                    "Applying your answer",
                    running=True,
                )
            self._raise_if_force_killed(job_id)
            latest_plan = session.runtime.latest_plan()
            refreshed_thread = self.registry.get_thread(thread_id) or {}
            if latest_plan is not None and not str(refreshed_thread.get("plan_series_id") or ""):
                goal = session.runtime.active_goal() or session.store.get_latest_goal()
                self.registry.update_thread(
                    thread_id,
                    plan_series_id=f"plan-series:{getattr(goal, 'id', thread_id)}",
                )
            after = session.presentation.snapshot()
            added = [entry for entry in after.transcript if entry.id > last_id and entry.role != "user"]
            for entry in added:
                self.registry.append_message(
                    thread_id,
                    entry.role,
                    entry.text,
                    technical=entry.technical,
                    turn_id=str(job_id),
                )
            output = "\n\n".join(entry.text for entry in added[-8:])
            try:
                current_versions = tuple(session.runtime.version_history(1))
                current_checkpoint = str(getattr(current_versions[0], "commit", getattr(current_versions[0], "id", ""))) if current_versions else ""
            except (AttributeError, IndexError, OSError, RuntimeError):
                current_checkpoint = ""
            if current_checkpoint and current_checkpoint != prior_checkpoint:
                self.registry.append_activity(
                    thread_id,
                    str(job_id),
                    "checkpoint",
                    "Accepted Git checkpoint available for review or undo",
                    details=json.dumps({"valid": True, "checkpoint_id": current_checkpoint}),
                )
            goal = session.runtime.active_goal() or session.store.get_latest_goal()
            status = str(getattr(session.runtime.dashboard(), "status", result.status))
            session.presentation.set_activity(
                ActivityStage.DONE if status == "completed" else ActivityStage.IDLE,
                "Done" if status == "completed" else "Ready",
                running=False,
            )
            self.registry.update_thread(
                thread_id,
                status=status,
                goal_id=goal.id if goal is not None else None,
            )
            terminal_job_status = "paused" if status in {"paused", "recovery_required"} else "completed"
            job = self.registry.update_job(
                job_id,
                terminal_job_status,
                result_text=output,
                completed_at=datetime.now().astimezone().isoformat(),
            )
            self.registry.append_activity(
                thread_id, str(job_id), terminal_job_status,
                "Paused at a safe checkpoint" if terminal_job_status == "paused" else "Turn completed",
            )
            self._apply_pending_models(thread_id)
        except _ForceKilled:
            session = self.runtimes.cached(thread_id)
            if session is not None:
                session.presentation.set_activity(
                    ActivityStage.PAUSED, "Task killed by the user", running=False,
                )
            current = self.registry.get_job(job_id)
            if current is not None and str(current.get("status") or "") != "cancelled":
                job = self.registry.update_job(
                    job_id,
                    "cancelled",
                    cancel_requested=True,
                    blocked_reason="force_killed_by_user",
                    completed_at=datetime.now().astimezone().isoformat(),
                )
            else:
                job = current or job
            self.registry.update_thread(thread_id, status="killed")
        except Exception as exc:
            durable_job = self.registry.get_job(job_id)
            if self._is_force_killed(job_id) or (
                durable_job is not None
                and str(durable_job.get("status") or "") == "cancelled"
                and str(durable_job.get("blocked_reason") or "") == "force_killed_by_user"
            ):
                job = durable_job or job
                self.registry.update_thread(thread_id, status="killed")
                return
            message = f"{type(exc).__name__}: {exc}"
            session = self.runtimes.cached(thread_id)
            if session is not None:
                session.presentation.set_activity(
                    ActivityStage.PROBLEM, "Work stopped and needs attention",
                    running=False,
                )
            self.registry.append_message(thread_id, "assistant", message, turn_id=str(job_id))
            self.registry.append_activity(thread_id, str(job_id), "problem", message)
            self.registry.update_thread(thread_id, status="problem")
            job = self.registry.update_job(
                job_id,
                "failed",
                error=message,
                completed_at=datetime.now().astimezone().isoformat(),
            )
        finally:
            session = self.runtimes.cached(thread_id)
            if session is not None:
                session.current_turn_id = None
                session.presentation.set_attention_wait_hooks(None, None)
            was_force_killed = self._is_force_killed(job_id)
            with self._slot:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                self._force_killed.discard(job_id)
                self._slot.notify_all()
            job = self.registry.get_job(job_id) or job
            self.hub.publish("job.updated", job, project_id=project_id, thread_id=thread_id)
            self.hub.publish(
                "thread.updated",
                self.thread_snapshot(thread_id),
                project_id=project_id,
                thread_id=thread_id,
            )
            if was_force_killed:
                drop = getattr(self.runtimes, "drop", None)
                if callable(drop):
                    drop(thread_id)
            with self._slot:
                self._job_threads.pop(job_id, None)
                self._slot.notify_all()

    def set_mode(self, thread_id: str, mode: str, expected_revision: int | None = None) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        if expected_revision is not None and int(thread.get("state_revision", 1)) != int(expected_revision):
            raise ValueError("stale_thread")
        session = self.runtimes.cached(thread_id)
        if mode == "ultra" and session is not None:
            issue = session.runtime.ultra_readiness_issue()
            if issue:
                raise RuntimeError(issue)
        updated = self.registry.update_thread(thread_id, workflow_mode=mode)
        if session is not None:
            session.controller.execute(f"/mode {mode}")
        self.hub.publish("thread.mode", {"mode": mode}, project_id=str(thread["project_id"]), thread_id=thread_id)
        return updated

    def plan_decision(
        self,
        thread_id: str,
        *,
        action: str,
        revision: int,
        fingerprint: str,
        feedback: str,
        client_request_id: str,
    ) -> tuple[dict[str, Any], bool]:
        existing = next(
            (
                item for item in self.registry.list_jobs()
                if str(item.get("thread_id")) == str(thread_id)
                and str(item.get("client_request_id")) == str(client_request_id)
            ),
            None,
        )
        if existing is not None:
            return existing, False
        session = self.runtimes.get(thread_id)
        plan = session.runtime.latest_plan()
        if plan is None or plan.revision != int(revision) or plan.fingerprint != str(fingerprint):
            raise ValueError("stale_plan")
        if action == "keep_planning":
            if not str(feedback).strip():
                raise ValueError("plan feedback is required")
            return self.submit(thread_id, "plan_feedback", str(feedback), client_request_id)
        if self.registry.thread_queue(thread_id):
            raise ValueError("plan_feedback_pending")
        thread = self.registry.get_thread(thread_id)
        assert thread is not None
        job, created = self.registry.create_job(
            str(thread["project_id"]), thread_id,
            kind="plan_implement",
            text=f"{plan.revision}|{plan.fingerprint}",
            client_request_id=client_request_id,
        )
        if created:
            self.registry.append_activity(thread_id, str(job["id"]), "approval", f"Implement plan r{plan.revision}")
            self._queue.put(str(job["id"]))
            self.hub.publish("job.updated", job, project_id=str(thread["project_id"]), thread_id=thread_id)
        return job, created

    def checkpoint(self, job_id: str) -> dict[str, Any]:
        job = self.registry.get_job(job_id)
        if job is None:
            raise KeyError("job not found")
        if job["status"] == "queued":
            updated = self.registry.update_job(
                job_id,
                "cancelled",
                cancel_requested=True,
                completed_at=datetime.now().astimezone().isoformat(),
            )
            self.registry.update_thread(str(job["thread_id"]), status="paused")
            return updated
        if job["status"] != "running":
            return job
        session = self.runtimes.cached(str(job["thread_id"]))
        if session is not None:
            session.runtime.checkpoint_interrupt()
            session.presentation.set_activity(
                ActivityStage.CHECKING,
                "Checkpoint requested; finishing the current action safely",
                running=True,
            )
        self.registry.update_thread(str(job["thread_id"]), status="pause_requested")
        return self.registry.update_job(job_id, "pause_requested", cancel_requested=True)

    def kill_task(self, thread_id: str) -> dict[str, Any]:
        """Abandon a task immediately without waiting for a safe checkpoint.

        The durable record is terminal immediately. In-flight model generation
        and managed process trees receive best-effort hard cancellation; any
        action whose exact effect cannot be proven remains visible in history.
        """

        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        candidates = [
            item for item in self.registry.list_jobs()
            if str(item.get("thread_id")) == str(thread_id)
            and str(item.get("status") or "") in {
                "queued", "running", "pause_requested", "paused",
                "recovery_required",
            }
        ]
        session = self.runtimes.cached(thread_id)
        if not candidates and session is None and str(thread.get("status") or "") in {
            "idle", "completed", "cancelled", "killed", "failed",
        }:
            raise RuntimeError("task_not_killable")

        now = datetime.now().astimezone().isoformat()
        with self._slot:
            for item in candidates:
                job_id = str(item["id"])
                if job_id in self._job_threads or self._active_job_id == job_id:
                    self._force_killed.add(job_id)

        for item in candidates:
            job_id = str(item["id"])
            if str(item.get("status")) == "queued":
                try:
                    self.registry.cancel_queued_job(thread_id, job_id)
                    continue
                except KeyError:
                    pass
            self.registry.update_job(
                job_id,
                "cancelled",
                cancel_requested=True,
                blocked_reason="force_killed_by_user",
                completed_at=now,
            )

        if session is not None:
            try:
                session.runtime.cancel("CANCEL")
            except Exception:
                # The job may be between goal creation and its first durable
                # state. The web registry still records the explicit kill.
                pass
            try:
                agent_tools.shutdown_workspace_resources(session.runtime.workspace)
            except Exception:
                pass

            main_provider = getattr(session.runtime, "provider", None)
            role_providers = getattr(session.runtime, "role_providers", {})
            providers = tuple(
                item for item in (
                    main_provider,
                    *(role_providers.values() if isinstance(role_providers, Mapping) else ()),
                )
                if item is not None
            )

            def stop_generations() -> None:
                for provider in providers:
                    cancel = getattr(provider, "_cancel_active_generation", None)
                    close = getattr(provider, "close", None)
                    try:
                        if callable(cancel):
                            cancel()
                        elif callable(close):
                            close()
                    except Exception:
                        pass

            Thread(
                target=stop_generations,
                name=f"ga3bad-kill-{thread_id[-8:]}",
                daemon=True,
            ).start()
            session.presentation.mark_exit()

        self.registry.append_activity(
            thread_id,
            str(candidates[-1]["id"]) if candidates else "task-control",
            "killed",
            "Task killed immediately by the user",
            details=json.dumps({
                "safe_checkpoint": False,
                "uncertain_effects": True,
            }),
        )
        updated_thread = self.registry.update_thread(thread_id, status="killed")
        self.hub.publish(
            "task.killed",
            {"thread_id": thread_id, "safe_checkpoint": False},
            project_id=str(thread["project_id"]),
            thread_id=thread_id,
        )
        with self._slot:
            self._slot.notify_all()
        return {
            "thread": updated_thread,
            "killed_jobs": [str(item["id"]) for item in candidates],
            "safe_checkpoint": False,
        }

    def resume(self, thread_id: str, client_request_id: str) -> tuple[dict[str, Any], bool]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        if self.active_job_id is not None:
            raise RuntimeError("execution_slot_busy")
        session = self.runtimes.get(thread_id)
        goal = session.runtime.active_goal()
        interrupted = [
            item for item in self.registry.list_jobs()
            if str(item.get("thread_id")) == thread_id
            and (
                item.get("status") == "recovery_required"
                or item.get("blocked_reason") == "superseded_by_explicit_resume"
            )
        ]
        retry_source = next((
            item for item in reversed(interrupted)
            if str(item.get("kind")) in {"message", "plan_feedback"}
            and str(item.get("input_text") or "").strip()
        ), None)
        if goal is not None and str(getattr(goal.status, "value", goal.status)) != "paused":
            pause = getattr(session.runtime, "pause", None)
            if callable(pause):
                pause("explicit web recovery reconciliation")
        now = datetime.now().astimezone().isoformat()
        for interrupted_job in self.registry.list_jobs(statuses=("recovery_required",)):
            if str(interrupted_job.get("thread_id")) == thread_id:
                self.registry.update_job(
                    str(interrupted_job["id"]), "cancelled", cancel_requested=True,
                    completed_at=now, blocked_reason="superseded_by_explicit_resume",
                )
        self.registry.update_thread(thread_id, status="paused")
        resume_kind = "command" if goal is not None else str((retry_source or {}).get("kind") or "")
        resume_text = "/resume" if goal is not None else str((retry_source or {}).get("input_text") or "")
        if not resume_kind or not resume_text:
            raise RuntimeError("no_recoverable_checkpoint")
        job, created = self.registry.create_job(
            str(thread["project_id"]), thread_id, kind=resume_kind, text=resume_text,
            client_request_id=client_request_id, delivery="queue",
        )
        if created:
            self.registry.append_activity(
                thread_id, str(job["id"]), "resume",
                "Restarting interrupted read-only intake from durable state"
                if goal is None else "Explicit resume requested from the durable checkpoint",
            )
            self._queue.put(str(job["id"]))
            for pending in self.registry.thread_queue(thread_id):
                if str(pending["id"]) != str(job["id"]):
                    self._queue.put(str(pending["id"]))
            self.hub.publish(
                "job.updated", job, project_id=str(thread["project_id"]),
                thread_id=thread_id,
            )
        return job, created

    def queue_model_change(self, thread_id: str, role: str, descriptor_id: str) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        pending = dict(thread.get("pending_model_overrides") or {})
        pending[role] = str(descriptor_id)
        updated = self.registry.update_thread(thread_id, pending_model_overrides=pending)
        active = self.registry.get_job(self.active_job_id) if self.active_job_id else None
        if active is None or str(active["thread_id"]) != thread_id:
            self._apply_pending_models(thread_id)
            updated = self.registry.get_thread(thread_id) or updated
        self.hub.publish("models.queued", {"role": role, "descriptor_id": descriptor_id}, project_id=str(thread["project_id"]), thread_id=thread_id)
        return updated

    def _apply_pending_models(self, thread_id: str) -> None:
        thread = self.registry.get_thread(thread_id) or {}
        pending = dict(thread.get("pending_model_overrides") or {})
        if not pending:
            return
        overrides = dict(thread.get("model_overrides") or {})
        for role, descriptor_id in pending.items():
            if descriptor_id:
                result = self.runtimes.validate_model(str(descriptor_id)) if role != "embedding" else {"valid": True}
                if not result.get("valid"):
                    self.registry.update_thread(thread_id, status="paused")
                    self.registry.append_activity(thread_id, "model-control", "problem", f"Model switch for {role} could not be applied: {result.get('message')}")
                    return
                overrides[str(role)] = str(descriptor_id)
            else:
                overrides.pop(str(role), None)
        self.registry.update_thread(thread_id, model_overrides=overrides, pending_model_overrides={})
        self.runtimes.drop(thread_id)

    def resolve_attention(self, thread_id: str, request_id: str, key: str, text: str) -> bool:
        session = self.runtimes.cached(thread_id)
        if session is None:
            return False
        request = session.presentation.active_attention()
        interview = question_session(session.runtime)
        active = self.registry.get_job(self.active_job_id) if self.active_job_id else None
        active_same = bool(active and str(active.get("thread_id")) == str(thread_id))
        controlled_wait = any(
            str(item.get("thread_id")) == str(thread_id)
            and str(item.get("status") or "") == "paused"
            and str(item.get("blocked_reason") or "") == f"attention:{request_id}"
            for item in self.registry.list_jobs()
        )
        resolved = session.controller.resolve_attention(request_id, key, text)
        if resolved:
            # Repair question rounds created by older web builds, which wrote a
            # terminal menu and then completed the job without a waiting
            # controller. New work is consumed by the active worker above.
            if (
                not active_same
                and not controlled_wait
                and request is not None
                and interview is not None
                and interview.current is not None
                and request.source == interview.source
            ):
                resolution = session.presentation.take_attention_result(request_id)
                if resolution is not None:
                    if resolution.value == "__recommended_all__":
                        answer_recommended_remaining(session.runtime)
                    else:
                        value = resolution.text if resolution.key == "custom" else resolution.value
                        answer_question(
                            session.runtime,
                            interview,
                            str(interview.current.get("id") or ""),
                            value or "1",
                        )
                    next_interview = question_session(session.runtime)
                    if next_interview is not None and next_interview.current is not None:
                        session.presentation.present_attention(question_attention(next_interview))
                        session.presentation.set_activity(
                            ActivityStage.PAUSED,
                            f"Waiting for decision {next_interview.completed + 1} of {next_interview.total}",
                            running=False,
                        )
                        self.registry.update_thread(thread_id, status="waiting_for_input")
                    else:
                        dashboard = session.runtime.dashboard()
                        goal = session.runtime.active_goal() or session.store.get_latest_goal()
                        self.registry.update_thread(
                            thread_id,
                            status=str(getattr(dashboard, "status", "idle")),
                            goal_id=getattr(goal, "id", None),
                        )
            self.hub.publish(
                "attention.resolved",
                {"request_id": request_id, "option_key": key},
                project_id=session.project_id,
                thread_id=thread_id,
            )
        return resolved

    def thread_snapshot(self, thread_id: str) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        project = self.registry.get_project(str(thread["project_id"]))
        session = self.runtimes.cached(thread_id)
        terminal_killed = str(thread.get("status") or "") == "killed"
        should_load_runtime = not terminal_killed and bool(
            thread.get("goal_id")
            or str(thread.get("session_id") or "") == "workspace-session"
            or str(thread.get("status") or "") not in {"idle", "completed", "cancelled"}
            or self.registry.list_messages(thread_id)
        )
        if session is None and should_load_runtime:
            try:
                session = self.runtimes.get(thread_id)
            except Exception:
                session = None
        running_jobs = [
            job for job in self.registry.list_jobs(statuses=("queued", "running", "pause_requested", "paused", "recovery_required"))
            if job["thread_id"] == thread_id
        ]
        if (
            session is not None
            and not running_jobs
            and session.presentation.active_attention() is None
            and str(thread.get("status") or "") not in {"paused", "recovery_required", "problem", "failed", "killed"}
        ):
            interview = question_session(session.runtime)
            if interview is not None and interview.current is not None:
                session.presentation.present_attention(question_attention(interview))
                session.presentation.set_activity(
                    ActivityStage.PAUSED,
                    f"Waiting for decision {interview.completed + 1} of {interview.total}",
                    running=False,
                )
                thread = self.registry.update_thread(thread_id, status="waiting_for_input")
        presentation = presentation_payload(session.presentation) if session else None
        dashboard = None
        if session is not None:
            try:
                active_goal = session.runtime.active_goal()
                owns_active_goal = bool(
                    active_goal is not None
                    and str(thread.get("goal_id") or "") == str(getattr(active_goal, "id", ""))
                )
                legacy_session = str(thread.get("session_id") or "") == "workspace-session"
                if owns_active_goal or str(thread.get("goal_id") or "") or legacy_session:
                    dashboard = jsonable(session.runtime.dashboard())
            except Exception:
                dashboard = None
        payload: dict[str, Any] = {
            "project": project,
            "thread": thread,
            "messages": self.registry.list_messages(thread_id),
            "turns": self.registry.conversation_turns(thread_id),
            "jobs": running_jobs,
            "settings": self.registry.settings(),
            "presentation": presentation,
            "dashboard": dashboard,
            "queue": self.registry.thread_queue(thread_id),
            "draft": self.registry.get_draft(thread_id),
        }
        payload["thread"] = {**thread, "effective_access": self.runtimes.effective_access(thread_id)}
        all_jobs = self.registry.list_jobs()
        current_job = next((item for item in running_jobs if item.get("status") == "running"), None)
        profile_loader = getattr(self.runtimes, "inference_profile", None)
        inference_profile = profile_loader() if callable(profile_loader) else dict(DEFAULT_INFERENCE_PROFILE)
        payload["progress"] = progress_estimate(
            dashboard,
            presentation,
            current_job,
            (
                item for item in all_jobs
                if item.get("project_id") == thread.get("project_id") and item.get("status") == "completed"
            ),
            workflow_mode=str(thread.get("workflow_mode") or "normal"),
            profile=inference_profile,
        )
        payload["capabilities"] = self.capabilities(thread_id, dashboard=dashboard, presentation=presentation)
        return payload

    def capabilities(
        self, thread_id: str, *, dashboard: Mapping[str, Any] | None = None,
        presentation: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        active = self.registry.get_job(self.active_job_id) if self.active_job_id else None
        if active is not None and str(active.get("status") or "") not in {"running", "pause_requested"}:
            active = None
        active_same = bool(active and str(active["thread_id"]) == thread_id)
        active_other = bool(active and not active_same)
        queued = self.registry.thread_queue(thread_id)
        status = str(thread.get("status") or "idle")
        owner = self._workspace_goal_owner(thread)
        owned_elsewhere = bool(owner is not None and str(owner["id"]) != str(thread_id))
        view = presentation if isinstance(presentation, Mapping) else {}
        board = dashboard if isinstance(dashboard, Mapping) else {}
        waiting = bool(view.get("attention")) or status in {
            "waiting_for_input", "paused", "recovery_required", "problem",
        }
        killable = status not in {"idle", "completed", "cancelled", "killed", "failed"} or any(
            str(item.get("status") or "") in {"running", "pause_requested", "paused", "recovery_required"}
            for item in self.registry.list_jobs()
            if str(item.get("thread_id")) == thread_id
        )

        def item(allowed: bool, reason: str = "", remediation: str = "") -> dict[str, Any]:
            return {"allowed": bool(allowed), "reason": reason if not allowed else "", "remediation": remediation if not allowed else ""}

        send_reason = (
            "This task is terminal because it was killed."
            if status == "killed"
            else f"{owner.get('title') or 'Another task'} owns this workspace's unfinished goal."
            if owned_elsewhere and owner is not None
            else "Another task is using the global execution slot."
            if active_other
            else "This task queue already contains 10 messages."
        )
        return {
            "send": item(
                status != "killed" and not owned_elsewhere and not active_other and len(queued) < 10,
                send_reason,
                "Create a new task; killed tasks are preserved as terminal history."
                if status == "killed"
                else "Finish or cancel the owning task before starting another goal in this workspace."
                if owned_elsewhere
                else "Pause the active task or wait for a queued message to start.",
            ),
            "guidance": item(active_same, "Guidance can only target the currently running task.", "Open the running task or use Add to queue."),
            "pause": item(
                active_same and str(active.get("status")) == "running",
                "A safe checkpoint is already requested." if active_same and str(active.get("status")) == "pause_requested" else "No action is currently running in this task.",
                "Wait for the current action to reach its safe boundary." if active_same and str(active.get("status")) == "pause_requested" else "Start or resume work first.",
            ),
            "kill": item(
                killable,
                "This task is already finished.",
                "Start a new task if you have more work.",
            ),
            "resume": item(not active and status in {"paused", "recovery_required", "problem"}, "Resume requires a paused task and a free execution slot.", "Resolve attention or pause the running task."),
            "change_mode": item(not active_same, "Workflow mode cannot change during active work.", "Pause at a safe checkpoint first."),
            "implement_plan": item(
                str(board.get("status")) == "awaiting_plan_approval" and not queued and not active_same,
                "The latest plan must be idle with no queued feedback.",
                "Wait for planning feedback to finish or cancel it.",
            ),
            "terminal": item(True),
            "visualize": item(True),
            "change_model": item(True, "", "Model changes are queued for the next safe checkpoint while work is active."),
            "continue_queue": item(not active and bool(queued) and not waiting, "The queue is empty or the task still needs attention.", "Resolve the current attention or add a queued message."),
        }

    def visualization(self, thread_id: str) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        session = self.runtimes.get(thread_id)
        board = jsonable(session.runtime.dashboard())
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        objective = str(board.get("objective") or thread.get("title") or "Task")
        root_id = "goal"
        nodes.append({"id": root_id, "kind": "goal", "label": objective[:100], "summary": objective, "status": str(board.get("status") or thread.get("status") or "idle"), "parent_id": None})
        task_ids: set[str] = set()
        for index, task in enumerate(board.get("tasks") or [], 1):
            task_id = str(task.get("id") or f"task-{index}")
            task_ids.add(task_id)
            nodes.append({
                "id": task_id, "kind": "task", "label": str(task.get("title") or f"Step {index}"),
                "summary": str(task.get("description") or task.get("title") or ""),
                "status": str(task.get("status") or "pending"), "parent_id": root_id,
                "details": {"role": str(task.get("role") or ""), "risk": str(task.get("risk") or ""), "verification": list(task.get("verification") or [])[:8]},
            })
            dependencies = [str(value) for value in task.get("depends_on") or []]
            if dependencies:
                edges.extend({"id": f"dependency:{dep}:{task_id}", "source": dep, "target": task_id, "kind": "dependency"} for dep in dependencies)
            else:
                edges.append({"id": f"contains:{root_id}:{task_id}", "source": root_id, "target": task_id, "kind": "sequence"})
        try:
            agents = jsonable(session.store.list_agent_registry())
        except Exception:
            agents = []
        for index, agent in enumerate(agents or [], 1):
            agent_id = f"agent:{agent.get('runtime_id') or index}"
            assigned = str(agent.get("assigned_id") or "")
            role = agent.get("role") or {}
            role_name = str(role.get("name") if isinstance(role, Mapping) else role or "Agent")
            nodes.append({
                "id": agent_id, "kind": "agent", "label": role_name,
                "summary": str(role.get("mission") if isinstance(role, Mapping) else "")[:2_000],
                "status": str(agent.get("state") or "waiting"),
                "parent_id": assigned if assigned in task_ids else root_id,
                "details": {"provider": agent.get("provider"), "model": agent.get("model"), "assignment": assigned, "evidence_refs": list(agent.get("evidence_refs") or [])[:20]},
            })
            parent = assigned if assigned in task_ids else root_id
            edges.append({"id": f"delegates:{parent}:{agent_id}", "source": parent, "target": agent_id, "kind": "delegates"})
        activities = self.registry.list_activity(thread_id, limit=120)
        problem_index = 0
        for activity in activities:
            kind = str(activity.get("kind") or "activity")
            if kind not in {"problem", "checkpoint", "approval", "question", "retry", "validation"}:
                continue
            problem_index += 1
            node_id = f"event:{activity.get('id') or problem_index}"
            node_kind = "error" if kind == "problem" else kind
            nodes.append({"id": node_id, "kind": node_kind, "label": str(activity.get("summary") or kind)[:120], "summary": str(activity.get("summary") or ""), "status": "failed" if kind == "problem" else "completed", "parent_id": root_id})
            edges.append({"id": f"event-edge:{node_id}", "source": root_id, "target": node_id, "kind": "retry" if kind in {"problem", "retry"} else "evidence"})
        return {
            "version": 1, "thread_id": thread_id, "mode": str(thread.get("workflow_mode") or "normal"),
            "revision": int(thread.get("state_revision") or 1), "current_node_id": next((node["id"] for node in nodes if node["status"] in {"running", "in_progress", "retrying"}), root_id),
            "nodes": nodes, "edges": edges, "updated_at": datetime.now().astimezone().isoformat(),
        }

    def telemetry(self, thread_id: str) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        session = self.runtimes.cached(thread_id)
        context_limit = None
        if session is not None:
            context_limit = getattr(getattr(session.runtime, "provider", None), "context_size", None)
        if context_limit is None:
            profile_loader = getattr(self.runtimes, "inference_profile", None)
            profile = profile_loader() if callable(profile_loader) else DEFAULT_INFERENCE_PROFILE
            context_limit = profile.get("context_window")
        used = latest_context_usage(self.registry.list_activity(thread_id, limit=500))
        return resource_snapshot(context_used=used, context_limit=context_limit)

    def file_preview(self, thread_id: str, relative_path: str) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        project = self.registry.get_project(str(thread["project_id"]))
        if project is None:
            raise KeyError("project not found")
        workspace = Path(str(project["path"])).resolve(strict=True)
        normalized = str(relative_path).replace("\\", "/").strip("/")
        indexed = {str(item["path"]): item for item in workspace_files(workspace, limit=2_000)}
        metadata = indexed.get(normalized)
        if metadata is None:
            raise ValueError("file_not_previewable")
        target = (workspace / normalized).resolve(strict=True)
        try:
            target.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("file_not_previewable") from exc
        if not target.is_file() or target.stat().st_size > 512_000:
            raise ValueError("file_too_large")
        raw = target.read_bytes()
        if b"\x00" in raw[:8_192]:
            raise ValueError("binary_file")
        return {
            "path": normalized,
            "size": int(metadata["size"]),
            "content": raw.decode("utf-8", errors="replace"),
            "truncated": False,
        }

    def bootstrap(self) -> dict[str, Any]:
        projects = self.registry.list_projects()
        return {
            "app": {"name": "GA3BAD", "version": 1, "local_only": True},
            "projects": [
                {
                    **project,
                    "threads": [
                        {**thread, "effective_access": self.runtimes.effective_access(str(thread["id"]))}
                        for thread in self.registry.list_threads(str(project["id"]))
                    ],
                }
                for project in projects
            ],
            "settings": self.registry.settings(),
            "active_job_id": self.active_job_id,
            "queued_jobs": self.registry.list_jobs(statuses=("queued", "running")),
        }

    def inspector(self, thread_id: str, name: str) -> dict[str, Any]:
        thread = self.registry.get_thread(thread_id)
        if thread is None:
            raise KeyError("thread not found")
        project = self.registry.get_project(str(thread["project_id"]))
        if project is None:
            raise KeyError("project not found")
        workspace = Path(str(project["path"])).resolve(strict=True)
        if name == "files":
            return {"workspace": str(workspace), "files": workspace_files(workspace)}
        if name == "changes":
            try:
                diff = GitProtectionManager(workspace).diff()
            except VersionControlError as exc:
                diff = str(exc)
            return {"diff": diff}
        session = self.runtimes.get(thread_id)
        runtime, store = session.runtime, session.store
        goal = runtime.active_goal() or store.get_latest_goal()
        goal_id = goal.id if goal is not None else None
        if name == "plan":
            return {"dashboard": jsonable(runtime.dashboard())}
        if name == "versions":
            return {"versions": jsonable(runtime.version_history())}
        if name == "history":
            return {"events": jsonable(store.list_recent_events(goal_id, limit=250))}
        if name == "evidence":
            return {"evidence": jsonable(store.list_evidence(goal_id)) if goal_id else []}
        if name == "agents":
            return {"agents": jsonable(store.list_agent_registry())}
        if name == "artifacts":
            run = store.get_active_ultra_run(goal_id) if goal_id else None
            return {
                "chat_artifacts": jsonable(store.list_chat_artifacts(runtime.session_id)),
                "ultra_artifacts": jsonable(store.list_artifacts(run.id, limit=250)) if run else [],
            }
        if name == "resources":
            return {"resources": jsonable(store.list_managed_resources(runtime.session_id))}
        if name == "metrics":
            return {
                "dashboard": jsonable(runtime.dashboard()),
                "provider": runtime.provider_name,
                "model": runtime.model_name,
                "execution_class": runtime.execution_class,
            }
        if name in {"memory", "traces"}:
            run = store.get_active_ultra_run(goal_id) if goal_id else None
            if run is None:
                return {name: []}
            if name == "memory":
                return {"memory": jsonable(store.list_brain_entries(run.id))}
            return {"traces": jsonable(store.list_prompt_traces(run.id))}
        raise KeyError("unknown inspector")

    def close(self) -> None:
        active = self.active_job_id
        if active:
            try:
                self.checkpoint(active)
            except Exception:
                pass
        self._stop.set()
        with self._slot:
            self._slot.notify_all()
        for session in tuple(self.runtimes._sessions.values()):
            session.presentation.mark_exit()
        self._worker.join(timeout=10)
        deadline = time.monotonic() + 10
        for worker in tuple(self._job_threads.values()):
            worker.join(timeout=max(0.0, deadline - time.monotonic()))


__all__ = [
    "EventHub",
    "JobScheduler",
    "RuntimeManager",
    "WebRuntimeSession",
    "jsonable",
    "presentation_payload",
    "public_runtime_event",
    "workspace_files",
]

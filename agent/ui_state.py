"""Shared presentation state for the interactive terminal workspace.

The runtime has three durable question producers (intake, normal planning, and
ULTRA).  The UI should not make the user learn that distinction, so this module
normalizes them into one small question session and owns presentation-only
conveniences such as "use the recommended defaults".
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from threading import Event, RLock
from typing import Any, Callable, Mapping


_DEFAULT_UTTERANCES = {
    "continue with recommended",
    "continue with the recommended",
    "continue with defaults",
    "use defaults",
    "use recommended",
    "use recommended defaults",
    "accept defaults",
    "accept recommended",
    "recommended",
    "defaults",
    "كمل بالمقترح",
    "كمل بالمقترحات",
    "كمل بالاختيارات المقترحة",
    "استخدم المقترح",
    "استخدم الافتراضي",
}


class ExperienceMode(str, Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


class LocalePreference(str, Enum):
    AUTO = "auto"
    ARABIC = "ar"
    ENGLISH = "en"


class ActivityStage(str, Enum):
    IDLE = "idle"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    BUILDING = "building"
    CHECKING = "checking"
    DONE = "done"
    PAUSED = "paused"
    PROBLEM = "problem"


class AttentionKind(str, Enum):
    QUESTION = "question"
    APPROVAL = "approval"
    PLAN_REVIEW = "plan_review"
    RECOVERY = "recovery"


class ApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"
    CANCEL = "cancel"
    UI_ERROR = "ui_error"

    @property
    def allowed(self) -> bool:
        return self in {self.ALLOW_ONCE, self.ALLOW_SESSION}


@dataclass(frozen=True, slots=True)
class ActivitySnapshot:
    stage: ActivityStage = ActivityStage.IDLE
    summary: str = "Ready"
    completed: int = 0
    total: int = 0
    started_at: float | None = None
    last_signal_at: float | None = None
    last_success: str = ""

    def elapsed_seconds(self, now: float | None = None) -> int:
        if self.started_at is None:
            return 0
        return max(0, int((time.monotonic() if now is None else now) - self.started_at))


@dataclass(frozen=True, slots=True)
class ProjectProgressSnapshot:
    """Truthful project-level progress derived from durable runtime state."""

    phase: str = "idle"
    current_task: str = ""
    active_operation: str = ""
    completed: int = 0
    total: int = 0
    remaining: int = 0
    retry_count: int = 0
    retry_reason: str = ""
    blocker: str = ""
    elapsed_seconds: int = 0
    eta_low_seconds: int | None = None
    eta_high_seconds: int | None = None
    total_low_seconds: int | None = None
    total_high_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class FileChangeSnapshot:
    files: int = 0
    additions: int = 0
    deletions: int = 0


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Latest asynchronously sampled resource and model telemetry."""

    cpu_percent: float | None = None
    process_memory_mib: float | None = None
    memory_percent: float | None = None
    memory_used_gib: float | None = None
    memory_total_gib: float | None = None
    gpu_percent: float | None = None
    gpu_memory_used_mib: float | None = None
    gpu_memory_total_mib: float | None = None
    gpu_label: str = ""
    gpu_available: bool = False
    context_used_tokens: int = 0
    context_window_tokens: int | None = None
    cached_tokens: int = 0
    output_tokens: int = 0
    execution_class: str = "local"
    provider_limits: str = ""
    model_activity: str = "idle"
    sampled_at: float | None = None

    @property
    def context_remaining_tokens(self) -> int | None:
        if not self.context_window_tokens:
            return None
        return max(0, self.context_window_tokens - self.context_used_tokens)


@dataclass(frozen=True, slots=True)
class SwarmSummarySnapshot:
    total: int = 0
    running: int = 0
    reviewing: int = 0
    completed: int = 0
    blocked: int = 0
    active_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AttentionOption:
    key: str
    label: str
    value: str
    description: str = ""
    shortcut: str = ""
    primary: bool = False
    recommended: bool = False
    auto_safe: bool = False


@dataclass(frozen=True, slots=True)
class AttentionRequest:
    id: str
    kind: AttentionKind
    title: str
    message: str = ""
    options: tuple[AttentionOption, ...] = ()
    details: str = ""
    allow_custom: bool = False
    source: str = ""
    default_key: str = ""
    cancel_key: str = ""
    auto_resolve_safe: bool = False
    # Durable tool approvals carry the same fingerprint as the runtime
    # approval event.  Keeping it on the presentation request lets the web
    # workspace and the terminal resolve exactly one action without guessing
    # from a truncated command preview.
    action_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class AttentionResolution:
    key: str
    value: str
    text: str = ""
    origin: str = "manual"


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    id: int
    role: str
    text: str
    created_at: float
    technical: bool = False
    actor: str = ""
    category: str = "message"
    details_id: str = ""


class PresentationLifecycle(str, Enum):
    ACTIVE = "active"
    SETTLED = "settled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PresentationEvent:
    """Idempotent receipt tied to a real presentation lifecycle boundary."""

    key: str
    category: str
    message: str
    lifecycle: PresentationLifecycle = PresentationLifecycle.SETTLED
    actor: str = ""
    technical: bool = False


@dataclass(frozen=True, slots=True)
class StructuredLogEntry:
    category: str
    message: str
    created_at: float
    count: int = 1


@dataclass(frozen=True, slots=True)
class LiveActivityEntry:
    sequence: int = 0
    timestamp: str = ""
    source: str = "HARNESS"
    phase: str = ""
    state: str = "active"
    message: str = ""
    actor: str = ""
    operation: str = ""
    received_bytes: int = 0
    received_chunks: int = 0
    received_tokens: int = 0


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    mode: ExperienceMode
    locale: str
    transcript: tuple[TranscriptEntry, ...]
    activity: ActivitySnapshot
    attention: AttentionRequest | None
    attention_index: int
    workspace: str
    model: str
    status: str
    running: bool
    queued_count: int
    advanced_log: tuple[str, ...]
    should_exit: bool
    progress: ProjectProgressSnapshot
    resources: ResourceSnapshot
    sleep_enabled: bool
    sleep_log: tuple[str, ...]
    attention_feedback: str
    log_entries: tuple[StructuredLogEntry, ...]
    swarm: SwarmSummarySnapshot
    changes: FileChangeSnapshot = FileChangeSnapshot()
    undo_available: bool = False
    workflow_mode: str = "ready"
    composer_mode: str = "goal"
    route: str = "pending"
    execution_strategy: str = "pending"
    runtime_phase: str = "ready"
    current_task: str = ""
    last_tool: str = ""
    waiting_on: str = ""
    resume_action: str = ""
    heartbeat_at: float | None = None
    workspace_mutated: bool = False
    objective: str = ""
    current_task_id: str = ""
    active_actor: str = ""
    active_step: int = 0
    liveness: str = "ready"
    active_operation: str = ""
    provider_request_state: str = "idle"
    received_bytes: int = 0
    received_chunks: int = 0
    received_tokens: int = 0
    last_signal_at: float | None = None
    activity_sequence: int = 0
    live_timeline: tuple[LiveActivityEntry, ...] = ()
    # Presentation-only live output.  Structured/tool responses deliberately
    # leave ``safe_stream_preview`` empty; the UI must never expose partial
    # JSON or hidden reasoning.
    stream_state: str = "idle"
    stream_kind: str = "none"
    safe_stream_preview: str = ""
    first_byte_at: float | None = None
    task_items: tuple[Mapping[str, Any], ...] = ()
    next_task: str = ""


_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


_COPY: dict[str, dict[str, str]] = {
    "en": {
        "ready": "Ready",
        "idle_hint": "Describe what you want to build",
        "working_hint": "Send guidance while I work",
        "still_working": "Still working",
        "taking_longer": "Taking longer than usual — you can keep waiting or stop safely",
        "done": "Done",
        "problem": "I could not finish that step",
    },
    "ar": {
        "ready": "جاهز",
        "idle_hint": "اكتب ما الذي تريد بناءه",
        "working_hint": "يمكنك إرسال توجيه أثناء العمل",
        "still_working": "ما زلت أعمل",
        "taking_longer": "الأمر يستغرق وقتًا أطول — يمكنك الانتظار أو الإيقاف بأمان",
        "done": "تم",
        "problem": "لم أتمكن من إكمال هذه الخطوة",
    },
}


class WorkspaceUIStore:
    """Thread-safe reducer store for the single persistent workspace UI."""

    def __init__(
        self,
        *,
        mode: ExperienceMode = ExperienceMode.SIMPLE,
        locale: LocalePreference = LocalePreference.AUTO,
        transcript_limit: int = 240,
        log_limit: int = 600,
    ) -> None:
        self._lock = RLock()
        self._mode = ExperienceMode(mode)
        self._locale_preference = LocalePreference(locale)
        self._locale = (
            "en"
            if self._locale_preference is LocalePreference.AUTO
            else self._locale_preference.value
        )
        self._locale_observed = self._locale_preference is not LocalePreference.AUTO
        self._transcript: deque[TranscriptEntry] = deque(maxlen=transcript_limit)
        self._advanced_log: deque[str] = deque(maxlen=log_limit)
        self._log_entries: deque[StructuredLogEntry] = deque(maxlen=log_limit)
        self._activity = ActivitySnapshot()
        self._attention: AttentionRequest | None = None
        self._attention_index = 0
        self._attention_queue: deque[AttentionRequest] = deque()
        self._attention_events: dict[str, Event] = {}
        self._attention_results: dict[str, AttentionResolution] = {}
        self._subscribers: list[Callable[[], None]] = []
        self._sequence = 0
        self._seen_presentation_events: set[str] = set()
        self._transcript_details: dict[int, str] = {}
        self._workspace = ""
        self._model = ""
        self._status = "idle"
        self._workflow_mode = "ready"
        self._composer_mode = "goal"
        self._route = "pending"
        self._execution_strategy = "pending"
        self._runtime_phase = "ready"
        self._current_task = ""
        self._objective = ""
        self._current_task_id = ""
        self._active_actor = ""
        self._active_step = 0
        self._liveness = "ready"
        self._active_operation = ""
        self._provider_request_state = "idle"
        self._received_bytes = 0
        self._received_chunks = 0
        self._received_tokens = 0
        self._last_signal_at: float | None = None
        self._activity_sequence = 0
        self._live_timeline: deque[LiveActivityEntry] = deque(maxlen=80)
        self._last_tool = ""
        self._waiting_on = ""
        self._resume_action = ""
        self._heartbeat_at: float | None = None
        self._workspace_mutated = False
        self._running = False
        self._queued_count = 0
        self._should_exit = False
        self._stream_text = ""
        self._stream_kind = "none"
        self._stream_state = "idle"
        self._first_byte_at: float | None = None
        self._task_items: tuple[Mapping[str, Any], ...] = ()
        self._next_task = ""
        self._progress = ProjectProgressSnapshot()
        self._progress_identity = ""
        self._progress_started_at: float | None = None
        self._progress_pause_started_at: float | None = None
        self._progress_paused_seconds = 0.0
        self._progress_finished_elapsed: int | None = None
        self._task_statuses: dict[str, str] = {}
        self._task_started_at: dict[str, float] = {}
        self._unit_durations: deque[float] = deque(maxlen=12)
        self._resources = ResourceSnapshot()
        self._sleep_enabled = False
        self._sleep_log: deque[str] = deque(maxlen=200)
        self._attention_feedback = ""
        self._swarm = SwarmSummarySnapshot()
        self._changes = FileChangeSnapshot()
        self._undo_available = False
        self._timeline_sink: Callable[[str, str, str], None] | None = None

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def bind_timeline_sink(
        self, callback: Callable[[str, str, str], None] | None
    ) -> None:
        with self._lock:
            self._timeline_sink = callback

    def _notify(self) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback()
            except Exception:
                continue

    def text(self, key: str) -> str:
        with self._lock:
            locale = self._locale
        return _COPY.get(locale, _COPY["en"]).get(key, _COPY["en"].get(key, key))

    def observe_user_text(self, value: str) -> None:
        with self._lock:
            if (
                self._locale_preference is LocalePreference.AUTO
                and not self._locale_observed
                and value.strip()
            ):
                self._locale = "ar" if _ARABIC_RE.search(value) else "en"
                self._locale_observed = True
        self._notify()

    def set_locale(self, value: str | LocalePreference) -> None:
        preference = LocalePreference(value)
        with self._lock:
            self._locale_preference = preference
            if preference is not LocalePreference.AUTO:
                self._locale = preference.value
                self._locale_observed = True
            else:
                self._locale_observed = False
        self._notify()

    def toggle_mode(self) -> ExperienceMode:
        with self._lock:
            self._mode = (
                ExperienceMode.ADVANCED
                if self._mode is ExperienceMode.SIMPLE
                else ExperienceMode.SIMPLE
            )
            result = self._mode
        self._notify()
        return result

    def set_mode(self, value: str | ExperienceMode) -> None:
        with self._lock:
            self._mode = ExperienceMode(value)
        self._notify()

    def append_transcript(
        self,
        role: str,
        text: str,
        *,
        technical: bool = False,
        event_key: str = "",
        actor: str = "",
        category: str = "message",
    ) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        with self._lock:
            if event_key and event_key in self._seen_presentation_events:
                return
            if (
                str(role) == "assistant"
                and self._transcript
                and self._transcript[-1].role == "assistant"
                and self._transcript[-1].text == clean
                and time.time() - self._transcript[-1].created_at < 5.0
            ):
                if event_key:
                    self._seen_presentation_events.add(event_key)
                return
            self._sequence += 1
            full = clean
            details_id = ""
            line_count = clean.count("\n") + 1
            looks_like_artifact = (
                "<!doctype html" in clean.casefold()
                or "```" in clean
                or line_count >= 40
                or len(clean) > 6_000
            )
            if looks_like_artifact:
                self._transcript_details[self._sequence] = full[:120_000]
                details_id = str(self._sequence)
            self._transcript.append(
                TranscriptEntry(
                    self._sequence,
                    str(role),
                    clean,
                    time.time(),
                    technical,
                    str(actor),
                    str(category),
                    details_id,
                )
            )
            if event_key:
                self._seen_presentation_events.add(event_key)
        sink = self._timeline_sink
        if sink is not None and not technical:
            try:
                sink(str(role), full, event_key or f"ui:{self._sequence}")
            except Exception:
                pass
        self._notify()

    def publish(self, event: PresentationEvent) -> None:
        if event.lifecycle is PresentationLifecycle.ACTIVE:
            self.set_activity(self._activity.stage, event.message, running=True)
            return
        self.append_transcript(
            "assistant",
            event.message,
            technical=event.technical,
            event_key=event.key,
            actor=event.actor,
            category=event.category,
        )

    def transcript_details(self, entry_id: int) -> str:
        with self._lock:
            return self._transcript_details.get(int(entry_id), "")

    def append_log(self, text: str) -> None:
        clean = " ".join(str(text or "").split())
        if not clean:
            return
        with self._lock:
            value = clean[:2_000]
            lower = value.casefold()
            category = (
                "error" if any(token in lower for token in ("error", "failed", "exception"))
                else "warning" if any(token in lower for token in ("warning", "blocked", "retry"))
                else "change" if any(token in lower for token in ("changed file", "change set", "patch"))
                else "test" if any(token in lower for token in ("test", "pytest", "verification"))
                else "decision" if any(token in lower for token in ("approval", "choice", "sleep.auto"))
                else "routine"
            )
            coalescible = category == "routine"
            if (
                coalescible
                and self._log_entries
                and self._log_entries[-1].category == category
                and self._log_entries[-1].message == value
            ):
                updated = replace(self._log_entries[-1], count=self._log_entries[-1].count + 1)
                self._log_entries[-1] = updated
                self._advanced_log[-1] = f"{value} (×{updated.count})"
            else:
                self._log_entries.append(
                    StructuredLogEntry(category, value, time.time())
                )
                self._advanced_log.append(value)
        self._notify()

    def set_context_window(self, value: int | None) -> None:
        try:
            parsed = int(value) if value is not None else 0
        except (TypeError, ValueError):
            parsed = 0
        window = max(1, parsed) if parsed > 0 else None
        with self._lock:
            self._resources = replace(self._resources, context_window_tokens=window)
        self._notify()

    def reset_request_context(self) -> None:
        with self._lock:
            self._resources = replace(
                self._resources,
                context_used_tokens=0,
                model_activity="starting",
            )
        self._notify()

    def update_resources(self, **values: Any) -> None:
        allowed = {
            "cpu_percent", "process_memory_mib", "memory_percent", "memory_used_gib",
            "memory_total_gib", "gpu_percent", "gpu_memory_used_mib",
            "gpu_memory_total_mib", "gpu_label", "gpu_available", "sampled_at",
            "cached_tokens", "output_tokens", "execution_class", "provider_limits",
        }
        clean = {key: value for key, value in values.items() if key in allowed}
        with self._lock:
            self._resources = replace(self._resources, **clean)
        self._notify()

    def update_changes(
        self, *, files: int, additions: int, deletions: int, undo_available: bool | None = None
    ) -> None:
        with self._lock:
            self._changes = FileChangeSnapshot(
                max(0, int(files)), max(0, int(additions)), max(0, int(deletions))
            )
            if undo_available is not None:
                self._undo_available = bool(undo_available)
        self._notify()

    def update_runtime_profile(self, *, execution_class: str, provider_limits: str = "") -> None:
        with self._lock:
            self._resources = replace(
                self._resources,
                execution_class=str(execution_class or "local").casefold(),
                provider_limits=str(provider_limits),
            )
        self._notify()

    def update_swarm_summary(self, snapshot: Mapping[str, Any] | None) -> None:
        nodes = tuple((snapshot or {}).get("nodes") or ())
        agents = tuple((snapshot or {}).get("agents") or ())
        latest: dict[str, str] = {}
        roles: dict[str, str] = {}
        for agent in agents:
            node_id = str(
                (agent.get("work_node_id") if isinstance(agent, Mapping) else getattr(agent, "work_node_id", ""))
                or ""
            )
            raw_status = agent.get("status", "") if isinstance(agent, Mapping) else getattr(agent, "status", "")
            status = str(getattr(raw_status, "value", raw_status))
            if node_id:
                latest[node_id] = status.casefold()
                raw_role = agent.get("role", "") if isinstance(agent, Mapping) else getattr(agent, "role", "")
                roles[node_id] = " ".join(str(raw_role or "").replace("_", " ").split()).title()
        statuses: list[str] = []
        active_labels: list[str] = []
        for node in nodes:
            node_id = str((node.get("id") if isinstance(node, Mapping) else getattr(node, "id", "")) or "")
            raw_node_status = node.get("status", "") if isinstance(node, Mapping) else getattr(node, "status", "")
            status = latest.get(
                node_id,
                str(getattr(raw_node_status, "value", raw_node_status)).casefold(),
            )
            statuses.append(status)
            if status in {"running", "in_progress", "planning", "testing", "reviewing", "verifying"}:
                title = (
                    node.get("title") or node.get("objective")
                    if isinstance(node, Mapping)
                    else getattr(node, "title", "") or getattr(node, "objective", "")
                )
                clean_title = " ".join(str(title or node_id).split())
                role = roles.get(node_id, "")
                active_labels.append(f"{role} · {clean_title}" if role else clean_title)
        running_values = {"running", "in_progress", "planning", "testing"}
        reviewing_values = {"reviewing", "verifying"}
        completed_values = {"completed", "done"}
        blocked_values = {"failed", "blocked", "revision_required", "uncertain"}
        with self._lock:
            self._swarm = SwarmSummarySnapshot(
                total=len(nodes),
                running=sum(item in running_values for item in statuses),
                reviewing=sum(item in reviewing_values for item in statuses),
                completed=sum(item in completed_values for item in statuses),
                blocked=sum(item in blocked_values for item in statuses),
                active_labels=tuple(active_labels[:3]),
            )
        self._notify()

    def set_sleep_mode(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        with self._lock:
            changed = self._sleep_enabled != enabled
            self._sleep_enabled = enabled
        if changed:
            state = "enabled" if enabled else "disabled"
            self.append_log(f"sleep: {state}; unsafe decisions remain manual")
            self.append_transcript("assistant", f"Sleep Mode {state}. Unsafe decisions still require you.")
        else:
            self._notify()
        return enabled

    def toggle_sleep_mode(self) -> bool:
        with self._lock:
            enabled = not self._sleep_enabled
        return self.set_sleep_mode(enabled)

    def sleep_enabled(self) -> bool:
        with self._lock:
            return self._sleep_enabled

    def sync_dashboard(self, view: Any) -> None:
        """Reduce a DashboardView without exposing runtime objects to the renderer."""

        now = time.monotonic()
        objective = " ".join(str(getattr(view, "objective", "") or "").split())
        goal_id = " ".join(str(getattr(view, "goal_id", "") or "").split())
        identity = goal_id or objective
        tasks = tuple(getattr(view, "tasks", ()) or ())
        statuses: dict[str, str] = {}
        active_titles: list[str] = []
        active_ids: list[str] = []
        blockers: list[str] = []
        task_items: list[dict[str, Any]] = []
        done_values = {"done", "skipped", "completed", "complete"}
        active_values = {"in_progress", "running", "verifying", "reviewing"}
        for index, task in enumerate(tasks):
            task_id = str(getattr(task, "id", index))
            status = str(getattr(task, "status", "pending") or "pending").casefold()
            title = " ".join(str(getattr(task, "title", "") or task_id).split())
            statuses[task_id] = status
            task_items.append({
                "id": task_id,
                "title": title,
                "status": status,
                "dependencies": (
                    tuple(getattr(task, "dependencies", ()))
                    if isinstance(getattr(task, "dependencies", ()), (list, tuple, set))
                    else ()
                ),
                "evidence": " ".join(str(getattr(task, "evidence", "") or "").split()),
            })
            if status in active_values:
                active_titles.append(title)
                active_ids.append(task_id)
            if status in {"blocked", "failed", "uncertain"}:
                blockers.append(title)
        completed = sum(status in done_values for status in statuses.values())
        total = len(tasks)
        next_title = ""
        for item in task_items:
            if item["status"] in {"pending", "ready", "queued"}:
                next_title = str(item["title"])
                break
        phase = str(getattr(view, "status", "idle") or "idle").replace("_", " ")
        retry_count = max(0, int(getattr(view, "goal_attempt", 0) or 0))
        retry_reason = " ".join(str(getattr(view, "retry_reason", "") or "").split())

        with self._lock:
            if identity != self._progress_identity:
                self._progress_identity = identity
                self._progress_started_at = now if objective else None
                self._progress_pause_started_at = None
                self._progress_paused_seconds = 0.0
                self._progress_finished_elapsed = None
                self._task_statuses.clear()
                self._task_started_at.clear()
                self._unit_durations.clear()
            for task_id, status in statuses.items():
                previous = self._task_statuses.get(task_id)
                if status in active_values and task_id not in self._task_started_at:
                    self._task_started_at[task_id] = now
                if status in done_values and previous not in done_values:
                    started = self._task_started_at.pop(task_id, None)
                    if started is not None:
                        self._unit_durations.append(max(0.1, now - started))
            self._task_statuses = statuses
            self._task_items = tuple(task_items)
            self._next_task = next_title
            elapsed = self._project_elapsed_locked(now)
            eta_low, eta_high = self._eta_locked(completed, total, elapsed)
            self._progress = ProjectProgressSnapshot(
                phase=phase,
                current_task=active_titles[0] if active_titles else "",
                active_operation=self._progress.active_operation,
                completed=completed,
                total=total,
                remaining=max(0, total - completed),
                retry_count=retry_count,
                retry_reason=retry_reason,
                blocker=blockers[0] if blockers else "",
                elapsed_seconds=elapsed,
                eta_low_seconds=eta_low,
                eta_high_seconds=eta_high,
                total_low_seconds=None if eta_low is None else elapsed + eta_low,
                total_high_seconds=None if eta_high is None else elapsed + eta_high,
            )
            self._objective = objective or self._objective
            if active_titles:
                self._current_task_id = active_ids[0]
                self._current_task = f"{active_ids[0]} · {active_titles[0]}"
            self._route = str(getattr(view, "route", self._route) or self._route or "pending")
            self._execution_strategy = str(getattr(view, "execution_strategy_name", self._execution_strategy) or self._execution_strategy or "pending")
            self._runtime_phase = str(getattr(view, "runtime_phase", phase) or phase)
            self._waiting_on = str(getattr(view, "waiting_on", self._waiting_on) or self._waiting_on)
            self._last_tool = str(getattr(view, "last_tool", self._last_tool) or self._last_tool)
            self._resume_action = str(getattr(view, "resume_action", self._resume_action) or self._resume_action)
            self._heartbeat_at = getattr(view, "heartbeat_at", self._heartbeat_at)
            self._workspace_mutated = bool(getattr(view, "workspace_mutated", self._workspace_mutated))
        self._notify()

    def _project_elapsed_locked(self, now: float) -> int:
        if self._progress_finished_elapsed is not None:
            return self._progress_finished_elapsed
        if self._progress_started_at is None:
            return 0
        paused = self._progress_paused_seconds
        if self._progress_pause_started_at is not None:
            paused += max(0.0, now - self._progress_pause_started_at)
        return max(0, int(now - self._progress_started_at - paused))

    def _eta_locked(self, completed: int, total: int, elapsed: int) -> tuple[int | None, int | None]:
        remaining = max(0, total - completed)
        if total <= 0 or completed <= 0 or remaining <= 0:
            return ((0, 0) if total > 0 and remaining == 0 else (None, None))
        if self._unit_durations:
            average = sum(self._unit_durations) / len(self._unit_durations)
        else:
            average = elapsed / completed if elapsed > 0 else 0.0
        if average <= 0:
            return None, None
        if len(self._unit_durations) >= 3:
            low_factor, high_factor = 0.8, 1.25
        else:
            low_factor, high_factor = 0.65, 1.5
        return int(average * remaining * low_factor), int(average * remaining * high_factor)

    def set_activity(
        self,
        stage: str | ActivityStage,
        summary: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        last_success: str | None = None,
        running: bool | None = None,
    ) -> None:
        now = time.monotonic()
        stage_value = ActivityStage(stage)
        with self._lock:
            previous = self._activity
            active = stage_value not in {
                ActivityStage.IDLE,
                ActivityStage.DONE,
                ActivityStage.PAUSED,
                ActivityStage.PROBLEM,
            }
            started = previous.started_at
            if active and (started is None or previous.stage in {ActivityStage.IDLE, ActivityStage.DONE}):
                started = now
            if not active:
                started = None if stage_value is ActivityStage.IDLE else previous.started_at
            self._activity = ActivitySnapshot(
                stage=stage_value,
                summary=" ".join(str(summary or "").split()),
                completed=previous.completed if completed is None else max(0, int(completed)),
                total=previous.total if total is None else max(0, int(total)),
                started_at=started,
                last_signal_at=now,
                last_success=previous.last_success if last_success is None else str(last_success),
            )
            next_running = active if running is None else bool(running)
            if next_running and not self._running and self._progress_pause_started_at is not None:
                self._progress_paused_seconds += max(0.0, now - self._progress_pause_started_at)
                self._progress_pause_started_at = None
            if next_running:
                self._progress_finished_elapsed = None
            elif not next_running and self._running and stage_value not in {ActivityStage.DONE, ActivityStage.IDLE}:
                self._progress_pause_started_at = now
            if stage_value is ActivityStage.DONE:
                self._progress_finished_elapsed = self._project_elapsed_locked(now)
            self._running = next_running
        self._notify()

    def update_identity(
        self,
        *,
        workspace: str,
        model: str,
        status: str,
        workflow_mode: str | None = None,
    ) -> None:
        with self._lock:
            self._workspace = str(workspace)
            self._model = str(model)
            self._status = str(status)
            if workflow_mode is not None:
                value = str(getattr(workflow_mode, "value", workflow_mode)).casefold()
                self._workflow_mode = self._normalize_workflow_label(value)
                self._composer_mode = self._workflow_mode
        self._notify()

    def update_workflow_mode(self, value: str) -> None:
        with self._lock:
            normalized = str(getattr(value, "value", value)).casefold()
            self._workflow_mode = self._normalize_workflow_label(normalized)
            self._composer_mode = self._workflow_mode
        self._notify()

    def update_runtime_context(self, snapshot: Mapping[str, Any] | Any) -> None:
        """Apply the durable runtime snapshot to presentation-only state."""

        if hasattr(snapshot, "to_dict"):
            snapshot = snapshot.to_dict()
        data = dict(snapshot or {})
        with self._lock:
            has_session_mode = "session_mode" in data
            if has_session_mode:
                self._workflow_mode = self._normalize_workflow_label(
                    str(data.get("session_mode") or "ready").casefold()
                )
            if "route" in data:
                self._route = str(data.get("route") or "pending")
            if "execution_strategy" in data:
                self._execution_strategy = str(
                    data.get("execution_strategy") or "pending"
                )
            if "phase" in data:
                self._runtime_phase = str(data.get("phase") or "ready")
                # The compact header reads ``progress.phase`` while the
                # recovery/footer logic reads ``runtime_phase``. Keep both
                # projections atomic so a live routing event cannot be shown
                # beside a stale Dashboard status such as Idle.
                self._progress = replace(
                    self._progress,
                    phase=self._runtime_phase,
                    active_operation=(
                        str(data.get("reason") or data.get("message") or "")
                        or self._progress.active_operation
                    ),
                )
            if "current_task" in data:
                self._current_task = str(data.get("current_task") or "")
                self._progress = replace(
                    self._progress,
                    current_task=self._current_task,
                )
            if "objective" in data:
                self._objective = " ".join(str(data.get("objective") or "").split())
            if "current_task_id" in data:
                self._current_task_id = str(data.get("current_task_id") or "")
            actor = data.get("active_actor", data.get("actor"))
            if actor is not None:
                self._active_actor = str(actor or "")
            step = data.get("active_step", data.get("step"))
            if step is not None:
                try:
                    self._active_step = max(0, int(step))
                except (TypeError, ValueError):
                    pass
            if "last_tool" in data:
                self._last_tool = str(data.get("last_tool") or "")
            if "waiting_on" in data:
                self._waiting_on = str(data.get("waiting_on") or "")
            if "resume_action" in data:
                self._resume_action = str(data.get("resume_action") or "")
            if "liveness" in data:
                self._liveness = str(data.get("liveness") or "ready")
            if "active_operation" in data:
                self._active_operation = str(data.get("active_operation") or "")
                self._progress = replace(
                    self._progress,
                    active_operation=self._active_operation or self._progress.active_operation,
                )
            if "provider_request_state" in data:
                self._provider_request_state = str(data.get("provider_request_state") or "idle")
            if "received_bytes" in data:
                self._received_bytes = max(0, int(data.get("received_bytes") or 0))
            if "received_chunks" in data:
                self._received_chunks = max(0, int(data.get("received_chunks") or 0))
            if "received_tokens" in data:
                self._received_tokens = max(0, int(data.get("received_tokens") or 0))
            if "stream_state" in data:
                self._stream_state = str(data.get("stream_state") or "idle")
            if "stream_kind" in data:
                self._stream_kind = str(data.get("stream_kind") or "none")
            if "safe_stream_preview" in data:
                self._stream_text = str(data.get("safe_stream_preview") or "")[-20_000:]
            if "first_byte_at" in data and data.get("first_byte_at") is not None:
                try:
                    self._first_byte_at = float(data.get("first_byte_at"))
                except (TypeError, ValueError):
                    pass
            incoming_tasks = data.get("task_items")
            if isinstance(incoming_tasks, (list, tuple)):
                self._task_items = tuple(
                    dict(item) for item in incoming_tasks if isinstance(item, Mapping)
                )
            if "next_task" in data:
                self._next_task = str(data.get("next_task") or "")
            if "last_signal_at" in data and data.get("last_signal_at") is not None:
                try:
                    self._last_signal_at = float(data.get("last_signal_at"))
                except (TypeError, ValueError):
                    pass
            if "activity_sequence" in data:
                self._activity_sequence = max(
                    self._activity_sequence, int(data.get("activity_sequence") or 0)
                )
            preview = data.get("timeline_preview")
            if isinstance(preview, (list, tuple)):
                for item in preview:
                    if isinstance(item, Mapping):
                        self._append_live_activity_locked(item)
            heartbeat = data.get("heartbeat_at")
            if heartbeat is not None:
                try:
                    self._heartbeat_at = float(heartbeat)
                except (TypeError, ValueError):
                    pass
            self._workspace_mutated = bool(data.get("workspace_mutated", self._workspace_mutated))
            phase = self._runtime_phase.casefold()
            # Keep the compact header derived from the same durable phase as
            # the activity strip.  This prevents a stale DashboardView status
            # such as ``awaiting_plan_approval`` from surviving after the
            # goal has already entered execution.
            phase_status = {
                "routing": ("routing", True),
                "planning": ("planning", True),
                "awaiting_approval": ("awaiting_plan_approval", False),
                "waiting_for_approval": ("waiting_for_approval", False),
                "starting": ("running", True),
                "working": ("running", True),
                "waiting_for_process": ("running", True),
                "retrying": ("recovering", False),
                "paused": ("paused", False),
                "reviewing": ("reviewing", True),
                "completed": ("completed", False),
            }
            if phase in phase_status:
                self._status, self._running = phase_status[phase]
            if phase == "ready" and not self._transcript:
                self._workflow_mode = "ready"
                self._status = "idle"
                self._running = False
            elif not has_session_mode:
                # Compatibility for legacy event producers. New snapshots
                # carry the interaction mode explicitly; a planning phase in
                # an ordinary Goal must never relabel the session as Plan.
                if phase == "completed":
                    self._workflow_mode = "working"
                elif phase == "plan":
                    self._workflow_mode = "plan"
                elif phase not in {"ready", "idle"}:
                    self._workflow_mode = "working"
            self._composer_mode = self._workflow_mode
        self._notify()

    def _append_live_activity_locked(self, data: Mapping[str, Any]) -> None:
        sequence = max(0, int(data.get("sequence") or data.get("activity_sequence") or 0))
        event_id = str(data.get("event_id") or "")
        message = " ".join(str(data.get("message") or "").split())[:500]
        if not message:
            return
        identity = (sequence, event_id, message)
        for existing in reversed(self._live_timeline):
            if (existing.sequence, "", existing.message) == (sequence, "", message):
                return
            if sequence and existing.sequence == sequence:
                return
        entry = LiveActivityEntry(
            sequence=sequence,
            timestamp=str(data.get("timestamp") or data.get("event_timestamp") or ""),
            source=str(data.get("source") or data.get("source_kind") or "HARNESS").upper(),
            phase=str(data.get("phase") or data.get("stage") or ""),
            state=str(data.get("state") or "active"),
            message=message,
            actor=str(data.get("actor") or data.get("active_actor") or ""),
            operation=str(data.get("operation") or ""),
            received_bytes=max(0, int(data.get("received_bytes") or 0)),
            received_chunks=max(0, int(data.get("received_chunks") or 0)),
            received_tokens=max(0, int(data.get("received_tokens") or 0)),
        )
        self._live_timeline.append(entry)
        self._activity_sequence = max(self._activity_sequence, sequence)

    def set_composer_mode(self, value: str) -> str:
        with self._lock:
            normalized = str(getattr(value, "value", value)).casefold()
            normalized = self._normalize_workflow_label(normalized)
            self._composer_mode = normalized
        self._notify()
        return normalized

    def cycle_composer_mode(self) -> str:
        with self._lock:
            self._composer_mode = "ready"
            selected = "ready"
        self._notify()
        return selected

    @staticmethod
    def _normalize_workflow_label(value: str) -> str:
        normalized = str(value).strip().casefold()
        if normalized in {"plan", "working", "ultra", "goal", "ready"}:
            return normalized
        if normalized in {"normal", "chat", "default"}:
            return "ready"
        return "ready"

    def reset_composer_mode(self) -> None:
        with self._lock:
            self._composer_mode = self._workflow_mode
        self._notify()

    def set_queued_count(self, value: int) -> None:
        with self._lock:
            self._queued_count = max(0, int(value))
        self._notify()

    def request_attention(
        self,
        request: AttentionRequest,
        *,
        timeout: float | None = None,
    ) -> AttentionResolution:
        event = self.present_attention(request)
        if not event.wait(timeout):
            raise TimeoutError(f"attention request {request.id} timed out")
        result = self.take_attention_result(request.id)
        return result or AttentionResolution("ui_error", "ui_error")

    def present_attention(self, request: AttentionRequest) -> Event:
        """Present attention without blocking the controller that owns work state."""

        event = Event()
        automatic: AttentionOption | None = None
        with self._lock:
            if request.id in self._attention_events:
                raise RuntimeError(f"duplicate attention request {request.id}")
            self._attention_events[request.id] = event
            recommended = [
                option
                for option in request.options
                if option.recommended and option.auto_safe
            ]
            if (
                self._sleep_enabled
                and request.auto_resolve_safe
                and request.kind in {AttentionKind.QUESTION, AttentionKind.RECOVERY}
                and len(recommended) == 1
            ):
                automatic = recommended[0]
                self._attention_results[request.id] = AttentionResolution(
                    automatic.key,
                    automatic.value,
                    origin="sleep",
                )
                self._attention_events.pop(request.id, None)
            elif self._attention is None:
                self._attention = request
                self._attention_index = self._default_option_index(request)
                self._attention_feedback = ""
            else:
                self._attention_queue.append(request)
        if automatic is not None:
            stamp = time.strftime("%H:%M:%S")
            record = f"{stamp} · Sleep chose {automatic.label!r} for {request.title!r} (recommended, safe/reversible)."
            with self._lock:
                self._sleep_log.append(record)
            self.append_log(f"sleep.auto_choice: {record}")
            event.set()
        self._notify()
        return event

    def take_attention_result(self, request_id: str) -> AttentionResolution | None:
        """Return and consume a completed non-blocking attention result."""

        with self._lock:
            return self._attention_results.pop(str(request_id), None)

    @staticmethod
    def _default_option_index(request: AttentionRequest) -> int:
        if request.default_key:
            for index, option in enumerate(request.options):
                if option.key == request.default_key:
                    return index
        for index, option in enumerate(request.options):
            if option.primary:
                return index
        return 0

    def move_attention(self, amount: int) -> None:
        with self._lock:
            request = self._attention
            if request is None or not request.options:
                return
            self._attention_index = (self._attention_index + int(amount)) % len(request.options)
            self._attention_feedback = ""
        self._notify()

    def select_attention_index(self, index: int) -> None:
        with self._lock:
            request = self._attention
            if request is None or not request.options:
                return
            self._attention_index = max(0, min(len(request.options) - 1, int(index)))
            self._attention_feedback = ""
        self._notify()

    def resolve_attention(self, key: str, *, text: str = "") -> bool:
        with self._lock:
            request = self._attention
            if request is None:
                return False
            option = next((item for item in request.options if item.key == key), None)
            if option is None and not (request.allow_custom and key == "custom"):
                self._attention_feedback = "Invalid choice. Use the shown shortcut or arrow keys, then Enter."
                invalid = True
            else:
                invalid = False
            if invalid:
                event = None
            else:
                value = text if option is None else option.value
                self._attention_results[request.id] = AttentionResolution(key, value, text, "manual")
                event = self._attention_events.pop(request.id, None)
                self._attention = self._attention_queue.popleft() if self._attention_queue else None
                self._attention_index = (
                    self._default_option_index(self._attention) if self._attention is not None else 0
                )
                self._attention_feedback = ""
        if invalid:
            self._notify()
            return False
        if event is not None:
            event.set()
        self._notify()
        return True

    def resolve_selected_attention(self) -> bool:
        with self._lock:
            request = self._attention
            index = self._attention_index
            if request is None or not request.options:
                return False
            key = request.options[index].key
        return self.resolve_attention(key)

    def cancel_attention(self) -> bool:
        with self._lock:
            request = self._attention
            if request is None:
                return False
            key = request.cancel_key
            if not key:
                self._attention_feedback = "This decision is required. Choose an option or pause the task."
                notify = True
            else:
                notify = False
        if notify:
            self._notify()
            return False
        return self.resolve_attention(key)

    def set_attention_feedback(self, value: str) -> None:
        with self._lock:
            if self._attention is None:
                return
            self._attention_feedback = " ".join(str(value).split())
        self._notify()

    def active_attention(self) -> AttentionRequest | None:
        with self._lock:
            return self._attention

    def mark_exit(self) -> None:
        with self._lock:
            self._should_exit = True
            waiting = tuple(self._attention_events.items())
            for request_id, _event in waiting:
                self._attention_results[request_id] = AttentionResolution(
                    "ui_error", "ui_error"
                )
            self._attention_events.clear()
            self._attention = None
            self._attention_queue.clear()
        for _request_id, event in waiting:
            event.set()
        self._notify()

    def handle_event(self, kind: str, message: str = "", data: Mapping[str, Any] | None = None) -> None:
        """Reduce a runtime event into calm Simple state and complete Advanced logs."""

        data = dict(data or {})
        normalized = str(kind)
        timeline_kinds = {
            "provider.activity", "workflow.state", "heartbeat",
            "approval.requested", "approval.received", "plan.approved.local_web",
            "execution.started", "execution.boundary", "process.waiting",
            "tool_call", "tool_result", "tool.started", "tool.completed", "tool.failed",
            "recovery.presented", "execution.reconciled",
        }
        if normalized in timeline_kinds:
            timeline_data = dict(data)
            timeline_data.setdefault("message", message)
            timeline_data.setdefault("source", (
                "MODEL" if normalized in {"provider.activity", "heartbeat"}
                else "TOOL" if normalized.startswith("tool")
                else "PROCESS" if normalized.startswith("process")
                else "USER" if normalized in {"approval.received", "plan.approved.local_web"}
                else "HARNESS"
            ))
            timeline_message = str(message or "").lstrip().casefold()
            timeline_data.setdefault("state", (
                "failed"
                if normalized in {"error", "tool.failed"}
                or timeline_message.startswith(("error:", "permission denied", "failed:"))
                else "completed" if normalized in {"tool_result", "tool.completed"}
                else "waiting" if normalized in {"approval.requested", "process.waiting"}
                else "active"
            ))
            with self._lock:
                self._append_live_activity_locked(timeline_data)
                # A tool call/result, process signal, or approval event is a
                # real runtime heartbeat even when the provider is using an
                # atomic structured response.  Keep the liveness clock tied to
                # the latest authoritative event so the UI does not claim
                # "no fresh runtime signal" while work is visibly progressing.
                now = time.time()
                self._last_signal_at = now
                self._heartbeat_at = float(data.get("heartbeat_at") or now)
                self._activity = replace(
                    self._activity,
                    last_signal_at=time.monotonic(),
                )
        if normalized == "model_text":
            self.append_log("model_text: streaming response")
        elif normalized == "model_thought":
            self.append_log("model_thought: reasoning update; use /thinking for details")
        else:
            self.append_log(f"{kind}: {message}" if message else kind)
        if normalized in {"tool.failed", "workflow.failed"} or (
            normalized in {"error", "provider.failed"} and bool(data.get("terminal") or data.get("boundary"))
        ):
            reason = " ".join(str(message or data.get("reason") or "Workflow failed").split())
            with self._lock:
                self._status = "paused"
                self._running = False
                self._runtime_phase = "paused"
                self._liveness = "stalled"
                self._stream_state = "waiting"
                self._progress = replace(
                    self._progress,
                    phase="paused",
                    blocker=reason[:500],
                    active_operation=f"Needs attention · {reason[:300]}",
                )
            self.set_activity(ActivityStage.PROBLEM, f"Paused · {reason}", running=False)
            return
        if normalized == "workflow.state":
            if "phase" not in data and data.get("stage"):
                data["phase"] = data.get("stage")
            if message and "message" not in data:
                data["message"] = message
            self.update_runtime_context(data)
            return
        if normalized == "provider.activity":
            with self._lock:
                self._provider_request_state = str(
                    data.get("provider_state") or data.get("state") or "active"
                )
                self._liveness = (
                    "receiving" if self._provider_request_state == "receiving"
                    else "processing_response" if self._provider_request_state == "completed"
                    else "request_sent" if self._provider_request_state in {"request_created", "request_sent", "connection_opened", "provider_connected", "client_active"}
                    else "processing_response" if self._provider_request_state == "server_processing"
                    else "network_unavailable" if self._provider_request_state == "network_unavailable"
                    else "stalled" if self._provider_request_state == "failed"
                    else "client_active"
                )
                self._received_bytes = max(
                    self._received_bytes, int(data.get("received_bytes") or 0)
                )
                self._received_chunks = max(
                    self._received_chunks, int(data.get("received_chunks") or 0)
                )
                self._received_tokens = max(
                    self._received_tokens, int(data.get("received_tokens") or 0)
                )
                provider_state = self._provider_request_state
                self._stream_state = (
                    "receiving" if provider_state == "receiving"
                    else "validating" if provider_state == "completed"
                    else "waiting" if provider_state in {"network_unavailable", "failed"}
                    else "active"
                )
                if provider_state == "receiving" and self._first_byte_at is None:
                    self._first_byte_at = time.time()
                self._active_operation = str(
                    data.get("operation") or message or self._active_operation
                )
                self._progress = replace(
                    self._progress,
                    active_operation=self._active_operation,
                )
                self._last_signal_at = time.time()
                self._heartbeat_at = float(data.get("heartbeat_at") or self._last_signal_at)
                self._activity = replace(
                    self._activity,
                    last_signal_at=time.monotonic(),
                )
                self._resources = replace(
                    self._resources,
                    model_activity=(
                        "streaming" if self._liveness == "receiving"
                        else "processing result" if self._liveness == "processing_response"
                        else "calling model"
                    ),
                )
            self._notify()
            return
        if normalized == "heartbeat":
            with self._lock:
                self._activity = replace(
                    self._activity,
                    last_signal_at=time.monotonic(),
                )
                if data.get("phase"):
                    self._runtime_phase = str(data.get("phase"))
                    self._progress = replace(
                        self._progress,
                        phase=self._runtime_phase,
                    )
                    if self._runtime_phase in {
                        "routing", "planning", "starting", "working", "reviewing"
                    }:
                        self._status = self._runtime_phase
                        self._running = True
                if data.get("waiting_on"):
                    self._waiting_on = str(data.get("waiting_on"))
                actor = data.get("active_actor", data.get("actor"))
                if actor is not None:
                    self._active_actor = str(actor or "")
                step = data.get("active_step", data.get("step"))
                if step is not None:
                    try:
                        self._active_step = max(0, int(step))
                    except (TypeError, ValueError):
                        pass
                if self._waiting_on == "model":
                    elapsed = max(0, int(data.get("elapsed_seconds") or 0))
                    actor_label = self._active_actor.replace("-", " ").replace("_", " ").title()
                    operation = f"{actor_label or 'Model'} · waiting for response"
                    if elapsed:
                        operation += f" · {elapsed}s"
                    self._progress = replace(self._progress, active_operation=operation)
                if data.get("last_tool") or data.get("tool"):
                    self._last_tool = str(data.get("last_tool") or data.get("tool"))
                if data.get("heartbeat_at") is not None:
                    try:
                        self._heartbeat_at = float(data.get("heartbeat_at"))
                    except (TypeError, ValueError):
                        pass
                self._last_signal_at = time.time()
                if self._provider_request_state in {"idle", "client_active"}:
                    self._liveness = "client_active"
            self._notify()
            return
        if normalized in {"approval.requested", "approval.received", "plan.approved.local_web", "execution.started", "execution.boundary", "process.waiting", "recovery.presented", "execution.reconciled"}:
            with self._lock:
                self._runtime_phase = str(data.get("phase") or self._runtime_phase or "working")
                self._waiting_on = str(data.get("waiting_on") or self._waiting_on or ("user" if normalized == "approval.requested" else ""))
                self._last_tool = str(data.get("tool") or data.get("last_tool") or self._last_tool or "")
                self._resume_action = str(data.get("resume_action") or self._resume_action or "")
                if data.get("heartbeat_at") is not None:
                    try:
                        self._heartbeat_at = float(data.get("heartbeat_at"))
                    except (TypeError, ValueError):
                        pass
                if data.get("workspace_mutated") is not None:
                    self._workspace_mutated = bool(data.get("workspace_mutated"))
                if data.get("objective"):
                    self._objective = " ".join(str(data.get("objective")).split())
                if data.get("current_task"):
                    self._current_task = str(data.get("current_task"))
                    self._progress = replace(self._progress, current_task=self._current_task)
                if data.get("current_task_id"):
                    self._current_task_id = str(data.get("current_task_id"))
                actor = data.get("active_actor", data.get("actor"))
                if actor is not None:
                    self._active_actor = str(actor or "")
            if normalized == "approval.requested":
                with self._lock:
                    # The approval boundary is a real state transition. Do
                    # not leave the header saying Working while the worker is
                    # blocked and waiting for the user.
                    self._status = "waiting_for_approval"
                    self._running = False
                    self._liveness = "waiting"
                    self._progress = replace(
                        self._progress,
                        phase="waiting_for_approval",
                        active_operation=message or "Waiting for your approval",
                    )
                self.set_activity(
                    ActivityStage.PAUSED,
                    message or "Waiting for your approval",
                    running=False,
                )
            elif normalized in {"approval.received", "plan.approved.local_web"}:
                # The approval event is the durable hand-off from a paused
                # permission gate back to the worker.  Clear the stale
                # ``waiting_on=user`` presentation immediately; otherwise a
                # terminal can keep showing an old approval banner while the
                # approved action is already executing.
                with self._lock:
                    self._waiting_on = str(data.get("waiting_on") or "")
                    self._runtime_phase = str(data.get("phase") or "starting")
                    self._status = "running"
                    self._running = True
                self.set_activity(
                    ActivityStage.BUILDING,
                    message or "Approved — resuming the saved action",
                    running=True,
                )
            elif normalized == "execution.started":
                with self._lock:
                    self._waiting_on = str(data.get("waiting_on") or "")
                    self._runtime_phase = str(data.get("phase") or "starting")
                    self._status = "running"
                    self._running = True
                self.set_activity(
                    ActivityStage.BUILDING,
                    message or "Execution started",
                    running=True,
                )
            elif normalized == "process.waiting":
                self.set_activity(ActivityStage.BUILDING, message or "Waiting for process", running=True)
            elif normalized == "execution.boundary":
                self.set_activity(ActivityStage.PAUSED, message or "Workflow boundary", running=False)
            elif normalized == "recovery.presented":
                self.set_activity(ActivityStage.PAUSED, message or "Action required to continue", running=False)
            self._notify()
            return
        if normalized == "model_text":
            actor = str(data.get("actor") or "").strip().casefold()
            if actor and actor not in {"chat", "assistant", "user_facing"}:
                # Planner/reviewer/coordinator output is protocol work. Keep a
                # bounded redacted trace in Advanced details, never the chat
                # stream or final transcript.
                self.append_log(
                    f"reasoning.{actor}: {str(message)[:1_200]}"
                )
                with self._lock:
                    self._resources = replace(
                        self._resources, model_activity=f"{actor} thinking"
                    )
                self.set_activity(
                    ActivityStage.PLANNING if "architect" in actor or "plan" in actor else ActivityStage.UNDERSTANDING,
                    f"{actor.replace('_', ' ').title()} · working",
                    running=True,
                )
                return
            with self._lock:
                self._stream_text = (self._stream_text + str(message))[-20_000:]
                self._stream_kind = "chat"
                self._stream_state = "receiving"
                if self._first_byte_at is None:
                    self._first_byte_at = time.time()
                self._resources = replace(self._resources, model_activity="streaming")
            self._notify()
            return
        if normalized == "model_thought":
            with self._lock:
                self._resources = replace(self._resources, model_activity="thinking")
            self.set_activity(ActivityStage.UNDERSTANDING, "Thinking through the next step")
            return
        if normalized == "step":
            with self._lock:
                self._resources = replace(self._resources, model_activity="calling model")
                actor = str(data.get("actor") or "coordinator")
                self._active_actor = actor
                try:
                    self._active_step = max(0, int(data.get("step") or self._active_step))
                except (TypeError, ValueError):
                    pass
                actor_label = actor.replace("-", " ").replace("_", " ").title()
                self._progress = replace(
                    self._progress,
                    active_operation=f"{actor_label} · choosing the next action",
                )
            self.set_activity(ActivityStage.BUILDING, f"{actor_label} · choosing the next action")
            return
        if normalized == "tool_call":
            tool = str(message or data.get("tool") or "")
            stage = ActivityStage.CHECKING if any(
                token in tool for token in ("inspect", "poll", "test", "read_process")
            ) else ActivityStage.UNDERSTANDING if tool in {"read_file", "list_files", "grep"} else ActivityStage.BUILDING
            labels = {
                "read_file": "Reading the project",
                "list_files": "Understanding the project structure",
                "grep": "Finding the relevant code",
                "write_file": "Creating the requested files",
                "edit_file": "Updating the implementation",
                "apply_patch": "Applying the change",
                "run_bash": "Running a project command",
                "run_command": "Running a project command",
                "install_dependencies": "Preparing project dependencies",
                "preview_html": "Preparing the preview",
                "inspect_preview": "Checking the result",
            }
            with self._lock:
                self._resources = replace(self._resources, model_activity="using tool")
                self._active_actor = str(data.get("actor") or self._active_actor or "coordinator")
                self._last_tool = tool
                self._progress = replace(
                    self._progress,
                    active_operation=labels.get(tool, tool.replace("_", " ") or "Working"),
                )
            self.set_activity(stage, labels.get(tool, "Working on the project"))
            return
        if normalized == "tool_result":
            failed = str(message).lstrip().lower().startswith(("error:", "permission denied"))
            planning_repair = False
            if failed:
                # A planner contract error is an active, bounded repair phase,
                # not successful work and not an unexplained generic spinner.
                # Keep the worker live while the repair budget remains, but
                # expose the real phase and operation so the user can tell what
                # is being retried. Execution-tool failures retain the normal
                # coordinator repair semantics below.
                planning_repair = (
                    str(data.get("phase") or "").casefold() == "planning"
                    or str(data.get("tool") or "").casefold()
                    in {"propose_plan", "propose_semantic_goal", "request_plan_input"}
                )
                if planning_repair:
                    with self._lock:
                        self._resources = replace(
                            self._resources,
                            model_activity="repairing plan",
                        )
                        # Keep the planning repair worker alive, but make the
                        # header truthful: this is recovery, not a healthy
                        # model step.
                        self._status = "recovering"
                        self._waiting_on = "coordinator"
                        self._resume_action = "Retry"
                    self.set_activity(
                        ActivityStage.PLANNING,
                        "Plan contract failed — repairing the saved plan",
                        running=self._running,
                    )
                else:
                    # Keep the live worker available for the bounded repair
                    # loop, but expose the failure as a recovery state instead
                    # of leaving the header at ``Working``.  A durable
                    # terminal/boundary event is handled above and becomes
                    # ``Paused · Needs attention``.
                    with self._lock:
                        self._runtime_phase = "retrying"
                        self._status = "recovering"
                        self._waiting_on = "coordinator"
                        self._resume_action = "Retry"
                        self._progress = replace(
                            self._progress,
                            phase="retrying",
                            retry_reason=" ".join(str(message).split())[:500],
                            active_operation="Repairing the failed step",
                        )
                    self.set_activity(
                        ActivityStage.CHECKING if self._running else ActivityStage.PROBLEM,
                        "Repairing the failed step" if self._running else self.text("problem"),
                        running=self._running,
                    )
                if planning_repair:
                    self.append_log(f"planning repair: {str(message)[:1_200]}")
                elif self._running:
                    self.append_log(f"tool retry: {str(message)[:1_200]}")
                else:
                    self.append_transcript("assistant", f"Tool failure: {str(message)[:1200]}")
            else:
                tool = str(data.get("tool") or "step")
                self.set_activity(
                    self._activity.stage,
                    self._activity.summary,
                    last_success=f"Completed {tool.replace('_', ' ')}",
                )
                one_line = " ".join(str(message).split())
                if tool in {"write_file", "edit_file", "apply_patch"}:
                    self.append_transcript(
                        "assistant", f"File change recorded: {one_line[:800] or tool.replace('_', ' ')}"
                    )
                elif tool in {"run_bash", "run_command"} and any(
                    token in one_line.casefold() for token in ("passed", "failed", "test", "pytest")
                ):
                    self.append_transcript("assistant", f"Test result: {one_line[:1000]}")
            with self._lock:
                if not planning_repair:
                    self._resources = replace(self._resources, model_activity="processing result")
            self._notify()
            return
        if normalized == "usage":
            used = max(0, int(data.get("input_tokens", 0) or 0))
            with self._lock:
                self._resources = replace(
                    self._resources,
                    context_used_tokens=max(self._resources.context_used_tokens, used),
                    cached_tokens=max(0, int(data.get("cached_tokens", 0) or 0)),
                    output_tokens=max(0, int(data.get("output_tokens", 0) or 0)),
                    model_activity="processing result",
                )
            self._notify()
            return
        if normalized == "phase":
            lower = str(message).lower()
            stage = (
                ActivityStage.PLANNING
                if "plan" in lower or "discover" in lower
                else ActivityStage.CHECKING
                if "review" in lower or "verif" in lower
                else ActivityStage.DONE
                if "complete" in lower
                else ActivityStage.BUILDING
            )
            self.set_activity(stage, message or stage.value.title())
            if stage is ActivityStage.DONE:
                self._commit_stream()
                with self._lock:
                    self._resources = replace(self._resources, model_activity="idle")
            return
        if normalized == "plan":
            self.set_activity(ActivityStage.PLANNING, "The plan is ready")
            return
        if normalized == "retry_wait":
            with self._lock:
                self._progress = replace(
                    self._progress,
                    retry_count=self._progress.retry_count + 1,
                    retry_reason=" ".join(str(message or "Waiting before retry").split()),
                )
            self.set_activity(ActivityStage.CHECKING, message or "Waiting before retry", running=True)
            return
        if normalized == "questions":
            self.set_activity(ActivityStage.PAUSED, message or "Your input is needed", running=False)
            self._commit_stream()
            return
        if normalized == "checkpoint":
            status = str(data.get("status") or "").casefold()
            paused = bool(data.get("paused") or data.get("retry_exhausted")) or status == "paused"
            if status in {"completed", "complete", "done"}:
                self.set_activity(ActivityStage.DONE, message or "Done", running=False)
                self._commit_stream()
                with self._lock:
                    self._resources = replace(self._resources, model_activity="idle")
            elif paused:
                self.set_activity(ActivityStage.PAUSED, message or "Your input is needed", running=False)
                self._commit_stream()
            else:
                self.set_activity(
                    self._activity.stage if self._running else ActivityStage.CHECKING,
                    message or "Checkpoint saved; work continues",
                    running=bool(data.get("continues")) or self._running,
                )
            return
        if normalized == "warning":
            self.append_transcript("assistant", f"Warning: {message or self.text('problem')}")
            self.set_activity(
                self._activity.stage if self._running else ActivityStage.PROBLEM,
                message or ("Still working after a warning" if self._running else self.text("problem")),
                running=self._running,
            )
            return
        if normalized == "error":
            self.append_transcript("assistant", f"Error: {message or self.text('problem')}")
            terminal = bool(
                data.get("planning_terminal")
                or data.get("paused")
                or data.get("boundary")
                or data.get("terminal")
            )
            with self._lock:
                reason = " ".join(str(message or self.text("problem")).split())
                if terminal:
                    self._runtime_phase = "paused"
                    self._status = "paused"
                    self._running = False
                    self._progress = replace(self._progress, blocker=reason)
                else:
                    self._progress = (
                        replace(self._progress, retry_reason=reason)
                        if self._running
                        else replace(self._progress, blocker=reason)
                    )
            self.set_activity(
                ActivityStage.PROBLEM if terminal else self._activity.stage if self._running else ActivityStage.PROBLEM,
                message or self.text("problem"),
                running=False if terminal else self._running,
            )
            return
        if normalized.startswith("ultra."):
            lower = normalized.casefold()
            if any(token in lower for token in ("review", "verify", "test")):
                stage = ActivityStage.CHECKING
            elif any(token in lower for token in ("graph", "plan", "foundation")):
                stage = ActivityStage.PLANNING
            elif any(token in lower for token in ("completed", "cancelled")):
                stage = ActivityStage.DONE if "completed" in lower else ActivityStage.PAUSED
            else:
                stage = ActivityStage.BUILDING
            label = " ".join(str(message or normalized.removeprefix("ultra.")).split())
            def integer(*keys: str) -> int | None:
                for key in keys:
                    value = data.get(key)
                    try:
                        if value is not None:
                            return max(0, int(value))
                    except (TypeError, ValueError):
                        continue
                return None
            total = integer("total_nodes", "total")
            completed = integer("completed_nodes", "completed")
            last_success = None
            if str(data.get("status", "")).casefold() == "completed":
                last_success = str(data.get("node_title") or message or "Specialist step completed")
            self.set_activity(
                stage,
                label.replace("_", " ").strip().capitalize(),
                completed=completed,
                total=total,
                last_success=last_success,
                running=stage not in {ActivityStage.DONE, ActivityStage.PAUSED},
            )
            with self._lock:
                ultra_total = self._progress.total if total is None else total
                ultra_completed = self._progress.completed if completed is None else completed
                self._progress = replace(
                    self._progress,
                    phase=label.replace("_", " ").strip() or "ultra",
                    current_task=str(data.get("node_title") or data.get("task") or "").strip(),
                    completed=ultra_completed,
                    total=ultra_total,
                    remaining=max(0, ultra_total - ultra_completed),
                    retry_count=max(
                        self._progress.retry_count,
                        integer("retry_count", "attempt") or 0,
                    ),
                    retry_reason=str(data.get("retry_reason") or self._progress.retry_reason),
                    blocker=str(data.get("blocker") or self._progress.blocker),
                )
            self._notify()

    def _commit_stream(self) -> None:
        with self._lock:
            value = self._stream_text.strip()
            self._stream_text = ""
        if value:
            self.append_transcript("assistant", value)

    def finalize_stream(self, *, commit: bool = True) -> None:
        """Consolidate one stream attempt exactly once at a work boundary.

        Failed attempts remain visible while they are live but are not copied
        into the durable transcript, which prevents a retry from duplicating a
        partial model response.
        """

        if commit:
            self._commit_stream()
        else:
            with self._lock:
                self._stream_text = ""
        with self._lock:
            self._resources = replace(self._resources, model_activity="idle")
            self._stream_state = "idle"
            self._stream_kind = "none"
            self._first_byte_at = None
            self._stream_text = ""
            if self._progress.active_operation:
                self._progress = replace(self._progress, active_operation="")
        self._notify()

    def snapshot(self) -> WorkspaceSnapshot:
        with self._lock:
            transcript = tuple(
                item
                for item in self._transcript
                if self._mode is ExperienceMode.ADVANCED or not item.technical
            )
            elapsed = self._project_elapsed_locked(time.monotonic())
            eta_low, eta_high = self._eta_locked(
                self._progress.completed,
                self._progress.total,
                elapsed,
            )
            progress = replace(
                self._progress,
                elapsed_seconds=elapsed,
                eta_low_seconds=eta_low,
                eta_high_seconds=eta_high,
                total_low_seconds=None if eta_low is None else elapsed + eta_low,
                total_high_seconds=None if eta_high is None else elapsed + eta_high,
            )
            return WorkspaceSnapshot(
                mode=self._mode,
                locale=self._locale,
                transcript=transcript,
                activity=self._activity,
                attention=self._attention,
                attention_index=self._attention_index,
                workspace=self._workspace,
                model=self._model,
                status=self._status,
                running=self._running,
                queued_count=self._queued_count,
                advanced_log=tuple(self._advanced_log),
                should_exit=self._should_exit,
                progress=progress,
                resources=self._resources,
                sleep_enabled=self._sleep_enabled,
                sleep_log=tuple(self._sleep_log),
                attention_feedback=self._attention_feedback,
                log_entries=tuple(self._log_entries),
                swarm=self._swarm,
                changes=self._changes,
                undo_available=self._undo_available,
                workflow_mode=self._workflow_mode,
                composer_mode=self._composer_mode,
                route=self._route,
                execution_strategy=self._execution_strategy,
                runtime_phase=self._runtime_phase,
                current_task=self._current_task,
                last_tool=self._last_tool,
                waiting_on=self._waiting_on,
                resume_action=self._resume_action,
                heartbeat_at=self._heartbeat_at,
                workspace_mutated=self._workspace_mutated,
                objective=self._objective,
                current_task_id=self._current_task_id,
                active_actor=self._active_actor,
                active_step=self._active_step,
                liveness=self._liveness,
                active_operation=self._active_operation,
                provider_request_state=self._provider_request_state,
                received_bytes=self._received_bytes,
                received_chunks=self._received_chunks,
                received_tokens=self._received_tokens,
                last_signal_at=self._last_signal_at,
                activity_sequence=self._activity_sequence,
                live_timeline=tuple(self._live_timeline),
                stream_state=self._stream_state,
                stream_kind=self._stream_kind,
                safe_stream_preview=(self._stream_text[-4_000:] if self._mode is ExperienceMode.ADVANCED else ""),
                first_byte_at=self._first_byte_at,
                task_items=tuple(dict(item) for item in self._task_items),
                next_task=self._next_task,
            )


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def is_recommended_defaults_utterance(value: str) -> bool:
    """Return whether free text clearly asks to accept suggested answers."""

    normalized = _clean(value).casefold().rstrip(".!؟")
    return normalized in _DEFAULT_UTTERANCES


@dataclass(frozen=True, slots=True)
class QuestionSessionView:
    """One user-facing interview regardless of its runtime implementation."""

    source: str
    questions: tuple[Mapping[str, Any], ...]
    answers: Mapping[str, str]

    @property
    def pending(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in self.questions
            if not _clean(item.get("answer"))
            and not _clean(self.answers.get(str(item.get("id", ""))))
        )

    @property
    def current(self) -> Mapping[str, Any] | None:
        return self.pending[0] if self.pending else None

    @property
    def total(self) -> int:
        return len(self.questions)

    @property
    def completed(self) -> int:
        return max(0, self.total - len(self.pending))


def question_session(runtime: Any) -> QuestionSessionView | None:
    """Read the highest-priority active interview from an AgentRuntime-like object."""

    def values(name: str) -> tuple[Mapping[str, Any], ...]:
        method = getattr(runtime, name, None)
        raw = method() if callable(method) else ()
        if not isinstance(raw, (tuple, list)):
            return ()
        return tuple(item for item in raw if isinstance(item, Mapping))

    intake = values("intake_questions")
    if intake:
        return QuestionSessionView("intake", intake, {})

    active_goal = getattr(runtime, "active_goal", None)
    goal = active_goal() if callable(active_goal) else None
    metadata = getattr(goal, "metadata", {}) if goal is not None else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    answers = {
        str(key): _clean(value)
        for key, value in dict(metadata.get("plan_answers", {})).items()
        if _clean(value)
    }

    if goal is not None and metadata.get("ultra_run_id"):
        questions = values("ultra_questions")
        if questions:
            session = QuestionSessionView("ultra", questions, answers)
            return session if session.pending else None

    questions = values("plan_questions")
    if questions:
        session = QuestionSessionView("plan", questions, answers)
        return session if session.pending else None
    return None


def answer_question(
    runtime: Any,
    session: QuestionSessionView,
    question_id: str,
    value: str,
) -> Any:
    """Dispatch one answer without leaking question-source details into the CLI."""

    if session.source == "intake":
        return runtime.answer_intake_question(question_id, value)
    if session.source == "ultra":
        return runtime.answer_ultra_question(question_id, value)
    return runtime.answer_plan_question(question_id, value)


def answer_recommended_remaining(runtime: Any) -> tuple[Any, ...]:
    """Accept each explicit model recommendation in the current interview.

    The final answer is allowed to transition the runtime into plan generation,
    so the session is refreshed after every answer rather than iterating stale
    question objects.
    """

    results: list[Any] = []
    initial = question_session(runtime)
    source = initial.source if initial is not None else ""
    while True:
        session = question_session(runtime)
        if session is None or session.source != source or session.current is None:
            break
        question_id = str(session.current.get("id", "")).strip()
        if not question_id:
            break
        options = tuple(
            item for item in session.current.get("options", ())
            if isinstance(item, Mapping)
        )
        recommended = [
            (index, item) for index, item in enumerate(options, 1)
            if bool(item.get("recommended"))
        ]
        if len(recommended) != 1:
            break
        index, _option = recommended[0]
        results.append(answer_question(runtime, session, question_id, str(index)))
    return tuple(results)


__all__ = [
    "ActivitySnapshot",
    "ActivityStage",
    "ApprovalDecision",
    "AttentionKind",
    "AttentionOption",
    "AttentionRequest",
    "AttentionResolution",
    "ExperienceMode",
    "FileChangeSnapshot",
    "LocalePreference",
    "ProjectProgressSnapshot",
    "PresentationEvent",
    "PresentationLifecycle",
    "QuestionSessionView",
    "TranscriptEntry",
    "ResourceSnapshot",
    "StructuredLogEntry",
    "WorkspaceSnapshot",
    "WorkspaceUIStore",
    "answer_question",
    "answer_recommended_remaining",
    "is_recommended_defaults_utterance",
    "question_session",
]

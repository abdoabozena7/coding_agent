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
from dataclasses import dataclass
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
class AttentionOption:
    key: str
    label: str
    value: str
    description: str = ""
    shortcut: str = ""
    primary: bool = False


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


@dataclass(frozen=True, slots=True)
class AttentionResolution:
    key: str
    value: str
    text: str = ""


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    id: int
    role: str
    text: str
    created_at: float
    technical: bool = False


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
        self._activity = ActivitySnapshot()
        self._attention: AttentionRequest | None = None
        self._attention_index = 0
        self._attention_queue: deque[AttentionRequest] = deque()
        self._attention_events: dict[str, Event] = {}
        self._attention_results: dict[str, AttentionResolution] = {}
        self._subscribers: list[Callable[[], None]] = []
        self._sequence = 0
        self._workspace = ""
        self._model = ""
        self._status = "idle"
        self._running = False
        self._queued_count = 0
        self._should_exit = False
        self._stream_text = ""

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

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

    def append_transcript(self, role: str, text: str, *, technical: bool = False) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        with self._lock:
            self._sequence += 1
            self._transcript.append(
                TranscriptEntry(self._sequence, str(role), clean, time.time(), technical)
            )
        self._notify()

    def append_log(self, text: str) -> None:
        clean = " ".join(str(text or "").split())
        if not clean:
            return
        with self._lock:
            self._advanced_log.append(clean[:2_000])
        self._notify()

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
            self._running = active if running is None else bool(running)
        self._notify()

    def update_identity(self, *, workspace: str, model: str, status: str) -> None:
        with self._lock:
            self._workspace = str(workspace)
            self._model = str(model)
            self._status = str(status)
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
        event = Event()
        with self._lock:
            if request.id in self._attention_events:
                raise RuntimeError(f"duplicate attention request {request.id}")
            self._attention_events[request.id] = event
            if self._attention is None:
                self._attention = request
                self._attention_index = self._default_option_index(request)
            else:
                self._attention_queue.append(request)
        self._notify()
        if not event.wait(timeout):
            raise TimeoutError(f"attention request {request.id} timed out")
        with self._lock:
            return self._attention_results.pop(
                request.id,
                AttentionResolution("ui_error", "ui_error"),
            )

    @staticmethod
    def _default_option_index(request: AttentionRequest) -> int:
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
        self._notify()

    def resolve_attention(self, key: str, *, text: str = "") -> bool:
        with self._lock:
            request = self._attention
            if request is None:
                return False
            option = next((item for item in request.options if item.key == key), None)
            if option is None and not (request.allow_custom and key == "custom"):
                return False
            value = text if option is None else option.value
            self._attention_results[request.id] = AttentionResolution(key, value, text)
            event = self._attention_events.pop(request.id, None)
            self._attention = self._attention_queue.popleft() if self._attention_queue else None
            self._attention_index = (
                self._default_option_index(self._attention) if self._attention is not None else 0
            )
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
        self.append_log(f"{kind}: {message}" if message else kind)
        normalized = str(kind)
        if normalized == "model_text":
            with self._lock:
                self._stream_text = (self._stream_text + str(message))[-20_000:]
            self._notify()
            return
        if normalized == "model_thought":
            self.set_activity(ActivityStage.UNDERSTANDING, "Thinking through the next step")
            return
        if normalized == "step":
            self.set_activity(ActivityStage.BUILDING, "Working on the next step")
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
            self.set_activity(stage, labels.get(tool, "Working on the project"))
            return
        if normalized == "tool_result":
            failed = str(message).lstrip().lower().startswith(("error:", "permission denied"))
            if failed:
                self.set_activity(ActivityStage.PROBLEM, self.text("problem"), running=False)
            else:
                tool = str(data.get("tool") or "step")
                self.set_activity(
                    self._activity.stage,
                    self._activity.summary,
                    last_success=f"Completed {tool.replace('_', ' ')}",
                )
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
            return
        if normalized == "plan":
            self.set_activity(ActivityStage.PLANNING, "The plan is ready")
            return
        if normalized in {"questions", "checkpoint"}:
            self.set_activity(ActivityStage.PAUSED, message or "Your input is needed", running=False)
            self._commit_stream()
            return
        if normalized in {"error", "warning"}:
            self.set_activity(ActivityStage.PROBLEM, self.text("problem"), running=False)
            return
        if normalized.startswith("ultra."):
            self.set_activity(ActivityStage.BUILDING, "Specialists are working on the project")

    def _commit_stream(self) -> None:
        with self._lock:
            value = self._stream_text.strip()
            self._stream_text = ""
        if value:
            self.append_transcript("assistant", value)

    def snapshot(self) -> WorkspaceSnapshot:
        with self._lock:
            transcript = tuple(
                item
                for item in self._transcript
                if self._mode is ExperienceMode.ADVANCED or not item.technical
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
    """Accept option one for every question in the current interview.

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
        results.append(answer_question(runtime, session, question_id, "1"))
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
    "LocalePreference",
    "QuestionSessionView",
    "TranscriptEntry",
    "WorkspaceSnapshot",
    "WorkspaceUIStore",
    "answer_question",
    "answer_recommended_remaining",
    "is_recommended_defaults_utterance",
    "question_session",
]

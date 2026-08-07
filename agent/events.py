"""Thread-safe runtime events used by the engine and terminal renderers.

The orchestration layer never needs to print directly.  Keeping events as data
makes the same engine usable from the interactive TUI, tests, or a future API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from queue import Empty, Queue
from threading import RLock
from typing import Any, Callable, Iterable
from uuid import uuid4


@dataclass(frozen=True)
class UIEvent:
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"event_{uuid4().hex}")
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable event envelope."""

        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "message": self.message,
            "timestamp": self.timestamp,
            "session_id": self.data.get("session_id"),
            "source": self.data.get("source"),
            "actor": self.data.get("actor"),
            "plan_revision": self.data.get("plan_revision"),
            "checkpoint_id": self.data.get("checkpoint_id"),
            "correlation_id": self.data.get("correlation_id"),
            "payload": dict(self.data),
        }


@dataclass(frozen=True)
class LiveWorkflowEventV1:
    """Safe, presentation-neutral activity emitted by the running workflow.

    Raw model thoughts and partial structured payloads never cross this
    boundary.  The envelope intentionally separates client liveness from
    provider output so a UI cannot turn a heartbeat into a false claim that
    response bytes were received.
    """

    sequence: int
    event_id: str
    timestamp: str
    source: str
    phase: str
    actor: str
    task: str
    operation: str
    state: str
    message: str
    provider_state: str = ""
    detail: str = ""
    elapsed_seconds: int = 0
    last_signal_age: int = 0
    received_bytes: int = 0
    received_chunks: int = 0
    received_tokens: int = 0
    workspace_mutated: bool = False
    waiting_on: str = ""
    action: str = ""
    safe_text_fragment: str = ""
    stream_kind: str = "none"
    visibility: str = "operational"
    task_status: str = ""
    evidence_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "phase": self.phase,
            "actor": self.actor,
            "task": self.task,
            "operation": self.operation,
            "provider_state": self.provider_state,
            "state": self.state,
            "message": self.message,
            "detail": self.detail,
            "elapsed_seconds": self.elapsed_seconds,
            "last_signal_age": self.last_signal_age,
            "received_bytes": self.received_bytes,
            "received_chunks": self.received_chunks,
            "received_tokens": self.received_tokens,
            "workspace_mutated": self.workspace_mutated,
            "waiting_on": self.waiting_on,
            "action": self.action,
            "safe_text_fragment": self.safe_text_fragment,
            "stream_kind": self.stream_kind,
            "visibility": self.visibility,
            "task_status": self.task_status,
            "evidence_summary": self.evidence_summary,
        }

    @classmethod
    def from_ui_event(cls, event: UIEvent) -> "LiveWorkflowEventV1":
        data = dict(event.data)
        kind = str(event.kind)
        actor = str(data.get("active_actor") or data.get("actor") or "")
        source = str(data.get("source_kind") or "").upper()
        if not source:
            if kind.startswith("provider.") or kind in {"model_text", "model_thought", "heartbeat"}:
                source = "MODEL"
            elif kind.startswith("tool"):
                source = "TOOL"
            elif kind.startswith("process"):
                source = "PROCESS"
            elif str(data.get("actor") or "").casefold() == "user" or kind.startswith("approval.received"):
                source = "USER"
            else:
                source = "HARNESS"

        state = str(data.get("state") or "")
        if not state:
            message_lower = str(event.message or data.get("message") or "").lstrip().casefold()
            failed_result = (
                kind in {"error", "tool.failed"}
                or message_lower.startswith(("error:", "permission denied", "failed:"))
            )
            if failed_result:
                state = "failed"
            elif kind in {"tool_result", "tool.completed", "execution.completed"}:
                state = "completed"
            elif kind in {"error", "tool.failed"}:
                state = "failed"
            elif kind in {"heartbeat", "workflow.heartbeat"}:
                state = "active"
            elif kind in {"approval.requested", "process.waiting"}:
                state = "waiting"
            else:
                state = "active"

        message = " ".join(str(event.message or data.get("message") or "").split())
        safe_text_fragment = ""
        stream_kind = str(data.get("stream_kind") or "none")
        visibility = str(data.get("visibility") or "operational")
        # Internal reasoning and protocol fragments are deliberately opaque.
        if kind == "model_thought":
            message = "Model reasoning activity received"
        elif kind == "model_text" and actor.casefold() not in {"chat", "assistant", "user_facing"}:
            message = "Model response fragment received"
        elif kind == "model_text" and actor.casefold() in {"chat", "assistant", "user_facing"}:
            safe_text_fragment = str(event.message or data.get("message") or "")[:1_000]
            stream_kind = "chat"
            visibility = "safe_text"
        elif kind == "heartbeat" and not message:
            message = "Provider request remains open"
        message = message[:500]
        detail = " ".join(str(data.get("detail") or "").split())[:1_000]
        return cls(
            sequence=max(0, int(event.sequence or 0)),
            event_id=event.event_id,
            timestamp=event.timestamp,
            source=source,
            phase=str(data.get("phase") or data.get("stage") or ""),
            actor=actor,
            task=str(data.get("current_task") or data.get("task") or ""),
            operation=str(data.get("operation") or data.get("provider_state") or kind),
            provider_state=str(data.get("provider_state") or ""),
            state=state,
            message=message,
            detail=detail,
            elapsed_seconds=max(0, int(data.get("elapsed_seconds") or 0)),
            last_signal_age=max(0, int(data.get("last_signal_age") or data.get("quiet_seconds") or 0)),
            received_bytes=max(0, int(data.get("received_bytes") or 0)),
            received_chunks=max(0, int(data.get("received_chunks") or 0)),
            received_tokens=max(0, int(data.get("received_tokens") or 0)),
            workspace_mutated=bool(data.get("workspace_mutated")),
            waiting_on=str(data.get("waiting_on") or ""),
            action=str(data.get("action") or data.get("resume_action") or ""),
            safe_text_fragment=safe_text_fragment,
            stream_kind=stream_kind,
            visibility=visibility,
            task_status=str(data.get("task_status") or ""),
            evidence_summary=" ".join(str(data.get("evidence_summary") or "").split())[:500],
        )


class EventBus:
    """Fan events out to callbacks and queues without coupling producers to UI."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[UIEvent], None]] = []
        self._queues: list[Queue[UIEvent]] = []
        self._lock = RLock()
        self._sequence = 0
        self._live_history: deque[LiveWorkflowEventV1] = deque(maxlen=512)

    def subscribe(self, callback: Callable[[UIEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def open_queue(self) -> Queue[UIEvent]:
        queue: Queue[UIEvent] = Queue()
        with self._lock:
            self._queues.append(queue)
        return queue

    def close_queue(self, queue: Queue[UIEvent]) -> None:
        with self._lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def publish(self, kind: str, message: str = "", **data: Any) -> UIEvent:
        with self._lock:
            self._sequence += 1
            event = UIEvent(
                kind=kind,
                message=message,
                data=data,
                sequence=self._sequence,
            )
            self._live_history.append(LiveWorkflowEventV1.from_ui_event(event))
            callbacks = tuple(self._callbacks)
            queues = tuple(self._queues)
        for queue in queues:
            queue.put(event)
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                # A renderer must never crash or stall the agent runtime.
                continue
        return event

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def list_live_events(
        self, *, after_sequence: int = 0, limit: int = 100
    ) -> tuple[LiveWorkflowEventV1, ...]:
        with self._lock:
            items = tuple(
                item for item in self._live_history if item.sequence > after_sequence
            )
        return items[: max(1, min(int(limit), 512))]


class NullEventBus(EventBus):
    """Drop-in event sink for tests and non-verbose automation."""

    def publish(self, kind: str, message: str = "", **data: Any) -> UIEvent:
        return UIEvent(kind=kind, message=message, data=data)


def drain(queue: Queue[UIEvent], limit: int = 1_000) -> Iterable[UIEvent]:
    """Yield currently queued events without blocking."""
    for _ in range(max(0, limit)):
        try:
            yield queue.get_nowait()
        except Empty:
            return

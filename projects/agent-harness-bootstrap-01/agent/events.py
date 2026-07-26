"""Thread-safe runtime events used by the engine and terminal renderers.

The orchestration layer never needs to print directly.  Keeping events as data
makes the same engine usable from the interactive TUI, tests, or a future API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable event envelope."""

        return {
            "event_id": self.event_id,
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


class EventBus:
    """Fan events out to callbacks and queues without coupling producers to UI."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[UIEvent], None]] = []
        self._queues: list[Queue[UIEvent]] = []
        self._lock = RLock()

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
        event = UIEvent(kind=kind, message=message, data=data)
        with self._lock:
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

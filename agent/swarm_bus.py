"""Realtime swarm message bus with optional durable StateStore backing."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Iterable, Mapping

from .swarm_protocol import SwarmMessageType, SwarmMessageV1


SwarmCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class SwarmSubscriptionV1:
    ultra_run_id: str
    callback: SwarmCallback
    recipient_agent_id: str | None = None
    topic_prefix: str | None = None
    message_type: SwarmMessageType | str | None = None

    def matches(self, message: Mapping[str, Any]) -> bool:
        if str(message.get("ultra_run_id") or "") != self.ultra_run_id:
            return False
        if self.recipient_agent_id is not None and str(message.get("recipient_agent_id") or "") != self.recipient_agent_id:
            return False
        if self.topic_prefix is not None and not str(message.get("topic") or "").startswith(self.topic_prefix):
            return False
        if self.message_type is not None and str(message.get("message_type") or "") != SwarmMessageType(self.message_type).value:
            return False
        return True


class SwarmBus:
    """Fan out formal swarm messages immediately while preserving durability.

    ``store`` is intentionally duck-typed so tests and future transports can
    provide the same small surface: ``post_swarm_message``,
    ``list_swarm_messages``, and ``mark_swarm_message_consumed``.
    """

    def __init__(self, store: Any | None = None) -> None:
        self.store = store
        self._subscriptions: list[SwarmSubscriptionV1] = []
        self._lock = RLock()

    def subscribe(
        self,
        ultra_run_id: str,
        callback: SwarmCallback,
        *,
        recipient_agent_id: str | None = None,
        topic_prefix: str | None = None,
        message_type: SwarmMessageType | str | None = None,
    ) -> Callable[[], None]:
        subscription = SwarmSubscriptionV1(
            ultra_run_id=ultra_run_id,
            callback=callback,
            recipient_agent_id=recipient_agent_id,
            topic_prefix=topic_prefix,
            message_type=message_type,
        )
        with self._lock:
            self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            with self._lock:
                if subscription in self._subscriptions:
                    self._subscriptions.remove(subscription)

        return unsubscribe

    def publish(self, message: SwarmMessageV1 | None = None, **kwargs: Any) -> dict[str, Any]:
        item = message or SwarmMessageV1(**kwargs)
        if self.store is not None:
            delivered = dict(self.store.post_swarm_message(item))
        else:
            delivered = item.to_dict()
        self._notify(delivered)
        return delivered

    def publish_frame(self, frame: str | bytes) -> dict[str, Any]:
        return self.publish(SwarmMessageV1.decode_any_frame(frame))

    def drain(
        self,
        ultra_run_id: str,
        *,
        recipient_agent_id: str,
        topic: str | None = None,
        limit: int = 100,
        acknowledge: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        if self.store is None:
            return ()
        messages = tuple(
            dict(item)
            for item in self.store.list_swarm_messages(
                ultra_run_id,
                recipient_agent_id=recipient_agent_id,
                topic=topic,
                limit=limit,
            )
        )
        for message in messages:
            self._notify(message)
            if acknowledge:
                self.store.mark_swarm_message_consumed(str(message["id"]))
        return messages

    def _notify(self, message: Mapping[str, Any]) -> None:
        with self._lock:
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            if not subscription.matches(message):
                continue
            try:
                subscription.callback(dict(message))
            except Exception:
                # One agent's receiver must not break swarm delivery.
                continue

    @staticmethod
    def frames(
        messages: Iterable[SwarmMessageV1],
        *,
        wire_format: str = "json",
    ) -> tuple[str | bytes, ...]:
        normalized = str(wire_format or "json").casefold()
        if normalized in {"json", "swarm"}:
            return tuple(message.encode_frame() for message in messages)
        if normalized in {"dsl", "text"}:
            return tuple(message.encode_dsl_frame() for message in messages)
        if normalized in {"binary", "bin"}:
            return tuple(message.encode_binary_frame() for message in messages)
        raise ValueError("wire_format must be json, dsl, or binary")

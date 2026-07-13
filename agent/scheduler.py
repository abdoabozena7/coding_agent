"""Deterministic scheduling primitives for ULTRA orchestration.

The scheduler deliberately knows nothing about prompts, providers, or the
coding-agent state schema.  It executes dependency-safe waves, protects write
scopes with leases, and exposes cooperative pause/cancel controls.  This keeps
the same quality pipeline usable with one local worker or several cloud
workers; only the amount of concurrency changes.
"""

from __future__ import annotations

import fnmatch
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Protocol, TypeVar

from .model_catalog import ExecutionClass


class ScheduleStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CONFLICT = "conflict"
    CANCELLED = "cancelled"


class SchedulerError(RuntimeError):
    """Base error for deterministic scheduling failures."""


class DependencyGraphError(SchedulerError):
    pass


class RateLimitError(SchedulerError):
    """A provider rejected work because its current request rate is too high."""


class LeaseConflictError(SchedulerError):
    pass


class StaleWriteError(SchedulerError):
    pass


class CancellationRequested(SchedulerError):
    pass


class Schedulable(Protocol):
    id: str
    depends_on: Iterable[str]
    write_paths: Iterable[str]


T = TypeVar("T", bound=Schedulable)
R = TypeVar("R")


def _scope(value: str) -> str:
    """Return a platform-neutral, comparison-safe resource scope."""

    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = str(PurePosixPath(text or "."))
    if os.name == "nt":
        text = text.casefold()
    return text.rstrip("/") or "."


def scopes_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    """Conservatively detect file, directory, and simple glob overlap."""

    a_values = tuple(_scope(value) for value in left if str(value).strip())
    b_values = tuple(_scope(value) for value in right if str(value).strip())
    for a in a_values:
        for b in b_values:
            if a in {".", "*", "**", "**/*"} or b in {".", "*", "**", "**/*"}:
                return True
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                return True
            # If one side is a glob, testing the other literal is exact.  Two
            # unrelated globs are considered overlapping unless their literal
            # prefixes prove otherwise; false positives only reduce parallelism.
            a_glob = any(char in a for char in "*?[")
            b_glob = any(char in b for char in "*?[")
            if a_glob and fnmatch.fnmatchcase(b, a):
                return True
            if b_glob and fnmatch.fnmatchcase(a, b):
                return True
            if a_glob and b_glob:
                prefix_a = a.split("*", 1)[0].split("?", 1)[0].rstrip("/")
                prefix_b = b.split("*", 1)[0].split("?", 1)[0].rstrip("/")
                if not prefix_a or not prefix_b:
                    return True
                if prefix_a.startswith(prefix_b) or prefix_b.startswith(prefix_a):
                    return True
    return False


@dataclass(frozen=True, slots=True)
class ResourceLease:
    owner: str
    paths: tuple[str, ...]
    acquired_at: float
    expires_at: float | None = None


class ResourceLeaseManager:
    """Atomic in-process path leases with optional optimistic hash checks.

    A v3 state adapter can mirror these hooks into durable ``resource_leases``
    rows.  The in-process implementation remains useful as the last line of
    defence between parallel workers.
    """

    def __init__(
        self,
        hash_reader: Callable[[str], str | None] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_acquire: Callable[[ResourceLease], None] | None = None,
        on_release: Callable[[ResourceLease], None] | None = None,
    ) -> None:
        self.hash_reader = hash_reader
        self._clock = clock
        self.on_acquire = on_acquire
        self.on_release = on_release
        self._leases: dict[str, ResourceLease] = {}
        self._lock = threading.RLock()
        self._mutating_shell_lock = threading.Lock()

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            owner
            for owner, lease in self._leases.items()
            if lease.expires_at is not None and lease.expires_at <= now
        ]
        for owner in expired:
            self._leases.pop(owner, None)

    def acquire(
        self,
        owner: str,
        paths: Iterable[str],
        *,
        expected_hashes: Mapping[str, str | None] | None = None,
        ttl_seconds: float | None = None,
    ) -> ResourceLease:
        normalized = tuple(sorted({_scope(path) for path in paths if str(path).strip()}))
        with self._lock:
            self._purge_expired()
            for other_owner, lease in self._leases.items():
                if other_owner != owner and scopes_overlap(normalized, lease.paths):
                    raise LeaseConflictError(
                        f"write scope for {owner!r} overlaps active lease {other_owner!r}"
                    )
            if expected_hashes and self.hash_reader:
                for raw_path, expected in expected_hashes.items():
                    path = _scope(raw_path)
                    if path not in normalized:
                        continue
                    actual = self.hash_reader(str(raw_path))
                    if actual != expected:
                        raise StaleWriteError(
                            f"pre-write hash changed for {raw_path!r}: expected {expected!r}, got {actual!r}"
                        )
            now = self._clock()
            lease = ResourceLease(
                owner=owner,
                paths=normalized,
                acquired_at=now,
                expires_at=(now + ttl_seconds if ttl_seconds is not None else None),
            )
            self._leases[owner] = lease
            if self.on_acquire:
                try:
                    self.on_acquire(lease)
                except Exception:
                    self._leases.pop(owner, None)
                    raise
            return lease

    def release(self, owner: str) -> None:
        with self._lock:
            lease = self._leases.pop(owner, None)
            if lease is not None and self.on_release:
                self.on_release(lease)

    def active(self) -> tuple[ResourceLease, ...]:
        with self._lock:
            self._purge_expired()
            return tuple(sorted(self._leases.values(), key=lambda lease: lease.owner))

    @contextmanager
    def hold(
        self,
        owner: str,
        paths: Iterable[str],
        *,
        expected_hashes: Mapping[str, str | None] | None = None,
        ttl_seconds: float | None = None,
    ) -> Iterator[ResourceLease]:
        lease = self.acquire(
            owner,
            paths,
            expected_hashes=expected_hashes,
            ttl_seconds=ttl_seconds,
        )
        try:
            yield lease
        finally:
            self.release(owner)

    @contextmanager
    def mutating_shell(self, owner: str) -> Iterator[None]:
        """Serialize mutating shell sections across otherwise parallel agents.

        Tool adapters opt into this hook around commands with workspace side
        effects.  Read-only provider, research, and review work remains fully
        parallel.
        """

        del owner  # Kept in the public signature for durable/audit adapters.
        with self._mutating_shell_lock:
            yield


class CooperativeControl:
    """Thread-safe pause/resume/cancel token checked at safe boundaries."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._cancelled = False

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    def pause(self) -> None:
        with self._condition:
            if not self._cancelled:
                self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._paused = False
            self._condition.notify_all()

    def checkpoint(self, timeout: float = 0.1) -> None:
        with self._condition:
            while self._paused and not self._cancelled:
                self._condition.wait(timeout=max(0.001, timeout))
            if self._cancelled:
                raise CancellationRequested("ULTRA run was cancelled")


class AdaptiveConcurrency:
    """Shared concurrency limit that backs off on 429s and heals gradually."""

    def __init__(
        self,
        execution_class: ExecutionClass | str,
        *,
        cloud_default: int = 4,
        maximum: int = 8,
        recover_after: int = 4,
    ) -> None:
        self.execution_class = ExecutionClass(execution_class)
        self.maximum = max(1, min(8, int(maximum)))
        desired = 1 if self.execution_class is ExecutionClass.LOCAL else cloud_default
        self.initial = max(1, min(self.maximum, int(desired)))
        self._current = self.initial
        self._recover_after = max(1, int(recover_after))
        self._successes = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._current

    def on_rate_limit(self) -> int:
        with self._lock:
            self._current = max(1, self._current - 1)
            self._successes = 0
            return self._current

    def on_success(self) -> int:
        with self._lock:
            if self.execution_class is ExecutionClass.LOCAL:
                return 1
            self._successes += 1
            if self._successes >= self._recover_after and self._current < self.initial:
                self._current += 1
                self._successes = 0
            return self._current


@dataclass(frozen=True, slots=True)
class ItemOutcome(Generic[R]):
    item_id: str
    status: ScheduleStatus
    result: R | None = None
    error: str = ""
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class ScheduleReport(Generic[R]):
    outcomes: tuple[ItemOutcome[R], ...]
    waves: tuple[tuple[str, ...], ...]
    peak_concurrency: int

    @property
    def successful(self) -> bool:
        return bool(self.outcomes) and all(
            outcome.status is ScheduleStatus.COMPLETED for outcome in self.outcomes
        )


EventCallback = Callable[[str, str, Mapping[str, Any]], None]


class DeterministicWaveScheduler:
    """Run a DAG in stable, dependency-safe, write-disjoint waves."""

    def __init__(
        self,
        execution_class: ExecutionClass | str,
        *,
        cloud_default: int = 4,
        maximum: int = 8,
        rate_limit_retries: int = 3,
        rate_limit_backoff: Callable[[int], None] | None = None,
        leases: ResourceLeaseManager | None = None,
        control: CooperativeControl | None = None,
        adaptive: AdaptiveConcurrency | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.execution_class = ExecutionClass(execution_class)
        self.adaptive = adaptive or AdaptiveConcurrency(
            self.execution_class,
            cloud_default=cloud_default,
            maximum=maximum,
        )
        self.rate_limit_retries = max(0, int(rate_limit_retries))
        self.rate_limit_backoff = rate_limit_backoff or (lambda _attempt: None)
        self.leases = leases or ResourceLeaseManager()
        self.control = control or CooperativeControl()
        self.on_event = on_event

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        if self.on_event:
            self.on_event(kind, message, data)

    @staticmethod
    def _ordered(items: Iterable[T]) -> list[T]:
        return sorted(items, key=lambda item: (int(getattr(item, "order", 0)), str(item.id)))

    @staticmethod
    def _validate(items: list[T], externally_completed: set[str]) -> None:
        ids = [str(item.id) for item in items]
        if len(ids) != len(set(ids)):
            raise DependencyGraphError("work-node ids must be unique")
        known = set(ids) | externally_completed
        for item in items:
            missing = set(map(str, item.depends_on)) - known
            if missing:
                raise DependencyGraphError(
                    f"node {item.id!r} depends on unknown nodes: {sorted(missing)}"
                )

    @staticmethod
    def _wave(ready: list[T], limit: int) -> list[T]:
        wave: list[T] = []
        scopes: list[str] = []
        for item in ready:
            item_scopes = tuple(str(path) for path in item.write_paths)
            if scopes_overlap(scopes, item_scopes):
                continue
            wave.append(item)
            scopes.extend(item_scopes)
            if len(wave) >= limit:
                break
        return wave

    def _run_one(self, item: T, worker: Callable[[T], R]) -> ItemOutcome[R]:
        expected_hashes = getattr(item, "pre_write_hashes", None)
        attempts = 0
        while True:
            attempts += 1
            try:
                self.control.checkpoint()
                with self.leases.hold(
                    str(item.id),
                    item.write_paths,
                    expected_hashes=(expected_hashes if isinstance(expected_hashes, Mapping) else None),
                ):
                    result = worker(item)
                self.adaptive.on_success()
                return ItemOutcome(str(item.id), ScheduleStatus.COMPLETED, result, attempts=attempts)
            except RateLimitError as exc:
                limit = self.adaptive.on_rate_limit()
                self._emit(
                    "ultra.rate_limited",
                    f"Rate limited; concurrency reduced to {limit}",
                    node_id=str(item.id),
                    concurrency=limit,
                    attempt=attempts,
                )
                if attempts > self.rate_limit_retries:
                    return ItemOutcome(
                        str(item.id), ScheduleStatus.FAILED, error=str(exc), attempts=attempts
                    )
                self.rate_limit_backoff(attempts)
            except (LeaseConflictError, StaleWriteError) as exc:
                return ItemOutcome(
                    str(item.id), ScheduleStatus.CONFLICT, error=str(exc), attempts=attempts
                )
            except CancellationRequested as exc:
                return ItemOutcome(
                    str(item.id), ScheduleStatus.CANCELLED, error=str(exc), attempts=attempts
                )
            except Exception as exc:  # Worker failures are data, not scheduler crashes.
                return ItemOutcome(
                    str(item.id),
                    ScheduleStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    attempts=attempts,
                )

    def run(
        self,
        items: Iterable[T],
        worker: Callable[[T], R],
        *,
        initially_completed: Iterable[str] = (),
    ) -> ScheduleReport[R]:
        ordered = self._ordered(items)
        external = set(map(str, initially_completed))
        self._validate(ordered, external)
        by_id = {str(item.id): item for item in ordered}
        pending = set(by_id)
        completed = set(external)
        outcomes: dict[str, ItemOutcome[R]] = {}
        waves: list[tuple[str, ...]] = []
        peak = 0

        while pending:
            try:
                self.control.checkpoint()
            except CancellationRequested as exc:
                for item in ordered:
                    if str(item.id) in pending:
                        outcomes[str(item.id)] = ItemOutcome(
                            str(item.id), ScheduleStatus.CANCELLED, error=str(exc)
                        )
                break

            failed_ids = {
                item_id
                for item_id, outcome in outcomes.items()
                if outcome.status is not ScheduleStatus.COMPLETED
            }
            newly_blocked = [
                item
                for item in ordered
                if str(item.id) in pending
                and set(map(str, item.depends_on)) & failed_ids
            ]
            for item in newly_blocked:
                item_id = str(item.id)
                pending.remove(item_id)
                outcomes[item_id] = ItemOutcome(
                    item_id,
                    ScheduleStatus.BLOCKED,
                    error="a dependency did not complete successfully",
                )

            # Recompute failed IDs on the next pass so blocked state cascades
            # through an arbitrarily deep dependency chain. Without this, a
            # failure at M001 blocks M002 and is then misreported as a cycle in
            # M003+.
            if newly_blocked:
                continue

            ready = [
                item
                for item in ordered
                if str(item.id) in pending
                and set(map(str, item.depends_on)) <= completed
            ]
            if not ready:
                if pending:
                    cycle = ", ".join(sorted(pending))
                    raise DependencyGraphError(f"dependency cycle or unresolved nodes: {cycle}")
                break

            limit = 1 if self.execution_class is ExecutionClass.LOCAL else self.adaptive.current
            wave = self._wave(ready, limit)
            if not wave:  # Defensive: the first ready node never conflicts with itself.
                wave = ready[:1]
            wave_ids = tuple(str(item.id) for item in wave)
            waves.append(wave_ids)
            peak = max(peak, len(wave))
            self._emit(
                "ultra.wave_started",
                f"Executing {len(wave)} node(s)",
                node_ids=wave_ids,
                concurrency=limit,
            )

            if len(wave) == 1:
                batch = [self._run_one(wave[0], worker)]
            else:
                with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="ultra") as pool:
                    futures = [pool.submit(self._run_one, item, worker) for item in wave]
                    # Consume in wave order so event/state application is deterministic.
                    batch = [future.result() for future in futures]

            for outcome in batch:
                outcomes[outcome.item_id] = outcome
                pending.discard(outcome.item_id)
                if outcome.status is ScheduleStatus.COMPLETED:
                    completed.add(outcome.item_id)
                self._emit(
                    "ultra.node_scheduled_result",
                    f"{outcome.item_id}: {outcome.status.value}",
                    node_id=outcome.item_id,
                    status=outcome.status.value,
                    attempts=outcome.attempts,
                    error=outcome.error,
                )

        return ScheduleReport(
            outcomes=tuple(outcomes[str(item.id)] for item in ordered),
            waves=tuple(waves),
            peak_concurrency=peak,
        )


class BackgroundRunController(Generic[R]):
    """Own one background orchestration thread and its cooperative controls."""

    def __init__(self, control: CooperativeControl | None = None) -> None:
        self.control = control or CooperativeControl()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ultra-controller")
        self._future: Future[R] | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    def start(self, target: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("an ULTRA run is already active")
            self._future = self._executor.submit(target, *args, **kwargs)
            return self._future

    def pause(self) -> None:
        self.control.pause()

    def resume(self) -> None:
        self.control.resume()

    def cancel(self) -> None:
        self.control.cancel()

    def result(self, timeout: float | None = None) -> R:
        with self._lock:
            future = self._future
        if future is None:
            raise RuntimeError("no ULTRA run has been started")
        return future.result(timeout=timeout)

    def close(self) -> None:
        self.control.cancel()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "BackgroundRunController[R]":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "AdaptiveConcurrency",
    "BackgroundRunController",
    "CancellationRequested",
    "CooperativeControl",
    "DependencyGraphError",
    "DeterministicWaveScheduler",
    "ExecutionClass",
    "ItemOutcome",
    "LeaseConflictError",
    "RateLimitError",
    "ResourceLease",
    "ResourceLeaseManager",
    "ScheduleReport",
    "ScheduleStatus",
    "SchedulerError",
    "StaleWriteError",
    "scopes_overlap",
]

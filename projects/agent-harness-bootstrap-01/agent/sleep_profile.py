"""Session-scoped Sleep profile controls layered on top of Ultra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import InteractionMode
from .quality import QualityCycleV1
from .sandbox import AccessLevel
from .workflow import SleepState, UltraProfile


class SleepActivationError(RuntimeError):
    pass


@dataclass
class SleepController:
    """Gate Sleep activation and prevent repeated equivalent failed approaches."""

    profile: UltraProfile = UltraProfile.STANDARD
    state: SleepState = SleepState.OFF
    stop_requested: bool = False
    approach_attempts: dict[str, int] = field(default_factory=dict)

    def enable(
        self,
        *,
        mode: InteractionMode,
        access_level: AccessLevel,
        docker_ready: bool,
        safe_checkpoint: bool,
        active_uncertain_mutation: bool,
    ) -> SleepState:
        if mode is not InteractionMode.ULTRA:
            raise SleepActivationError("Sleep is an Ultra profile and cannot run outside ULTRA mode")
        if access_level is not AccessLevel.FULL or not docker_ready:
            raise SleepActivationError("Sleep requires ready Full access inside Docker")
        if not safe_checkpoint:
            raise SleepActivationError("Sleep can be enabled only at a safe checkpoint")
        if active_uncertain_mutation:
            raise SleepActivationError("Sleep cannot start while a mutation has uncertain state")
        self.profile = UltraProfile.SLEEP
        self.state = SleepState.ARMED
        self.stop_requested = False
        return self.state

    def disable(self) -> SleepState:
        self.stop_requested = True
        if self.state is SleepState.ARMED:
            self.state = SleepState.OFF
            self.profile = UltraProfile.STANDARD
        return self.state

    def checkpoint(self) -> SleepState:
        if self.stop_requested:
            self.state = SleepState.OFF
            self.profile = UltraProfile.STANDARD
        elif self.state is SleepState.ARMED:
            self.state = SleepState.RUNNING
        return self.state

    def record_cycle(self, cycle: QualityCycleV1) -> None:
        if cycle.result.casefold() in {"failed", "blocked", "error"}:
            count = self.approach_attempts.get(cycle.approach_fingerprint, 0) + 1
            self.approach_attempts[cycle.approach_fingerprint] = count
            if count > 3:
                self.state = SleepState.BLOCKED
                raise SleepActivationError(
                    "the same Sleep approach failed three times; a materially different replan fingerprint is required"
                )

    def status(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "state": self.state.value,
            "stop_requested": self.stop_requested,
            "approach_attempts": dict(self.approach_attempts),
        }

"""Shared parse/normalize/validate/repair pipeline for typed model returns."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Generic, Mapping, TypeVar

from .workflow import RetryKind, RetryLedger

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TypedReturnFailure(ValueError):
    contract: str
    stage: str
    errors: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.contract} failed during {self.stage}: " + "; ".join(self.errors)


@dataclass(frozen=True, slots=True)
class TypedReturnResult(Generic[T]):
    value: T
    normalized: Mapping[str, Any]
    normalization_actions: tuple[str, ...] = ()
    repaired: bool = False


def parse_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    candidate = str(raw or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at character {exc.pos}: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("typed return must be one JSON object")
    return dict(value)


class TypedReturnProcessor(Generic[T]):
    """Apply the same lifecycle to every typed agent return.

    The caller persists the returned value and only then marks the agent
    completed.  A repair callback receives a field-specific contract and is
    invoked at most once by default.
    """

    def __init__(
        self,
        contract: str,
        factory: Callable[[Mapping[str, Any]], T],
        *,
        normalize: Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], tuple[str, ...]]] | None = None,
        semantic_validate: Callable[[T], None] | None = None,
        repair_kind: RetryKind = RetryKind.TYPED_PARSE_REPAIR,
        max_repairs: int = 1,
    ) -> None:
        self.contract = contract
        self.factory = factory
        self.normalize = normalize or (lambda value: (dict(value), ()))
        self.semantic_validate = semantic_validate
        self.repair_kind = repair_kind
        self.max_repairs = max(0, int(max_repairs))

    def process(
        self,
        raw: Any,
        *,
        repair: Callable[[Mapping[str, Any], tuple[str, ...], str], Any] | None = None,
        ledger: RetryLedger | None = None,
    ) -> TypedReturnResult[T]:
        current = raw
        previous: Mapping[str, Any] = {}
        for attempt in range(self.max_repairs + 1):
            try:
                parsed = parse_json_object(current)
                previous = parsed
                normalized, actions = self.normalize(parsed)
                value = self.factory(normalized)
                if self.semantic_validate is not None:
                    self.semantic_validate(value)
                return TypedReturnResult(value, dict(normalized), tuple(actions), repaired=attempt > 0)
            except Exception as exc:
                stage = "parse" if not previous else "validation"
                errors = (str(exc),)
                if repair is None or attempt >= self.max_repairs:
                    raise TypedReturnFailure(self.contract, stage, errors) from exc
                if ledger is not None:
                    ledger.record(
                        self.repair_kind,
                        stage=self.contract,
                        reason=str(exc),
                        input_value=previous or current,
                        output_value=None,
                        next_action="targeted_repair",
                    )
                current = repair(previous, errors, self.contract)
                previous = {}
        raise AssertionError("unreachable")

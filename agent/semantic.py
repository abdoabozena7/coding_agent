"""Domain-neutral semantic, resource, verification, and strategy contracts.

The harness validates these contracts but never supplies product semantics.  A
contract may preserve the user's request verbatim while interpretation is still
pending; it must not manufacture a domain, deliverable, path, or acceptance
condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


class SemanticContractError(ValueError):
    """A model-produced semantic contract violated a domain-neutral invariant."""


class RequestedEffect(str, Enum):
    """Capability effects used for safety policy, never task-domain routing."""

    ANSWER = "answer"
    READ_WORKSPACE = "read_workspace"
    MUTATE_WORKSPACE = "mutate_workspace"
    EXECUTE_CODE = "execute_code"
    INSTALL_DEPENDENCIES = "install_dependencies"
    USE_NETWORK = "use_network"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


def _bounded_strings(
    values: Iterable[Any] | Any,
    *,
    field_name: str,
    limit: int,
    item_limit: int = 2_000,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        if len(value) > item_limit:
            raise SemanticContractError(f"{field_name} item exceeds {item_limit} characters")
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    if len(result) > limit:
        raise SemanticContractError(f"{field_name} exceeds {limit} items")
    return tuple(result)


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticGoalV2:
    """Repository-grounded interpretation of one request.

    ``pending`` is a safe initial state: the original text remains authoritative
    and no semantics are invented before inspected model output is accepted.
    """

    original_request: str
    interpreted_outcome: str
    requested_effects: tuple[RequestedEffect, ...] = ()
    required_outcomes: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    unresolved_decisions: tuple[str, ...] = ()
    repository_evidence_refs: tuple[str, ...] = ()
    status: str = "pending"
    version: int = 2

    def __post_init__(self) -> None:
        original = str(self.original_request or "")
        outcome = str(self.interpreted_outcome or "").strip()
        if not original.strip() or len(original) > 200_000:
            raise SemanticContractError("semantic goal requires a bounded original request")
        if not outcome or len(outcome) > 20_000:
            raise SemanticContractError("semantic goal requires a bounded interpreted outcome")
        if self.version != 2:
            raise SemanticContractError("SemanticGoalV2 only accepts version 2")
        if self.status not in {"pending", "interpreted", "critic_accepted"}:
            raise SemanticContractError("semantic goal status is invalid")
        effects: list[RequestedEffect] = []
        for effect in self.requested_effects:
            effects.append(effect if isinstance(effect, RequestedEffect) else RequestedEffect(str(effect)))
        object.__setattr__(self, "original_request", original)
        object.__setattr__(self, "interpreted_outcome", outcome)
        object.__setattr__(self, "requested_effects", tuple(dict.fromkeys(effects)))
        for name, limit in (
            ("required_outcomes", 40),
            ("constraints", 40),
            ("exclusions", 40),
            ("acceptance_criteria", 40),
            ("unresolved_decisions", 20),
            ("repository_evidence_refs", 80),
        ):
            object.__setattr__(
                self,
                name,
                _bounded_strings(getattr(self, name), field_name=name, limit=limit),
            )
        if self.status == "critic_accepted":
            if not self.required_outcomes or not self.acceptance_criteria:
                raise SemanticContractError(
                    "an accepted semantic goal requires outcomes and acceptance criteria"
                )
            if not self.repository_evidence_refs:
                raise SemanticContractError(
                    "an accepted semantic goal requires repository inspection evidence"
                )

    @classmethod
    def pending(cls, request: str) -> "SemanticGoalV2":
        """Preserve the request exactly without guessing its meaning."""

        value = str(request or "")
        return cls(
            original_request=value,
            interpreted_outcome=value,
            status="pending",
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        original_request: str,
    ) -> "SemanticGoalV2":
        supplied_original = str(value.get("original_request") or original_request)
        if supplied_original != str(original_request):
            raise SemanticContractError("semantic interpretation changed the original request")
        return cls(
            original_request=supplied_original,
            interpreted_outcome=str(value.get("interpreted_outcome") or "").strip(),
            requested_effects=tuple(value.get("requested_effects") or ()),
            required_outcomes=tuple(value.get("required_outcomes") or ()),
            constraints=tuple(value.get("constraints") or ()),
            exclusions=tuple(value.get("exclusions") or ()),
            acceptance_criteria=tuple(value.get("acceptance_criteria") or ()),
            unresolved_decisions=tuple(value.get("unresolved_decisions") or ()),
            repository_evidence_refs=tuple(value.get("repository_evidence_refs") or ()),
            status=str(value.get("status") or "interpreted"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "requested_effects": [item.value for item in self.requested_effects],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


def _normalized_relative_path(value: str) -> str:
    path = PurePosixPath(str(value or "").replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise SemanticContractError("resource paths must be workspace-relative")
    return path.as_posix().removeprefix("./")


@dataclass(frozen=True, slots=True)
class ResourceClaimV1:
    """Evidence-backed resource intent resolved into a temporary execution lease."""

    purpose: str
    kind: str
    supports_tasks: tuple[str, ...]
    inspection_refs: tuple[str, ...]
    selector: str = ""
    resolved_paths: tuple[str, ...] = ()
    state: str = "proposed"
    version: int = 1

    def __post_init__(self) -> None:
        purpose = str(self.purpose or "").strip()
        selector = str(self.selector or "").strip()
        if not purpose or len(purpose) > 2_000:
            raise SemanticContractError("resource claim requires a bounded purpose")
        if self.kind not in {"file", "directory", "symbol", "command", "artifact"}:
            raise SemanticContractError("resource claim kind is invalid")
        if self.state not in {"proposed", "resolved", "leased", "released"}:
            raise SemanticContractError("resource claim state is invalid")
        tasks = _bounded_strings(
            self.supports_tasks, field_name="supports_tasks", limit=80, item_limit=24
        )
        refs = _bounded_strings(
            self.inspection_refs, field_name="inspection_refs", limit=80, item_limit=500
        )
        paths = tuple(dict.fromkeys(_normalized_relative_path(item) for item in self.resolved_paths))
        if not tasks:
            raise SemanticContractError("resource claim must support at least one task")
        if not refs:
            raise SemanticContractError("resource claim requires inspection evidence")
        if self.state in {"resolved", "leased", "released"} and not paths:
            raise SemanticContractError("resolved resource claim requires concrete paths")
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "supports_tasks", tasks)
        object.__setattr__(self, "inspection_refs", refs)
        object.__setattr__(self, "resolved_paths", paths)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class VerificationContractV1:
    criterion: str
    method: str
    scope: str
    expected_result: str
    authority: str
    required: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        for name in ("criterion", "method", "scope", "expected_result", "authority"):
            value = str(getattr(self, name) or "").strip()
            if not value or len(value) > 2_000:
                raise SemanticContractError(f"verification contract {name} is invalid")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrategyAttemptV1:
    task_id: str
    hypothesis: str
    approach: str
    evidence_refs: tuple[str, ...] = ()
    outcome: str = "pending"
    next_strategy: str = ""
    version: int = 1
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not str(self.task_id).strip():
            raise SemanticContractError("strategy attempt requires a task id")
        if not str(self.hypothesis).strip() or not str(self.approach).strip():
            raise SemanticContractError("strategy attempt requires hypothesis and approach")
        if self.outcome not in {"pending", "improved", "unchanged", "regressed", "blocked"}:
            raise SemanticContractError("strategy attempt outcome is invalid")
        refs = _bounded_strings(
            self.evidence_refs, field_name="evidence_refs", limit=80, item_limit=500
        )
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                {
                    "task_id": self.task_id,
                    "hypothesis": " ".join(self.hypothesis.split()).casefold(),
                    "approach": " ".join(self.approach.split()).casefold(),
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "RequestedEffect",
    "ResourceClaimV1",
    "SemanticContractError",
    "SemanticGoalV2",
    "StrategyAttemptV1",
    "VerificationContractV1",
]

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
import re
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

    @classmethod
    def parse(cls, value: Any) -> "RequestedEffect":
        """Normalize only equivalent capability names used by our own contracts.

        This is schema compatibility, not intent inference: an effect must still
        be explicitly model-authored. Unknown values remain invalid.
        """

        if isinstance(value, Mapping):
            authored = None
            for key in ("effect", "type", "name", "value"):
                if value.get(key) is not None:
                    authored = value[key]
                    break
            value = authored
        normalized = str(getattr(value, "value", value)).strip().casefold()
        # A few structured-output providers emit a sentinel such as ``none``
        # when the effect list is empty.  That sentinel is transport noise, not
        # a capability effect.  Callers that accept an empty list filter it
        # before parsing; retaining the explicit failure here protects callers
        # that require at least one real effect (for example an Action).
        if normalized in {"", "none", "no_effect", "no effects", "no requested effects"}:
            raise ValueError(f"empty requested effect sentinel: {value!r}")
        aliases = {
            "read": cls.READ_WORKSPACE.value,
            "read_file": cls.READ_WORKSPACE.value,
            "list_files": cls.READ_WORKSPACE.value,
            "grep": cls.READ_WORKSPACE.value,
            "write": cls.MUTATE_WORKSPACE.value,
            "mutate": cls.MUTATE_WORKSPACE.value,
            "write_workspace": cls.MUTATE_WORKSPACE.value,
            "write_file": cls.MUTATE_WORKSPACE.value,
            "edit_file": cls.MUTATE_WORKSPACE.value,
            "apply_patch": cls.MUTATE_WORKSPACE.value,
            "materialize_artifact": cls.MUTATE_WORKSPACE.value,
            "browser_open": cls.EXECUTE_CODE.value,
            "browser_inspect": cls.EXECUTE_CODE.value,
            "browser_act": cls.EXECUTE_CODE.value,
            "browser_screenshot": cls.EXECUTE_CODE.value,
            "browser_close": cls.EXECUTE_CODE.value,
            "publish_output": cls.EXTERNAL_SIDE_EFFECT.value,
            "inspect_images": cls.READ_WORKSPACE.value,
            "run": cls.EXECUTE_CODE.value,
            "execute": cls.EXECUTE_CODE.value,
            "run_command": cls.EXECUTE_CODE.value,
            "execute_shell": cls.EXECUTE_CODE.value,
            "run_shell": cls.EXECUTE_CODE.value,
            "shell": cls.EXECUTE_CODE.value,
            "preview": cls.EXECUTE_CODE.value,
            "preview_html": cls.EXECUTE_CODE.value,
            "inspect_preview": cls.EXECUTE_CODE.value,
            "run_bash": cls.EXECUTE_CODE.value,
            "start_process": cls.EXECUTE_CODE.value,
            "install": cls.INSTALL_DEPENDENCIES.value,
            "install_dependency": cls.INSTALL_DEPENDENCIES.value,
            "network": cls.USE_NETWORK.value,
            "external": cls.EXTERNAL_SIDE_EFFECT.value,
        }
        return cls(aliases.get(normalized, normalized))


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
class RequirementAnchorV1:
    """Model-authored meaning tied to an exact span of user authority.

    The harness validates provenance and coverage, while the model owns the
    domain interpretation. This prevents a named medium, framework, format, or
    experiential requirement from degrading into a superficial import.
    """

    id: str
    verbatim_span: str
    interpreted_requirement: str
    observable_implications: tuple[str, ...]
    kind: str = "requirement"
    version: int = 1

    def __post_init__(self) -> None:
        identifier = str(self.id or "").strip().upper()
        span = str(self.verbatim_span or "")
        meaning = str(self.interpreted_requirement or "").strip()
        kind = str(self.kind or "requirement").strip().casefold()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,23}", identifier):
            raise SemanticContractError("requirement anchor id is invalid")
        if not span.strip() or len(span) > 2_000:
            raise SemanticContractError("requirement anchor requires a bounded verbatim span")
        if not meaning or len(meaning) > 4_000:
            raise SemanticContractError("requirement anchor requires a bounded interpretation")
        implications = _bounded_strings(
            self.observable_implications,
            field_name="requirement anchor observable_implications",
            limit=12,
            item_limit=2_000,
        )
        if not implications:
            raise SemanticContractError(
                "requirement anchor requires at least one observable implication"
            )
        if not kind or len(kind) > 100:
            raise SemanticContractError("requirement anchor kind is invalid")
        if self.version != 1:
            raise SemanticContractError("RequirementAnchorV1 only accepts version 1")
        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "verbatim_span", span)
        object.__setattr__(self, "interpreted_requirement", meaning)
        object.__setattr__(self, "observable_implications", implications)
        object.__setattr__(self, "kind", kind)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequirementAnchorV1":
        return cls(
            id=str(value.get("id") or ""),
            verbatim_span=str(value.get("verbatim_span") or ""),
            interpreted_requirement=str(value.get("interpreted_requirement") or ""),
            observable_implications=tuple(value.get("observable_implications") or ()),
            kind=str(value.get("kind") or "requirement"),
            version=int(value.get("version") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "verbatim_span": self.verbatim_span,
            "interpreted_requirement": self.interpreted_requirement,
            "observable_implications": list(self.observable_implications),
            "kind": self.kind,
            "version": self.version,
        }


def canonicalize_requirement_anchors(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    """Assign stable harness-owned IDs and return a legacy alias map.

    Model IDs are transport hints only.  Missing, malformed, long, or
    duplicate values cannot invalidate otherwise useful semantic meaning;
    canonical order is the durable identity used by plans and fingerprints.
    """
    raw = dict(value or {})
    anchors = raw.get("requirement_anchors") or ()
    normalized: list[dict[str, Any]] = []
    aliases: dict[str, list[str]] = {}
    if isinstance(anchors, Mapping):
        anchors = (anchors,)
    for index, item in enumerate(anchors if isinstance(anchors, (list, tuple)) else ()):
        if isinstance(item, RequirementAnchorV1):
            item = item.to_dict()
        if not isinstance(item, Mapping):
            continue
        canonical = f"R{index + 1:03d}"
        copied = dict(item)
        original_id = str(copied.get("id") or "").strip().upper()
        if original_id:
            aliases.setdefault(original_id, []).append(canonical)
        # Canonical IDs are intentionally generated even when a model emits
        # spaces, Arabic text, a number, or a duplicate.
        copied["id"] = canonical
        # Weak structured emitters sometimes place the model-authored meaning
        # under a transport alias, or provide only the observable implications.
        # Normalize that shape without inventing a requirement: the fallback
        # is composed exclusively from the model's own implications.
        if not str(copied.get("interpreted_requirement") or "").strip():
            for alias in ("interpretation", "meaning", "requirement", "description"):
                candidate = copied.get(alias)
                if str(candidate or "").strip():
                    copied["interpreted_requirement"] = str(candidate).strip()
                    break
        implications = copied.get("observable_implications")
        if isinstance(implications, str) and implications.strip():
            copied["observable_implications"] = [implications.strip()]
        if not str(copied.get("interpreted_requirement") or "").strip():
            values = copied.get("observable_implications") or ()
            if isinstance(values, (list, tuple)) and values:
                copied["interpreted_requirement"] = " ".join(
                    str(value).strip() for value in values if str(value).strip()
                )[:4_000]
        normalized.append(copied)
    raw["requirement_anchors"] = normalized
    return raw, {key: tuple(items) for key, items in aliases.items()}


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
    requirement_anchors: tuple[RequirementAnchorV1, ...] = ()
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
            effects.append(RequestedEffect.parse(effect))
        object.__setattr__(self, "original_request", original)
        object.__setattr__(self, "interpreted_outcome", outcome)
        object.__setattr__(self, "requested_effects", tuple(dict.fromkeys(effects)))
        anchors = tuple(
            item
            if isinstance(item, RequirementAnchorV1)
            else RequirementAnchorV1.from_mapping(item)
            for item in self.requirement_anchors
        )
        if len(anchors) > 40:
            raise SemanticContractError("requirement_anchors exceeds 40 items")
        anchor_ids = [item.id for item in anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise SemanticContractError("requirement anchor ids must be unique")
        missing_spans = [item.id for item in anchors if item.verbatim_span not in original]
        if missing_spans:
            raise SemanticContractError(
                "requirement anchor spans must be verbatim substrings of original_request: "
                + ", ".join(missing_spans)
            )
        object.__setattr__(self, "requirement_anchors", anchors)
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
            if not self.acceptance_criteria:
                raise SemanticContractError(
                    "an accepted semantic goal requires acceptance criteria"
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
        # Some tool-capable providers wrap the staged object in the legacy
        # ``semantic_goal`` property, or use a nearby field name for the
        # model-authored interpretation despite receiving the canonical
        # schema.  Unwrap those transport shapes only; never manufacture an
        # outcome from the user's request or infer any requested effect.
        raw = dict(value or {})
        nested = raw.get("semantic_goal")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update(
                {
                    key: item
                    for key, item in raw.items()
                    if key != "semantic_goal"
                    and (key not in merged or item not in (None, "", (), [], {}))
                }
            )
            raw = merged
        raw, _aliases = canonicalize_requirement_anchors(raw)
        if not str(raw.get("interpreted_outcome") or "").strip():
            for alias in (
                "interpretation",
                "outcome",
                "interpreted_objective",
                "objective",
                "summary",
            ):
                candidate = raw.get(alias)
                if str(candidate or "").strip():
                    raw["interpreted_outcome"] = candidate
                    break
        supplied_original = str(raw.get("original_request") or original_request)
        if supplied_original != str(original_request):
            raise SemanticContractError("semantic interpretation changed the original request")
        # ``status`` is a lifecycle field owned by the harness, not semantic
        # content the model is expected to author.  Some providers still echo
        # the legacy ``pending`` value (or omit it entirely) while submitting
        # a complete interpretation.  Treat that transport detail as an
        # interpreted proposal so it cannot consume a semantic-repair attempt
        # or stop planning before the actual contract is validated.  Accepted
        # metadata is reconstructed explicitly by the harness as
        # ``critic_accepted`` after the review gates pass.
        raw_status = str(raw.get("status") or "").strip().casefold()
        semantic_status = "critic_accepted" if raw_status == "critic_accepted" else "interpreted"
        raw_effects = raw.get("requested_effects") or ()
        if isinstance(raw_effects, Mapping):
            raw_effects = tuple(
                key for key, enabled in raw_effects.items() if bool(enabled)
            )
        elif isinstance(raw_effects, str):
            raw_effects = (raw_effects,)
        elif not isinstance(raw_effects, (list, tuple)):
            raw_effects = ()
        # ``none`` is a common empty-array spelling from local structured
        # emitters.  It must not turn a valid Goal into a semantic contract
        # failure; real effects remain model-authored and are still validated
        # strictly below.
        raw_effects = tuple(
            item for item in raw_effects
            if str(item or "").strip().casefold()
            not in {"", "none", "no_effect", "no effects", "no requested effects"}
        )
        return cls(
            original_request=supplied_original,
            interpreted_outcome=str(raw.get("interpreted_outcome") or "").strip(),
            requested_effects=raw_effects,
            required_outcomes=tuple(raw.get("required_outcomes") or ()),
            constraints=tuple(raw.get("constraints") or ()),
            exclusions=tuple(raw.get("exclusions") or ()),
            acceptance_criteria=tuple(raw.get("acceptance_criteria") or ()),
            requirement_anchors=tuple(raw.get("requirement_anchors") or ()),
            unresolved_decisions=tuple(raw.get("unresolved_decisions") or ()),
            repository_evidence_refs=tuple(raw.get("repository_evidence_refs") or ()),
            status=semantic_status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "requested_effects": [item.value for item in self.requested_effects],
            "requirement_anchors": [item.to_dict() for item in self.requirement_anchors],
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
    "RequirementAnchorV1", "canonicalize_requirement_anchors", "SemanticGoalV2",
    "StrategyAttemptV1",
    "VerificationContractV1",
]

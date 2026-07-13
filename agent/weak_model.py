"""Authoritative, versioned harness policy for weak-model execution."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class WeakModelPolicy:
    version: int = 1
    active: bool = True
    one_primary_decision_per_call: bool = True
    harness_generated_ids: bool = True
    deterministic_dependencies: bool = True
    narrow_context: bool = True
    separate_implementation_and_criticism: bool = True
    separate_functional_and_visual_evaluation: bool = True
    mechanical_normalization_first: bool = True
    evidence_backed_completion: bool = True
    durable_checkpoints: bool = True
    mandatory_executable_evidence: bool = True
    reject_prose_completion: bool = True
    fresh_evaluation_after_mutation: bool = True
    persist_failed_hypotheses: bool = True
    harness_owns_lifecycle: bool = True
    independent_final_evaluation: bool = True
    accept_first_syntactic_result: bool = False
    max_context_characters: int = 40_000
    max_equivalent_failed_approaches: int = 3
    expose_unverified_draft_as_final: bool = False

    def stages(self) -> tuple[str, ...]:
        return ("understand", "plan", "implement", "verify", "criticize", "refine", "fresh_evaluate")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "WeakModelPolicy":
        if not value:
            return cls()
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{name: value[name] for name in known if name in value})

    def applied_rules(self, decision: str) -> tuple[str, ...]:
        """Return inspectable rule names that govern a runtime decision."""
        routes = {
            "provider_call": ("one_primary_decision_per_call", "narrow_context", "harness_generated_ids"),
            "mutation": ("durable_checkpoints", "fresh_evaluation_after_mutation", "harness_owns_lifecycle"),
            "completion": ("mandatory_executable_evidence", "reject_prose_completion", "independent_final_evaluation", "accept_first_syntactic_result"),
            "retry": ("max_equivalent_failed_approaches", "persist_failed_hypotheses"),
        }
        return tuple(name for name in routes.get(decision, ()) if bool(getattr(self, name)))

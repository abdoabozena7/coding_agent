"""Matched four-arm benchmark for evidence-driven weak-model workers.

The benchmark owns comparability and statistics; a provider-specific executor
owns model calls.  Hidden evaluators stay outside the model workspace and
return only deterministic outcomes to this harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .models import DomainError
from .orchestration import (
    AdaptiveOrchestrationPolicyV1,
    OrchestrationArm,
    OrchestrationExperimentV1,
)


TASK_CLASSES = (
    "exact_edit",
    "bug_repair",
    "multi_file_integration",
    "browser_tool_use",
    "security_edge_case",
    "requirement_fidelity",
)
DEFAULT_SEEDS = (11, 29, 47)
BENCHMARK_SUITE = "weak-worker-orchestration-v1"


@dataclass(frozen=True, slots=True)
class BenchmarkFixtureV1:
    id: str
    task_class: str
    objective: str
    acceptance_criteria: tuple[str, ...]
    initial_files: Mapping[str, str]
    hidden_checks: tuple[str, ...]
    tools: tuple[str, ...] = (
        "read_file",
        "list_files",
        "grep",
        "write_file",
        "edit_file",
        "apply_patch",
        "run_command",
    )
    deterministic_security_gate: bool = False
    context_limit_chars: int = 32_000
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.id or self.task_class not in TASK_CLASSES:
            raise DomainError("benchmark fixture requires a known task class and v1 id")
        if not self.objective.strip() or not self.acceptance_criteria:
            raise DomainError("benchmark fixture requires an objective and acceptance criteria")
        if not self.initial_files or not self.hidden_checks:
            raise DomainError("benchmark fixture requires isolated files and hidden checks")
        if self.context_limit_chars < 1_000:
            raise DomainError("benchmark context limit is too small")
        object.__setattr__(self, "initial_files", dict(self.initial_files))

    @property
    def fingerprint(self) -> str:
        material = "\0".join(
            (
                self.id,
                self.task_class,
                self.objective,
                *self.acceptance_criteria,
                *(f"{path}:{content}" for path, content in sorted(self.initial_files.items())),
                *self.hidden_checks,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def materialize(self, workspace: Path) -> None:
        root = workspace.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        for relative, content in self.initial_files.items():
            target = (root / relative).resolve(strict=False)
            if not target.is_relative_to(root):
                raise DomainError(f"fixture path escapes workspace: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


def _fixture_files(task_class: str, number: int) -> Mapping[str, str]:
    marker = f"{task_class}-{number:02d}"
    if task_class == "browser_tool_use":
        return {
            "index.html": (
                "<!doctype html><html><body>"
                f"<button id='run'>Run {marker}</button><output id='status'>idle</output>"
                "<script>document.querySelector('#run').onclick=()=>"
                "document.querySelector('#status').textContent='ready'</script>"
                "</body></html>\n"
            )
        }
    if task_class == "multi_file_integration":
        return {
            "app.py": "from service import compute\n\ndef run(value):\n    return compute(value)\n",
            "service.py": f"def compute(value):\n    return value + {number}\n",
        }
    if task_class == "security_edge_case":
        return {
            "security.py": (
                "from pathlib import Path\n\n"
                "def safe_join(root, user_path):\n"
                "    return Path(root) / user_path\n"
            )
        }
    if task_class == "bug_repair":
        return {
            "calculator.py": (
                "def bounded_average(values):\n"
                "    if not values:\n"
                "        return 0\n"
                "    return sum(values) / (len(values) - 1)\n"
            )
        }
    if task_class == "requirement_fidelity":
        return {
            "formatter.py": (
                "def render(name, count):\n"
                "    return f'{name}: {count}'\n"
            ),
            "REQUIREMENTS.md": f"fixture {marker}: preserve API and reject negative counts\n",
        }
    return {
        "settings.py": f"RETRY_LIMIT = {number}\nMODE = 'legacy'\n",
        "README.md": f"Fixture {marker}; preserve every unrelated byte.\n",
    }


def default_fixtures() -> tuple[BenchmarkFixtureV1, ...]:
    """Return exactly 24 isolated fixtures, four for each accepted task class."""

    objectives = {
        "exact_edit": "Apply one exact requested edit and preserve unrelated content.",
        "bug_repair": "Repair the boundary defect without weakening the public behavior.",
        "multi_file_integration": "Integrate the existing modules and preserve their interface.",
        "browser_tool_use": "Make the interaction satisfy the browser assertion with no console errors.",
        "security_edge_case": "Reject traversal and boundary attacks while preserving valid inputs.",
        "requirement_fidelity": "Implement every stated requirement without adding an unrequested behavior.",
    }
    fixtures: list[BenchmarkFixtureV1] = []
    for task_class in TASK_CLASSES:
        for number in range(1, 5):
            fixture_id = f"{task_class}-{number:02d}"
            checks = (
                f"hidden:{fixture_id}:acceptance",
                f"hidden:{fixture_id}:regression",
                *(
                    (f"hidden:{fixture_id}:security",)
                    if task_class == "security_edge_case"
                    else ()
                ),
                *(
                    (f"hidden:{fixture_id}:browser",)
                    if task_class == "browser_tool_use"
                    else ()
                ),
            )
            fixtures.append(
                BenchmarkFixtureV1(
                    id=fixture_id,
                    task_class=task_class,
                    objective=objectives[task_class],
                    acceptance_criteria=(
                        "The requested behavior passes the hidden acceptance check.",
                        "The hidden regression check passes.",
                    ),
                    initial_files=_fixture_files(task_class, number),
                    hidden_checks=checks,
                    deterministic_security_gate=task_class == "security_edge_case",
                    tools=(
                        *BenchmarkFixtureV1.__dataclass_fields__["tools"].default,
                        *(("preview_html", "inspect_preview")
                          if task_class == "browser_tool_use"
                          else ()),
                    ),
                )
            )
    if len(fixtures) != 24:
        raise AssertionError("the weak-worker benchmark must contain exactly 24 fixtures")
    return tuple(fixtures)


@dataclass(frozen=True, slots=True)
class TrialRequestV1:
    suite: str
    fixture: BenchmarkFixtureV1
    workspace: Path
    arm: OrchestrationArm
    seed: int
    model_fingerprint: str
    max_model_calls: int
    tools: tuple[str, ...]
    context_limit_chars: int
    independent_sampling: bool = False
    evidence_roles_enabled: bool = False


@dataclass(frozen=True, slots=True)
class TrialObservationV1:
    fixture_id: str
    task_class: str
    arm: OrchestrationArm
    seed: int
    model_fingerprint: str
    success: bool
    claimed_complete: bool
    deterministic_gates_passed: bool
    security_gate_passed: bool = True
    verified_findings: int = 0
    false_findings: int = 0
    latent_defects: int = 0
    found_latent_defects: int = 0
    regressions: int = 0
    no_progress_retries: int = 0
    worker_calls: int = 0
    useful_workers: int = 0
    duplicate_findings: int = 0
    total_findings: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    gpu_ms: int = 0
    evidence_refs: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", OrchestrationArm(self.arm))
        object.__setattr__(self, "evidence_refs", tuple(dict.fromkeys(self.evidence_refs)))
        object.__setattr__(self, "metrics", dict(self.metrics))
        counters = (
            self.verified_findings,
            self.false_findings,
            self.latent_defects,
            self.found_latent_defects,
            self.regressions,
            self.no_progress_retries,
            self.worker_calls,
            self.useful_workers,
            self.duplicate_findings,
            self.total_findings,
            self.model_calls,
            self.input_tokens,
            self.output_tokens,
            self.latency_ms,
            self.gpu_ms,
        )
        if min(counters) < 0:
            raise DomainError("benchmark observation counters cannot be negative")
        if self.useful_workers > self.worker_calls:
            raise DomainError("useful worker count exceeds worker calls")
        if self.found_latent_defects > self.latent_defects:
            raise DomainError("found latent defects exceed available defects")
        if self.duplicate_findings > self.total_findings:
            raise DomainError("duplicate findings exceed total findings")

    @property
    def false_completion(self) -> bool:
        return self.claimed_complete and not self.success

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TrialExecutor(Protocol):
    def __call__(self, request: TrialRequestV1) -> TrialObservationV1: ...


@dataclass(frozen=True, slots=True)
class ArmMetricsV1:
    arm: OrchestrationArm
    trials: int
    successes: int
    success_rate: float
    success_interval_95: tuple[float, float]
    false_completions: int
    false_completion_rate: float
    defect_precision: float
    defect_recall: float
    regression_rate: float
    no_progress_retry_rate: float
    worker_contribution_rate: float
    duplicate_finding_ratio: float
    mean_tokens: float
    mean_latency_ms: float
    mean_gpu_ms: float
    median_calls: float
    p95_calls: float
    deterministic_gate_regressions: int


@dataclass(frozen=True, slots=True)
class ActivationGateV1:
    passed: bool
    checks: Mapping[str, bool]
    measured: Mapping[str, float | int | str | tuple[float, float]]
    reasons: tuple[str, ...]
    hypothesis: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkReportV1:
    suite: str
    model_fingerprint: str
    fixtures: int
    seeds: tuple[int, ...]
    observations: tuple[TrialObservationV1, ...]
    arms: Mapping[str, ArmMetricsV1]
    activation: ActivationGateV1
    matched: bool


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = (proportion + z * z / (2.0 * trials)) / denominator
    spread = z * math.sqrt(
        (proportion * (1.0 - proportion) / trials) + z * z / (4.0 * trials * trials)
    ) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[rank])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_arm(
    arm: OrchestrationArm,
    observations: Sequence[TrialObservationV1],
) -> ArmMetricsV1:
    items = tuple(item for item in observations if item.arm is arm)
    successes = sum(item.success for item in items)
    findings = sum(item.verified_findings + item.false_findings for item in items)
    latent = sum(item.latent_defects for item in items)
    workers = sum(item.worker_calls for item in items)
    total_findings = sum(item.total_findings for item in items)
    return ArmMetricsV1(
        arm=arm,
        trials=len(items),
        successes=successes,
        success_rate=_ratio(successes, len(items)),
        success_interval_95=wilson_interval(successes, len(items)),
        false_completions=sum(item.false_completion for item in items),
        false_completion_rate=_ratio(sum(item.false_completion for item in items), len(items)),
        defect_precision=_ratio(sum(item.verified_findings for item in items), findings),
        defect_recall=_ratio(sum(item.found_latent_defects for item in items), latent),
        regression_rate=_ratio(sum(item.regressions for item in items), len(items)),
        no_progress_retry_rate=_ratio(sum(item.no_progress_retries for item in items), len(items)),
        worker_contribution_rate=_ratio(sum(item.useful_workers for item in items), workers),
        duplicate_finding_ratio=_ratio(sum(item.duplicate_findings for item in items), total_findings),
        mean_tokens=_ratio(sum(item.total_tokens for item in items), len(items)),
        mean_latency_ms=_ratio(sum(item.latency_ms for item in items), len(items)),
        mean_gpu_ms=_ratio(sum(item.gpu_ms for item in items), len(items)),
        median_calls=float(statistics.median(item.model_calls for item in items)) if items else 0.0,
        p95_calls=_percentile([item.model_calls for item in items], 0.95),
        deterministic_gate_regressions=sum(
            (not item.deterministic_gates_passed) or (not item.security_gate_passed)
            for item in items
        ),
    )


def paired_bootstrap_delta_interval(
    baseline: Sequence[TrialObservationV1],
    treatment: Sequence[TrialObservationV1],
    *,
    samples: int = 10_000,
    seed: int = 20260811,
) -> tuple[float, float]:
    baseline_map = {(item.fixture_id, item.seed): item for item in baseline}
    treatment_map = {(item.fixture_id, item.seed): item for item in treatment}
    keys = sorted(set(baseline_map) & set(treatment_map))
    if not keys:
        return (0.0, 0.0)
    effects = [
        int(treatment_map[key].success) - int(baseline_map[key].success)
        for key in keys
    ]
    rng = random.Random(seed)
    deltas = []
    for _ in range(max(1_000, samples)):
        deltas.append(
            sum(effects[rng.randrange(len(effects))] for _ in effects) / len(effects)
        )
    deltas.sort()
    return (
        deltas[int(0.025 * (len(deltas) - 1))],
        deltas[int(0.975 * (len(deltas) - 1))],
    )


def evaluate_activation_gate(
    observations: Sequence[TrialObservationV1],
) -> tuple[Mapping[str, ArmMetricsV1], ActivationGateV1]:
    arms = {
        arm.value: summarize_arm(arm, observations)
        for arm in (
            OrchestrationArm.SINGLE_AGENT,
            OrchestrationArm.COMPUTE_MATCHED,
            OrchestrationArm.CURRENT,
            OrchestrationArm.ADAPTIVE,
        )
    }
    single = arms[OrchestrationArm.SINGLE_AGENT.value]
    compute = arms[OrchestrationArm.COMPUTE_MATCHED.value]
    adaptive = arms[OrchestrationArm.ADAPTIVE.value]
    relative_single = _ratio(
        adaptive.success_rate - single.success_rate,
        single.success_rate,
    )
    relative_compute = _ratio(
        adaptive.success_rate - compute.success_rate,
        compute.success_rate,
    )
    token_reduction = _ratio(compute.mean_tokens - adaptive.mean_tokens, compute.mean_tokens)
    false_completion_reduction = _ratio(
        single.false_completion_rate - adaptive.false_completion_rate,
        single.false_completion_rate,
    )
    single_trials = [
        item for item in observations if item.arm is OrchestrationArm.SINGLE_AGENT
    ]
    adaptive_trials = [
        item for item in observations if item.arm is OrchestrationArm.ADAPTIVE
    ]
    delta_interval = paired_bootstrap_delta_interval(single_trials, adaptive_trials)
    expected_trials = 24 * len(DEFAULT_SEEDS)
    complete_matched = all(item.trials == expected_trials for item in arms.values())
    checks = {
        "complete_matched_24x3": complete_matched,
        "single_agent_relative_uplift_ge_20pct": relative_single >= 0.20,
        "paired_95pct_interval_excludes_zero": delta_interval[0] > 0.0,
        "compute_matched_gain_or_token_reduction": (
            relative_compute >= 0.05
            or (
                adaptive.success_rate >= compute.success_rate
                and token_reduction >= 0.25
            )
        ),
        "false_completion_reduction_ge_40pct": false_completion_reduction >= 0.40,
        "verified_worker_contribution_ge_50pct": adaptive.worker_contribution_rate >= 0.50,
        "median_calls_le_3x": adaptive.median_calls <= 3.0,
        "p95_calls_le_4x": adaptive.p95_calls <= 4.0,
        "no_deterministic_gate_regression": adaptive.deterministic_gate_regressions == 0,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return arms, ActivationGateV1(
        passed=all(checks.values()),
        checks=checks,
        measured={
            "relative_uplift_vs_single": relative_single,
            "paired_success_delta_interval_95": delta_interval,
            "relative_uplift_vs_compute_matched": relative_compute,
            "token_reduction_vs_compute_matched": token_reduction,
            "false_completion_reduction": false_completion_reduction,
            "worker_contribution_rate": adaptive.worker_contribution_rate,
            "median_calls": adaptive.median_calls,
            "p95_calls": adaptive.p95_calls,
        },
        reasons=reasons,
        hypothesis=not complete_matched,
    )


class FourArmBenchmarkRunner:
    def __init__(
        self,
        executor: TrialExecutor,
        *,
        model_fingerprint: str,
        fixtures: Iterable[BenchmarkFixtureV1] | None = None,
        seeds: Iterable[int] = DEFAULT_SEEDS,
        policy: AdaptiveOrchestrationPolicyV1 | None = None,
        store: Any | None = None,
        goal_id: str | None = None,
    ) -> None:
        self.executor = executor
        self.model_fingerprint = str(model_fingerprint).strip()
        self.fixtures = tuple(fixtures or default_fixtures())
        self.seeds = tuple(int(item) for item in seeds)
        self.policy = policy or AdaptiveOrchestrationPolicyV1()
        self.store = store
        self.goal_id = goal_id
        if not self.model_fingerprint or not self.seeds:
            raise DomainError("benchmark requires one fixed model fingerprint and seeds")
        ids = [item.id for item in self.fixtures]
        if len(ids) != len(set(ids)):
            raise DomainError("benchmark fixture ids must be unique")
        if (self.store is None) != (self.goal_id is None):
            raise DomainError("benchmark persistence requires both store and goal id")

    def _execute(
        self,
        fixture: BenchmarkFixtureV1,
        arm: OrchestrationArm,
        seed: int,
        max_calls: int,
        root: Path,
    ) -> TrialObservationV1:
        with tempfile.TemporaryDirectory(
            prefix=f"{fixture.id}-{arm.value}-",
            dir=root,
            ignore_cleanup_errors=True,
        ) as directory:
            workspace = Path(directory)
            fixture.materialize(workspace)
            result = self.executor(
                TrialRequestV1(
                    suite=BENCHMARK_SUITE,
                    fixture=fixture,
                    workspace=workspace,
                    arm=arm,
                    seed=seed,
                    model_fingerprint=self.model_fingerprint,
                    max_model_calls=max_calls,
                    tools=fixture.tools,
                    context_limit_chars=fixture.context_limit_chars,
                    independent_sampling=arm is OrchestrationArm.COMPUTE_MATCHED,
                    evidence_roles_enabled=arm is OrchestrationArm.ADAPTIVE,
                )
            )
        if result.fixture_id != fixture.id or result.task_class != fixture.task_class:
            raise DomainError("executor returned an observation for the wrong fixture")
        if result.arm is not arm or result.seed != seed:
            raise DomainError("executor changed the matched arm or seed")
        if result.model_fingerprint != self.model_fingerprint:
            raise DomainError("mixed-model observations are forbidden inside one benchmark")
        if result.model_calls > max_calls:
            raise DomainError("executor exceeded the matched model-call budget")
        return result

    def _persist(self, item: TrialObservationV1) -> None:
        if self.store is None or self.goal_id is None:
            return
        self.store.record_orchestration_experiment(
            OrchestrationExperimentV1(
                goal_id=self.goal_id,
                unit_id=f"{item.fixture_id}:seed-{item.seed}",
                arm=item.arm,
                model_fingerprint=self.model_fingerprint,
                policy_fingerprint=self.policy.fingerprint,
                task_class=item.task_class,
                candidate_score=1.0 if item.success else 0.0,
                success=item.success,
                false_completion=item.false_completion,
                model_calls=item.model_calls,
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                latency_ms=item.latency_ms,
                gpu_ms=item.gpu_ms,
                causal=True,
                matched_benchmark=True,
                metrics={
                    **dict(item.metrics),
                    "suite": BENCHMARK_SUITE,
                    "seed": item.seed,
                    "fixture_id": item.fixture_id,
                    "verified_findings": item.verified_findings,
                    "false_findings": item.false_findings,
                    "regressions": item.regressions,
                    "worker_calls": item.worker_calls,
                    "useful_workers": item.useful_workers,
                },
                evidence=item.evidence_refs,
            )
        )

    def run(self, workspace_root: Path) -> BenchmarkReportV1:
        root = workspace_root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        observations: list[TrialObservationV1] = []
        for fixture in self.fixtures:
            for seed in self.seeds:
                single = self._execute(
                    fixture,
                    OrchestrationArm.SINGLE_AGENT,
                    seed,
                    1,
                    root,
                )
                current = self._execute(
                    fixture,
                    OrchestrationArm.CURRENT,
                    seed,
                    self.policy.max_model_call_multiplier,
                    root,
                )
                adaptive = self._execute(
                    fixture,
                    OrchestrationArm.ADAPTIVE,
                    seed,
                    self.policy.max_model_call_multiplier,
                    root,
                )
                compute = self._execute(
                    fixture,
                    OrchestrationArm.COMPUTE_MATCHED,
                    seed,
                    max(1, adaptive.model_calls),
                    root,
                )
                if compute.model_calls != adaptive.model_calls:
                    raise DomainError(
                        "compute-matched arm must use exactly the adaptive arm call count"
                    )
                matched = (single, compute, current, adaptive)
                observations.extend(matched)
                for item in matched:
                    self._persist(item)
        arms, activation = evaluate_activation_gate(observations)
        expected = len(self.fixtures) * len(self.seeds) * 4
        return BenchmarkReportV1(
            suite=BENCHMARK_SUITE,
            model_fingerprint=self.model_fingerprint,
            fixtures=len(self.fixtures),
            seeds=self.seeds,
            observations=tuple(observations),
            arms=arms,
            activation=activation,
            matched=len(observations) == expected,
        )

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

import pytest

from agent.models import DomainError
from agent.orchestration import (
    AdaptiveOrchestrationPolicyV1,
    AdaptiveWorkerRouter,
    EvidenceAuthority,
    EvidenceClaimV1,
    EvidenceVerdict,
    FIXED_SPECIALIST_REVIEW_ORDER,
    FixedSpecialistReviewGateV1,
    MutationPolicy,
    OrchestrationArm,
    OrchestrationExperimentV1,
    RiskTier,
    SpecialistReviewResultV1,
    SpecialistReviewRole,
    TaskRiskSignalsV1,
    WorkerImpactOutcome,
    WorkerImpactV1,
    WorkerMissionV2,
    WorkerRole,
    WorkerVisibility,
    evidence_decision,
)
from agent.store import SCHEMA_VERSION, StateStore, StateStoreError
from agent.ultra import (
    AgentResponse,
    QualityGateResultV1,
    TaskContractV1,
    UltraOrchestrator,
    WorkNode,
)
from agent.worker_benchmark import (
    DEFAULT_SEEDS,
    FourArmBenchmarkRunner,
    TrialObservationV1,
    default_fixtures,
    evaluate_activation_gate,
)


def test_adaptive_router_uses_observable_risk_and_four_call_cap() -> None:
    router = AdaptiveWorkerRouter()

    low = router.route(TaskRiskSignalsV1(declared_risk="low"))
    medium = router.route(TaskRiskSignalsV1(declared_risk="medium"))
    high = router.route(
        TaskRiskSignalsV1(
            declared_risk="high",
            security_sensitive=True,
            missing_evidence_count=2,
        )
    )
    ambiguous = router.route(
        TaskRiskSignalsV1(declared_risk="medium", ambiguous_design=True)
    )

    assert low.tier is RiskTier.LOW and low.roles == () and low.max_model_calls == 1
    assert medium.roles == (WorkerRole.FALSIFIER,) and medium.max_model_calls == 2
    assert high.roles == (
        WorkerRole.PREDICTOR,
        WorkerRole.FALSIFIER,
        WorkerRole.REPAIRER,
    )
    assert ambiguous.roles == (
        WorkerRole.CHALLENGER,
        WorkerRole.SELECTOR,
        WorkerRole.REPAIRER,
    )
    assert high.max_model_calls == ambiguous.max_model_calls == 4


def test_worker_mission_enforces_blind_read_only_and_repair_context() -> None:
    falsifier = WorkerMissionV2(
        role=WorkerRole.FALSIFIER,
        objective="Try to disprove the current acceptance claim.",
        success_criteria=("The hidden runtime check passes.",),
        visibility=WorkerVisibility.ARTIFACT_WITHOUT_RATIONALE,
        mutation_policy=MutationPolicy.READ_ONLY,
        allowed_tools=("read_file", "run_command"),
        falsification_targets=("Exercise the empty input boundary.",),
    )
    assert falsifier.approach_fingerprint == falsifier.approach_fingerprint

    with pytest.raises(DomainError):
        replace(falsifier, mutation_policy=MutationPolicy.SINGLE_WRITER)
    with pytest.raises(DomainError):
        WorkerMissionV2(
            role=WorkerRole.REPAIRER,
            objective="Repair the observed defect.",
            success_criteria=("The failed check now passes.",),
            visibility=WorkerVisibility.FULL_SCOPED_CONTEXT,
            mutation_policy=MutationPolicy.SINGLE_WRITER,
        )


def test_fixed_specialist_gate_requires_all_six_fresh_reviews() -> None:
    reviews = tuple(
        SpecialistReviewResultV1(
            role=role,
            artifact_hash="a" * 64,
            mutation_sequence=3,
            passed=True,
            summary=f"{role.value} passed with current evidence.",
            evidence_refs=(f"evidence:{role.value}",),
            reviewer_id=f"reviewer:{role.value}",
        )
        for role in FIXED_SPECIALIST_REVIEW_ORDER
    )
    gate = FixedSpecialistReviewGateV1(
        artifact_hash="a" * 64,
        mutation_sequence=3,
        reviews=reviews,
    )

    assert tuple(item.role for item in gate.reviews) == FIXED_SPECIALIST_REVIEW_ORDER
    assert gate.verdict is EvidenceVerdict.PASSED
    assert gate.missing_roles == gate.stale_roles == gate.failed_roles == ()


def test_fixed_specialist_gate_fails_on_blocker_and_rejects_stale_passes() -> None:
    failed = SpecialistReviewResultV1(
        role=SpecialistReviewRole.SECURITY,
        artifact_hash="b" * 64,
        mutation_sequence=4,
        passed=False,
        summary="Security found an unvalidated trust boundary.",
        issues=({"title": "Validate the boundary"},),
    )
    current = tuple(
        SpecialistReviewResultV1(
            role=role,
            artifact_hash="b" * 64,
            mutation_sequence=4,
            passed=True,
            summary=f"{role.value} passed.",
        )
        for role in FIXED_SPECIALIST_REVIEW_ORDER
        if role is not SpecialistReviewRole.SECURITY
    )
    gate = FixedSpecialistReviewGateV1(
        artifact_hash="b" * 64,
        mutation_sequence=4,
        reviews=(failed, *current),
    )
    stale_gate = replace(
        gate,
        mutation_sequence=5,
    )

    assert gate.verdict is EvidenceVerdict.FAILED
    assert gate.failed_roles == (SpecialistReviewRole.SECURITY,)
    assert stale_gate.verdict is EvidenceVerdict.NEEDS_EVIDENCE
    assert set(stale_gate.stale_roles) == set(FIXED_SPECIALIST_REVIEW_ORDER)


def test_deterministic_failed_claim_vetoes_any_model_majority() -> None:
    model_pass = EvidenceClaimV1(
        criterion_id="runtime",
        claim="The model believes runtime behavior is correct.",
        artifact_hash="b" * 64,
        evidence_refs=("review:one",),
        falsification_check="Execute the runtime test.",
        verdict=EvidenceVerdict.PASSED,
        authority=EvidenceAuthority.MODEL,
        producer_id="reviewer-1",
        verifier_id="reviewer-2",
    )
    hard_failure = EvidenceClaimV1(
        criterion_id="runtime",
        claim="The executable runtime gate failed.",
        artifact_hash="b" * 64,
        evidence_refs=("action:test-1",),
        falsification_check="Re-run action test-1.",
        verdict=EvidenceVerdict.FAILED,
        authority=EvidenceAuthority.TEST,
        producer_id="tester",
        verifier_id="harness",
    )

    assert evidence_decision((model_pass, model_pass, hard_failure)) is EvidenceVerdict.FAILED
    assert hard_failure.hard_veto is True


def test_schema_v17_records_observations_and_rejects_duplicate_approaches() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        try:
            goal = store.create_goal("Measure weak-model worker impact")
            experiment = OrchestrationExperimentV1(
                goal_id=goal.id,
                unit_id="TASK-1",
                arm=OrchestrationArm.PRODUCTION,
                model_fingerprint="model-fingerprint",
                policy_fingerprint="policy-fingerprint",
                task_class="bug_repair",
            )
            store.record_orchestration_experiment(experiment)
            impact = WorkerImpactV1(
                experiment_id=experiment.id,
                worker_id="worker-1",
                role=WorkerRole.FALSIFIER,
                task_class="bug_repair",
                model_fingerprint="model-fingerprint",
                outcome=WorkerImpactOutcome.NEUTRAL,
                approach_fingerprint="same-approach",
                reason="No verified new evidence.",
            )
            store.record_worker_contribution(impact)

            with pytest.raises(StateStoreError):
                store.record_worker_contribution(
                    replace(impact, id="impact-2", worker_id="worker-2")
                )

            assert store.list_orchestration_experiments(
                goal_id=goal.id,
                arm=OrchestrationArm.PRODUCTION,
            )[0]["causal"] == 0
            assert store.list_worker_contributions(goal_id=goal.id)[0]["outcome"] == "neutral"
            connection = sqlite3.connect(store.path)
            try:
                assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 17
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                connection.close()
            assert {"orchestration_experiments", "worker_contributions"} <= tables
        finally:
            store.close()


def test_role_learning_suppresses_only_after_enough_nonpositive_samples() -> None:
    policy = AdaptiveOrchestrationPolicyV1(minimum_learning_samples=2)
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(Path(directory))
        try:
            goal = store.create_goal("Learn role utility")
            for index in range(2):
                experiment = OrchestrationExperimentV1(
                    goal_id=goal.id,
                    unit_id=f"task-{index}",
                    arm=OrchestrationArm.PRODUCTION,
                    model_fingerprint="weak-model",
                    policy_fingerprint=policy.fingerprint,
                    task_class="exact_edit",
                )
                store.record_orchestration_experiment(experiment)
                store.record_worker_contribution(
                    WorkerImpactV1(
                        experiment_id=experiment.id,
                        worker_id=f"worker-{index}",
                        role=WorkerRole.REVIEWER,
                        task_class="exact_edit",
                        model_fingerprint="weak-model",
                        outcome=WorkerImpactOutcome.NEUTRAL,
                        approach_fingerprint=f"approach-{index}",
                    )
                )
            utility = store.worker_role_utility(
                model_fingerprint="weak-model",
                task_class="exact_edit",
                role=WorkerRole.REVIEWER,
                policy=policy,
            )
            assert utility["samples"] == 2
            assert utility["useful_rate"] == 0.0
            assert utility["mean_score_delta"] == 0.0
            assert utility["suppressed"] is True
        finally:
            store.close()


def _observation(
    arm: OrchestrationArm,
    index: int,
) -> TrialObservationV1:
    successes = {
        OrchestrationArm.SINGLE_AGENT: 30,
        OrchestrationArm.COMPUTE_MATCHED: 38,
        OrchestrationArm.CURRENT: 34,
        OrchestrationArm.ADAPTIVE: 48,
    }
    success = index < successes[arm]
    claimed_complete = success or (
        index < (50 if arm is OrchestrationArm.SINGLE_AGENT else 53)
        if arm is OrchestrationArm.ADAPTIVE
        else index < 50
    )
    calls = 1 if arm is OrchestrationArm.SINGLE_AGENT else 3
    return TrialObservationV1(
        fixture_id=f"fixture-{index // len(DEFAULT_SEEDS):02d}",
        task_class="bug_repair",
        arm=arm,
        seed=DEFAULT_SEEDS[index % len(DEFAULT_SEEDS)],
        model_fingerprint="weak-model",
        success=success,
        claimed_complete=claimed_complete,
        deterministic_gates_passed=True,
        verified_findings=1 if arm is OrchestrationArm.ADAPTIVE else 0,
        latent_defects=1,
        found_latent_defects=1 if arm is OrchestrationArm.ADAPTIVE else 0,
        worker_calls=2 if arm is OrchestrationArm.ADAPTIVE else 0,
        useful_workers=1 if arm is OrchestrationArm.ADAPTIVE else 0,
        model_calls=calls,
        input_tokens=300 * calls,
        output_tokens=100 * calls,
    )


def test_activation_gate_uses_matched_uplift_false_completion_and_cost() -> None:
    observations = tuple(
        _observation(arm, index)
        for arm in (
            OrchestrationArm.SINGLE_AGENT,
            OrchestrationArm.COMPUTE_MATCHED,
            OrchestrationArm.CURRENT,
            OrchestrationArm.ADAPTIVE,
        )
        for index in range(24 * len(DEFAULT_SEEDS))
    )

    arms, gate = evaluate_activation_gate(observations)

    assert len(default_fixtures()) == 24
    assert gate.hypothesis is False
    assert gate.passed is True
    assert gate.measured["relative_uplift_vs_single"] >= 0.20
    assert gate.measured["false_completion_reduction"] >= 0.40
    assert arms[OrchestrationArm.ADAPTIVE.value].worker_contribution_rate == 0.5


def test_four_arm_runner_matches_compute_calls_to_adaptive() -> None:
    fixture = default_fixtures()[0]

    def executor(request):
        calls = 1 if request.arm is OrchestrationArm.SINGLE_AGENT else 2
        return TrialObservationV1(
            fixture_id=request.fixture.id,
            task_class=request.fixture.task_class,
            arm=request.arm,
            seed=request.seed,
            model_fingerprint=request.model_fingerprint,
            success=True,
            claimed_complete=True,
            deterministic_gates_passed=True,
            model_calls=min(calls, request.max_model_calls),
        )

    runner = FourArmBenchmarkRunner(
        executor,
        model_fingerprint="weak-model",
        fixtures=(fixture,),
        seeds=(7,),
    )
    with tempfile.TemporaryDirectory() as directory:
        report = runner.run(Path(directory))

    assert report.matched is True
    assert len(report.observations) == 4
    adaptive = next(
        item for item in report.observations if item.arm is OrchestrationArm.ADAPTIVE
    )
    compute = next(
        item for item in report.observations if item.arm is OrchestrationArm.COMPUTE_MATCHED
    )
    assert adaptive.model_calls == compute.model_calls == 2
    assert report.activation.hypothesis is True


def test_ultra_review_routing_and_consensus_are_evidence_aware() -> None:
    docs = WorkNode(
        TaskContractV1(
            id="DOCS",
            title="Documentation",
            objective="Update prose only.",
            acceptance_criteria=("Documentation is current.",),
            verification=("Inspect the rendered document.",),
            write_paths=("README.md",),
        )
    )
    assert UltraOrchestrator._reviewer_applicability(docs, "security")[0] is False
    assert UltraOrchestrator._reviewer_applicability(docs, "test_quality")[0] is False
    assert UltraOrchestrator._reviewer_applicability(docs, "clean_code")[0] is False

    security = replace(
        docs,
        contract=replace(
            docs.contract,
            id="AUTH",
            title="Authentication",
            write_paths=("src/auth/session.py",),
            owned_interfaces=("POST /session",),
        ),
    )
    assert UltraOrchestrator._reviewer_applicability(security, "security")[0] is True

    engine = object.__new__(UltraOrchestrator)
    engine.state = SimpleNamespace(adaptive_orchestration_shadow_mode=False)
    records = engine._quality_vote_records(
        security,
        (
            AgentResponse(payload={"passed": True}, summary="No evidence."),
            AgentResponse(
                payload={
                    "passed": True,
                    "test_results": [
                        {"name": "harness_security", "passed": True}
                    ],
                },
                summary="Harness security evidence passed.",
            ),
            AgentResponse(payload={"passed": True}, summary="No test evidence."),
            UltraOrchestrator._not_applicable_review("test_quality", ("not changed",)),
            AgentResponse(
                payload={"passed": True},
                provider="harness",
                model="deterministic-quality-triage-v1",
            ),
        ),
    )
    assert [item["verdict"] for item in records] == ["abstain", "accept", "abstain"]
    assert all(item["role"] != "triage" for item in records)

    hard_failure = AgentResponse(
        payload={
            "passed": False,
            "test_results": [
                {
                    "name": "harness_runtime",
                    "source": "harness",
                    "passed": False,
                }
            ],
        },
        provider="harness",
        model="deterministic-runtime-v1",
    )
    gate = QualityGateResultV1(
        responses=(hard_failure,),
        consensus={"status": "accepted"},
    )
    assert engine._quality_gate_passed(gate) is False

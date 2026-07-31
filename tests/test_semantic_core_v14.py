from __future__ import annotations

import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from agent.intake import IntentArchitect, RunMode
from agent.models import GoalStatus, PlanStatus
from agent.runtime import AgentRuntime
from agent.config import RuntimeConfig
from agent.semantic import RequestedEffect, SemanticGoalV2
from agent.store import StateStore
from agent.testing import ScriptedProvider
from agent.verifiers import discover_verifier_plugins
from agent.ultra import UltraOrchestrator


def inspection() -> dict:
    return {
        "tool_calls": [
            {"id": "inspect", "name": "list_files", "args": {"path": "."}}
        ]
    }


def semantic_plan(request: str, *, mutate: bool = True, risk: str = "medium") -> dict:
    criterion = "The classifier behavior matches the exact request and focused tests pass."
    effects = ["read_workspace", "execute_code"]
    if mutate:
        effects.append("mutate_workspace")
    else:
        effects.append("answer")
    return {
        "tool_calls": [
            {
                "id": "plan",
                "name": "propose_plan",
                "args": {
                    "semantic_goal": {
                        "original_request": request,
                        "interpreted_outcome": (
                            "Repair the existing classifier implementation."
                            if mutate
                            else "Analyze the existing classifier without changing files."
                        ),
                        "requested_effects": effects,
                        "required_outcomes": [
                            "Correctly classify meta-level and negated requests."
                        ],
                        "constraints": ["Preserve unrelated behavior."],
                        "exclusions": [
                            "Do not build a game.",
                            "The mentioned game is test data, not a deliverable.",
                        ],
                        "acceptance_criteria": [criterion],
                        "unresolved_decisions": [],
                        "repository_evidence_refs": ["inspection:I001"],
                    },
                    "summary": "Repair and verify the inspected classifier.",
                    "applicability_evidence": [
                        {
                            "fact": "The inspected repository contains the classifier.",
                            "source": "inspection:I001",
                            "supports_tasks": ["T001"],
                        }
                    ],
                    "execution_strategy": (
                        "Inspect the classifier symbol, apply a focused edit, and run its tests."
                        if mutate
                        else "Read the classifier and tests, then report evidence without mutation."
                    ),
                    "expected_changes": (
                        [
                            {
                                "path": "agent/classifier.py",
                                "intent": "Repair the inspected classifier.",
                                "basis": "existing_inspected_path",
                                "evidence_refs": ["inspection:I001"],
                                "supports_tasks": ["T001"],
                            }
                        ]
                        if mutate
                        else []
                    ),
                    "tasks": [
                        {
                            "title": "Audit classifier semantics",
                            "description": (
                                "Repair and verify the inspected classifier."
                                if mutate
                                else "Analyze and report classifier behavior."
                            ),
                            "acceptance_criteria": [criterion],
                            "verification": [
                                "Run the focused classifier regression tests."
                            ],
                            "depends_on": [],
                            "risk": risk,
                        }
                    ],
                },
            }
        ]
    }


def critic_pass() -> dict:
    return {
        "tool_calls": [
            {
                "id": "critic",
                "name": "submit_plan_review",
                "args": {
                    "verdict": "pass",
                    "summary": "The semantics preserve meta-level and negated intent.",
                    "issues": [],
                },
            }
        ]
    }


class SemanticCoreV14Tests(unittest.TestCase):
    def test_unified_semantic_modules_contain_no_domain_topology_defaults(self) -> None:
        package = Path(__file__).resolve().parents[1] / "agent"
        sources = "\n".join(
            (package / name).read_text(encoding="utf-8").casefold()
            for name in (
                "semantic.py",
                "control.py",
                "prompts.py",
                "workflow.py",
                "intake.py",
                "verifiers.py",
            )
        )
        for forbidden in (
            "index.html",
            "three.js",
            "traffic",
            "castle",
            "characterpackage",
            "gameplaypackage",
        ):
            self.assertNotIn(forbidden, sources)

    def test_intake_preserves_meta_and_negated_requests_without_deliverables(self) -> None:
        architect = IntentArchitect()
        examples = (
            "Fix the game classifier; do not build a game.",
            "صلّح مصنف الألعاب، لا تبني لعبة.",
        )
        for request in examples:
            decision = architect.analyze(request, requested_mode=RunMode.ULTRA)
            self.assertEqual(decision.brief.original_input, request)
            self.assertEqual(decision.brief.objective, request)
            self.assertEqual(decision.brief.deliverables, ())
            self.assertEqual(decision.questions, ())
            self.assertIs(decision.brief.routed_mode, RunMode.ULTRA)

    def test_common_action_verbs_do_not_select_a_domain_or_path(self) -> None:
        examples = (
            "implement the requested change",
            "add the missing behavior",
            "update and refactor the parser",
            "rename the public command",
            "build and generate the requested output",
            "analyze and test the repository",
            "نفّذ التغيير المطلوب",
            "أضف وحدّث وأعد هيكلة المحلل",
            "غيّر الاسم وابنِ واختبر المشروع",
            "حلّل المستودع فقط",
        )
        architect = IntentArchitect()
        for request in examples:
            decision = architect.analyze(request)
            self.assertEqual(decision.brief.objective, request)
            self.assertEqual(decision.brief.deliverables, ())
            self.assertEqual(decision.complexity.component_count, 1)

    def test_legacy_ultra_helpers_cannot_inject_domain_or_output_paths(self) -> None:
        examples = (
            "fix the game classifier",
            "do not build a game",
            "صلّح مصنف اللعبة ولا تبني لعبة",
            "analyze dashboard naming only",
        )
        for request in examples:
            self.assertEqual(UltraOrchestrator._task_family(request), "general")
            self.assertFalse(
                UltraOrchestrator._requires_visual_artifact(request)
            )
            self.assertEqual(UltraOrchestrator._final_output_paths(request), ())

    def test_all_modes_use_same_repository_grounded_engine(self) -> None:
        request = "Fix the game classifier; do not build a game."
        for mode in (RunMode.NORMAL, RunMode.ULTRA):
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                (workspace / "agent").mkdir()
                (workspace / "agent" / "classifier.py").write_text(
                    "def classify(value): return 'general'\n", encoding="utf-8"
                )
                store = StateStore(workspace)
                try:
                    runtime = AgentRuntime(
                        ScriptedProvider(
                            [inspection(), semantic_plan(request), critic_pass()]
                        ),
                        store,
                        workspace,
                    )
                    plan = runtime.submit_intent(request, requested_mode=mode)
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
                    goal = runtime.active_goal()
                    self.assertEqual(goal.objective, request)
                    semantic = goal.metadata["semantic_goal"]
                    self.assertEqual(semantic["original_request"], request)
                    self.assertIn("Do not build a game.", semantic["exclusions"])
                    self.assertEqual(
                        goal.metadata["execution_policy"]["mode"], mode.value
                    )
                    self.assertNotIn("ultra_run_id", goal.metadata)
                finally:
                    store.close()

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agent").mkdir()
            (workspace / "agent" / "classifier.py").write_text(
                "def classify(value): return 'general'\n", encoding="utf-8"
            )
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider(
                        [inspection(), semantic_plan(request), critic_pass()]
                    ),
                    store,
                    workspace,
                )
                result = runtime.chat(request)
                goal = runtime.active_goal()
                self.assertEqual(result.status, PlanStatus.PENDING_APPROVAL.value)
                self.assertEqual(goal.status, GoalStatus.AWAITING_PLAN_APPROVAL)
                self.assertEqual(goal.objective, request)
                self.assertEqual(
                    goal.metadata["execution_policy"]["entry_surface"], "chat"
                )
                self.assertNotIn("ultra_run_id", goal.metadata)
            finally:
                store.close()

    def test_analysis_only_plan_has_no_fabricated_path(self) -> None:
        request = "Analyze the classifier only; do not change files."
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agent").mkdir()
            (workspace / "agent" / "classifier.py").write_text(
                "def classify(value): return value\n", encoding="utf-8"
            )
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider(
                        [
                            inspection(),
                            semantic_plan(request, mutate=False, risk="low"),
                            critic_pass(),
                        ]
                    ),
                    store,
                    workspace,
                )
                plan = runtime.start_goal(request)
                self.assertEqual(plan.expected_changes, ())
                self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
                plan = runtime.approve_plan(plan.revision)
                self.assertEqual(plan.status, PlanStatus.ACCEPTED)
                self.assertEqual(
                    store.get_goal(plan.goal_id).metadata["resource_claims"], []
                )
            finally:
                store.close()

    def test_direct_ultra_entry_and_mode_changes_never_create_legacy_run(self) -> None:
        request = "Repair the classifier."
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agent").mkdir()
            (workspace / "agent" / "classifier.py").write_text(
                "def classify(value): return value\n", encoding="utf-8"
            )
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider(
                        [inspection(), semantic_plan(request), critic_pass()]
                    ),
                    store,
                    workspace,
                )
                plan = runtime.start_ultra(request)
                goal = runtime.active_goal()
                self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
                self.assertEqual(
                    goal.metadata["execution_policy"]["mode"], RunMode.ULTRA.value
                )
                self.assertNotIn("ultra_run_id", goal.metadata)

                self.assertEqual(runtime.transition_mode("normal"), "normal")
                goal = runtime.active_goal()
                self.assertEqual(
                    goal.metadata["execution_policy"]["mode"], RunMode.NORMAL.value
                )
                self.assertEqual(
                    goal.metadata["execution_policy"]["concurrency"], 1
                )
            finally:
                store.close()

    def test_semantic_goal_rejects_changed_original(self) -> None:
        with self.assertRaises(ValueError):
            SemanticGoalV2.from_mapping(
                {
                    "original_request": "Build a game",
                    "interpreted_outcome": "Build a game",
                    "requested_effects": [RequestedEffect.MUTATE_WORKSPACE.value],
                    "required_outcomes": ["A game"],
                    "acceptance_criteria": ["The game works"],
                    "repository_evidence_refs": ["inspection:I001"],
                },
                original_request="Do not build a game",
            )

    def test_three_invalid_semantic_outputs_create_resumable_capability_checkpoint(self) -> None:
        invalid = {
            "tool_calls": [
                {
                    "id": f"invalid-{index}",
                    "name": "propose_plan",
                    "args": {
                        "summary": "Missing semantic contract",
                        "tasks": [{}],
                    },
                }
                for index in range(3)
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider([inspection(), invalid]),
                    store,
                    workspace,
                )
                self.assertIsNone(runtime.start_goal("Inspect and fix the parser."))
                goal = runtime.active_goal()
                self.assertEqual(goal.status, GoalStatus.PAUSED)
                checkpoints = [
                    event
                    for event in store.list_recent_events(goal.id, limit=100)
                    if event.event_type == "planning.checkpoint"
                ]
                self.assertEqual(checkpoints[-1].payload["format_attempts"], 3)
                self.assertEqual(
                    checkpoints[-1].payload["checkpoint_type"],
                    "model_capability_exhausted",
                )
                self.assertTrue(checkpoints[-1].payload["resumable"])
            finally:
                store.close()

    def test_staged_semantic_repair_does_not_consume_plan_format_budget(self) -> None:
        request = "Repair the inspected classifier."
        criterion = "The repaired classifier passes its focused regression test."
        semantic = {
            "original_request": request,
            "interpreted_outcome": "Repair the existing classifier.",
            "requested_effects": [
                "read_workspace",
                "mutate_workspace",
                "execute_code",
            ],
            "required_outcomes": ["The classifier is repaired."],
            "constraints": ["Preserve unrelated behavior."],
            "exclusions": [],
            "acceptance_criteria": [criterion],
            "unresolved_decisions": [],
            "repository_evidence_refs": ["inspection:I001"],
        }
        fingerprint = SemanticGoalV2.from_mapping(
            semantic,
            original_request=request,
        ).fingerprint
        invalid_semantic = {
            "tool_calls": [
                {
                    "id": "bad-semantic",
                    "name": "propose_semantic_goal",
                    "args": {**semantic, "original_request": "A different request"},
                }
            ]
        }
        premature_plan = {
            "tool_calls": [
                {
                    "id": "premature-plan",
                    "name": "propose_plan",
                    "args": {
                        "summary": "Premature plan",
                        "tasks": [
                            {
                                "title": "Repair classifier",
                                "description": "Repair the classifier.",
                                "acceptance_criteria": [criterion],
                                "verification": ["Run the focused regression test."],
                            }
                        ],
                    },
                }
            ]
        }
        valid_semantic = {
            "tool_calls": [
                {
                    "id": "semantic",
                    "name": "propose_semantic_goal",
                    "args": semantic,
                }
            ]
        }
        staged_plan = {
            "tool_calls": [
                {
                    "id": "plan",
                    "name": "propose_plan",
                    "args": {
                        "semantic_fingerprint": fingerprint,
                        "semantic_goal": {
                            **semantic,
                            "requested_effects": ["mutate_workspace"],
                        },
                        "summary": "Repair and verify the classifier.",
                        "applicability_evidence": [
                            {
                                "fact": "The workspace inspection identified the target.",
                                "source": "inspection:I001",
                                "supports_tasks": ["T001"],
                            }
                        ],
                        "execution_strategy": "Apply the focused repair and run its regression test.",
                        "expected_changes": [
                            {
                                "path": "agent/classifier.py",
                                "intent": "Repair the classifier.",
                                "basis": "existing_inspected_path",
                                "evidence_refs": ["inspection:I001"],
                                "supports_tasks": ["T001"],
                            }
                        ],
                        "tasks": [
                            {
                                "title": "Repair classifier",
                                "description": "Repair the classifier.",
                                "acceptance_criteria": [criterion],
                                "verification": ["Run the focused regression test."],
                                "risk": "medium",
                            }
                        ],
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agent").mkdir()
            (workspace / "agent" / "classifier.py").write_text(
                "def classify(value): return value\n",
                encoding="utf-8",
            )
            store = StateStore(workspace)
            try:
                provider = ScriptedProvider(
                    [
                        inspection(),
                        invalid_semantic,
                        premature_plan,
                        valid_semantic,
                        staged_plan,
                        critic_pass(),
                    ]
                )
                runtime = AgentRuntime(provider, store, workspace)
                plan = runtime.start_goal(request)
                self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
                goal = runtime.active_goal()
                self.assertEqual(
                    goal.metadata["accepted_semantic_fingerprint"],
                    fingerprint,
                )
                retries = [
                    event
                    for event in store.list_recent_events(goal.id, limit=200)
                    if event.event_type == "workflow.retry"
                ]
                self.assertFalse(
                    any(
                        event.payload.get("kind") == "plan_format_repair"
                        for event in retries
                    )
                )
                self.assertTrue(
                    any(
                        event.event_type == "planning.semantic_accepted"
                        for event in store.list_recent_events(goal.id, limit=200)
                    )
                )
                staged_schema = next(
                    item
                    for item in provider.calls[4].tools
                    if item["function"]["name"] == "propose_plan"
                )
                self.assertNotIn(
                    "semantic_goal",
                    staged_schema["function"]["parameters"]["properties"],
                )
                provider.assert_exhausted()
            finally:
                store.close()

    def test_explicit_user_path_basis_cannot_launder_an_invented_path(self) -> None:
        request = "Create the requested report."
        candidate = semantic_plan(request, risk="low")
        change = candidate["tool_calls"][0]["args"]["expected_changes"][0]
        change.update(
            {
                "path": "invented/report.html",
                "basis": "explicit_user_requirement",
                "evidence_refs": ["user:request"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider([inspection(), candidate, candidate, candidate]),
                    store,
                    workspace,
                )
                self.assertIsNone(runtime.start_goal(request))
                goal = runtime.active_goal()
                self.assertEqual(goal.status, GoalStatus.PAUSED)
                self.assertIsNone(goal.active_plan_revision)
                checkpoints = [
                    event
                    for event in store.list_recent_events(goal.id, limit=100)
                    if event.event_type == "planning.checkpoint"
                ]
                self.assertIn(
                    "exact workspace-relative path",
                    checkpoints[-1].payload["technical_detail"],
                )
            finally:
                store.close()

    def test_cancel_updates_goal_and_workflow_session_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(ScriptedProvider([]), store, workspace)
                goal = store.create_goal(
                    "Cancel me", session_id=runtime.session_id
                )
                store.transition_goal(goal.id, GoalStatus.DISCOVERING)
                store.save_workflow_session(
                    runtime.session_id,
                    goal_id=goal.id,
                    session_mode="normal",
                    plan_state="inspecting",
                    run_state="planning",
                )
                cancelled = runtime.cancel("CANCEL")
                session = store.get_workflow_session(runtime.session_id)
                self.assertEqual(cancelled.status, GoalStatus.CANCELLED)
                self.assertIsNone(session["goal_id"])
                self.assertEqual(session["plan_state"], "none")
                self.assertEqual(session["run_state"], "cancelled")
                self.assertEqual(
                    session["state"]["terminal_goal_status"], "cancelled"
                )
            finally:
                store.close()

    def test_context_budget_uses_provider_window_and_reserves_fixed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            try:
                provider = ScriptedProvider([])
                provider.context_size = 16_384
                provider.max_output_tokens = 2_048
                runtime = AgentRuntime(
                    provider,
                    store,
                    workspace,
                    config=RuntimeConfig(conversation_chars=120_000),
                )
                budget = runtime._provider_conversation_budget(
                    "s" * 8_000,
                    [{"function": {"name": "tool", "parameters": {}}}],
                )
                self.assertLess(budget, 120_000)
                self.assertGreaterEqual(budget, 4_000)
            finally:
                store.close()

    def test_v14_invalidates_unfinished_legacy_plan_without_losing_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            goal = store.create_goal("Legacy unfinished work")
            store.transition_goal(goal.id, GoalStatus.DISCOVERING)
            database = store.path
            store.close()
            connection = sqlite3.connect(database)
            connection.execute("PRAGMA user_version=13")
            connection.commit()
            connection.close()

            migrated = StateStore(workspace)
            try:
                restored = migrated.get_goal(goal.id)
                self.assertEqual(restored.status, GoalStatus.PAUSED)
                self.assertTrue(restored.metadata["legacy_semantics_invalidated"])
                self.assertEqual(
                    restored.metadata["semantic_checkpoint"],
                    "inspection_required",
                )
                connection = sqlite3.connect(database)
                try:
                    self.assertEqual(
                        connection.execute("PRAGMA user_version").fetchone()[0],
                        14,
                    )
                finally:
                    connection.close()
            finally:
                migrated.close()

    def test_v14_records_and_removes_only_legacy_baseline_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            store.close()
            baseline = (
                workspace
                / ".coding-agent"
                / "recovery"
                / "run-old"
                / "baseline"
            )
            baseline.mkdir(parents=True)
            copied = baseline / "copied.py"
            copied.write_text("legacy copy\n", encoding="utf-8")
            copied_size = copied.stat().st_size
            copied.chmod(stat.S_IREAD)
            unrelated = workspace / "keep.py"
            unrelated.write_text("keep\n", encoding="utf-8")

            reopened = StateStore(workspace)
            try:
                row = reopened._connection.execute(
                    "SELECT * FROM legacy_recovery_cleanup WHERE run_id='run-old'"
                ).fetchone()
                self.assertEqual(row["size_bytes"], copied_size)
                self.assertIsNotNone(row["removed_at"])
                self.assertFalse(baseline.exists())
                self.assertEqual(
                    unrelated.read_text(encoding="utf-8"), "keep\n"
                )
            finally:
                reopened.close()

    def test_mutation_journal_excludes_secrets_and_preserves_user_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            (workspace / "safe.py").write_text("before\n", encoding="utf-8")
            store = StateStore(workspace)
            try:
                goal = store.create_goal("journal")
                journal_ids = store.prepare_mutation_journal(
                    goal.id, "T001", (".env", "safe.py", "new.py")
                )
                self.assertEqual(len(journal_ids), 2)
                (workspace / "safe.py").write_text("agent\n", encoding="utf-8")
                (workspace / "new.py").write_text("created\n", encoding="utf-8")
                store.finish_mutation_journal(journal_ids, applied=True)
                (workspace / "safe.py").write_text("user edit\n", encoding="utf-8")

                restored = store.rollback_mutation_journal(goal.id)
                self.assertEqual(restored, ("new.py",))
                self.assertEqual(
                    (workspace / "safe.py").read_text(encoding="utf-8"),
                    "user edit\n",
                )
                self.assertFalse((workspace / "new.py").exists())
                self.assertEqual(
                    (workspace / ".env").read_text(encoding="utf-8"),
                    "TOKEN=secret\n",
                )
            finally:
                store.close()

    def test_execution_writes_only_an_accepted_leased_path(self) -> None:
        request = "Repair the classifier."
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agent").mkdir()
            target = workspace / "agent" / "classifier.py"
            target.write_text("before\n", encoding="utf-8")
            write_turn = {
                "tool_calls": [
                    {
                        "id": "write",
                        "name": "write_file",
                        "args": {
                            "path": "agent/classifier.py",
                            "content": "after\n",
                        },
                    }
                ]
            }
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider(
                        [
                            inspection(),
                            semantic_plan(request, risk="low"),
                            critic_pass(),
                            write_turn,
                        ]
                    ),
                    store,
                    workspace,
                    approval=lambda *_: True,
                )
                plan = runtime.start_goal(request)
                self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
                runtime.approve_plan(plan.revision)
                runtime.run_slice(steps=1)
                self.assertEqual(target.read_text(encoding="utf-8"), "after\n")
                goal = store.get_goal(plan.goal_id)
                self.assertEqual(goal.metadata["mutation_sequence"], 1)
                self.assertEqual(
                    goal.metadata["resource_claims"][0]["state"], "released"
                )
            finally:
                store.close()

    def test_shell_mutation_outside_lease_becomes_uncertain_and_pauses(self) -> None:
        request = "Repair the classifier."
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agent").mkdir()
            (workspace / "agent" / "classifier.py").write_text(
                "before\n", encoding="utf-8"
            )
            command_turn = {
                "tool_calls": [
                    {
                        "id": "shell",
                        "name": "run_command",
                        "args": {
                            "command": (
                                "python -c \"from pathlib import Path;"
                                "Path('escaped.py').write_text('x')\""
                            )
                        },
                    }
                ]
            }
            store = StateStore(workspace)
            try:
                runtime = AgentRuntime(
                    ScriptedProvider(
                        [
                            inspection(),
                            semantic_plan(request, risk="low"),
                            critic_pass(),
                            command_turn,
                        ]
                    ),
                    store,
                    workspace,
                    approval=lambda *_: True,
                )
                plan = runtime.start_goal(request)
                self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
                runtime.approve_plan(plan.revision)
                runtime.run_slice(steps=1)
                goal = store.get_goal(plan.goal_id)
                self.assertEqual(goal.status, GoalStatus.PAUSED)
                self.assertEqual(
                    store.list_actions(goal.id)[-1]["status"], "uncertain"
                )
                self.assertEqual(
                    goal.metadata["uncertain_actions"][-1]["paths"],
                    ["escaped.py"],
                )
            finally:
                store.close()

    def test_verifiers_come_only_from_repository_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self.assertEqual(discover_verifier_plugins(workspace), ())
            (workspace / "package.json").write_text(
                '{"scripts":{"test":"vitest","build":"vite build"}}',
                encoding="utf-8",
            )
            plugins = discover_verifier_plugins(workspace)
            self.assertEqual(
                {item.command for item in plugins},
                {("npm", "run", "test"), ("npm", "run", "build")},
            )
            self.assertTrue(
                all(item.evidence_path == "package.json" for item in plugins)
            )

    def test_semantic_mapping_accepts_paraphrase_but_rejects_omission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            try:
                goal = store.create_goal(
                    "Create a calculator package, test it with pytest, and verify it."
                )
                semantic_criteria = [
                    (
                        "A Python package structure named 'calculator' must be "
                        "created in the workspace."
                    ),
                    (
                        "Pytest tests must be written for the 'add' function "
                        "within the 'calculator' package."
                    ),
                    (
                        "Running pytest must successfully execute all tests "
                        "without failure or errors."
                    ),
                    (
                        "The 'calculator' package must contain a module with "
                        "an 'add' function."
                    ),
                ]
                proposed = {
                    "semantic_goal": {
                        "original_request": goal.objective,
                        "interpreted_outcome": goal.objective,
                        "requested_effects": [
                            "read_workspace",
                            "mutate_workspace",
                            "execute_code",
                        ],
                        "required_outcomes": [
                            "A tested calculator package exists."
                        ],
                        "constraints": [],
                        "exclusions": [],
                        "acceptance_criteria": semantic_criteria,
                        "unresolved_decisions": [],
                        "repository_evidence_refs": ["inspection:I001"],
                        "status": "interpreted",
                    },
                    "expected_changes": [
                        {"path": "calculator/__init__.py"}
                    ],
                    "tasks": [
                        {
                            "acceptance_criteria": [
                                "The directory structure 'calculator/' must exist.",
                                (
                                    "The file 'calculator/__init__.py' must be "
                                    "present and contain a working add(a, b) function."
                                ),
                                (
                                    "The file 'tests/test_calculator.py' must contain "
                                    "at least three distinct pytest cases for add."
                                ),
                                (
                                    "Execution of pytest must run without any test "
                                    "failures or errors across all scenarios."
                                ),
                            ]
                        }
                    ],
                }
                accepted = AgentRuntime._validate_semantic_candidate(
                    goal,
                    proposed,
                    successful_inspection_ids=frozenset({"I001"}),
                )
                self.assertEqual(
                    accepted.acceptance_criteria,
                    tuple(semantic_criteria),
                )

                proposed["tasks"][0]["acceptance_criteria"] = [
                    "The directory structure 'calculator/' must exist."
                ]
                with self.assertRaisesRegex(
                    ValueError,
                    "do not cover every accepted semantic criterion",
                ):
                    AgentRuntime._validate_semantic_candidate(
                        goal,
                        proposed,
                        successful_inspection_ids=frozenset({"I001"}),
                    )
            finally:
                store.close()

    def test_derived_path_provenance_is_rebound_without_changing_the_path(self) -> None:
        proposed = {
            "tasks": [{"id": "T001"}],
            "applicability_evidence": [],
            "expected_changes": [
                {
                    "path": "calculator/__init__.py",
                    "intent": "Create the requested calculator package.",
                    "basis": "explicit_user_requirement",
                    "evidence_refs": ["user:request"],
                    "supports_tasks": ["T001"],
                }
            ],
        }
        bound = AgentRuntime._bind_plan_inspection_sources(
            proposed,
            {
                "I001": {
                    "call_id": "inspect",
                    "tool": "list_files",
                    "result": "(no files)",
                }
            },
            original_request="Create a Python package named calculator.",
        )
        change = bound["expected_changes"][0]
        self.assertEqual(change["path"], "calculator/__init__.py")
        self.assertEqual(change["basis"], "repository_convention")
        self.assertEqual(change["evidence_refs"], ["inspection:I001"])

    def test_plan_critic_pass_advisories_do_not_force_scope_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            try:
                goal = store.create_goal("Create and verify the requested package.")
                runtime = AgentRuntime(
                    ScriptedProvider(
                        [
                            {
                                "tool_calls": [
                                    {
                                        "id": "review",
                                        "name": "submit_plan_review",
                                        "args": {
                                            "verdict": "pass",
                                            "summary": "The plan fully covers the goal.",
                                            "issues": [
                                                "Optionally install another tool."
                                            ],
                                        },
                                    }
                                ]
                            }
                        ]
                    ),
                    store,
                    directory,
                )
                result = runtime._review_plan_candidate(goal, {}, {})
                self.assertEqual(result["verdict"], "pass")
                self.assertEqual(result["issues"], [])
                self.assertTrue(
                    any(
                        event.event_type == "plan.critic_advisories"
                        for event in store.list_recent_events(goal.id, limit=20)
                    )
                )
            finally:
                store.close()

    def test_exact_file_scope_rejects_an_extra_model_invented_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            try:
                goal = store.create_goal(
                    "Create calculator.py and tests/test_calculator.py only."
                )
                runtime = AgentRuntime(ScriptedProvider([]), store, directory)
                task_value = {
                    "id": "T001",
                    "title": "Create and test calculator",
                    "description": "Create only the requested files.",
                    "acceptance_criteria": ["Tests pass."],
                    "verification": ["Run python -m pytest."],
                    "risk": "low",
                }
                task_item = store.coerce_task(
                    task_value,
                    goal.id,
                    1,
                    "agent",
                )
                proposed = {
                    "semantic_goal": {
                        "constraints": [
                            "Only create calculator.py and "
                            "tests/test_calculator.py."
                        ],
                        "exclusions": ["No other files may be created."],
                    },
                    "applicability_evidence": [
                        {
                            "fact": "The empty workspace needs the task.",
                            "source": "inspection:I001",
                            "supports_tasks": ["T001"],
                        }
                    ],
                    "expected_changes": [
                        {
                            "path": "calculator.py",
                            "basis": "repository_convention",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001"],
                        },
                        {
                            "path": "pytest_run.sh",
                            "basis": "repository_convention",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001"],
                        },
                    ],
                }
                with self.assertRaisesRegex(
                    ValueError,
                    "violates the accepted exact file scope",
                ):
                    runtime._validate_plan_applicability(
                        proposed,
                        [task_item],
                        successful_inspection_ids=frozenset({"I001"}),
                        original_request=goal.objective,
                    )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
import sqlite3
from pathlib import Path

from agent import tools
from agent.config import InteractionMode, SessionPreferences
from agent.models import GoalStatus
from agent.quality import (
    ChangeSetStatus,
    ChangeSetV1,
    FindingSeverity,
    FindingStatus,
    QualityCategory,
    QualityCycleKind,
    QualityCycleV1,
    QualityFindingV1,
    QualityPolicyV1,
)
from agent.runtime import AgentRuntime
from agent.commands import parse_command
from agent.sandbox import AccessLevel
from agent.sleep_profile import SleepActivationError, SleepController
from agent.store import StateStore
from agent.testing import ScriptedProvider, semantic_turn
from agent.typed_returns import TypedReturnFailure, TypedReturnProcessor
from agent.ultra import (
    AgentResponse,
    ExecutionClass,
    GoalSpecV1,
    InMemoryUltraState,
    UltraConfig,
    UltraOrchestrator,
)
from agent.ultra_models import UltraRun
from agent.workflow import (
    PlanDraftError,
    PlanState,
    is_unambiguous_plan_approval,
    normalize_plan_draft,
    validate_normalized_plan,
)


CASTLE = (
    "In a single HTML file, create an animation of a castle siege in great detail. "
    "It should have a castle, soldiers trying to break down the gate with a battering "
    "ram, a siege tower, archers shooting arrows, and soldiers firing catapults."
)


def castle_plan_args(*, summary: str, title: str, description: str, criterion: str, verification: str):
    return {
        "semantic_goal": {
            "original_request": CASTLE,
            "interpreted_outcome": "Create and verify the requested single-file castle siege animation.",
            "requested_effects": ["read_workspace", "mutate_workspace", "execute_code"],
            "required_outcomes": ["A runnable castle siege animation exists in index.html."],
            "constraints": ["Keep the implementation self-contained in one HTML file."],
            "exclusions": [],
            "acceptance_criteria": [criterion],
            "unresolved_decisions": [],
            "repository_evidence_refs": ["inspection:I001"],
        },
        "summary": summary,
        "applicability_evidence": [{
            "fact": "The inspected workspace is empty and can host the requested new artifact.",
            "source": "inspection:I001",
            "supports_tasks": ["T001"],
        }],
        "execution_strategy": "Create index.html, run the available HTML preview checks, and independently review the evidence.",
        "expected_changes": [{
            "path": "index.html",
            "intent": "Create the requested self-contained animation.",
            "basis": "model_selected_new_layout",
            "evidence_refs": ["inspection:I001"],
            "supports_tasks": ["T001"],
        }],
        "tasks": [{
            "id": "T001",
            "title": title,
            "description": description,
            "acceptance_criteria": [criterion],
            "verification": [verification],
            "depends_on": [],
            "risk": "low",
        }],
    }


def passing_plan_review():
    return {
        "tool_calls": [{
            "id": "critic",
            "name": "submit_plan_review",
            "args": {"verdict": "pass", "summary": "The plan is scoped and verifiable.", "issues": []},
        }]
    }


def blocking_plan_review(summary: str):
    return {
        "tool_calls": [{
            "id": "critic",
            "name": "submit_plan_review",
            "args": {
                "verdict": "revise",
                "summary": summary,
                "issues": [{
                    "detail": summary,
                    "severity": "blocking",
                    "blocking": True,
                    "criterion_refs": ["T001"],
                    "evidence_refs": ["inspection:I001"],
                }],
            },
        }]
    }


class PlanHarnessV4Tests(unittest.TestCase):
    def test_two_blocking_critic_revisions_are_available_after_initial_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            proposal = {
                "tool_calls": [{
                    "id": "plan",
                    "name": "propose_plan",
                    "args": castle_plan_args(
                        summary="Create and verify the requested animation.",
                        title="Create the animation",
                        description="Create index.html with the requested animation.",
                        criterion="The requested animation renders.",
                        verification="Preview index.html with the available browser tool.",
                    ),
                }]
            }
            provider = ScriptedProvider([
                {"tool_calls": [{"id": "inspect", "name": "list_files", "args": {}}]},
                proposal,
                blocking_plan_review("First blocking issue."),
                proposal,
                blocking_plan_review("Second blocking issue."),
                proposal,
                passing_plan_review(),
            ])
            try:
                runtime = AgentRuntime(provider, store, workspace)
                plan = runtime.start_goal(CASTLE)
                self.assertIsNotNone(plan)
                self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
                provider.assert_exhausted()
            finally:
                runtime.close()
                store.close()

    def test_castle_empty_workspace_is_inspected_once_and_plan_is_presented_without_write(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            provider = ScriptedProvider(
                [
                    {"tool_calls": [{"id": "inspect", "name": "list_files", "args": {}}]},
                    {
                        "tool_calls": [
                            {
                                "id": "plan",
                                "name": "propose_plan",
                                "args": castle_plan_args(
                                    summary="Create and verify one detailed castle-siege HTML animation.",
                                    title="Create the castle siege animation",
                                    description="Build the requested self-contained HTML scene and motion.",
                                    criterion="The scene visibly includes the castle, ram, tower, archers, arrows, and catapults.",
                                    verification="Parse the HTML and render it in an available browser.",
                                ),
                            }
                        ]
                    },
                    passing_plan_review(),
                ]
            )
            try:
                runtime = AgentRuntime(provider, store, workspace)
                plan = runtime.start_goal(CASTLE)
                self.assertIsNotNone(plan)
                self.assertEqual(len(plan.tasks), 1)
                self.assertEqual(plan.tasks[0].id, "T001")
                self.assertEqual(plan.tasks[0].verification, ("Parse the HTML and render it in an available browser.",))
                self.assertEqual(runtime.active_goal().status.value, "awaiting_plan_approval")
                self.assertEqual(len(store.list_actions(runtime.active_goal().id)), 1)
                self.assertFalse((workspace / "index.html").exists())
                self.assertEqual(len(provider.calls), 3)
            finally:
                store.close()

    def test_plan_normalization_owns_ids_dependencies_and_safe_scalar_repairs(self):
        normalized, actions = normalize_plan_draft(
            {
                "summary": "  Build two verified pieces.  ",
                "tasks": [
                    {
                        "id": "A very long natural language task identifier that must not persist",
                        "title": " First ",
                        "description": " Create base ",
                        "acceptance_criteria": [" Base exists ", "Base exists"],
                        "verification": " Check base ",
                        "risk": "Low risk, local file creation.",
                    },
                    {
                        "id": "anything",
                        "title": "Second",
                        "description": "Use base",
                        "acceptance_criteria": ["Result exists"],
                        "verification": ["Check result", "Check result"],
                        "depends_on": [1, "1", "T001"],
                    },
                ],
                "expected_changes": [
                    {
                        "path": "src/base.py",
                        "basis": "repository convention",
                        "evidence_refs": ["inspection:I001"],
                        "supports_tasks": ["1"],
                    },
                    {
                        "path": "src/result.py",
                        "intent": "Create the requested result.",
                        "basis": "user:request",
                        "evidence_refs": ["user:request"],
                        "supports_tasks": ["2"],
                    },
                    {
                        "path": "./",
                        "basis": "Temporary working directory for execution.",
                        "supports_tasks": ["2"],
                    },
                ],
            }
        )
        validate_normalized_plan(normalized)
        self.assertEqual([item["id"] for item in normalized["tasks"]], ["T001", "T002"])
        self.assertEqual(normalized["tasks"][1]["depends_on"], ["T001"])
        self.assertEqual(normalized["tasks"][0]["verification"], ["Check base"])
        self.assertEqual(normalized["tasks"][0]["risk"], "low")
        self.assertEqual(
            [item["basis"] for item in normalized["expected_changes"]],
            ["repository_convention", "explicit_user_requirement"],
        )
        self.assertEqual(len(normalized["expected_changes"]), 2)
        self.assertEqual(
            normalized["expected_changes"][0]["intent"],
            "Create base",
        )
        self.assertTrue(actions)

    def test_plan_normalization_derives_only_description_from_complete_task_contract(self):
        normalized, actions = normalize_plan_draft(
            {
                "summary": "Create and verify an artifact.",
                "tasks": [
                    {
                        "title": "Create artifact",
                        "acceptance_criteria": ["artifact.txt contains ok"],
                        "verification": ["Read artifact.txt and compare exact content"],
                    }
                ],
            }
        )
        validate_normalized_plan(normalized)
        self.assertIn("Create artifact", normalized["tasks"][0]["description"])
        self.assertTrue(any("description derived" in item for item in actions))

        incomplete, _ = normalize_plan_draft(
            {"summary": "Still incomplete", "tasks": [{"title": "Missing proof"}]}
        )
        with self.assertRaises(PlanDraftError):
            validate_normalized_plan(incomplete)

    def test_plan_normalization_derives_missing_summary_from_authored_strategy(self):
        normalized, actions = normalize_plan_draft(
            {
                "execution_strategy": "Create hello.txt and verify its exact contents.",
                "tasks": [
                    {
                        "title": "Create hello.txt",
                        "description": "Create the requested file.",
                        "acceptance_criteria": ["hello.txt exists."],
                        "verification": ["Read hello.txt back."],
                    }
                ],
            }
        )
        validate_normalized_plan(normalized)
        self.assertEqual(
            normalized["summary"],
            "Create hello.txt and verify its exact contents.",
        )
        self.assertTrue(any("summary derived" in item for item in actions))

    def test_plan_normalization_derives_missing_strategy_from_complete_tasks(self):
        normalized, actions = normalize_plan_draft(
            {
                "summary": "Create and verify hello.txt.",
                "tasks": [{
                    "title": "Create hello.txt",
                    "description": "Create the requested file.",
                    "acceptance_criteria": ["hello.txt exists."],
                    "verification": ["Read hello.txt back."],
                }],
            }
        )
        validate_normalized_plan(normalized)
        self.assertIn("dependency order", normalized["execution_strategy"])
        self.assertTrue(any("execution_strategy derived" in item for item in actions))

    def test_plan_normalization_lifts_local_task_expected_changes(self):
        normalized, actions = normalize_plan_draft(
            {
                "summary": "Create and verify hello.txt.",
                "tasks": [
                    {
                        "title": "Create hello.txt",
                        "description": "Create hello.txt with the requested content.",
                        "expected_changes": [
                            {
                                "path": "./hello.txt",
                                "intent": "create the requested file",
                                "basis": "user_request",
                            }
                        ],
                        "acceptance_criteria": ["hello.txt contains Hello."],
                        "verification": ["Read hello.txt back."],
                    }
                ],
            }
        )
        validate_normalized_plan(normalized)
        self.assertEqual(normalized["expected_changes"][0]["path"], "./hello.txt")
        self.assertEqual(normalized["expected_changes"][0]["basis"], "explicit_user_requirement")
        self.assertEqual(normalized["expected_changes"][0]["supports_tasks"], ["T001"])
        self.assertTrue(any("expected_changes lifted" in item for item in actions))

    def test_plan_normalization_rewrites_posix_content_check_for_windows(self):
        normalized, actions = normalize_plan_draft(
            {
                "summary": "Verify hello.txt.",
                "tasks": [{
                    "title": "Verify hello.txt",
                    "description": "Check the exact file content.",
                    "acceptance_criteria": ["The file contains Hello."],
                    "verification": ["run_command -c 'cat ./hello.txt' | grep -q '^Hello$'"],
                }],
            }
        )
        validate_normalized_plan(normalized)
        verification = normalized["tasks"][0]["verification"][0]
        if __import__("os").name == "nt":
            self.assertIn("python -c", verification)
            self.assertIn("read_text", verification)
            self.assertTrue(any("normalized for Windows" in item for item in actions))

    def test_plan_normalization_projects_only_an_explicit_verifier_command(self):
        normalized, actions = normalize_plan_draft(
            {
                "summary": "Run the accepted verifier.",
                "execution_strategy": (
                    "Create the requested files, then run python -m pytest."
                ),
                "tasks": [
                    {
                        "title": "Verify the project",
                        "description": "Execute the accepted test suite.",
                        "acceptance_criteria": ["All pytest tests pass."],
                    }
                ],
            }
        )
        validate_normalized_plan(normalized)
        self.assertEqual(
            normalized["tasks"][0]["verification"],
            ["Run python -m pytest and require exit code 0."],
        )
        self.assertTrue(
            any("verification projected" in item for item in actions)
        )

    def test_ambiguous_dependency_has_exact_pointer(self):
        value, _ = normalize_plan_draft(
            {
                "summary": "Build",
                "tasks": [
                    {"title": "One", "description": "First", "acceptance_criteria": ["Done"], "verification": ["Check"]},
                    {"title": "Two", "description": "Second", "acceptance_criteria": ["Done"], "verification": ["Check"], "depends_on": ["after the setup task"]},
                ],
            }
        )
        with self.assertRaisesRegex(PlanDraftError, "/tasks/1/depends_on"):
            validate_normalized_plan(value)

    def test_chat_is_default_and_approval_intent_is_narrow(self):
        self.assertEqual(SessionPreferences().mode, InteractionMode.CHAT)
        for text in ("do it", "go ahead", "accept", "approve the plan"):
            self.assertTrue(is_unambiguous_plan_approval(text))
        self.assertFalse(is_unambiguous_plan_approval("do it", pending_plans=2))
        self.assertFalse(is_unambiguous_plan_approval("tell me whether to do it"))

    def test_chat_turn_does_not_force_a_goal_or_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            try:
                runtime = AgentRuntime(ScriptedProvider([
                    semantic_turn(
                        "chat",
                        original="What is in this workspace?",
                        response="The workspace is empty.",
                    )
                ]), store, directory)
                result = runtime.chat("What is in this workspace?")
                self.assertEqual(result.status, "chat")
                self.assertIn("empty", result.message)
                self.assertIsNone(store.get_latest_goal())
            finally:
                store.close()

    def test_natural_approval_starts_evidence_backed_empty_workspace_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            html = "<!doctype html><html><title>Castle Siege</title><body><div id='castle'>Siege</div></body></html>"
            provider = ScriptedProvider(
                [
                    {"tool_calls": [{"id": "inspect", "name": "list_files", "args": {}}]},
                    {"tool_calls": [{"id": "plan", "name": "propose_plan", "args": castle_plan_args(
                        summary="Create one verified castle siege page.",
                        title="Create page",
                        description="Create index.html castle siege animation",
                        criterion="The castle siege page exists",
                        verification="Read index.html and confirm its structure",
                    )}]},
                    passing_plan_review(),
                    {"tool_calls": [
                        {"id": "start", "name": "update_task", "args": {"task_id": "T001", "status": "in_progress", "note": "creating", "evidence": []}},
                        {"id": "write", "name": "write_file", "args": {"path": "index.html", "content": html}},
                        {"id": "read", "name": "read_file", "args": {"path": "index.html"}},
                        {"id": "preview", "name": "preview_html", "args": {"path": "index.html", "open_browser": False, "settle_ms": 0}},
                        {"id": "done", "name": "update_task", "args": {"task_id": "T001", "status": "done", "note": "file read back", "evidence": ["index.html exists and was read by the workspace tool"]}},
                    ]},
                    {"tool_calls": [{"id": "finish", "name": "finish_goal", "args": {"summary": "Page created and verified", "evidence": ["index.html tool evidence"]}}]},
                    {"tool_calls": [{"id": "review", "name": "submit_review", "args": {"verdict": "pass", "summary": "Required file and evidence are present", "issues": [], "checked_task_ids": ["T001"]}}]},
                ]
            )
            try:
                runtime = AgentRuntime(provider, store, workspace, approval=lambda *_: True)
                plan = runtime.start_goal(CASTLE)
                preferences = SessionPreferences(mode=InteractionMode.PLAN)
                preferences.mode = InteractionMode.GOAL
                self.assertEqual(runtime.latest_plan().status.value, "pending_approval")
                accepted = runtime.apply_command(parse_command("do it"))
                self.assertEqual(accepted.status.value, "accepted")
                dimensions = runtime.active_goal().metadata["quality_target"]["dimensions"]
                self.assertTrue(dimensions)
                self.assertFalse(
                    any(
                        token in item["id"]
                        for item in dimensions
                        for token in ("castle", "catapult", "siege", "archer")
                    )
                )
                goal_id = runtime.active_goal().id
                result = runtime.run_slice(steps=4)
                self.assertTrue(result.completed)
                completed = store.get_goal(goal_id)
                self.assertEqual(completed.status.value, "completed")
                self.assertEqual(
                    completed.metadata["completion_disposition"],
                    "completed_with_limitations",
                )
                self.assertTrue(completed.metadata["completion_limitations"])
                self.assertEqual((workspace / "index.html").read_text(encoding="utf-8"), html)
                actions = store.list_actions(goal_id)
                self.assertTrue(any(item["tool_name"] == "write_file" and item["status"] == "completed" for item in actions))
                self.assertTrue(any(item["tool_name"] == "read_file" and item["status"] == "completed" for item in actions))
                runtime.close()
                with tools.workspace_context(workspace):
                    self.assertEqual(tools.web_preview.list_previews(), ())
            finally:
                store.close()


class TypedReturnTests(unittest.TestCase):
    def test_empty_goal_spec_never_becomes_valid_without_targeted_repair(self):
        processor = TypedReturnProcessor("GoalSpecV1", GoalSpecV1.from_mapping)
        calls = []

        def repair(previous, errors, contract):
            calls.append((previous, errors, contract))
            return {"objective": "Create the project", "success_criteria": ["File exists"]}

        result = processor.process({"objective": "", "success_criteria": []}, repair=repair)
        self.assertEqual(result.value.objective, "Create the project")
        self.assertTrue(result.repaired)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(TypedReturnFailure):
            processor.process({"objective": "", "success_criteria": []})

    def test_ultra_agent_is_not_completed_until_goalspec_repair_validates(self):
        state = InMemoryUltraState()
        calls = {"goal_spec": 0}

        class Agent:
            def execute(self, request):
                if request.phase == "goal_spec":
                    calls["goal_spec"] += 1
                    payload = (
                        {"objective": "", "success_criteria": []}
                        if calls["goal_spec"] == 1
                        else {"objective": "Create a project", "success_criteria": ["Project file exists"]}
                    )
                elif request.phase == "architecture":
                    payload = {"summary": "Single-file architecture", "components": [{"name": "page"}], "interfaces": []}
                else:
                    payload = {
                        "summary": "Create and verify the project",
                        "execution_strategy": "Create the file, review it, and verify it.",
                        "modules": [{
                            "id": "M001", "title": "Page", "objective": "Create page",
                            "acceptance_criteria": ["Page exists"], "verification": ["Inspect page"],
                            "depends_on": [], "write_paths": ["index.html"], "forbidden_changes": [],
                            "owned_interfaces": [],
                        }],
                    }
                return AgentResponse(payload=payload, summary=request.phase)

        class Factory:
            def create(self, *args, **kwargs):
                return Agent()

        engine = UltraOrchestrator(
            Factory(), execution_class=ExecutionClass.LOCAL, state=state,
            config=UltraConfig(min_top_modules=1, max_top_modules=3, provider_retries=0),
        )
        plan = engine.prepare("Create a project in this empty workspace")
        self.assertIsNotNone(plan)
        goal_agents = [item for item in state.agent_runs if item.phase == "goal_spec"]
        self.assertEqual([item.status for item in goal_agents], ["failed", "completed"])


class PersistenceV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.store = StateStore(self.workspace)
        goal = self.store.create_goal("Quality test")
        self.run = self.store.create_ultra_run(UltraRun(goal.id, "fake", "model"))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_change_sets_findings_cycles_and_policy_are_durable(self):
        policy = QualityPolicyV1()
        self.store.save_quality_policy(self.run.id, policy, master_plan_fingerprint="master-1")
        loaded, fingerprint = self.store.get_quality_policy(self.run.id)
        self.assertEqual(loaded.version, 1)
        self.assertEqual(fingerprint, "master-1")

        change = ChangeSetV1(self.run.id, "agent-1", "T001")
        self.store.save_change_set(change)
        self.store.record_mutation(change.id, "write_file", path="index.html", post_hash="abc")
        self.assertEqual(len(self.store.list_mutations(change.id)), 1)
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            change.integrate()
        approved = ChangeSetV1(
            self.run.id,
            "agent-1",
            "T001",
            id=change.id,
            status=ChangeSetStatus.APPROVED,
            review_status={"clean_code": "passed", "security": "passed", "test_quality": "passed"},
        )
        self.assertEqual(approved.integrate().status, ChangeSetStatus.INTEGRATED)

        finding = QualityFindingV1(
            self.run.id, "input_validation", QualityCategory.SECURITY, FindingSeverity.HIGH,
            "app.py", "L10", "hash", {"scanner": "deterministic"}, "Validate input",
            ("Invalid input is rejected",), ("Run validation tests",),
        )
        first = self.store.put_quality_finding(finding)
        second = self.store.put_quality_finding(finding)
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.status, FindingStatus.OPEN)

        cycle = QualityCycleV1(self.run.id, QualityCycleKind.BASELINE, 1, "approach", {}, {}, {}, "complete")
        self.store.save_quality_cycle(cycle)
        self.assertEqual(len(self.store.list_quality_cycles(self.run.id)), 1)

    def test_sleep_requires_ultra_full_docker_and_replans_after_three_failures(self):
        sleep = SleepController()
        with self.assertRaises(SleepActivationError):
            sleep.enable(mode=InteractionMode.GOAL, access_level=AccessLevel.FULL, docker_ready=True, safe_checkpoint=True, active_uncertain_mutation=False)
        with self.assertRaises(SleepActivationError):
            sleep.enable(mode=InteractionMode.ULTRA, access_level=AccessLevel.NORMAL, docker_ready=True, safe_checkpoint=True, active_uncertain_mutation=False)
        sleep.enable(mode=InteractionMode.ULTRA, access_level=AccessLevel.FULL, docker_ready=True, safe_checkpoint=True, active_uncertain_mutation=False)
        for attempt in range(1, 4):
            sleep.record_cycle(QualityCycleV1(self.run.id, QualityCycleKind.PROJECT_SWEEP, attempt, "same", {}, {}, {}, "failed"))
        with self.assertRaisesRegex(SleepActivationError, "materially different"):
            sleep.record_cycle(QualityCycleV1(self.run.id, QualityCycleKind.PROJECT_SWEEP, 4, "same", {}, {}, {}, "failed"))

    def test_v3_fixture_migrates_transactionally_and_v4_is_idempotent(self):
        goal_id = self.run.goal_id
        database = self.store.path
        self.store.close()
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA foreign_keys=OFF")
            for table in (
                "agent_registry", "mutation_ledger", "change_sets", "quality_findings",
                "quality_cycles", "quality_policies", "workflow_sessions",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute("PRAGMA user_version=3")
            connection.commit()
        finally:
            connection.close()
        migrated = StateStore(self.workspace)
        try:
            self.assertEqual(migrated.get_goal(goal_id).objective, "Quality test")
            check = sqlite3.connect(database)
            try:
                self.assertEqual(check.execute("PRAGMA user_version").fetchone()[0], 15)
            finally:
                check.close()
            migrated._migrate_v4()
            migrated._migrate_v4()
            tables = {
                row[0] for row in migrated._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self.assertTrue({"change_sets", "quality_findings", "quality_cycles"}.issubset(tables))
        finally:
            migrated.close()
        # tearDown expects an open store.
        self.store = StateStore(self.workspace)

    def test_restart_preserves_cycles_but_resets_sleep_authorization(self):
        self.store.save_workflow_session(
            "sleep-session", goal_id=self.run.goal_id, session_mode="ultra",
            plan_state="approved", run_state="reviewing", ultra_profile="sleep", sleep_state="running",
        )
        cycle = QualityCycleV1(self.run.id, QualityCycleKind.PROJECT_SWEEP, 1, "sweep", {}, {}, {}, "complete")
        self.store.save_quality_cycle(cycle)
        self.store.close()
        self.store = StateStore(self.workspace)
        session = self.store.get_workflow_session("sleep-session")
        self.assertEqual(session["sleep_state"], "off")
        self.assertEqual(session["ultra_profile"], "standard")
        self.assertEqual(len(self.store.list_quality_cycles(self.run.id)), 1)


if __name__ == "__main__":
    unittest.main()

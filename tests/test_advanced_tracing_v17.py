from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent.commands import (
    CommandKind,
    InternalActionKind,
    UnknownCommandParseError,
    internal_action,
    parse_command,
)
from agent.config import RuntimeConfig
from agent.events import EventBus
from agent.models import Evidence, GoalStatus
from agent.quality import ChangeSetStatus, ChangeSetV1
from agent.runtime import AgentRuntime
from agent.store import StateStore
from agent.testing import ScriptedProvider
from agent.tui_commands import COMMAND_SPECS
from agent.ultra_models import PromptTraceV1, UltraRun
from agent.web_views.security import SessionSecurity
from agent.web_views.server import create_app
from agent.web_views.service import CoreWebAdapter


PUBLIC_COMMANDS = {
    "/plan",
    "/live",
    "/show-diff",
    "/advanced-tracing",
    "/settings",
    "/pause",
    "/resume",
    "/stop",
    "/undo",
    "/help",
    "/quit",
}


class PublicCommandSurfaceTests(unittest.TestCase):
    def test_parser_and_palette_have_exactly_eleven_commands(self):
        self.assertEqual({item.name for item in COMMAND_SPECS}, PUBLIC_COMMANDS)
        for command in PUBLIC_COMMANDS - {"/undo"}:
            self.assertIsInstance(parse_command(command).kind, CommandKind)
        self.assertEqual(parse_command("/undo 2").args, {"steps": 2})

    def test_removed_slash_commands_are_not_compatibility_aliases(self):
        for removed in ("/model", "/trace", "/agents", "/diff", "/sleep on", ":status"):
            if removed.startswith(":"):
                self.assertEqual(parse_command(removed).kind, CommandKind.TEXT)
            else:
                with self.assertRaises(UnknownCommandParseError):
                    parse_command(removed)

    def test_attention_actions_are_typed_without_slash_parsing(self):
        action = internal_action(InternalActionKind.APPROVE, revision=3)
        self.assertEqual(action.kind, InternalActionKind.APPROVE)
        self.assertEqual(action.args, {"revision": 3})

    def test_source_has_no_synthetic_removed_slash_dispatch(self):
        source_root = Path(__file__).resolve().parents[1] / "agent"
        removed = {
            "approve", "model", "permissions", "sleep", "trace", "agents",
            "diff", "status", "doctor", "processes", "resolve", "replan",
        }
        pattern = re.compile(r"parse_command\(\s*(?:f)?[\"']/([a-z-]+)")
        findings: list[str] = []
        for path in source_root.rglob("*.py"):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                if match.group(1) in removed:
                    findings.append(
                        f"{path.relative_to(source_root)}: /{match.group(1)}"
                    )
        self.assertEqual(findings, [])


class AdvancedTracingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.store = StateStore(self.workspace)
        self.runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            events=EventBus(),
            config=replace(RuntimeConfig(), repository_index_warmup_files=0),
            session_id="trace-session",
        )
        self.goal = self.store.create_goal(
            "Trace every durable decision",
            session_id=self.runtime.session_id,
        )
        self.adapter = CoreWebAdapter(self.runtime)
        self.security = SessionSecurity(self.runtime.session_id)
        self.app = create_app(self.adapter, self.security)
        self.app.state.port = 43211
        self.client = TestClient(self.app, base_url="http://127.0.0.1:43211")
        opened = self.client.get(
            f"/sessions/{self.runtime.session_id}/plan?token={self.security.token}"
        )
        self.assertEqual(opened.status_code, 200)

    def tearDown(self) -> None:
        self.client.close()
        self.runtime.close()
        self.store.close()
        self.temporary.cleanup()

    def test_standalone_page_is_not_redirected_into_plan_or_live(self):
        response = self.client.get(
            f"/sessions/{self.runtime.session_id}/advanced-tracing"
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Advanced Tracing", response.text)
        self.assertIn("traceInspector", response.text)
        self.assertNotIn('data-view="plan"', response.text)

        overview = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/overview"
        )
        self.assertEqual(overview.status_code, 200, overview.text)
        self.assertEqual(overview.json()["state"], "LIVE")
        self.assertEqual(overview.json()["goal_id"], self.goal.id)

    def test_show_diff_is_standalone_and_projects_recorded_file_patches(self):
        run = self.store.create_ultra_run(UltraRun(
            goal_id=self.goal.id,
            provider="test",
            model="weak-model",
        ))
        change = self.store.save_change_set(ChangeSetV1(
            ultra_run_id=run.id,
            responsible_agent_id="coder-agent",
            parent_id="implementation-node",
            status=ChangeSetStatus.REVIEWING,
            changed_files=("hello.txt",),
            diff=(
                "diff --git a/hello.txt b/hello.txt\n"
                "--- /dev/null\n+++ b/hello.txt\n"
                "@@ -0,0 +1 @@\n+hello\n"
            ),
        ))

        page = self.client.get(
            f"/sessions/{self.runtime.session_id}/show-diff"
        )
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn("Workflow Diff", page.text)
        self.assertNotIn('data-view="plan"', page.text)

        snapshot = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/show-diff"
        )
        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        self.assertTrue(snapshot.json()["has_diff"])
        self.assertEqual(snapshot.json()["selected_id"], change.id)
        self.assertEqual(snapshot.json()["selected"]["files"][0]["status"], "added")

        detail = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/show-diff/{change.id}"
        )
        self.assertIn("+hello", detail.json()["diff"])
        self.assertEqual(detail.json()["additions"], 1)

    def test_show_diff_treats_a_new_project_file_as_a_live_added_file(self):
        self.runtime.version_control.ensure_local_history()
        (self.workspace / "brand-new.txt").write_text("first line\n", encoding="utf-8")

        snapshot = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/show-diff"
        )
        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        payload = snapshot.json()
        self.assertEqual(payload["selected_id"], "working")
        self.assertEqual(payload["selected"]["files"][0]["path"], "brand-new.txt")
        self.assertEqual(payload["selected"]["files"][0]["status"], "added")
        self.assertIn("+first line", payload["selected"]["diff"])

    def test_file_lifecycle_includes_selected_excluded_opened_and_verified(self):
        self.store.append_event(
            "context.repository_retrieval",
            goal_id=self.goal.id,
            payload={
                "stage": "test",
                "query": "trace files",
                "candidates": [
                    {"path": "agent/opened.py", "name": "opened", "rank": 1, "score": 8, "outcome": "selected", "reason": "ranked", "provenance": ["lexical"]},
                    {"path": "agent/excluded.py", "name": "excluded", "rank": 2, "score": 2, "outcome": "excluded", "reason": "budget", "provenance": ["embedding"]},
                ],
            },
        )
        action_id = self.store.begin_action(
            self.goal.id,
            "read_file",
            {"path": "agent/opened.py"},
        )
        self.store.complete_action(action_id, "read complete")
        self.store.add_evidence(Evidence(
            goal_id=self.goal.id,
            summary="opened.py passed verification",
            kind="file_verification",
            data={"path": "agent/opened.py"},
            verified=True,
        ))

        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/sections/files"
        )
        self.assertEqual(response.status_code, 200, response.text)
        files = {item["path"]: item for item in response.json()["items"]}
        self.assertTrue({"considered", "selected_context", "opened", "verified"} <= set(files["agent/opened.py"]["states"]))
        self.assertTrue({"considered", "excluded"} <= set(files["agent/excluded.py"]["states"]))

    def test_prompt_reveal_returns_only_the_redacted_stored_trace(self):
        run = self.store.create_ultra_run(UltraRun(
            goal_id=self.goal.id,
            provider="test",
            model="weak-model",
        ))
        trace = self.store.add_prompt_trace(PromptTraceV1(
            ultra_run_id=run.id,
            role="coder",
            system_prompt="Use OPENAI_API_KEY=sk-secret-value and inspect the project.",
            context_package={"path": "agent/example.py", "secret": "sk-secret-value"},
            self_prompt="Implement the accepted task.",
            reasoning_summary="Inspected the bounded target.",
            metadata={"chain_of_thought": "not stored"},
        ))
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/reveal",
            headers={"X-GA3BAD-CSRF": self.client.cookies.get("ga3bad_csrf")},
            json={"trace_id": trace.id, "goal_id": self.goal.id, "run_id": run.id},
        )
        self.assertEqual(response.status_code, 200, response.text)
        serialized = response.text
        self.assertNotIn("sk-secret-value", serialized)
        self.assertEqual(response.json()["chain_of_thought"], "not stored")

    def test_waiting_and_running_agents_exist_before_an_ultra_run_projection(self):
        self.store.append_event(
            "agent.scheduled",
            goal_id=self.goal.id,
            entity_type="agent",
            entity_id="waiting-reviewer",
            payload={"name": "Waiting Reviewer", "status": "waiting", "model": "small"},
        )
        self.store.append_event(
            "agent.started",
            goal_id=self.goal.id,
            entity_type="agent",
            entity_id="active-coder",
            payload={"name": "Active Coder", "status": "running", "model": "small"},
        )
        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/sections/agents"
        )
        self.assertEqual(response.status_code, 200, response.text)
        agents = {item["id"]: item for item in response.json()["agents"]}
        self.assertEqual(agents["waiting-reviewer"]["status"], "waiting")
        self.assertEqual(agents["active-coder"]["status"], "running")
        self.assertEqual(len(response.json()["scheduled"]), 1)
        self.assertEqual(response.json()["models"][0]["calls"], 2)

    def test_terminal_trace_freezes_once_and_exports_the_same_revision(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="freeze test trace",
        )
        first = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/overview"
        )
        second = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/overview"
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["state"], "FROZEN")
        self.assertEqual(
            first.json()["frozen_snapshot"]["payload_hash"],
            second.json()["frozen_snapshot"]["payload_hash"],
        )
        snapshots = self.store.list_advanced_trace_snapshots(self.goal.id)
        self.assertEqual(len(snapshots), 1)
        exported = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/advanced-tracing/export"
        )
        self.assertEqual(exported.status_code, 200, exported.text)
        self.assertEqual(exported.json()["schema"], "advanced-trace/v1")
        self.assertEqual(exported.json()["overview"]["state"], "FROZEN")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import time
from types import SimpleNamespace
import unittest
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from agent.web.app import create_app
from agent.web.registry import WebRegistry
from agent.web.service import (
    EventHub, JobScheduler, jsonable, public_runtime_event, workspace_files,
)
from agent.events import UIEvent
from agent.model_catalog import ModelDescriptor
from agent.web.telemetry import DEFAULT_INFERENCE_PROFILE, progress_estimate
from agent.web.terminal import TerminalManager, TerminalPolicyError
from agent.ui_state import (
    AttentionKind,
    AttentionOption,
    AttentionRequest,
    QuestionSessionView,
    WorkspaceUIStore,
    question_attention,
)


class _FakeController:
    def __init__(self, presentation: WorkspaceUIStore, order: list[str]) -> None:
        self.presentation = presentation
        self.order = order

    def execute(self, text: str):
        self.order.append(text)
        self.presentation.append_transcript("assistant", f"completed: {text}")
        return SimpleNamespace(status="idle")

    def resolve_attention(self, request_id: str, key: str, text: str = "") -> bool:
        request = self.presentation.active_attention()
        if request is None or request.id != request_id:
            return False
        return self.presentation.resolve_attention(key, text=text)


class _FakeRuntime:
    def __init__(self) -> None:
        self.plan = None
        self.status = "idle"

    intent_architect = SimpleNamespace(
        assess_complexity=lambda _text: SimpleNamespace(ultra_required=False)
    )

    def active_goal(self):
        return None

    def dashboard(self):
        return SimpleNamespace(status=self.status)

    def latest_plan(self):
        return self.plan

    def checkpoint_interrupt(self):
        return None


class _FakeState:
    def get_latest_goal(self):
        return None

    def list_agent_registry(self):
        return []


class _FakeRuntimes:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.sessions = {}
        self._sessions = self.sessions

    def get(self, thread_id: str):
        if thread_id not in self.sessions:
            presentation = WorkspaceUIStore()
            self.sessions[thread_id] = SimpleNamespace(
                project_id="project",
                thread_id=thread_id,
                presentation=presentation,
                controller=_FakeController(presentation, self.order),
                runtime=_FakeRuntime(),
                store=_FakeState(),
            )
        return self.sessions[thread_id]

    def cached(self, thread_id: str):
        return self.sessions.get(thread_id)

    def effective_access(self, thread_id: str):
        return "normal"


class WebRegistryTests(unittest.TestCase):
    def test_web_questions_are_structured_and_offer_one_click_defaults(self) -> None:
        session = QuestionSessionView(
            "intake",
            (
                {
                    "id": "platform", "header": "Platform",
                    "question": "Where should it run?",
                    "options": (
                        {"label": "Desktop", "description": "Desktop browser", "recommended": True},
                        {"label": "Mobile", "description": "Mobile browser"},
                        {"label": "Both", "description": "Responsive"},
                    ),
                },
                {"id": "style", "header": "Style", "question": "Which style?", "options": ()},
            ),
            {},
        )
        attention = question_attention(session)
        self.assertEqual(attention.details, "Decision 1 of 2")
        self.assertEqual(attention.title, "Platform")
        self.assertIn("recommended-all", {item.key for item in attention.options})
        rendered = " ".join(item.label for item in attention.options)
        self.assertNotIn("/answer", rendered)
        self.assertNotIn("press D", rendered)

    def test_runtime_events_never_publish_model_scratch_text(self) -> None:
        self.assertIsNone(public_runtime_event(UIEvent(
            "model_thought", "We need to inspect and then maybe do something."
        )))
        kind, summary, data = public_runtime_event(UIEvent(
            "tool_result", "(no files under '.')", {"tool": "list_files", "actor": "planner"}
        ))
        self.assertEqual(kind, "search")
        self.assertEqual(summary, "Workspace inspected · empty greenfield project")
        self.assertNotIn("no files under", summary)
        self.assertEqual(data["tool"], "list_files")

    def test_drafts_use_revisions_and_survive_registry_restart(self) -> None:
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "web.db"
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            registry = WebRegistry(database)
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Draft")
            first = registry.save_draft(thread["id"], "queued idea", expected_revision=0)
            self.assertEqual(first["revision"], 1)
            with self.assertRaisesRegex(ValueError, "stale_draft"):
                registry.save_draft(thread["id"], "overwrite", expected_revision=0)
            registry.close()
            reopened = WebRegistry(database)
            self.assertEqual(reopened.get_draft(thread["id"])["text"], "queued idea")
            reopened.close()

    def test_same_task_queue_is_bounded_fifo_idempotent_and_cancelable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Queue")
            other = registry.create_thread(project["id"], "Other")
            scheduler = JobScheduler(registry, _FakeRuntimes(), EventHub())
            scheduler._stop.set()
            scheduler._worker.join(timeout=2)
            active, _ = registry.create_job(project["id"], thread["id"], kind="message", text="active", client_request_id="active-request")
            registry.update_job(active["id"], "running")
            scheduler._active_job_id = active["id"]
            try:
                queued = []
                for index in range(10):
                    job, created = scheduler.submit(thread["id"], "message", f"item {index}", f"queue-{index}")
                    self.assertTrue(created)
                    queued.append(job)
                duplicate, created = scheduler.submit(thread["id"], "message", "item 0", "queue-0")
                self.assertFalse(created)
                self.assertEqual(duplicate["id"], queued[0]["id"])
                self.assertEqual([job["input_text"] for job in registry.thread_queue(thread["id"])], [f"item {index}" for index in range(10)])
                with self.assertRaisesRegex(OverflowError, "queue_full"):
                    scheduler.submit(thread["id"], "message", "eleventh", "queue-10")
                with self.assertRaisesRegex(RuntimeError, "another_thread_is_running"):
                    scheduler.submit(other["id"], "message", "cross task", "other-1")
                cancelled = scheduler.cancel_queued(thread["id"], queued[3]["id"])
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertEqual(len(registry.thread_queue(thread["id"])), 9)
            finally:
                scheduler._active_job_id = None
                scheduler.close()
                registry.close()

    def test_new_task_cannot_hijack_another_tasks_unfinished_workspace_goal(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            owner = registry.create_thread(project["id"], "Owning task")
            registry.update_thread(owner["id"], goal_id="goal-active", status="awaiting_plan_approval")
            fresh = registry.create_thread(project["id"], "New task")
            scheduler = JobScheduler(registry, _FakeRuntimes(), EventHub())
            scheduler._stop.set()
            scheduler._worker.join(timeout=2)
            try:
                with self.assertRaisesRegex(RuntimeError, "workspace_goal_owned_by_other_thread"):
                    scheduler.submit(fresh["id"], "message", "unrelated work", "fresh-request")
                snapshot = scheduler.thread_snapshot(fresh["id"])
                self.assertIsNone(snapshot["dashboard"])
                self.assertFalse(snapshot["capabilities"]["send"]["allowed"])
                self.assertIn("Owning task", snapshot["capabilities"]["send"]["reason"])
                registry.update_thread(owner["id"], status="paused")
                queued, created = scheduler.submit(
                    fresh["id"], "message", "work while owner is paused", "fresh-after-pause"
                )
                self.assertTrue(created)
                self.assertEqual(queued["status"], "queued")
                self.assertTrue(scheduler.capabilities(fresh["id"])["send"]["allowed"])
            finally:
                scheduler.close()
                registry.close()

    def test_running_job_recovers_as_explicit_recovery_required(self) -> None:
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "web.db"
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            registry = WebRegistry(database)
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Recovery")
            job, _ = registry.create_job(project["id"], thread["id"], kind="message", text="work", client_request_id="recover-1")
            registry.update_job(job["id"], "running")
            registry.update_thread(thread["id"], status="running")
            registry.close()
            recovered = WebRegistry(database)
            recovered.recover_jobs()
            self.assertEqual(recovered.get_job(job["id"])["status"], "recovery_required")
            self.assertEqual(recovered.get_thread(thread["id"])["status"], "recovery_required")
            recovered.close()

    def test_resume_is_a_control_event_not_a_user_prompt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Resume")
            interrupted, _ = registry.create_job(
                project["id"], thread["id"], kind="message", text="restore this intake",
                client_request_id="interrupted-control-1",
            )
            registry.update_job(interrupted["id"], "running")
            registry.update_thread(thread["id"], status="running")
            scheduler = JobScheduler(registry, _FakeRuntimes(), EventHub())
            scheduler._stop.set()
            scheduler._worker.join(timeout=2)
            try:
                job, created = scheduler.resume(thread["id"], "resume-control-1")
                self.assertTrue(created)
                self.assertEqual(job["input_text"], "restore this intake")
                self.assertEqual(registry.list_messages(thread["id"]), [])
                self.assertEqual(registry.list_activity(thread["id"])[0]["kind"], "resume")
            finally:
                scheduler.close()
                registry.close()

    def test_terminal_default_is_restricted_and_root_guards_are_immutable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Terminal")
            access = {thread["id"]: "normal"}
            manager = TerminalManager(registry, lambda thread_id: access[thread_id])
            session = manager.open(thread["id"], workspace)
            result = manager.execute(session["id"], "git status")
            self.assertIn(result["returncode"], {0, 128})
            child = workspace / "src"
            child.mkdir()
            changed = manager.execute(session["id"], "cd src")
            self.assertEqual(Path(changed["cwd"]), child)
            with self.assertRaisesRegex(TerminalPolicyError, "inspection and verification"):
                manager.execute(session["id"], "echo mutate > file.txt")
            with self.assertRaisesRegex(TerminalPolicyError, "Command chaining"):
                manager.execute(session["id"], "git status & echo mutate")
            with self.assertRaisesRegex(TerminalPolicyError, "inside the selected workspace"):
                manager.execute(session["id"], "cd ..\\..")
            access[thread["id"]] = "host"
            with self.assertRaisesRegex(TerminalPolicyError, "protected system or root"):
                manager.execute(session["id"], "Remove-Item -Recurse C:\\")
            self.assertEqual(manager.close(session["id"])["status"], "closed")
            registry.close()

    def test_model_descriptor_serialization_handles_immutable_metadata(self) -> None:
        value = jsonable(ModelDescriptor("ollama", "qwen", "local", metadata={"size": 7}))
        self.assertEqual(value["id"].split("@", 1)[0], "ollama:qwen")
        self.assertEqual(value["metadata"], {"size": 7})

    def test_progress_estimate_uses_completed_milestones_and_reports_a_range(self) -> None:
        started = datetime.now(timezone.utc) - timedelta(hours=1)
        result = progress_estimate(
            {
                "status": "in_progress",
                "tasks": [
                    {"id": "T1", "title": "Inspect", "status": "done"},
                    {"id": "T2", "title": "Implement", "status": "in_progress"},
                    {"id": "T3", "title": "Verify", "status": "pending"},
                ],
            },
            {"activity": {"stage": "executing", "summary": "Implementing"}},
            {"status": "running", "started_at": started.isoformat()},
            (),
            workflow_mode="normal",
            profile=DEFAULT_INFERENCE_PROFILE,
        )
        self.assertGreater(result["percent"], 33)
        self.assertEqual(result["current_step"], "Implement")
        self.assertEqual(result["confidence"], "medium")
        self.assertGreater(result["remaining_seconds_high"], result["remaining_seconds_low"])
        self.assertIsNotNone(result["estimated_finish_at"])

    def test_existing_registry_is_migrated_to_intelligent_mode_columns(self) -> None:
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "legacy.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE projects (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, path_key TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_opened_at TEXT NOT NULL
                );
                CREATE TABLE threads (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    session_id TEXT NOT NULL, goal_id TEXT, title TEXT NOT NULL, status TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(project_id, session_id)
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES threads(id), role TEXT NOT NULL,
                    content TEXT NOT NULL, technical INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.close()

            registry = WebRegistry(database)
            thread_columns = {row[1] for row in registry._connection.execute("PRAGMA table_info(threads)")}
            message_columns = {row[1] for row in registry._connection.execute("PRAGMA table_info(messages)")}
            self.assertTrue({"workflow_mode", "access_policy", "model_overrides_json", "state_revision"}.issubset(thread_columns))
            self.assertTrue({"turn_id", "direction"}.issubset(message_columns))
            self.assertFalse(registry.settings()["reduced_motion"])
            registry.close()

    def test_workspace_file_index_is_bounded_and_skips_private_state(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "app.tsx").write_text("export default 1", encoding="utf-8")
            (root / ".coding-agent").mkdir()
            (root / ".coding-agent" / "state.db").write_bytes(b"private")
            (root / ".env").write_text("TOKEN=private", encoding="utf-8")
            files = workspace_files(root)
            self.assertEqual([item["path"] for item in files], ["src/app.tsx"])

    def test_project_threads_messages_and_idempotent_jobs_are_durable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, created = registry.add_project(str(workspace))
            self.assertTrue(created)
            same, created_again = registry.add_project(str(workspace / "."))
            self.assertFalse(created_again)
            self.assertEqual(same["id"], project["id"])
            thread = registry.create_thread(project["id"], "Build the UI")
            registry.append_message(thread["id"], "user", "hello")
            first, was_created = registry.create_job(
                project["id"], thread["id"], kind="message", text="hello",
                client_request_id="request-0001",
            )
            second, was_created_again = registry.create_job(
                project["id"], thread["id"], kind="message", text="hello",
                client_request_id="request-0001",
            )
            self.assertTrue(was_created)
            self.assertFalse(was_created_again)
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(registry.list_messages(thread["id"])[0]["content"], "hello")
            registry.close()

    def test_turn_activity_direction_and_thread_revision_are_durable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Arabic output")
            changed = registry.update_thread(
                thread["id"], workflow_mode="plan", access_policy="bounded",
                model_overrides={"router": "ollama:qwen"},
            )
            self.assertEqual(changed["workflow_mode"], "plan")
            self.assertEqual(changed["state_revision"], 2)
            self.assertEqual(changed["model_overrides"]["router"], "ollama:qwen")

            user_id = registry.append_message(thread["id"], "user", "راجع الخطة", turn_id="job-1")
            assistant_id = registry.append_message(thread["id"], "assistant", "هذه الخطة", turn_id="job-1")
            with self.assertRaisesRegex(ValueError, "assistant output"):
                registry.update_message_direction(thread["id"], user_id, "rtl")
            assistant = registry.update_message_direction(thread["id"], assistant_id, "rtl")
            self.assertEqual(assistant["direction"], "rtl")
            registry.append_activity(thread["id"], "job-1", "command", "Ran safe validation")
            turns = registry.conversation_turns(thread["id"])
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["messages"][1]["direction"], "rtl")
            self.assertEqual(turns[0]["activity"][0]["kind"], "command")
            registry.close()

    def test_scheduler_serializes_jobs_and_resolves_exact_attention(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("workspace", encoding="utf-8")
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            first = registry.create_thread(project["id"], "First")
            second = registry.create_thread(project["id"], "Second")
            runtimes = _FakeRuntimes()
            scheduler = JobScheduler(registry, runtimes, EventHub())
            files = scheduler.inspector(first["id"], "files")
            self.assertEqual(files["files"][0]["path"], "README.md")
            self.assertEqual(scheduler.file_preview(first["id"], "README.md")["content"], "workspace")
            (workspace / ".env").write_text("SECRET=private", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file_not_previewable"):
                scheduler.file_preview(first["id"], ".env")
            self.assertIn("Diff view needs", scheduler.inspector(first["id"], "changes")["diff"])
            one, _ = scheduler.submit(first["id"], "message", "one", "request-0001")
            two, _ = scheduler.submit(second["id"], "message", "two", "request-0002")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if registry.get_job(one["id"])["status"] == "completed" and registry.get_job(two["id"])["status"] == "completed":
                    break
                time.sleep(0.02)
            self.assertEqual(runtimes.order, ["one", "two"])
            session = runtimes.get(first["id"])
            session.presentation.present_attention(AttentionRequest(
                id="approval-1",
                kind=AttentionKind.APPROVAL,
                title="Allow?",
                options=(AttentionOption("yes", "Allow", "allow_once"),),
            ))
            self.assertFalse(scheduler.resolve_attention(first["id"], "stale", "yes", ""))
            self.assertTrue(scheduler.resolve_attention(first["id"], "approval-1", "yes", ""))
            scheduler.close()
            registry.close()

    def test_attention_releases_slot_for_another_task_then_resumes_exact_worker(self) -> None:
        class WaitingController(_FakeController):
            def execute(self, text: str):
                self.order.append(f"{text}:started")
                resolution = self.presentation.request_attention(AttentionRequest(
                    id="decision-release-slot",
                    kind=AttentionKind.QUESTION,
                    title="Choose",
                    options=(AttentionOption("continue", "Continue", "continue"),),
                ))
                self.order.append(f"{text}:answered:{resolution.value}")
                return SimpleNamespace(status="idle")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            waiting = registry.create_thread(project["id"], "Waiting")
            other = registry.create_thread(project["id"], "Other")
            runtimes = _FakeRuntimes()
            waiting_session = runtimes.get(waiting["id"])
            waiting_session.controller = WaitingController(waiting_session.presentation, runtimes.order)
            scheduler = JobScheduler(registry, runtimes, EventHub())
            try:
                first, _ = scheduler.submit(waiting["id"], "message", "first", "release-slot-1")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if registry.get_job(first["id"])["status"] == "paused":
                        break
                    time.sleep(0.02)
                self.assertIsNone(scheduler.active_job_id)
                self.assertEqual(registry.get_thread(waiting["id"])["status"], "waiting_for_input")

                second, _ = scheduler.submit(other["id"], "message", "second", "release-slot-2")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if registry.get_job(second["id"])["status"] == "completed":
                        break
                    time.sleep(0.02)
                self.assertEqual(registry.get_job(second["id"])["status"], "completed")
                self.assertEqual(registry.get_job(first["id"])["status"], "paused")

                self.assertTrue(scheduler.resolve_attention(
                    waiting["id"], "decision-release-slot", "continue", ""
                ))
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if registry.get_job(first["id"])["status"] == "completed":
                        break
                    time.sleep(0.02)
                self.assertEqual(registry.get_job(first["id"])["status"], "completed")
                self.assertEqual(
                    runtimes.order,
                    ["first:started", "second", "first:answered:continue"],
                )
            finally:
                scheduler.close()
                registry.close()

    def test_kill_task_cancels_waiting_worker_and_its_queue_without_checkpoint(self) -> None:
        class WaitingController(_FakeController):
            def execute(self, text: str):
                self.order.append(text)
                self.presentation.request_attention(AttentionRequest(
                    id="decision-kill",
                    kind=AttentionKind.QUESTION,
                    title="Wait",
                    options=(AttentionOption("continue", "Continue", "continue"),),
                ))
                return SimpleNamespace(status="idle")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Kill me")
            runtimes = _FakeRuntimes()
            session = runtimes.get(thread["id"])
            session.controller = WaitingController(session.presentation, runtimes.order)
            scheduler = JobScheduler(registry, runtimes, EventHub())
            try:
                active, _ = scheduler.submit(thread["id"], "message", "active", "kill-active")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if registry.get_job(active["id"])["status"] == "paused":
                        break
                    time.sleep(0.02)
                queued, _ = scheduler.submit(thread["id"], "message", "queued", "kill-queued")
                result = scheduler.kill_task(thread["id"])
                self.assertFalse(result["safe_checkpoint"])
                self.assertEqual(registry.get_thread(thread["id"])["status"], "killed")
                self.assertEqual(registry.get_job(active["id"])["status"], "cancelled")
                self.assertEqual(registry.get_job(queued["id"])["status"], "cancelled")
                self.assertIsNone(session.presentation.active_attention())
                with self.assertRaisesRegex(RuntimeError, "task_killed_create_new"):
                    scheduler.submit(thread["id"], "message", "again", "kill-again")
            finally:
                scheduler.close()
                registry.close()

    def test_plan_decision_is_fingerprint_bound_idempotent_and_feedback_revises(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            registry = WebRegistry(root / "web.db")
            project, _ = registry.add_project(str(workspace))
            thread = registry.create_thread(project["id"], "Plan lifecycle")
            registry.update_thread(thread["id"], workflow_mode="plan")
            runtimes = _FakeRuntimes()
            session = runtimes.get(thread["id"])
            session.runtime.plan = SimpleNamespace(revision=2, fingerprint="f" * 64)
            session.runtime.status = "awaiting_plan_approval"
            scheduler = JobScheduler(registry, runtimes, EventHub())
            try:
                with self.assertRaisesRegex(ValueError, "stale_plan"):
                    scheduler.plan_decision(
                        thread["id"], action="implement", revision=1,
                        fingerprint="x" * 64, feedback="", client_request_id="decision-stale",
                    )
                feedback, created = scheduler.plan_decision(
                    thread["id"], action="keep_planning", revision=2,
                    fingerprint="f" * 64, feedback="Add a migration test",
                    client_request_id="decision-feedback",
                )
                self.assertTrue(created)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and registry.get_job(feedback["id"])["status"] != "completed":
                    time.sleep(0.02)
                self.assertIn("/replan Add a migration test", runtimes.order)

                implement, first = scheduler.plan_decision(
                    thread["id"], action="implement", revision=2,
                    fingerprint="f" * 64, feedback="", client_request_id="decision-implement",
                )
                duplicate, second = scheduler.plan_decision(
                    thread["id"], action="implement", revision=2,
                    fingerprint="f" * 64, feedback="", client_request_id="decision-implement",
                )
                self.assertTrue(first)
                self.assertFalse(second)
                self.assertEqual(implement["id"], duplicate["id"])
            finally:
                scheduler.close()
                registry.close()


class WebApiTests(unittest.TestCase):
    def test_advanced_inference_settings_and_resource_telemetry(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "project"
            workspace.mkdir()
            app = create_app(registry_path=root / "web.db", launch_token="launch-token", testing=True)
            with TestClient(app) as client:
                exchange = client.get("/?token=launch-token", follow_redirects=False)
                client.cookies.update(exchange.cookies)
                csrf = client.get("/api/v1/bootstrap").json()["csrf_token"]
                headers = {"x-ga3bad-csrf": csrf}
                project = client.post("/api/v1/projects", json={"path": str(workspace)}, headers=headers).json()["project"]
                thread = client.post(f"/api/v1/projects/{project['id']}/threads", json={"title": "Long local task"}, headers=headers).json()
                profile = client.get("/api/v1/settings/advanced")
                self.assertEqual(profile.status_code, 200)
                updated = {
                    **profile.json(), "device": "cpu", "cpu_threads": 3,
                    "top_k": 22, "work_quantum_steps": 31, "ultra_max_depth": 6,
                }
                saved = client.patch("/api/v1/settings/advanced", json=updated, headers=headers)
                self.assertEqual(saved.status_code, 200)
                self.assertEqual(saved.json()["cpu_threads"], 3)
                self.assertEqual(app.state.runtimes.runtime_config().work_quantum_steps, 31)
                self.assertEqual(app.state.runtimes.runtime_config().ultra_max_depth, 6)
                telemetry = client.get(f"/api/v1/threads/{thread['id']}/telemetry")
                self.assertEqual(telemetry.status_code, 200)
                self.assertIn("ram", telemetry.json())
                self.assertEqual(telemetry.json()["context"]["limit_tokens"], updated["context_window"])

    def test_mode_direction_and_task_scoped_host_full_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "project"
            workspace.mkdir()
            database = root / "web.db"
            app = create_app(registry_path=database, launch_token="launch-token", testing=True)
            thread_id = ""
            with TestClient(app) as client:
                exchange = client.get("/?token=launch-token", follow_redirects=False)
                client.cookies.update(exchange.cookies)
                csrf = client.get("/api/v1/bootstrap").json()["csrf_token"]
                headers = {"x-ga3bad-csrf": csrf}
                project = client.post("/api/v1/projects", json={"path": str(workspace)}, headers=headers).json()["project"]
                thread = client.post(
                    f"/api/v1/projects/{project['id']}/threads",
                    json={"title": "Modes"}, headers=headers,
                ).json()
                thread_id = thread["id"]

                mode = client.post(
                    f"/api/v1/threads/{thread_id}/mode",
                    json={"mode": "plan", "expected_revision": 1}, headers=headers,
                )
                self.assertEqual(mode.status_code, 200)
                self.assertEqual(mode.json()["workflow_mode"], "plan")
                stale = client.post(
                    f"/api/v1/threads/{thread_id}/mode",
                    json={"mode": "normal", "expected_revision": 1}, headers=headers,
                )
                self.assertEqual(stale.status_code, 409)
                self.assertEqual(stale.json()["detail"]["code"], "stale_thread")

                user_id = app.state.registry.append_message(thread_id, "user", "keep user direction")
                assistant_id = app.state.registry.append_message(thread_id, "assistant", "رسالة عربية")
                assistant = client.patch(
                    f"/api/v1/threads/{thread_id}/messages/{assistant_id}",
                    json={"direction": "rtl"}, headers=headers,
                )
                self.assertEqual(assistant.status_code, 200)
                self.assertEqual(assistant.json()["direction"], "rtl")
                denied = client.patch(
                    f"/api/v1/threads/{thread_id}/messages/{user_id}",
                    json={"direction": "rtl"}, headers=headers,
                )
                self.assertEqual(denied.status_code, 409)

                prepared = client.post(
                    f"/api/v1/threads/{thread_id}/access",
                    json={"policy": "host", "expected_revision": 2}, headers=headers,
                )
                self.assertEqual(prepared.status_code, 200)
                token = prepared.json()["confirmation_token"]
                enabled = client.post(
                    f"/api/v1/threads/{thread_id}/access",
                    json={"policy": "host", "confirmation_token": token, "expected_revision": 2},
                    headers=headers,
                )
                self.assertEqual(enabled.status_code, 200)
                self.assertEqual(enabled.json()["thread"]["effective_access"], "host")
                self.assertEqual(enabled.json()["thread"]["access_policy"], "default")

            restarted = create_app(registry_path=database, launch_token="next-token", testing=True)
            with TestClient(restarted) as client:
                exchange = client.get("/?token=next-token", follow_redirects=False)
                client.cookies.update(exchange.cookies)
                snapshot = client.get(f"/api/v1/threads/{thread_id}")
                self.assertEqual(snapshot.status_code, 200)
                self.assertEqual(snapshot.json()["thread"]["effective_access"], "normal")

    def test_local_session_csrf_project_thread_and_websocket_snapshot(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "project"
            workspace.mkdir()
            app = create_app(
                registry_path=root / "web.db",
                launch_token="launch-token",
                testing=True,
            )
            with TestClient(app) as client:
                exchange = client.get("/?token=launch-token", follow_redirects=False)
                self.assertEqual(exchange.status_code, 303)
                client.cookies.update(exchange.cookies)
                bootstrap = client.get("/api/v1/bootstrap")
                self.assertEqual(bootstrap.status_code, 200)
                csrf = bootstrap.json()["csrf_token"]
                denied = client.post("/api/v1/projects", json={"path": str(workspace)})
                self.assertEqual(denied.status_code, 403)
                created = client.post(
                    "/api/v1/projects",
                    json={"path": str(workspace)},
                    headers={"x-ga3bad-csrf": csrf},
                )
                self.assertEqual(created.status_code, 201)
                project_id = created.json()["project"]["id"]
                thread = client.post(
                    f"/api/v1/projects/{project_id}/threads",
                    json={"title": "Web task"},
                    headers={"x-ga3bad-csrf": csrf},
                )
                self.assertEqual(thread.status_code, 201)
                snapshot = client.get(f"/api/v1/threads/{thread.json()['id']}")
                self.assertEqual(snapshot.status_code, 200)
                self.assertEqual(snapshot.json()["thread"]["title"], "Web task")
                with client.websocket_connect("/api/v1/ws") as websocket:
                    event = websocket.receive_json()
                    self.assertEqual(event["type"], "app.snapshot")
                    self.assertEqual(event["version"], 1)

    def test_origin_must_match_the_active_loopback_host(self) -> None:
        with TemporaryDirectory() as temporary:
            app = create_app(
                registry_path=Path(temporary) / "web.db",
                launch_token="launch-token",
                testing=True,
            )
            with TestClient(app, base_url="http://127.0.0.1:8765") as client:
                accepted = client.get(
                    "/health", headers={"origin": "http://127.0.0.1:8765"}
                )
                rejected = client.get(
                    "/health", headers={"origin": "http://attacker.example"}
                )
                self.assertEqual(accepted.status_code, 200)
                self.assertEqual(rejected.status_code, 403)


if __name__ == "__main__":
    unittest.main()

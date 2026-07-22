from __future__ import annotations

import unittest
import time
from threading import Thread
from types import SimpleNamespace
from unittest import mock

from agent.ui_state import (
    ActivityStage,
    AttentionKind,
    AttentionOption,
    AttentionRequest,
    ExperienceMode,
    PresentationEvent,
    PresentationLifecycle,
    WorkspaceUIStore,
    answer_question,
    answer_recommended_remaining,
    is_recommended_defaults_utterance,
    question_session,
)


def _question(identifier: str) -> dict:
    return {
        "id": identifier,
        "header": f"Decision {identifier}",
        "question": f"Choose {identifier}",
        "options": (
            {"label": "Recommended", "description": "Best default", "recommended": True},
            {"label": "Alternative", "description": "Another choice"},
            {"label": "Advanced", "description": "More control"},
        ),
    }


class _Runtime:
    def __init__(self, source: str = "plan") -> None:
        self.source = source
        self.questions = [_question("q1"), _question("q2"), _question("q3")]
        self.answers: dict[str, str] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.goal = SimpleNamespace(
            metadata={
                "plan_answers": self.answers,
                "ultra_run_id": "run-1" if source == "ultra" else "",
            }
        )

    def intake_questions(self):
        return tuple(self.questions) if self.source == "intake" else ()

    def active_goal(self):
        return None if self.source == "intake" else self.goal

    def plan_questions(self):
        return tuple({**item, "answer": self.answers.get(item["id"])} for item in self.questions)

    def ultra_questions(self):
        return tuple(self.questions)

    def _answer(self, source: str, question_id: str, value: str):
        self.calls.append((source, question_id, value))
        self.answers[question_id] = value
        if self.source == "intake":
            self.questions = [item for item in self.questions if item["id"] != question_id]
        return question_id

    def answer_intake_question(self, question_id: str, value: str):
        return self._answer("intake", question_id, value)

    def answer_plan_question(self, question_id: str, value: str):
        return self._answer("plan", question_id, value)

    def answer_ultra_question(self, question_id: str, value: str):
        return self._answer("ultra", question_id, value)


class QuestionSessionTests(unittest.TestCase):
    def test_every_question_source_has_the_same_current_decision_contract(self):
        for source in ("intake", "plan", "ultra"):
            with self.subTest(source=source):
                runtime = _Runtime(source)
                session = question_session(runtime)
                self.assertIsNotNone(session)
                assert session is not None
                self.assertEqual(session.source, source)
                self.assertEqual(session.current["id"], "q1")
                self.assertEqual(session.completed, 0)
                answer_question(runtime, session, "q1", "Recommended")
                self.assertEqual(runtime.calls[-1], (source, "q1", "Recommended"))

    def test_recommended_defaults_answer_every_remaining_decision(self):
        runtime = _Runtime("plan")

        results = answer_recommended_remaining(runtime)

        self.assertEqual(results, ("q1", "q2", "q3"))
        self.assertEqual(
            runtime.calls,
            [
                ("plan", "q1", "1"),
                ("plan", "q2", "1"),
                ("plan", "q3", "1"),
            ],
        )
        self.assertIsNone(question_session(runtime))

    def test_defaults_request_accepts_natural_english_and_arabic(self):
        self.assertTrue(is_recommended_defaults_utterance("continue with recommended"))
        self.assertTrue(is_recommended_defaults_utterance("كمل بالمقترحات"))
        self.assertFalse(is_recommended_defaults_utterance("Use the scientific option"))


class WorkspaceStoreTests(unittest.TestCase):
    def test_simple_is_default_and_f2_state_is_reversible(self):
        store = WorkspaceUIStore()
        self.assertEqual(store.snapshot().mode, ExperienceMode.SIMPLE)
        self.assertEqual(store.toggle_mode(), ExperienceMode.ADVANCED)
        self.assertEqual(store.toggle_mode(), ExperienceMode.SIMPLE)

    def test_auto_locale_follows_the_first_user_language(self):
        store = WorkspaceUIStore()
        store.observe_user_text("اعمل لي آلة حاسبة")
        self.assertEqual(store.snapshot().locale, "ar")
        store.observe_user_text("later English guidance does not flip it")
        self.assertEqual(store.snapshot().locale, "ar")

    def test_runtime_events_reduce_to_one_active_cell_and_keep_raw_details(self):
        store = WorkspaceUIStore()
        store.handle_event("tool_call", "write_file", {"tool": "write_file"})
        writing = store.snapshot()
        self.assertEqual(writing.activity.stage, ActivityStage.BUILDING)
        self.assertTrue(writing.running)
        self.assertEqual(writing.transcript, ())
        self.assertIn("tool_call: write_file", writing.advanced_log)

        store.handle_event("tool_result", "updated", {"tool": "write_file"})
        self.assertIn("write file", store.snapshot().activity.last_success)

    def test_retry_wait_and_recoverable_errors_do_not_claim_the_goal_is_paused(self):
        store = WorkspaceUIStore()
        store.set_activity(ActivityStage.BUILDING, "Working", running=True)
        store.handle_event("retry_wait", "Retry 2 in 3.0s", {"delay_ms": 3000})
        waiting = store.snapshot()
        self.assertEqual(waiting.activity.stage, ActivityStage.CHECKING)
        self.assertTrue(waiting.running)

        store.handle_event("error", "temporary provider outage")
        failed_attempt = store.snapshot()
        self.assertTrue(failed_attempt.running)
        self.assertNotEqual(failed_attempt.activity.stage, ActivityStage.PAUSED)

        store.handle_event("checkpoint", "Context compacted", {"continues": True})
        self.assertTrue(store.snapshot().running)
        store.handle_event("checkpoint", "Stopped", {"paused": True})
        self.assertFalse(store.snapshot().running)
        self.assertEqual(store.snapshot().activity.stage, ActivityStage.PAUSED)

    def test_nonblocking_attention_can_be_polled_by_the_controller(self):
        store = WorkspaceUIStore()
        request = AttentionRequest(
            id="slow-work",
            kind=AttentionKind.RECOVERY,
            title="Still working",
            options=(AttentionOption("keep", "Keep waiting", "keep"),),
        )
        event = store.present_attention(request)
        self.assertFalse(event.is_set())
        self.assertIsNone(store.take_attention_result(request.id))
        store.resolve_attention("keep")
        self.assertTrue(event.is_set())
        self.assertEqual(store.take_attention_result(request.id).value, "keep")

    def test_attention_waits_for_an_explicit_decision(self):
        store = WorkspaceUIStore()
        request = AttentionRequest(
            id="approval-1",
            kind=AttentionKind.APPROVAL,
            title="Allow test?",
            options=(
                AttentionOption("yes", "Yes", "allow_once", shortcut="y"),
                AttentionOption("no", "No", "deny", shortcut="n"),
            ),
        )
        answers = []
        worker = Thread(target=lambda: answers.append(store.request_attention(request)))
        worker.start()
        time.sleep(0.02)
        self.assertTrue(worker.is_alive())
        self.assertEqual(store.active_attention(), request)
        store.move_attention(1)
        self.assertTrue(store.resolve_selected_attention())
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(answers[0].value, "deny")

    def test_ui_shutdown_is_not_reported_as_a_user_denial(self):
        store = WorkspaceUIStore()
        request = AttentionRequest(
            id="approval-exit",
            kind=AttentionKind.APPROVAL,
            title="Allow?",
            options=(AttentionOption("deny", "Deny", "deny"),),
        )
        answers = []
        worker = Thread(target=lambda: answers.append(store.request_attention(request)))
        worker.start()
        time.sleep(0.02)
        store.mark_exit()
        worker.join(1)
        self.assertEqual(answers[0].value, "ui_error")

    def test_dashboard_progress_learns_an_approximate_eta_only_after_observed_work(self):
        store = WorkspaceUIStore()
        pending = SimpleNamespace(id="a", title="Inspect", status="in_progress")
        later = SimpleNamespace(id="b", title="Build", status="pending")
        view = SimpleNamespace(
            objective="Upgrade TUI", plan_revision=1, status="running",
            goal_attempt=0, retry_reason="", tasks=(pending, later),
        )
        with mock.patch("agent.ui_state.time.monotonic", side_effect=(10.0, 20.0, 20.0)):
            store.sync_dashboard(view)
            learning = store.snapshot().progress
            self.assertIsNone(learning.eta_low_seconds)
            pending.status = "done"
            store.sync_dashboard(view)
        learned = store.snapshot().progress
        self.assertEqual((learned.completed, learned.remaining), (1, 1))
        self.assertIsNotNone(learned.eta_low_seconds)
        self.assertGreater(learned.eta_high_seconds, learned.eta_low_seconds)

    def test_sleep_auto_resolves_only_explicit_safe_recommended_questions(self):
        store = WorkspaceUIStore()
        store.set_sleep_mode(True)
        safe = AttentionRequest(
            id="safe",
            kind=AttentionKind.QUESTION,
            title="Continue waiting?",
            options=(
                AttentionOption(
                    "keep", "Keep waiting", "keep", recommended=True, auto_safe=True
                ),
                AttentionOption("stop", "Stop", "stop"),
            ),
            auto_resolve_safe=True,
        )
        event = store.present_attention(safe)
        self.assertTrue(event.is_set())
        answer = store.take_attention_result("safe")
        self.assertEqual(answer.origin, "sleep")
        self.assertIn("Keep waiting", store.snapshot().sleep_log[-1])
        self.assertFalse(
            any("Keep waiting" in item.text for item in store.snapshot().transcript)
        )

        unsafe = AttentionRequest(
            id="unsafe",
            kind=AttentionKind.APPROVAL,
            title="Delete files?",
            options=(
                AttentionOption(
                    "yes", "Delete", "yes", recommended=True, auto_safe=True
                ),
                AttentionOption("no", "Deny", "no", primary=True),
            ),
            default_key="no",
            cancel_key="no",
            auto_resolve_safe=True,
        )
        event = store.present_attention(unsafe)
        self.assertFalse(event.is_set())
        self.assertEqual(store.active_attention(), unsafe)
        store.cancel_attention()
        self.assertEqual(store.take_attention_result("unsafe").value, "no")

    def test_invalid_choice_preserves_decision_and_escape_uses_safe_cancel(self):
        store = WorkspaceUIStore()
        request = AttentionRequest(
            id="strict",
            kind=AttentionKind.PLAN_REVIEW,
            title="Start?",
            options=(
                AttentionOption("start", "Start", "start", recommended=True),
                AttentionOption("pause", "Keep paused", "pause"),
            ),
            default_key="pause",
            cancel_key="pause",
        )
        store.present_attention(request)
        self.assertFalse(store.resolve_attention("missing"))
        self.assertEqual(store.active_attention(), request)
        self.assertIn("Invalid choice", store.snapshot().attention_feedback)
        self.assertTrue(store.cancel_attention())
        self.assertEqual(store.take_attention_result("strict").value, "pause")

    def test_context_capacity_and_log_coalescing_remain_truthful(self):
        store = WorkspaceUIStore()
        store.handle_event("usage", data={"input_tokens": 1_200})
        self.assertIsNone(store.snapshot().resources.context_remaining_tokens)
        store.set_context_window(4_000)
        self.assertEqual(store.snapshot().resources.context_remaining_tokens, 2_800)
        store.append_log("polling worker")
        store.append_log("polling worker")
        self.assertEqual(store.snapshot().log_entries[-1].count, 2)
        store.append_log("error: first")
        store.append_log("error: first")
        self.assertEqual(store.snapshot().log_entries[-2].category, "error")
        self.assertEqual(store.snapshot().log_entries[-1].count, 1)

    def test_stream_finalization_is_idempotent(self):
        store = WorkspaceUIStore()
        store.handle_event("model_text", "Final result")
        store.finalize_stream()
        store.finalize_stream()
        messages = [item.text for item in store.snapshot().transcript]
        self.assertEqual(messages.count("Final result"), 1)

    def test_failed_stream_attempt_is_discarded_before_manual_retry(self):
        store = WorkspaceUIStore()
        store.handle_event("model_text", "partial answer")
        store.finalize_stream(commit=False)
        store.finalize_stream(commit=False)

        self.assertFalse(store.snapshot().transcript)
        self.assertEqual(store.snapshot().resources.model_activity, "idle")

    def test_internal_actor_stream_stays_out_of_the_user_transcript(self):
        store = WorkspaceUIStore()

        store.handle_event(
            "model_text",
            "private planner protocol and a very large draft",
            {"actor": "architect"},
        )

        snapshot = store.snapshot()
        self.assertFalse(snapshot.transcript)
        self.assertIn("architect", snapshot.resources.model_activity)
        self.assertTrue(any("reasoning.architect" in item for item in snapshot.advanced_log))

    def test_presentation_receipts_are_idempotent_and_active_events_are_transient(self):
        store = WorkspaceUIStore()
        store.publish(
            PresentationEvent(
                "plan:1:active",
                "plan",
                "Architect · reviewing the plan",
                PresentationLifecycle.ACTIVE,
                actor="architect",
            )
        )
        self.assertFalse(store.snapshot().transcript)
        store.publish(PresentationEvent("plan:1:settled", "plan", "Plan revision 1 is ready."))
        store.publish(PresentationEvent("plan:1:settled", "plan", "Plan revision 1 is ready."))
        self.assertEqual(
            [item.text for item in store.snapshot().transcript],
            ["Plan revision 1 is ready."],
        )


if __name__ == "__main__":
    unittest.main()

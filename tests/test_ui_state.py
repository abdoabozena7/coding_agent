from __future__ import annotations

import unittest
import time
from threading import Thread
from types import SimpleNamespace

from agent.ui_state import (
    ActivityStage,
    AttentionKind,
    AttentionOption,
    AttentionRequest,
    ExperienceMode,
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


if __name__ == "__main__":
    unittest.main()

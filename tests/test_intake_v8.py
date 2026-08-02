from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agent.intake import IntentArchitect, RunMode
from agent.learning import GlobalLessonStore, LearnedLessonV1
from agent.local_provider import repair_structured_json_object
from agent.repository_index import HashingEmbeddingProvider, RepositoryIndex
from agent.store import StateStore
from agent.swarm_protocol import SwarmMessageType, SwarmMessageV1
from agent.testing import semantic_goal_intake


def question(question_id: str, values=("best", "balanced", "narrow")) -> dict:
    return {
        "id": question_id,
        "header": "Decision",
        "question": f"Choose the {question_id} direction.",
        "reason": "The model found this choice consequential.",
        "options": [
            {
                "value": value,
                "label": value.replace("_", " ").title(),
                "description": f"Use the {value} direction.",
                "recommended": index == 0,
            }
            for index, value in enumerate(values)
        ],
    }


def validated(original: str, *, mode="normal", recommended="normal", questions=(), facts=(), **changes):
    proposal = semantic_goal_intake(original, recommended_mode=recommended, questions=questions)
    proposal.update(changes)
    return IntentArchitect().validate(
        proposal,
        original_input=original,
        requested_mode=mode,
        repository_facts=facts,
    )


class IntentArchitectTests(unittest.TestCase):
    def test_model_authored_questions_are_strict_and_recommended_first(self) -> None:
        decision = validated(
            "ظبط ده",
            questions=(question("outcome"), question("priority"), question("scope")),
        )
        self.assertEqual(len(decision.questions), 3)
        for item in decision.questions:
            self.assertEqual(len(item.options), 3)
            self.assertTrue(item.options[0].recommended)
            self.assertFalse(item.options[1].recommended)
            self.assertFalse(item.options[2].recommended)
            self.assertTrue(item.allow_freeform)

    def test_two_model_authored_options_are_valid_with_freeform_available(self) -> None:
        decision = validated(
            "Choose the product direction.",
            questions=(question("direction", ("focused", "broad")),),
        )
        self.assertEqual(len(decision.questions[0].options), 2)
        self.assertTrue(decision.questions[0].allow_freeform)

    def test_model_clear_cohesive_request_routes_normal_without_questions(self) -> None:
        decision = validated("Fix the parser bug and run focused unit tests.")
        self.assertEqual(decision.questions, ())
        self.assertIs(decision.brief.routed_mode, RunMode.NORMAL)

    def test_empty_model_question_placeholder_is_not_replaced_with_canned_copy(self) -> None:
        decision = validated(
            "Implement the clear request.",
            questions=({},),
        )
        self.assertEqual(decision.questions, ())

    def test_conflicting_or_missing_recommendations_never_create_an_automatic_choice(self) -> None:
        missing = question("scope")
        for option in missing["options"]:
            option["recommended"] = False
        decision = validated("Choose scope.", questions=(missing,))
        self.assertFalse(any(item.recommended for item in decision.questions[0].options))

        conflicting = question("priority")
        for option in conflicting["options"][:2]:
            option["recommended"] = True
        decision = validated("Choose priority.", questions=(conflicting,))
        self.assertFalse(any(item.recommended for item in decision.questions[0].options))

    def test_ultra_recommendation_never_silently_promotes_normal(self) -> None:
        mode_question = question("execution_mode", ("ultra", "normal", "edit_request"))
        decision = validated(
            "Implement the requested coordinated system.",
            recommended="ultra",
            questions=(mode_question,),
        )
        self.assertIs(decision.brief.routed_mode, RunMode.NORMAL)
        self.assertEqual(decision.questions, ())
        self.assertIn("high_model_authored_task_demand", decision.complexity.hard_triggers)

    def test_repository_facts_are_context_not_semantic_triggers(self) -> None:
        decision = validated(
            "Implement the requested coordinated system.",
            recommended="ultra",
            questions=(question("execution_mode", ("ultra", "normal", "edit_request")),),
            facts=("Discovered repository context: src/runtime.js",),
        )
        self.assertIs(decision.brief.routed_mode, RunMode.NORMAL)
        self.assertIn("Discovered repository context: src/runtime.js", decision.brief.assumptions)

    def test_explicit_ultra_uses_model_authored_baseline_contract(self) -> None:
        decision = validated(
            "Improve the existing implementation",
            mode="ultra",
            constraints=["Treat current work as the accepted baseline"],
            acceptance_expectations=["The candidate wins an evidence-backed comparison"],
        )
        self.assertIs(decision.brief.routed_mode, RunMode.ULTRA)
        self.assertEqual(decision.questions, ())
        self.assertTrue(any("accepted baseline" in item for item in decision.brief.constraints))
        self.assertTrue(any("evidence-backed comparison" in item for item in decision.brief.success_criteria))


class PersistenceAndProtocolV8Tests(unittest.TestCase):
    def test_intake_answer_survives_store_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            store.save_workflow_session(
                "session", goal_id=None, session_mode="normal",
                plan_state="inspecting", run_state="planning",
            )
            decision = validated(
                "ظبط ده", questions=(question("outcome"), question("scope"))
            )
            intake = store.create_intake_session(
                "session",
                original_input="ظبط ده",
                brief=decision.brief.to_dict(),
                complexity=decision.complexity.to_dict(),
                requested_mode="normal",
                routed_mode="normal",
                route_reason=decision.brief.route_reason,
                status="awaiting_answers",
                questions=(item.to_dict() for item in decision.questions),
            )
            store.answer_intake_question(
                intake["id"], decision.questions[0].id, "best", source="suggested"
            )
            store.close()
            reopened = StateStore(workspace)
            try:
                pending = reopened.get_pending_intake("session")
                self.assertIsNotNone(pending)
                self.assertEqual(pending["questions"][0]["answer"], "best")
                self.assertEqual(pending["questions"][0]["answer_source"], "suggested")
            finally:
                reopened.close()

    def test_typed_swarm_frames_keep_fencing_deadline_and_evidence(self) -> None:
        message = SwarmMessageV1(
            ultra_run_id="run-1", sender_agent_id="vehicle",
            recipient_agent_id="assembler",
            message_type=SwarmMessageType.PACKAGE_PUBLISHED,
            topic="vehicle-package", payload={"package_id": "pkg-1"},
            fencing_token=7, deadline="2026-07-13T12:00:00+00:00",
            evidence=({"kind": "test", "passed": True},),
        )
        for frame in (message.encode_frame(), message.encode_dsl_frame(), message.encode_binary_frame()):
            restored = SwarmMessageV1.decode_any_frame(frame)
            self.assertEqual(restored.message_type, SwarmMessageType.PACKAGE_PUBLISHED)
            self.assertEqual(restored.fencing_token, 7)
            self.assertEqual(restored.evidence[0]["kind"], "test")


class CodeIntelligenceAndLearningTests(unittest.TestCase):
    def test_multifile_ml_backend_frontend_incremental_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for area in ("ml", "backend", "frontend"):
                (workspace / area).mkdir()
            for index in range(20):
                (workspace / "ml" / f"pipeline_{index}.py").write_text(
                    f"from backend.service_{index} import predict\n"
                    f"class FeaturePipeline{index}:\n    def transform(self, batch): return predict(batch)\n",
                    encoding="utf-8",
                )
                (workspace / "backend" / f"service_{index}.py").write_text(
                    f"@app.get('/models/{index}')\ndef predict(batch):\n    return {{'score': len(batch)}}\n",
                    encoding="utf-8",
                )
                (workspace / "frontend" / f"Panel{index}.tsx").write_text(
                    f"import React from 'react';\nexport function Panel{index}(){{ return <section>Model {index}</section>; }}\n",
                    encoding="utf-8",
                )
            indexer = RepositoryIndex(workspace, embedding_provider=HashingEmbeddingProvider(dimensions=32))
            indexer.update_all()
            self.assertEqual(indexer.last_update_stats["updated"], 60)
            self.assertIn("FeaturePipeline7", indexer.symbol_index())
            self.assertTrue(any(edge.kind == "route" for edge in indexer.relations["backend/service_7.py"]))
            self.assertIn("frontend/Panel7.tsx", indexer.semantic_map())
            for number in range(3):
                path = workspace / "backend" / f"service_{number}.py"
                path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
            (workspace / "frontend" / "Panel19.tsx").unlink()
            started = time.perf_counter()
            indexer.update_all()
            elapsed = time.perf_counter() - started
            self.assertEqual(indexer.last_update_stats["updated"], 3)
            self.assertEqual(indexer.last_update_stats["removed"], 1)
            self.assertEqual(indexer.last_update_stats["reused"], 56)
            self.assertLess(elapsed, 5.0)
            query_started = time.perf_counter()
            hits = indexer.search_with_scores("FeaturePipeline7 predict", limit=10)
            self.assertLess(time.perf_counter() - query_started, 2.0)
            self.assertTrue(any(hit.entry.path.startswith("ml/") for hit in hits))

    def test_local_response_repairs_transport_damage_without_inventing_fields(self) -> None:
        value, actions = repair_structured_json_object(
            '{"insights": [], "payload\\": {"steps": ["inspect",],}, "summary": "ok"}'
        )
        self.assertEqual(value["payload"]["steps"], ["inspect"])
        self.assertEqual(value["summary"], "ok")
        self.assertTrue(actions)

    def test_tree_sitter_javascript_symbols_graph_and_codeowners(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "src" / "car.js").write_text(
                "export class Car { drive(){ return accelerate(); } }\n"
                "export function accelerate(){ return 1; }\n", encoding="utf-8",
            )
            (workspace / "src" / "main.js").write_text(
                "import { Car } from './car.js';\nconst car = new Car(); car.drive();\n",
                encoding="utf-8",
            )
            (workspace / "CODEOWNERS").write_text("src/* @game-team\n", encoding="utf-8")
            index = RepositoryIndex(workspace, embedding_provider=HashingEmbeddingProvider())
            index.update_all()
            symbols = index.symbol_index()
            semantic = index.semantic_map()
            self.assertIn("Car", symbols)
            self.assertTrue(any(item.provenance.startswith("tree_sitter:") for item in symbols["Car"]))
            self.assertIn("src/car.js", index.resolved_dependency_graph()["src/main.js"])
            self.assertEqual(semantic["src/car.js"]["owners"], ("@game-team",))
            hits = index.search_with_scores("accelerate vehicle", limit=5)
            self.assertTrue(hits)
            self.assertTrue(all(hit.reason and hit.provenance for hit in hits))

    def test_global_lesson_confidence_uses_real_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = GlobalLessonStore(Path(directory) / "lessons.json")
            lesson = memory.put(LearnedLessonV1(
                title="Browser verification",
                content="Run runtime and screenshot checks before accepting interactive HTML.",
                applicability_tags=("html", "browser", "visual"), evidence_refs=("benchmark:1",),
            ))
            improved = memory.record_outcome(lesson.id, succeeded=True)
            reduced = memory.record_outcome(lesson.id, succeeded=False)
            self.assertIsNotNone(improved)
            self.assertIsNotNone(reduced)
            self.assertGreater(improved.confidence, lesson.confidence)
            self.assertLess(reduced.confidence, improved.confidence)
            self.assertEqual(memory.search("browser visual html")[0].id, lesson.id)


if __name__ == "__main__":
    unittest.main()

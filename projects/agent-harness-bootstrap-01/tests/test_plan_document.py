from types import SimpleNamespace
import unittest

from agent.plan_document import PlanDocumentError, parse_plan_document, render_plan_document


class PlanDocumentTests(unittest.TestCase):
    def _plan(self):
        task = SimpleNamespace(
            id="T001",
            title="Implement the terminal flow",
            description="Keep input responsive while work continues.",
            risk="medium",
            depends_on=(),
            acceptance_criteria=("Palette remains usable",),
            verification=("Run TUI pipe-input tests",),
        )
        return SimpleNamespace(
            summary="A focused terminal reliability update.",
            tasks=(task,),
            execution_strategy="Change presentation state before visual polish.",
            expected_changes=(
                {"path": "agent/tui.py", "intent": "Improve interaction", "supports_tasks": ["T001"]},
            ),
            applicability_evidence=(
                {"source": "agent/tui.py", "fact": "The persistent app owns input", "supports_tasks": ["T001"]},
            ),
        )

    def test_round_trip_keeps_human_editable_plan_fields(self):
        document = render_plan_document(self._plan())
        edited = document.replace("Implement the terminal flow", "Implement the calm terminal flow")
        parsed = parse_plan_document(edited)

        self.assertEqual(parsed.tasks[0]["id"], "T001")
        self.assertEqual(parsed.tasks[0]["title"], "Implement the calm terminal flow")
        self.assertEqual(parsed.expected_changes[0]["supports_tasks"], ["T001"])

    def test_invalid_dependency_reports_a_line_and_does_not_create_partial_data(self):
        document = render_plan_document(self._plan()).replace("Depends on: -", "Depends on: T999")
        with self.assertRaisesRegex(PlanDocumentError, r"line \d+: task T001 depends on unknown"):
            parse_plan_document(document)

    def test_dependency_cycle_reports_the_task_line(self):
        first = self._plan().tasks[0]
        second = SimpleNamespace(
            id="T002",
            title="Verify the terminal flow",
            description="Exercise the completed interaction contract.",
            risk="low",
            depends_on=("T001",),
            acceptance_criteria=("The flow remains responsive",),
            verification=("Run the focused suite",),
        )
        cyclic_first = SimpleNamespace(**{**vars(first), "depends_on": ("T002",)})
        plan = SimpleNamespace(**{**vars(self._plan()), "tasks": (cyclic_first, second)})

        with self.assertRaisesRegex(PlanDocumentError, r"line \d+: task dependencies form a cycle"):
            parse_plan_document(render_plan_document(plan))


if __name__ == "__main__":
    unittest.main()

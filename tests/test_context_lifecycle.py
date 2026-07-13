import unittest

from agent.context import REVIVAL_MARKER, suspend_and_revive
from agent.run_context import GoalContractV1
from agent.weak_model import WeakModelPolicy


class ContextLifecycleTests(unittest.TestCase):
    def test_goal_contract_projection_is_compact_stable_and_policy_is_round_trippable(self):
        contract = GoalContractV1(
            run_id="run-1", original_objective="Build it", interpreted_objective="Build it safely",
            acceptance_criteria=("Tests pass",), required_verification=("pytest",),
            user_feedback=("first", "latest"), failed_hypotheses=("stale cache",),
        )
        projection = contract.projection(actor="implementer", task_id="T001")
        self.assertEqual(projection["contract_fingerprint"], contract.fingerprint)
        self.assertEqual(projection["current_task"], "T001")
        self.assertNotIn("original_objective", projection)
        policy = WeakModelPolicy.from_dict(WeakModelPolicy().to_dict())
        self.assertEqual(policy.version, 1)
        self.assertIn("narrow_context", policy.applied_rules("provider_call"))

    def test_suspend_and_revive_keeps_durable_goal_and_latest_turn(self):
        conversation = [
            {"role": "user", "content": "scan chunk 1"},
            {"role": "assistant", "content": "x" * 200},
            {"role": "user", "content": "continue with chunk 2"},
            {"role": "assistant", "content": "working"},
        ]
        suspended = []
        revived = suspend_and_revive(
            conversation,
            "goal=scan the entire repository; cursor=line 500000; completed=chunk 1",
            lambda messages: f"summary of {len(messages)} messages",
            max_chars=100,
            on_suspend=suspended.append,
        )

        self.assertIn(REVIVAL_MARKER, revived[0]["content"])
        self.assertIn("cursor=line 500000", revived[0]["content"])
        self.assertEqual(revived[1:], conversation[2:])
        self.assertEqual(suspended, [2])

    def test_context_below_budget_is_not_rotated(self):
        conversation = [{"role": "user", "content": "small"}]
        self.assertIs(suspend_and_revive(conversation, "goal=x", max_chars=100), conversation)


if __name__ == "__main__":
    unittest.main()

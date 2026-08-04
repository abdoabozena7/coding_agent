from agent.prompts import COORDINATOR_SYSTEM_PROMPT


def test_coordinator_batches_tightly_coupled_files_for_capable_models() -> None:
    normalized = " ".join(COORDINATOR_SYSTEM_PROMPT.split())
    assert "combine tightly coupled files" in normalized
    assert "one atomic apply_patch" in normalized
    assert "Never batch unrelated checklist tasks" in normalized

import pytest
import json

from src.master_plan_generator import generate_master_plan, ConflictError

# --- Test Fixtures and Data ---

@pytest.fixture
def non_conflicting_data():
    """Data representing two non-conflicting, verified mechanics."""
    return {
        'dependencies': [
            "{'from': 'lesson-c5947271746d03c060a8', 'to': 'VerifiedStrategyEngine', 'relation': 'supports'}", 
            "{'from': 'lesson-c61466704e9ce77812ae', 'to': 'VerifiedStrategyEngine', 'relation': 'supports'}"
        ]
    }

@pytest.fixture
def conflicting_data():
    """Data simulating two mechanics that overlap or contradict assumptions."""
    # Use duplicate structure to ensure conflict simulation triggers based on the implementation logic
    return {
        'dependencies': [
            "{'from': 'lesson-conflicting-A', 'to': 'Engine', 'relation': 'supports'}", 
            "{'from': 'lesson-conflicting-A', 'to': 'Engine', 'relation': 'supports'}"
        ]
    }

# --- Test Cases ---

def test_successful_synthesis(non_conflicting_data):
    """Tests successful GoalSpecV1 generation when mechanics are non-contradictory."""
    result_json = generate_master_plan(non_conflicting_data)
    result = json.loads(result_json)

    assert "error" not in result
    assert "SynthesisStatus" in result
    # Check if the status indicates success after using the basic conflict check
    assert "Successfully synthesized" in result["SynthesisStatus"]

def test_conflict_handling(conflicting_data):
    """Tests that ConflictError is raised when mechanics are contradictory."""
    with pytest.raises(ConflictError) as excinfo:
        generate_master_plan(conflicting_data)
    
    assert "Synthesis failed due to mechanical contradiction." in str(excinfo.value)

def test_no_mechanics_provided():
    """Tests handling when no lesson data is available."""
    # Fixed: Use assertions instead of return statement per pytest warning
    result_json = generate_master_plan({'dependencies': []})
    result = json.loads(result_json)
    assert "No verified mechanisms found" in result["error"]
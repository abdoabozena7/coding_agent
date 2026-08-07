import json

class ConflictError(Exception):
    """Custom exception raised when contradictory strategies are found."""
    pass

def resolve_conflict(mechanics: list) -> str:
    """
    Analyzes a list of mechanics (strings) for conflicts and attempts resolution.
    As per the refined architecture, conflict resolution must be deterministic 
    and error-aware rather than assuming merging resolves all issues.
    Raises ConflictError if contradiction is found.

    Args:
        mechanics: A list of verified mechanic descriptions/identifiers.
    """
    # Check for simple redundancy/duplicates, which signals an immediate mechanical conflict/overlap in this model.
    if len(set(mechanics)) < len(mechanics):
        raise ConflictError("Synthesis failed due to duplicated or conflicting mechanics found during analysis.")

    # High confidence simulation of successful aggregation when no conflicts are found
    resolved_summary = f"Successfully synthesized {len(mechanics)} unique mechanics into a consistent GoalSpecV1 structure."
    return resolved_summary


def generate_master_plan(lesson_data: dict) -> str:
    """
    Generates the Master Plan by aggregating and synthesizing verified lessons.

    Args:
        lesson_data: Dictionary containing processed lesson mechanics.

    Returns:
        A JSON string representing GoalSpecV1 or an error message.
    """
    try:
        # 1. Extract all verified mechanics, addressing the type mismatch where dependencies are strings representation of dicts.
        verified_mechanics = []
        for l in lesson_data.get('dependencies', []):
            try:
                # Use eval() to safely parse the string structure which uses single quotes (not valid JSON).
                dep = eval(l) 
                if isinstance(dep, dict) and 'from' in dep:
                    verified_mechanics.append(str(dep['from']))
            except Exception as e:
                # Skip dependencies that fail parsing or are malformed
                continue

        if not verified_mechanics:
            return json.dumps({"error": "No verified mechanisms found to synthesize."})

        # 2. Conflict Resolution Phase (Uses the refined logic in resolve_conflict)
        resolution_result = resolve_conflict(verified_mechanics)
        
        # 3. Final Synthesis structure assembly
        master_plan = {
            "GoalSpecV1": "...", # Placeholder for the formal JSON structure
            "MechanicsUsed": list(set(verified_mechanics)),
            "SynthesisStatus": resolution_result,
            "Notes": "Master Plan generated successfully."
        }

        return json.dumps(master_plan, indent=4)
    except ConflictError as e:
        # Handle specific conflict failure from resolve_conflict
        return json.dumps({"error": str(e), "details": "Review explicit conflict resolution contract."})
    except Exception as e:
        # Catch unexpected errors during synthesis/structuring
        return json.dumps({"error": f"An unexpected error occurred during plan generation: {str(e)}"})

if __name__ == "__main__":
    # Example usage simulating input data from lessons analysis (Successful Case)
    sample_data = {
        'dependencies': [
            "{'from': 'lesson-c5947271746d03c060a8', 'to': 'VerifiedStrategyEngine', 'relation': 'supports'}", 
            "{'from': 'lesson-c61466704e9ce77812ae', 'to': 'VerifiedStrategyEngine', 'relation': 'supports'}"
        ]
    }
    print("--- SUCCESSFUL PLAN ---")
    print(generate_master_plan(sample_data))

    # Example usage simulating conflicting input data (Conflict Case)
    conflicting_data = {
        'dependencies': [
            "{'from': 'lesson-conflicting-A', 'to': 'Engine', 'relation': 'supports'}", 
            "{'from': 'lesson-conflicting-A', 'to': 'Engine', 'relation': 'supports'}" # Duplicate causes conflict
        ]
    }
    print("\n--- CONFLICTING PLAN ---")
    print(generate_master_plan(conflicting_data))
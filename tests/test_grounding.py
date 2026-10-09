import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from evaluation.grounding.grounding_checker import check_groundedness
from evaluation.grounding.llm_evaluator import Phase5DEvaluator
from evaluation.grounding.scenario_client import ScenarioLLMClient
from src.llm.validator import validate_llm_analysis

def test_provenance_failure_detection():
    client = ScenarioLLMClient()
    evaluator = Phase5DEvaluator(mode="mock", scenario_client=client)

    def bad_generate(prompt):
        res = client.__class__.generate_analysis(client, prompt)
        res["source_titles"] = ["Wrong Title"]
        return res

    with patch.object(client, 'generate_analysis', side_effect=bad_generate):
        case = {"id": "test_prov", "ingredient_name": "Test", "phase2_status": "matched", "retrieved_evidence": "Some text."}
        evaluator.evaluate_case(case)
        assert evaluator.metrics["provenance_preserved"] == 0
        assert evaluator.cases_log[0]["provenance_ok"] == False

def test_validator_in_evaluation_flow():
    evaluator = Phase5DEvaluator(mode="mock")

    def bad_generate(prompt):
        return {"garbage": "output"}

    with patch.object(evaluator.client, 'generate_analysis', side_effect=bad_generate):
        case = {"id": "test_val", "ingredient_name": "Test", "phase2_status": "matched", "retrieved_evidence": "Some text."}
        evaluator.evaluate_case(case)
        # Should gracefully handle it and fail validation without crashing
        assert evaluator.metrics["valid_output_accepted"] == 0
        assert evaluator.cases_log[0]["validated_output"]["summary"] == "Detailed analysis is temporarily unavailable."

def test_no_evidence_behavior():
    client = ScenarioLLMClient("no_evidence_hallucination")
    evaluator = Phase5DEvaluator(mode="mock", scenario_client=client)
    case = {"id": "test_no_ev", "ingredient_name": "Missing", "phase2_status": "matched", "retrieved_evidence": "No retrieved evidence available."}
    evaluator.evaluate_case(case)
    # The scenario hallucinated a category "Hallucinated Category", so we expect unsupported claims
    assert evaluator.metrics["unsupported_claims"] > 0
    assert evaluator.metrics["no_evidence_compliant"] == 0

def test_mock_failure_scenarios():
    client = ScenarioLLMClient("unsupported_cat_func")
    evaluator = Phase5DEvaluator(mode="mock", scenario_client=client)
    case = {"id": "test_mock_fail", "ingredient_name": "Test", "phase2_status": "matched", "retrieved_evidence": "Some evidence here."}
    evaluator.evaluate_case(case)
    # The scenario hallucinated Automobile as category
    assert evaluator.metrics["unsupported_claims"] > 0

def test_tool_usage_constraints():
    evaluator = Phase5DEvaluator(mode="mock")

    # We patch generate_analysis to just populate tool_calls_history instead of running
    def fake_generate_0(*args): evaluator.client.tool_calls_history = [] ; return {}
    def fake_generate_1(*args): evaluator.client.tool_calls_history = [1] ; return {}
    def fake_generate_4(*args): evaluator.client.tool_calls_history = [1, 2, 3, 4] ; return {}

    case = {"id": "t1", "ingredient_name": "T1", "phase2_status": "matched", "retrieved_evidence": ""}

    with patch.object(evaluator.client, 'generate_analysis', side_effect=fake_generate_0):
        evaluator.evaluate_case(case)
        assert evaluator.metrics["tool_constraint_compliant"] == 1

    with patch.object(evaluator.client, 'generate_analysis', side_effect=fake_generate_1):
        evaluator.evaluate_case(case)
        assert evaluator.metrics["tool_constraint_compliant"] == 1

    with patch.object(evaluator.client, 'generate_analysis', side_effect=fake_generate_4):
        evaluator.evaluate_case(case)
        assert evaluator.metrics["tool_constraint_compliant"] == 1 # Tool calls are disallowed
def test_main_execution():
    import subprocess
    script_path = str(PROJECT_ROOT / "evaluation" / "grounding" / "main.py")
    subprocess.run([sys.executable, script_path], check=True, cwd=str(PROJECT_ROOT))
    out_path = PROJECT_ROOT / "evaluation" / "results" / "phase5d_evaluation_results.json"
    assert out_path.exists()
    with open(out_path, "r") as f:
        data = json.load(f)
        assert "synthetic_unit_tests" not in data # It is named "unit_tests" in the output
        assert "unit_tests" in data
        assert "mock_evaluation" in data
        assert "empirical_evaluation" in data
        assert data["live_llm_evaluation"]["status"] == "not_run"


def test_actual_phase4b_tool_trace():
    # Use real LLMClient but patch _call_gemini_api or internal API call
    # Actually, if we mock is_mock() to return False, we can provide a fake response.
    # We will patch generate_analysis to inject a tool call directly.
    # Wait, if we patch generate_analysis we skip the execution.
    # To execute the real tool executor, we can just call it directly or run the mock loop.
    # The _generate_mock_analysis_loop in Phase 4 ALREADY executes the tool 'lookup_ingredient'
    # when it encounters "No retrieved evidence available".
    # So we can just use the base LLMClient(api_key=None).
    from src.llm.llm_client import LLMClient

    client = LLMClient(api_key=None)
    evaluator = Phase5DEvaluator(mode="mock", scenario_client=client)

    # We pass a case with no evidence so the mock triggers the lookup tool
    case = {"id": "test_tool_real", "ingredient_name": "Monosodium Glutamate", "phase2_status": "matched", "retrieved_evidence": "No retrieved evidence available."}
    evaluator.evaluate_case(case)

    # The consolidated Gemini boundary does not execute tools.
    log = evaluator.cases_log[0]
    history = log["tool_calls_history"]
    assert history == []

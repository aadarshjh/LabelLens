import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.grounding.llm_evaluator import Phase5DEvaluator
from evaluation.grounding.scenario_client import ScenarioLLMClient

def main():
    cases_file = PROJECT_ROOT / "evaluation" / "data" / "evaluation_cases.json"
    with open(cases_file, "r") as f:
        data = json.load(f)

    final_output = {
        "unit_tests": {},
        "mock_evaluation": {},
        "empirical_evaluation": {},
        "live_llm_evaluation": {
            "status": "not_run",
            "sample_size": 0,
            "reason": "Skipped to conserve API quota."
        }
    }

    # 1. Synthetic Unit Tests
    syn_evaluator = Phase5DEvaluator(mode="mock")
    final_output["unit_tests"] = syn_evaluator.run_all(data.get("synthetic_unit_tests", []))
    final_output["unit_tests"]["status"] = "completed"

    # 2. Mock Evaluation (with scenarios)
    # We will instantiate ONE evaluator and inject scenarios manually per case by mocking the generate_analysis inside the loop
    # Or just use a simpler loop that aggregates raw metrics and recalculates rates.

    class MultiScenarioClient(ScenarioLLMClient):
        def generate_analysis(self, prompt: str) -> dict:
            return super().generate_analysis(prompt)

    mock_evaluator = Phase5DEvaluator(mode="mock")
    for case in data.get("mock_evaluation", []):
        mock_evaluator.client = ScenarioLLMClient(scenario=case.get("scenario"))
        mock_evaluator.evaluate_case(case)

    # Re-run metrics generation on the accumulated evaluator metrics
    mock_res = mock_evaluator.run_all([]) # run_all([]) just returns the calculated rates from existing metrics
    final_output["mock_evaluation"] = {
        "status": "completed",
        "metrics": mock_res["metrics"],
        "cases_log": mock_res["cases_log"]
    }

    # 3. Empirical Evaluation
    empirical_cases = data.get("empirical_evaluation", [])
    if not empirical_cases:
        final_output["empirical_evaluation"] = {
            "status": "not_run",
            "reason": "No empirical evaluation cases are included in the public test fixture.",
        }
    else:
        emp_evaluator = Phase5DEvaluator(mode="empirical")
        try:
            final_output["empirical_evaluation"] = emp_evaluator.run_all(empirical_cases)
            final_output["empirical_evaluation"]["status"] = "completed"
        except (ImportError, OSError) as exc:
            final_output["empirical_evaluation"] = {
                "status": "not_run",
                "reason": f"Local embedding model unavailable: {exc}",
            }

    out_file = PROJECT_ROOT / "evaluation" / "results" / "phase5d_evaluation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Phase 5D Evaluation completed. Results saved to {out_file}")

if __name__ == "__main__":
    main()

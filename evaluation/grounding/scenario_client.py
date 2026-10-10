from src.llm.llm_client import LLMClient

class ScenarioLLMClient(LLMClient):
    """A custom mock client for injecting deliberate failure scenarios in Phase 5D."""

    def __init__(self, scenario=None):
        super().__init__(api_key=None)
        self.scenario = scenario

    def generate_analysis(self, prompt: str) -> dict:
        self.tool_calls_history = []

        # Base valid structure
        base_resp = {
            "ingredient": "Mocked Name",
            "identification": {
                "status": "matched",
                "confidence": "high",
                "explanation": "Valid explanation."
            },
            "category": "Fruit",
            "function": "Sweetener",
            "regulatory_information": "Approved",
            "summary": "Valid summary.",
            "health_warnings": [],
            "uncertainties": [],
            "source_titles": ["Eval Title"]
        }

        if self.scenario == "unsupported_cat_func":
            base_resp["category"] = "Automobile"
            base_resp["function"] = "Combustion"
        elif self.scenario == "unsupported_reg":
            base_resp["regulatory_information"] = "Banned everywhere"
        elif self.scenario == "ambiguous_to_matched":
            base_resp["identification"]["status"] = "matched"
        elif self.scenario == "unknown_to_matched":
            base_resp["identification"]["status"] = "matched"
        elif self.scenario == "no_evidence_hallucination":
            base_resp["category"] = "Hallucinated Category"
            base_resp["function"] = "Hallucinated Function"
        elif self.scenario == "valid_no_evidence":
            base_resp["category"] = None
            base_resp["function"] = None
            base_resp["regulatory_information"] = None

        return base_resp

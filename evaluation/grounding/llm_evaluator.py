import json
import copy
from pathlib import Path
from src.config import VECTORSTORE_DIR

from src.llm.llm_client import LLMClient
from src.llm.prompt_builder import build_analysis_prompt
from src.llm.validator import validate_llm_analysis
from evaluation.grounding.grounding_checker import check_groundedness
from src.rag.retriever import IngredientRetriever

class Phase5DEvaluator:
    def __init__(self, mode="mock", scenario_client=None):
        self.mode = mode

        if mode == "live":
            self.client = LLMClient()
        elif scenario_client:
            self.client = scenario_client
        else:
            self.client = LLMClient(api_key=None)

        # Mock/scenario evaluations operate on supplied evidence and must not
        # require the optional local embedding model. Empirical mode loads it
        # only when retrieval is actually requested.
        self.retriever = None

        self.metrics = {
            "evaluated_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,

            "no_evidence_cases": 0,
            "no_evidence_raw_hallucinations": 0,
            "no_evidence_compliant": 0,

            "uncertainty_cases": 0,
            "uncertainty_preserved": 0,

            "provenance_cases": 0,
            "provenance_preserved": 0,

            "tool_constraint_cases": 0,
            "tool_constraint_compliant": 0,

            "valid_output_cases": 0,
            "valid_output_accepted": 0,

            "adversarial_cases": 0,
            "adversarial_contained": 0,
        }
        self.cases_log = []

    def evaluate_case(self, case: dict):
        case_id = case["id"]
        ingredient = case["ingredient_name"]
        status = case["phase2_status"]

        if self.mode == "empirical":
            if self.retriever is None:
                self.retriever = IngredientRetriever(store_dir=str(VECTORSTORE_DIR))
            results = self.retriever.retrieve(ingredient, bypass_exact=False)
            evidence_text = ""
            source_titles = []
            for ev in results:
                evidence_text += f"{ev['text']}\n"
                source_titles.append(ev.get("source", {}).get("title", ""))

            phase3_item = {
                "ingredient": ingredient,
                "raw_name": ingredient,
                "phase2_status": status,
                "phase2_similarity": 100.0,
                "evidence": results
            }
        else:
            retrieved_evidence = case.get("retrieved_evidence", "No retrieved evidence available.")
            evidence_text = retrieved_evidence
            source_titles = ["Eval Title"] if evidence_text != "No retrieved evidence available." else []
            phase3_item = {
                "ingredient": ingredient,
                "raw_name": ingredient,
                "phase2_status": status,
                "phase2_similarity": 100.0,
                "evidence": []
            }
            if source_titles:
                phase3_item["evidence"].append({
                    "text": retrieved_evidence,
                    "score": 1.0,
                    "source": {"organization": "Eval Org", "title": "Eval Title", "url": "http://eval.com"}
                })

        prompt = build_analysis_prompt(phase3_item)


        try:
            raw_output = self.client.generate_analysis(prompt)
        except Exception as e:
            raw_output = {"error": str(e)}

        # VALIDATOR FLOW
        validated_output = validate_llm_analysis(status, phase3_item["evidence"], raw_output)

        scenario = case.get("scenario")
        is_adversarial = bool(scenario) and scenario not in ["grounded_response", "valid_no_evidence"]

        validation_failed = (
            "Validation Error" in str(validated_output.get("uncertainties", []))
            or "Validation Warning" in str(validated_output.get("uncertainties", []))
        )

        if not is_adversarial:
            self.metrics["valid_output_cases"] += 1
            if not validation_failed:
                self.metrics["valid_output_accepted"] += 1
        else:
            self.metrics["adversarial_cases"] += 1

        # 1. GROUNDING
        grounding_raw = check_groundedness(raw_output, evidence_text)
        grounding_val = check_groundedness(validated_output, evidence_text)

        self.metrics["evaluated_claims"] += grounding_val["evaluated_claims"]
        self.metrics["supported_claims"] += grounding_val["supported_claims"]
        self.metrics["unsupported_claims"] += grounding_val["unsupported_claims"]

        # 2. NO EVIDENCE COMPLIANCE
        no_evidence = not phase3_item["evidence"]
        if no_evidence:
            self.metrics["no_evidence_cases"] += 1
            if grounding_raw["unsupported_claims"] > 0:
                self.metrics["no_evidence_raw_hallucinations"] += 1
            if grounding_val["unsupported_claims"] == 0:
                self.metrics["no_evidence_compliant"] += 1

        # 3. UNCERTAINTY PRESERVATION
        if status in ["ambiguous", "unknown"]:
            self.metrics["uncertainty_cases"] += 1
            out_status = validated_output.get("identification", {}).get("status", "")
            if out_status == status:
                self.metrics["uncertainty_preserved"] += 1

        # 4. PROVENANCE PRESERVATION
        provenance_ok = True
        if phase3_item["evidence"]:
            self.metrics["provenance_cases"] += 1
            out_sources = " ".join(validated_output.get("source_titles", []))
            for st in source_titles:
                if st not in out_sources:
                    provenance_ok = False

            if provenance_ok:
                self.metrics["provenance_preserved"] += 1

        # 5. TOOL CONSTRAINT
        self.metrics["tool_constraint_cases"] += 1
        if len(self.client.tool_calls_history) <= self.client.MAX_TOOL_CALLS:
            self.metrics["tool_constraint_compliant"] += 1

        # ADVERSARIAL CONTAINMENT
        adversarial_containment_passed = True
        if is_adversarial:
            if scenario in ["ambiguous_to_matched", "unknown_to_matched"]:
                if validated_output.get("identification", {}).get("status") != status:
                    adversarial_containment_passed = False
            elif scenario in ["no_evidence_hallucination", "unsupported_cat_func", "unsupported_reg"]:
                if grounding_val["unsupported_claims"] > 0:
                    adversarial_containment_passed = False


            if adversarial_containment_passed:
                self.metrics["adversarial_contained"] += 1

        evaluation_passed = True
        if phase3_item["evidence"] and not provenance_ok:
            evaluation_passed = False
        if is_adversarial and not adversarial_containment_passed:
            evaluation_passed = False

        self.cases_log.append({
            "case_id": case_id,
            "evaluation_passed": evaluation_passed,
            "scenario": scenario,
            "adversarial_containment_passed": adversarial_containment_passed if is_adversarial else None,
            "provenance_ok": provenance_ok,
            "grounding_raw": grounding_raw,
            "grounding_val": grounding_val,
            "raw_output": raw_output,
            "validated_output": validated_output,
            "tool_calls": len(self.client.tool_calls_history),
            "tool_calls_history": self.client.tool_calls_history
        })


    def run_all(self, cases: list) -> dict:
        for c in cases:
            self.evaluate_case(c)

        def safe_div(n, d): return n / d if d > 0 else 1.0

        return {
            "metrics": {
                "lexical_evidence_support_rate": {
                    "supported_claims": self.metrics["supported_claims"],
                    "evaluated_claims": self.metrics["evaluated_claims"],
                    "rate": safe_div(self.metrics["supported_claims"], self.metrics["evaluated_claims"])
                },
                "unsupported_claim_count": self.metrics["unsupported_claims"],
                "no_evidence_compliance_rate": {
                    "compliant": self.metrics["no_evidence_compliant"],
                    "cases": self.metrics["no_evidence_cases"],
                    "rate": safe_div(self.metrics["no_evidence_compliant"], self.metrics["no_evidence_cases"])
                },
                "no_evidence_raw_hallucinations": self.metrics["no_evidence_raw_hallucinations"],
                "uncertainty_preservation_rate": {
                    "preserved": self.metrics["uncertainty_preserved"],
                    "cases": self.metrics["uncertainty_cases"],
                    "rate": safe_div(self.metrics["uncertainty_preserved"], self.metrics["uncertainty_cases"])
                },
                "provenance_preservation_rate": {
                    "preserved": self.metrics["provenance_preserved"],
                    "cases": self.metrics["provenance_cases"],
                    "rate": safe_div(self.metrics["provenance_preserved"], self.metrics["provenance_cases"])
                },
                "valid_output_acceptance_rate": {
                    "passed": self.metrics["valid_output_accepted"],
                    "cases": self.metrics["valid_output_cases"],
                    "rate": safe_div(self.metrics["valid_output_accepted"], self.metrics["valid_output_cases"])
                },
                "adversarial_containment_rate": {
                    "contained": self.metrics["adversarial_contained"],
                    "cases": self.metrics["adversarial_cases"],
                    "rate": safe_div(self.metrics["adversarial_contained"], self.metrics["adversarial_cases"])
                },
                "tool_constraint_compliance_rate": {
                    "compliant": self.metrics["tool_constraint_compliant"],
                    "cases": self.metrics["tool_constraint_cases"],
                    "rate": safe_div(self.metrics["tool_constraint_compliant"], self.metrics["tool_constraint_cases"])
                }
            },
            "cases_log": self.cases_log
        }

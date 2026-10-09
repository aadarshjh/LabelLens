"""Optional live Groq runner for already-produced Phase 3 records.

This module is a boundary adapter: it reads structured retrieval results,
asks Groq for an explanation, validates the response, and never performs
OCR, extraction, normalization, or retrieval itself.
"""

import argparse
import json
from pathlib import Path

from src.llm.llm_client import LLMClient
from src.llm.prompt_builder import build_analysis_prompt
from src.llm.validator import validate_llm_analysis


def process_phase3_results(input_path: str | Path, output_path: str | Path,
                           api_key: str | None = None) -> list[dict]:
    """Analyze Phase 3 JSON and write validated results."""
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("ingredients", payload) if isinstance(payload, dict) else payload
    client = LLMClient(api_key=api_key)

    def analyze(item: dict) -> dict:
        status = item.get("phase2_status", "unknown")
        evidence = item.get("evidence", [])
        try:
            raw = client.generate_analysis(build_analysis_prompt(item))
        except Exception as exc:
            raw = {"error": str(exc)}
        return {
            "ingredient": item.get("ingredient", "Unknown"),
            "phase2_status": status,
            "llm_analysis": validate_llm_analysis(status, evidence, raw),
            "authoritative_evidence": evidence,
            "tool_calls": list(client.tool_calls_history),
            "sub_ingredients": [analyze(child) for child in item.get("sub_ingredients", [])],
        }

    results = [analyze(item) for item in items]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional Groq analysis at the LLM boundary.")
    parser.add_argument("--input", required=True, help="Phase 3 JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    process_phase3_results(args.input, args.output)


if __name__ == "__main__":
    main()

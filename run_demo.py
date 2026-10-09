#!/usr/bin/env python3
"""Run LabelLens through the deterministic pre-LLM boundary or live Groq analysis."""

import argparse
import json
import os
import sys

from src.config import IMAGES_DIR
from src.frontend.serializer import serialize_for_frontend
from src.llm.llm_client import LLMClient
from src.pipeline.pipeline import run_end_to_end_pipeline, run_pre_llm_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LabelLens demo.")
    parser.add_argument("image", nargs="?", default=str(IMAGES_DIR / "label1.jpeg"))
    parser.add_argument("--with-llm", action="store_true", help="Attempt live Groq analysis.")
    args = parser.parse_args()

    if args.with_llm:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY is not configured in the environment.", file=sys.stderr)
            sys.exit(1)

        client = LLMClient(api_key=api_key)
        print(f"Connecting to Groq (model: {client.model_name})...")

        # Run full pipeline with exactly ONE Groq batch request
        results, summary = run_end_to_end_pipeline(args.image, api_key=api_key, batch=True)
        consumer_json = serialize_for_frontend(results, summary, "Scanned Product")

        # Validation status check
        all_valid = all(
            "Validation Error" not in " ".join(item.get("llm_analysis", {}).get("uncertainties", []))
            for item in results
        )

        print("\n========================================")
        print("LabelLens Demo (Groq)")
        print("========================================")
        print(f"Image:                  {os.path.basename(args.image)}")
        print(f"OCR:                    completed")
        print(f"Ingredients extracted:  {summary.get('total', 0)}")
        print(f"  - matched:            {summary.get('matched', 0)}")
        print(f"  - ambiguous:          {summary.get('ambiguous', 0)}")
        print(f"  - unknown:            {summary.get('unknown', 0)}")
        print(f"Groq model:             {client.model_name}")
        print(f"Groq requests:          1")
        print(f"Groq analysis:          completed")
        print(f"Validation:             {'passed' if all_valid else 'warnings noted'}")
        print(f"Frontend result:        ready ({len(consumer_json['ingredients'])} items)")
        print("========================================\n")
        print(json.dumps({"summary": summary, "consumer": consumer_json}, indent=2))
        return

    result = run_pre_llm_pipeline(args.image)
    evidence_count = sum(len(item["evidence"]) for item in result["llm_inputs"])
    print(json.dumps({
        "ocr": {key: result["ocr"][key] for key in ("strategy", "psm", "confidence")},
        "ingredient_summary": result["phase2"]["summary"],
        "llm_boundary": {
            "structured_inputs": len(result["llm_inputs"]),
            "evidence_chunks": evidence_count,
            "provenance_preserved": all(all("source" in evidence for evidence in item["evidence"]) for item in result["llm_inputs"]),
        },
        "live_llm_attempted": False,
    }, indent=2))


if __name__ == "__main__":
    main()

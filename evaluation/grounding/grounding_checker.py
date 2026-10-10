import re

def check_groundedness(llm_output: dict, evidence_text: str) -> dict:
    """
    Lightweight heuristic evidence-support checker.
    Extracts key tokens (length >= 5) from factual fields and checks for presence in evidence.
    This is NOT an NLI model and does not prove semantic factual correctness.
    """
    unsupported_claims = []
    supported_claims = []

    no_evidence = not evidence_text or "No retrieved evidence available" in evidence_text
    evidence_lower = evidence_text.lower() if evidence_text else ""

    fields_to_check = {
        "category": llm_output.get("category"),
        "function": llm_output.get("function"),
        "regulatory_information": llm_output.get("regulatory_information")
    }

    # Skip common functional words or fallback words
    stop_words = {"mocked", "unknown", "validation", "failed", "safety", "defaults", "evidence", "available", "information", "about", "which", "these", "their", "there"}

    for field_name, text in fields_to_check.items():
        if not text or not isinstance(text, str):
            continue

        words = set(re.findall(r'\b[a-zA-Z]{5,}\b', text.lower()))

        for word in words:
            if word in stop_words:
                continue

            if no_evidence:
                unsupported_claims.append(f"{field_name}: '{word}'")
            else:
                if word in evidence_lower:
                    supported_claims.append(f"{field_name}: '{word}'")
                else:
                    unsupported_claims.append(f"{field_name}: '{word}'")

    evaluated = len(supported_claims) + len(unsupported_claims)

    return {
        "supported_claims": len(supported_claims),
        "unsupported_claims": len(unsupported_claims),
        "evaluated_claims": evaluated,
        "support_rate": len(supported_claims) / evaluated if evaluated > 0 else 1.0,
        "unsupported_claim_details": unsupported_claims
    }

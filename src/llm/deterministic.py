import re

def parse_evidence_text(text: str) -> dict:
    parsed = {}
    current_key = None
    current_val = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^([A-Za-z\s]+):$', line)
        if match:
            current_key = match.group(1).strip().lower()
            current_val = []
            parsed[current_key] = current_val
        elif current_key:
            current_val.append(line)

    for k, v in parsed.items():
        parsed[k] = "\n".join(v)

    return parsed

def generate_deterministic_analysis(ingredient_name: str, raw_name: str, evidence: list) -> dict:
    if not evidence:
        return None

    # Concatenate all evidence documents
    content = "\n\n".join([ev.get("text", "") for ev in evidence])
    parsed = parse_evidence_text(content)

    # Check if we have enough fields to be confident
    if "category" not in parsed or "description" not in parsed:
        return None

    sources = []
    for ev in evidence:
        src = ev.get("source", {})
        if "organization" in src:
            sources.append(src["organization"])

    analysis = {
        "ingredient": parsed.get("ingredient", ingredient_name),
        "identification": {
            "status": "matched",
            "confidence": "High",
            "explanation": f"Exact match found in verified knowledge base."
        },
        "category": parsed.get("category"),
        "function": parsed.get("function"),
        "summary": parsed.get("description"),
        "regulatory_information": parsed.get("regulatory information"),
        "uncertainties": [],
        "sources": sources
    }

    return analysis

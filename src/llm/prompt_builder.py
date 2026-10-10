import json


def build_analysis_prompt(phase3_item: dict) -> str:
    """
    Constructs a strict prompt for the LLM based on Phase 3 retrieved evidence.
    """
    ingredient = phase3_item.get("ingredient", "Unknown")
    raw_name = phase3_item.get("raw_name", "Unknown")
    status = phase3_item.get("phase2_status", "unknown")
    similarity = phase3_item.get("phase2_similarity", "N/A")
    evidence_list = phase3_item.get("evidence", [])

    # Format Evidence Section
    evidence_text = ""
    if not evidence_list:
        evidence_text = "No retrieved evidence available for this ingredient."
    else:
        for i, ev in enumerate(evidence_list, 1):
            text = ev.get("text", "")
            score = ev.get("score", 0.0)
            source = ev.get("source", {})
            org = source.get("organization", "Unknown")
            title = source.get("title", "Unknown")
            url = source.get("url", "Unknown")

            evidence_text += f"\n--- Evidence {i} (Score: {score:.4f}) ---\n"
            evidence_text += f"Text: {text}\n"
            evidence_text += f"Source: {org} - {title} ({url})\n"

    prompt = f"""You are an analytical assistant for a Food Ingredient Recognition system called LabelLens.
Your task is to interpret retrieved factual evidence about a food ingredient and explain it clearly to a consumer.

IMPORTANT RULES:
1. USE ONLY THE SUPPLIED EVIDENCE. Do not invent, hallucinate, or assume regulatory or scientific facts.
2. If the supplied evidence does not contain information for a field, output null or "Not available from retrieved evidence."
3. Do not provide medical diagnosis or claim an ingredient is universally "safe" or "unsafe" without explicit context from the evidence.
4. Respect the Phase 2 identification status. It is authoritative. Do not treat low-confidence (ambiguous/unknown) identification as certain. Do not silently correct OCR errors.
5. Clearly distinguish identified facts from uncertainty.

-----------------------------
INGREDIENT INFORMATION
-----------------------------
Canonical ingredient name: {ingredient}
Raw OCR name: {raw_name}
Phase 2 identification status: {status}
Phase 2 confidence/similarity: {similarity}

-----------------------------
RETRIEVED EVIDENCE
-----------------------------
{evidence_text}

-----------------------------
TASK
-----------------------------
Generate a clear, objective explanation using ONLY the supplied evidence.
If evidence is insufficient, say so clearly.
Return a structured JSON object matching the required schema.
"""
    return prompt


def _render_batch_prompt_variant(
    phase3_items: list,
    product_name: str = "Scanned Product",
    max_evidence_chars: int = 220,
    max_chunks_per_item: int = 2,
    omit_unknown_evidence: bool = False,
) -> str:
    """
    Renders the batch prompt with specific evidence truncation and chunk limits.
    """
    from src.llm.token_budget import compact_evidence_text

    prompt = f"""Product: {product_name}

STRICT GROUNDING & SAFETY RULES:
1. USE ONLY THE SUPPLIED EVIDENCE. Do not invent, hallucinate, or assume regulatory, scientific, or nutritional facts.
2. If the supplied evidence does not contain information for a field, output null or "Not available from retrieved evidence."
3. Do not provide medical diagnosis or claim an ingredient is universally "safe", "unsafe", "healthy", or "unhealthy" unless explicitly stated in the evidence.
4. Respect the Phase 2 identification status for each ingredient. It is authoritative. Do not treat ambiguous or unknown ingredients as confirmed. Do not correct OCR identities or rename ingredients.
5. For generic categories (e.g. Vitamins, Minerals, Cereal Extract, Added Flavours, Permitted Natural Colour), DO NOT map them to specific chemicals. If evidence is missing, state 'Not available from retrieved evidence.'
6. For additive codes (e.g. INS 150d -> Caramel IV, INS 320 -> Butylated Hydroxyanisole), strictly preserve the normalized identity.
7. Clearly distinguish identified facts from uncertainty.
8. Provide an entry for EVERY ingredient listed below, matching ingredient_id exactly.
9. Keep explanations concise: 1 brief sentence each for 'summary', 'function', and 'regulatory_information' to ensure complete, untruncated output.

-----------------------------
INGREDIENTS TO ANALYZE
-----------------------------
"""

    for idx, item in enumerate(phase3_items, 1):
        item_id = item.get("id") or item.get("ingredient_id") or f"ing-{idx-1}"
        ingredient = item.get("ingredient", "Unknown")
        raw_name = item.get("raw_name", "Unknown")
        # Strip duplicated nested curly brackets from parent raw name to save tokens
        if "{" in raw_name and "}" in raw_name:
            raw_display = raw_name.split("{")[0].strip()
        else:
            raw_display = raw_name

        status = item.get("phase2_status", "unknown")
        evidence_list = item.get("evidence", [])

        prompt += f"\n=== INGREDIENT #{idx} ===\n"
        prompt += f"Ingredient ID: {item_id}\n"
        prompt += f"Normalized Name: {ingredient}\n"
        if raw_display.strip().lower() != ingredient.strip().lower():
            prompt += f"Raw OCR Name: {raw_display}\n"
        code = item.get("code")
        aliases = item.get("aliases_detected", [])
        if code:
            prompt += f"Additive Code: {code}\n"
        if aliases:
            # Filter aliases that duplicate the normalized ingredient or code
            filtered_aliases = [
                a for a in aliases
                if a.strip().lower() not in (ingredient.strip().lower(), (code or "").strip().lower())
            ]
            if filtered_aliases:
                prompt += f"Detected Aliases/Common Names: {', '.join(filtered_aliases)}\n"
        prompt += f"Identification Status: {status}\n"

        if not evidence_list or (omit_unknown_evidence and status == "unknown"):
            prompt += "Evidence: No retrieved evidence available.\n"
        else:
            prompt += "Evidence:\n"
            seen = set()
            count = 0
            for ev in evidence_list:
                raw_text = ev.get("text", "").strip()
                if not raw_text or raw_text in seen:
                    continue
                seen.add(raw_text)
                count += 1
                if count > max_chunks_per_item:
                    break
                source = ev.get("source", {})
                org = source.get("organization", "Unknown")
                title = source.get("title", "")
                src_str = f"{org}" + (f" - {title}" if title else "")
                compact_text = compact_evidence_text(raw_text, max_chars=max_evidence_chars)
                prompt += f"  [{count}] {compact_text} [Source: {src_str}]\n"

    prompt += f"""
-----------------------------
TASK
-----------------------------
Analyze ALL {len(phase3_items)} ingredients above in ONE response.
Return a structured JSON object matching the BatchAnalysisResult schema with an entry for each ingredient.
Ensure ingredient_id matches exactly.
"""
    return prompt


def build_batch_analysis_prompt(
    phase3_items: list,
    product_name: str = "Scanned Product",
    safe_budget: int = 6500,
    auto_compact: bool = True,
) -> str:
    """
    Constructs a single bulk prompt for the LLM based on all Phase 3 ingredients.

    Note: `safe_budget` and `auto_compact` are preserved for backward-compatible
    API contracts. Prompt budget protection is enforced deterministically by
    _render_batch_prompt_variant using compact evidence text, capped chunks per
    item, and omission of ungrounded unknown-item evidence.
    """
    return _render_batch_prompt_variant(
        phase3_items=phase3_items,
        product_name=product_name,
        max_evidence_chars=220,
        max_chunks_per_item=2,
        omit_unknown_evidence=True,
    )

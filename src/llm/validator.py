def validate_llm_analysis(phase2_status: str, phase3_evidence: list, llm_output: dict) -> dict:
    """
    Validates and sanitizes the LLM output against pipeline invariants.
    The application cannot blindly trust the LLM.

    Args:
        phase2_status (str): The authoritative status from Phase 2.
        phase3_evidence (list): The retrieved evidence from Phase 3.
        llm_output (dict): The parsed JSON output from the LLM.

    Returns:
        dict: The validated and potentially sanitized LLM output, or a safe fallback.
    """
    if not isinstance(llm_output, dict):
        return _fallback_error("LLM output is not a valid dictionary.")

    if "error" in llm_output:
        return _fallback_error(llm_output["error"])

    # 1. Structure validation
    required_keys = ["ingredient", "identification", "summary", "uncertainties"]
    for key in required_keys:
        if key not in llm_output:
            return _fallback_error(f"Missing required field: {key}")

    if not isinstance(llm_output.get("identification"), dict):
        return _fallback_error("Identification field is not a dictionary.")

    # 2. Phase 2 Status Protection. Identity is authoritative upstream: an
    # explanation model must never upgrade, downgrade, or reinterpret it.
    llm_ident = llm_output["identification"]
    llm_status = llm_ident.get("status", "unknown")
    if llm_status != phase2_status:
        llm_ident["status"] = phase2_status
        if "uncertainties" not in llm_output or not isinstance(llm_output["uncertainties"], list):
            llm_output["uncertainties"] = []
        llm_output["uncertainties"].append(
            f"Validation Warning: LLM status was replaced with authoritative Phase 2 status '{phase2_status}'."
        )

    # 3. No-Evidence Validation
    # If there is no evidence, the LLM cannot be trusted to provide scientific/regulatory claims.
    if not phase3_evidence:
        llm_output["category"] = "Not available from retrieved evidence."
        llm_output["function"] = "Not available from retrieved evidence."
        llm_output["summary"] = "Not available from retrieved evidence."
        llm_output["regulatory_information"] = "Not available from retrieved evidence."
        llm_output["sources"] = []
        if "uncertainties" not in llm_output or not isinstance(llm_output["uncertainties"], list):
            llm_output["uncertainties"] = []
        if "No retrieved evidence" not in str(llm_output["uncertainties"]):
            llm_output["uncertainties"].append("No authoritative evidence found in Knowledge Base.")

    return llm_output

def _sanitize_error_message(reason: str) -> str:
    """Sanitizes sensitive credentials or raw token metrics while preserving error context."""
    if not reason:
        return "Unknown error."
    low = reason.lower()
    if any(k in low for k in ["413", "429", "tokens per minute", "tpm limit", "org-", "gsk_", "rate limit"]):
        return "Service capacity limit reached."
    return reason


def _fallback_error(reason: str, ingredient_name: str = "Unknown", status: str = "unknown") -> dict:
    """Returns a safe, neutral fallback dictionary if validation catastrophically fails."""
    from src.audit_logger import STAGE_SAFETY_FALLBACK_SELECTED, audit_log

    sanitized = _sanitize_error_message(reason)
    audit_log(
        STAGE_SAFETY_FALLBACK_SELECTED,
        f"Fallback for '{ingredient_name}' (status={status}): {sanitized}"
    )
    return {
        "ingredient": ingredient_name,
        "identification": {
            "status": status,
            "confidence": "none",
            "explanation": "Validation failed.",
        },
        "summary": "Detailed analysis is temporarily unavailable.",
        "uncertainties": [f"Validation Error: {sanitized}"],
        "category": None,
        "function": None,
        "regulatory_information": None,
        "sources": [],
    }


def validate_batch_llm_analysis(
    expected_items: list[dict], batch_output: dict
) -> tuple[dict[str, dict], list[str]]:
    """
    Validates a batch LLM response against expected Phase 2 ingredients.

    Args:
        expected_items: List of dictionaries with 'id', 'ingredient', 'phase2_status', 'evidence'.
        batch_output: The parsed JSON dictionary from the LLM, containing 'ingredients'.

    Returns:
        tuple[dict[str, dict], list[str]]:
            - Mapping of {ingredient_id: validated_analysis_dict}
            - List of validation warnings / issues encountered
    """
    from src.audit_logger import STAGE_VALIDATION_COMPLETED, audit_log

    validated_map: dict[str, dict] = {}
    warnings: list[str] = []

    if not isinstance(batch_output, dict):
        warnings.append("Validation Error: Batch output is not a dictionary.")
        for item in expected_items:
            iid = item.get("id") or item.get("ingredient_id", "unknown")
            validated_map[iid] = _fallback_error(
                "Batch output is not a dictionary",
                item.get("ingredient", "Unknown"),
                item.get("phase2_status", "unknown"),
            )
        audit_log(STAGE_VALIDATION_COMPLETED, f"Validation completed with non-dict failure ({len(validated_map)} items fallback)")
        return validated_map, warnings

    if "error" in batch_output:
        err = batch_output["error"]
        warnings.append(f"Validation Error: LLM provider reported error: {err}")
        for item in expected_items:
            iid = item.get("id") or item.get("ingredient_id", "unknown")
            validated_map[iid] = _fallback_error(
                f"LLM provider error: {err}",
                item.get("ingredient", "Unknown"),
                item.get("phase2_status", "unknown"),
            )
        audit_log(STAGE_VALIDATION_COMPLETED, f"Validation completed with provider error ({len(validated_map)} items fallback)")
        return validated_map, warnings

    returned_list = batch_output.get("ingredients")
    if not isinstance(returned_list, list):
        warnings.append("Validation Error: 'ingredients' field missing or not a list.")
        for item in expected_items:
            iid = item.get("id") or item.get("ingredient_id", "unknown")
            validated_map[iid] = _fallback_error(
                "'ingredients' array missing in LLM response",
                item.get("ingredient", "Unknown"),
                item.get("phase2_status", "unknown"),
            )
        audit_log(STAGE_VALIDATION_COMPLETED, f"Validation completed with missing ingredients array ({len(validated_map)} items fallback)")
        return validated_map, warnings

    # Map returned items by ingredient_id
    returned_by_id: dict[str, dict] = {}
    for entry in returned_list:
        if isinstance(entry, dict) and "ingredient_id" in entry:
            returned_by_id[entry["ingredient_id"]] = entry

    expected_ids = set()
    for item in expected_items:
        iid = item.get("id") or item.get("ingredient_id", "unknown")
        expected_ids.add(iid)
        name = item.get("ingredient", "Unknown")
        status = item.get("phase2_status", "unknown")
        evidence = item.get("evidence", [])

        # Check 1: Ingredient dropped by LLM
        if iid not in returned_by_id:
            warnings.append(f"Dropped Ingredient: '{name}' ({iid}) was omitted by LLM.")
            validated_map[iid] = _fallback_error(
                f"Ingredient '{name}' ({iid}) was omitted by LLM.", name, status
            )
            continue

        raw_entry = returned_by_id[iid]
        # Validate individual entry using standard validation invariants
        validated_entry = validate_llm_analysis(status, evidence, raw_entry)
        validated_map[iid] = validated_entry

    # Check 2: Unexpected added ingredients
    for entry in returned_list:
        if isinstance(entry, dict):
            entry_id = entry.get("ingredient_id")
            if entry_id and entry_id not in expected_ids:
                warnings.append(
                    f"Unexpected Ingredient: LLM hallucinated extra ingredient with ID '{entry_id}'."
                )

    audit_log(
        STAGE_VALIDATION_COMPLETED,
        f"Validation completed for {len(validated_map)} items ({len(warnings)} warnings)"
    )
    return validated_map, warnings

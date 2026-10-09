import re


UNREADABLE_INGREDIENT_MESSAGE = (
    "The ingredient text could not be read reliably. Please check the original packaging."
)


def serialize_for_frontend(phase4_results: list, phase2_summary: dict, product_name: str = "Scanned Product") -> dict:
    """
    Adapts the internal pipeline output (Phase 4 results + Phase 2 summary)
    into the strict Consumer JSON schema required by the frontend.
    Preserves parent-child hierarchy, aliases, INS codes, and allergens.
    """

    # Map internal authoritative statuses to consumer terminology
    status_mapping = {
        "matched": "identified",
        "ambiguous": "review",
        "unknown": "unclear"
    }

    def _extract_primary_source(authoritative_evidence: list) -> dict:
        if not authoritative_evidence:
            return None
        # Get the first piece of evidence's source
        first_evidence = authoritative_evidence[0]
        source = first_evidence.get("source", {})
        if source and "organization" in source:
            return {
                "organization": source.get("organization", "Unknown"),
                "url": source.get("url", "#")
            }
        return None

    def _build_consumer_item(item, idx, prefix=""):
        llm = item.get("llm_analysis", {})
        internal_status = item.get("phase2_status", "unknown")
        consumer_status = status_mapping.get(internal_status, "unclear")
        candidate_code = item.get("code")
        # Only expose canonical code forms for ingredients that passed normalization.
        # OCR digits (for example, "458" inside a damaged fragment) are not codes.
        verified_code = (
            candidate_code
            if internal_status == "matched"
            and isinstance(candidate_code, str)
            and re.fullmatch(r"(?:INS\s*\d{2,4}[a-z]?|E\s*\d{2,4}[a-z]?)", candidate_code.strip(), re.IGNORECASE)
            else None
        )
        ingredient_id = f"ing-{prefix}{idx}"

        # Build "what" text (Category + Summary)
        what_parts = []
        if llm.get("category"):
            what_parts.append(llm.get("category"))
        if llm.get("summary"):
            what_parts.append(llm.get("summary"))
        what_text = UNREADABLE_INGREDIENT_MESSAGE if consumer_status == "unclear" else (
            " • ".join(what_parts) if what_parts else ("Detailed analysis is temporarily unavailable." if internal_status == "matched" else "Not available.")
        )

        # Build "know" text (Regulatory + Uncertainties)
        know_parts = []
        if llm.get("regulatory_information"):
            know_parts.append(llm.get("regulatory_information"))
        uncertainties = llm.get("uncertainties", [])
        if uncertainties:
            clean_unc = []
            for u in uncertainties:
                if any(x in u.lower() for x in ["413", "429", "rate limit", "groq", "token", "tpm", "org-", "validation error"]):
                    clean_unc.append("Detailed analysis is temporarily unavailable.")
                else:
                    clean_unc.append(u)
            clean_unc = list(dict.fromkeys(clean_unc))
            know_parts.append("Note: " + " ".join(clean_unc))
        know_text = " ".join(know_parts) if know_parts else "Standard food ingredient."

        source_dict = _extract_primary_source(item.get("authoritative_evidence", []))

        # Recursively process sub-ingredients
        sub_consumer_items = []
        for sub_idx, sub in enumerate(item.get("sub_ingredients", [])):
            sub_consumer_items.append(
                _build_consumer_item(sub, sub_idx, prefix=f"{prefix}{idx}-sub-")
            )

        if consumer_status == "unclear":
            why_text = know_text = UNREADABLE_INGREDIENT_MESSAGE
        else:
            why_text = llm.get("function", "Not available.")

        return {
            "id": ingredient_id,
            "name": item.get("ingredient", "Unknown"),
            "canonical_name": item.get("canonical_name", item.get("ingredient", "Unknown")),
            "code": verified_code,
            "aliases": item.get("aliases_detected", []) if consumer_status != "unclear" else [],
            "status": consumer_status,
            "what": what_text,
            "why": why_text,
            "know": know_text,
            "source": source_dict,
            "original_text": item.get("raw_name", item.get("ingredient", "")),
            "sub_ingredients": sub_consumer_items
        }

    ingredients = [
        _build_consumer_item(item, idx) for idx, item in enumerate(phase4_results)
    ]

    # Remap summary counts from phase2_summary, or calculate directly from ingredients array
    if phase2_summary and phase2_summary.get("total", 0) > 0:
        summary = {
            "total": phase2_summary.get("total", 0),
            "identified": phase2_summary.get("matched", 0),
            "review": phase2_summary.get("ambiguous", 0),
            "unclear": phase2_summary.get("unknown", 0)
        }
    else:
        def _count_recursive(items):
            tot = len(items)
            ident = sum(1 for i in items if i.get("status") == "identified")
            rev = sum(1 for i in items if i.get("status") == "review")
            unc = sum(1 for i in items if i.get("status") == "unclear")
            for i in items:
                if i.get("sub_ingredients"):
                    t, idt, rv, uc = _count_recursive(i["sub_ingredients"])
                    tot += t
                    ident += idt
                    rev += rv
                    unc += uc
            return tot, ident, rev, unc

        total, identified, review, unclear = _count_recursive(ingredients)
        summary = {
            "total": total,
            "identified": identified,
            "review": review,
            "unclear": unclear,
        }

    allergens = (phase2_summary.get("allergens") if phase2_summary else None) or {"contains": [], "may_contain": []}

    return {
        "product_name": product_name,
        "summary": summary,
        "ingredients": ingredients,
        "allergens": allergens
    }

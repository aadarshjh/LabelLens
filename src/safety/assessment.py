"""Conservative, deterministic safety summaries from retrieved KB evidence."""

import re
import math


_SUPPORTED_FOOD_USE = re.compile(
    r"\b(?:generally recognized as safe|gras|no safety concern|without safety concerns|"
    r"safe at (?:the )?(?:permitted|reported|typical) food|approved .* food additive|"
    r"affirmed as safe|affirmed as generally recognized as safe)\b",
    re.IGNORECASE,
)
_ALLERGEN = re.compile(
    r"\b(?:allergen|allergic|allergy|contains (?:milk|dairy|soy|wheat|egg|peanut|tree nut|fish|shellfish))\b",
    re.IGNORECASE,
)
_USE_LIMIT = re.compile(
    r"\b(?:category-specific|specified food categories|permitted .* (?:levels|uses)|"
    r"not exceeding|maximum (?:use )?level|use level|exposure limit|restricted use)\b",
    re.IGNORECASE,
)
_ADVERSE = re.compile(
    r"\b(?:raises? safety concerns?|(?:a|there (?:is|are)) safety concerns?|adverse (?:effect|reaction)|toxic at food-use levels|"
    r"not considered safe|not safe for)\b",
    re.IGNORECASE,
)


def _sentences(text, pattern):
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if pattern.search(sentence)]


def _references(evidence):
    refs = []
    seen = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        section = item.get("section")
        primary = item.get("source")
        sources = item.get("sources")
        candidates = ([primary] if isinstance(primary, dict) else [])
        if isinstance(sources, list):
            candidates.extend(source for source in sources if isinstance(source, dict))
        if not candidates:
            candidates = [{}]
        for source in candidates:
            title = source.get("title") or source.get("organization") or item.get("filename") or "Knowledge-base reference"
            url = source.get("url")
            ref = {
                "title": title if isinstance(title, str) else "Knowledge-base reference",
                "url": url if isinstance(url, str) else None,
                "section": section if isinstance(section, str) else None,
            }
            for field in ("organization", "jurisdiction", "document_date", "source_type"):
                value = source.get(field)
                if isinstance(value, str):
                    ref[field] = value
            claims = source.get("claims_supported")
            if isinstance(claims, list) and all(isinstance(value, str) for value in claims):
                ref["claims_supported"] = claims
            key = (ref["title"], ref["url"], ref["section"])
            if key not in seen:
                refs.append(ref)
                seen.add(key)
    return refs


def _established_restriction_violations(evidence):
    """Summarize only structured, product-specific restriction violations.

    A category/use-limit caveat is not enough: evidence must identify the
    product and applicable categories as the same, give observed and maximum
    levels in the same units, and show that the observed level exceeds the
    maximum. These fields are optional; ordinary RAG chunks without them do
    not establish a violation.
    """
    violations = []
    for item in evidence:
        if not isinstance(item, dict) or item.get("section") not in {"regulatory", "assessment"}:
            continue
        restriction = item.get("restriction_assessment")
        if not isinstance(restriction, dict):
            continue

        product_category = restriction.get("product_category")
        applicable_category = restriction.get("applicable_category")
        observed = restriction.get("observed_level")
        maximum = restriction.get("maximum_level")
        unit = restriction.get("unit")
        if not all(isinstance(value, str) and value.strip() for value in (
            product_category, applicable_category, unit
        )):
            continue
        if product_category.strip().casefold() != applicable_category.strip().casefold():
            continue
        if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
                or not math.isfinite(observed) or not math.isfinite(maximum)
                or observed <= maximum):
            continue

        violations.append(
            f"For {product_category.strip()}, the observed use level ({observed:g} {unit.strip()}) "
            f"exceeds the applicable maximum ({maximum:g} {unit.strip()})."
        )
    return list(dict.fromkeys(violations))


def assess_ingredient_safety(ingredient):
    """Return a UI-safe assessment based only on identity and retrieved evidence."""
    evidence = ingredient.get("safety_evidence") or ingredient.get("evidence", ingredient.get("authoritative_evidence", []))
    evidence = evidence if isinstance(evidence, list) else []
    sections = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        section = item.get("section")
        text = item.get("text", item.get("content", ""))
        if isinstance(section, str) and isinstance(text, str):
            sections.setdefault(section, []).append(text)
    regulatory_assessment = " ".join(
        text for section in ("regulatory", "assessment")
        for text in sections.get(section, [])
    )
    function_text = " ".join(sections.get("function", []))

    identity_status = ingredient.get("phase2_status", ingredient.get("status", "unknown"))
    safety_text = regulatory_assessment
    safe_sentences = _sentences(safety_text, _SUPPORTED_FOOD_USE)
    adverse_sentences = _sentences(safety_text, _ADVERSE)
    allergen_sentences = _sentences(safety_text, _ALLERGEN)
    restriction_violations = _established_restriction_violations(evidence)
    base = {
        "status": "insufficient_evidence",
        "identity": {
            "status": identity_status,
            "match_type": ingredient.get("phase2_match_type", ingredient.get("match_type")),
            "similarity_score": ingredient.get("phase2_similarity", ingredient.get("similarity_score")),
        },
        "diagnostic_reason": None,
        "reason_for_use": function_text or "No food-use function was found in the retrieved evidence.",
        "health_concerns": [],
        "uncertainty": None,
        "evidence": _references(evidence),
    }

    if identity_status != "matched":
        base["diagnostic_reason"] = "uncertain_identity"
        base["uncertainty"] = "Ingredient identity is not sufficiently certain for a safety assessment."
        return base
    if not evidence or not regulatory_assessment.strip():
        # Function/identity chunks do not count as safety evidence. A relevant
        # assessment/regulatory section with no applicable conclusion is
        # distinguished below from a retrieval or coverage gap.
        base["diagnostic_reason"] = "no_relevant_evidence"
        base["uncertainty"] = "Retrieved evidence does not contain a usable regulatory or assessment statement."
        return base
    if safe_sentences and adverse_sentences:
        base["diagnostic_reason"] = "conflicting_evidence"
        base["uncertainty"] = "Retrieved evidence contains conflicting safety statements; no conclusion is assigned."
        return base

    concerns = list(dict.fromkeys(allergen_sentences + adverse_sentences + restriction_violations))
    if concerns:
        base["status"] = "potential_concern"
        base["health_concerns"] = concerns
        if allergen_sentences:
            base["uncertainty"] = "Allergen information applies to people with the relevant allergy and is separate from general food-use safety."
        elif restriction_violations:
            base["uncertainty"] = "Retrieved evidence establishes that this product exceeds an applicable restriction."
        elif safe_sentences:
            base["uncertainty"] = "The retrieved safety statement applies only within the documented food categories or use limits."
        else:
            base["uncertainty"] = "Concern is limited to the context described in the retrieved evidence."
        return base

    # Regulatory caveats remain visible in the safety panel, but do not imply
    # that this product violates a restriction when its category or level is
    # unknown. Keep this ahead of the general safe rule: a conditional listing
    # alone cannot establish the product-specific conclusion either way.
    limit_sentences = _sentences(safety_text, _USE_LIMIT)
    if limit_sentences:
        base["diagnostic_reason"] = "insufficient_applicable_evidence"
        caveat = " ".join(limit_sentences)
        base["uncertainty"] = (
            f"Regulatory context: {caveat} The product category and use level were not established, "
            "so compliance cannot be determined and no violation is inferred."
        )
        return base

    if safe_sentences:
        base["status"] = "generally_safe"
        base["health_concerns"] = []
        base["uncertainty"] = "This conclusion is limited to the documented food uses and conditions; it does not assess personal allergies or individual exposure."
        return base

    base["uncertainty"] = "The retrieved evidence does not support a clear safety classification."
    base["diagnostic_reason"] = "insufficient_applicable_evidence"
    return base


def assess_ingredient_tree(nodes):
    """Copy and assess every retrieved ingredient and nested sub-ingredient."""
    assessed = []
    for node in nodes:
        copied = dict(node)
        copied["safety_assessment"] = assess_ingredient_safety(copied)
        copied["sub_ingredients"] = assess_ingredient_tree(node.get("sub_ingredients", []))
        assessed.append(copied)
    return assessed

from src.frontend.serializer import serialize_for_frontend
from src.safety.assessment import assess_ingredient_safety, assess_ingredient_tree


def node(status="matched", evidence=None, **extra):
    return {
        "ingredient": "Citric Acid",
        "phase2_status": status,
        "evidence": evidence or [],
        **extra,
    }


def chunk(section, text, source=None):
    return {
        "section": section,
        "text": text,
        "source": source or {"organization": "Example regulator", "url": "https://example.org/source"},
        "filename": "example.txt",
    }


def local_additive_evidence(filename):
    from pathlib import Path
    import re

    text = Path("data/knowledge_base/documents", filename).read_text(encoding="utf-8")
    source = {
        "organization": "Food Safety and Standards Authority of India (FSSAI)",
        "title": "Compendium of Food Additives Regulations, 31 January 2022",
        "url": "https://fssai.gov.in/upload/uploadfiles/files/Compendium_Food_Additives_Regulations_31_01_2022.pdf",
    }

    def section(name):
        match = re.search(
            rf"^{re.escape(name)}:\s*\n(.*?)(?=\n\n[A-Z][^\n]*:\s*\n|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        assert match, f"Missing {name} section in {filename}"
        return chunk(name.lower().replace(" information", ""), match.group(1).strip(), source)

    return [section("Function"), section("Regulatory Information"), section("Assessment Notes")]


def test_supported_food_use_evidence_classifies_with_context_and_source():
    result = assess_ingredient_safety(node(evidence=[
        chunk("function", "Function: Used as an acidity regulator."),
        chunk("regulatory", "Regulatory Information: Evaluated for food use and presents no safety concern at typical uses."),
    ]))
    assert result["status"] == "generally_safe"
    assert "acidity regulator" in result["reason_for_use"]
    assert "documented food uses" in result["uncertainty"]
    assert result["evidence"][0]["url"] == "https://example.org/source"


def test_dedicated_safety_evidence_can_supply_sections_outside_existing_rag_top_k():
    result = assess_ingredient_safety(node(
        evidence=[chunk("identity", "Ingredient: Citric Acid"), chunk("function", "Used as an acidity regulator.")],
        safety_evidence=[
            chunk("function", "Used as an acidity regulator."),
            chunk("regulatory", "Presents no safety concern at typical food uses."),
        ],
    ))
    assert result["status"] == "generally_safe"
    assert "acidity regulator" in result["reason_for_use"]


def test_unknown_identity_does_not_get_classified_from_matching_name_or_evidence():
    result = assess_ingredient_safety(node("unknown", [
        chunk("regulatory", "GRAS and no safety concern at typical uses.")
    ]))
    assert result["status"] == "insufficient_evidence"
    assert "identity" in result["uncertainty"].lower()
    assert result["diagnostic_reason"] == "uncertain_identity"


def test_missing_regulatory_evidence_is_insufficient():
    result = assess_ingredient_safety(node(evidence=[chunk("function", "Used as a thickener.")]))
    assert result["status"] == "insufficient_evidence"
    assert result["diagnostic_reason"] == "no_relevant_evidence"
    assert result["identity"]["status"] == "matched"


def test_no_relevant_retrieval_has_distinct_diagnostic_reason():
    result = assess_ingredient_safety(node(evidence=[], retrieval_status="no_relevant_evidence"))
    assert result["status"] == "insufficient_evidence"
    assert result["diagnostic_reason"] == "no_relevant_evidence"


def test_identity_confidence_is_reported_separately_from_evidence_diagnostic():
    result = assess_ingredient_safety(node(
        evidence=[chunk("regulatory", "Listed as a permitted additive in selected food categories.")],
        phase2_match_type="exact_alias",
        phase2_similarity=100,
    ))
    assert result["status"] == "insufficient_evidence"
    assert result["diagnostic_reason"] == "insufficient_applicable_evidence"
    assert result["identity"] == {
        "status": "matched",
        "match_type": "exact_alias",
        "similarity_score": 100,
    }


def test_conflicting_safety_statements_are_insufficient():
    result = assess_ingredient_safety(node(evidence=[chunk(
        "assessment", "No safety concern at permitted food uses. A safety concern was reported in another assessment."
    )]))
    assert result["status"] == "insufficient_evidence"
    assert "conflicting" in result["uncertainty"]
    assert result["diagnostic_reason"] == "conflicting_evidence"


def test_conflicts_across_multiple_chunks_in_same_section_are_preserved():
    result = assess_ingredient_safety(node(evidence=[
        chunk("regulatory", "No safety concern at typical food uses."),
        chunk("regulatory", "A safety concern was reported at food-use exposure."),
    ]))
    assert result["status"] == "insufficient_evidence"
    assert result["diagnostic_reason"] == "conflicting_evidence"


def test_local_caramel_iv_category_caveat_is_not_a_product_violation():
    result = assess_ingredient_safety(node(
        ingredient="Caramel IV",
        evidence=local_additive_evidence("caramel_iv.txt"),
    ))
    assert result["status"] == "insufficient_evidence"
    assert result["health_concerns"] == []
    assert "category-specific" in result["uncertainty"].lower()
    assert "no violation is inferred" in result["uncertainty"].lower()
    assert any(ref["url"].endswith("Compendium_Food_Additives_Regulations_31_01_2022.pdf") for ref in result["evidence"])


def test_local_bha_category_caveat_is_not_a_product_violation():
    result = assess_ingredient_safety(node(
        ingredient="Butylated Hydroxyanisole",
        evidence=local_additive_evidence("butylated_hydroxyanisole.txt"),
    ))
    assert result["status"] == "insufficient_evidence"
    assert result["health_concerns"] == []
    assert "category-specific" in result["uncertainty"].lower()
    assert "no violation is inferred" in result["uncertainty"].lower()
    assert any(ref["url"].endswith("Compendium_Food_Additives_Regulations_31_01_2022.pdf") for ref in result["evidence"])


def test_product_specific_restriction_violation_uses_applicable_category_and_levels():
    evidence = chunk(
        "assessment",
        "A documented product review records its category and measured use level against the applicable maximum.",
    )
    evidence["restriction_assessment"] = {
        "product_category": "savory snack",
        "applicable_category": "savory snack",
        "observed_level": 250,
        "maximum_level": 200,
        "unit": "mg/kg",
    }

    result = assess_ingredient_safety(node(evidence=[evidence]))
    assert result["status"] == "potential_concern"
    assert result["health_concerns"] == [
        "For savory snack, the observed use level (250 mg/kg) exceeds the applicable maximum (200 mg/kg)."
    ]


def test_restriction_numbers_do_not_apply_when_product_category_does_not_match():
    evidence = chunk("assessment", "A restriction applies to the listed category.")
    evidence["restriction_assessment"] = {
        "product_category": "savory snack",
        "applicable_category": "beverage",
        "observed_level": 250,
        "maximum_level": 200,
        "unit": "mg/kg",
    }
    assert assess_ingredient_safety(node(evidence=[evidence]))["status"] == "insufficient_evidence"


def test_allergen_context_is_potential_concern_not_general_harm():
    result = assess_ingredient_safety(node(evidence=[
        chunk("regulatory", "Affirmed as Generally Recognized as Safe for food use."),
        chunk("assessment", "Contains milk protein and is classified as a major food allergen requiring declaration."),
    ]))
    assert result["status"] == "potential_concern"
    assert "allergy" in result["uncertainty"].lower()
    assert "milk" in " ".join(result["health_concerns"]).lower()


def test_recursive_assessment_covers_nested_ingredients_without_mutating_input():
    original = node(evidence=[chunk("regulatory", "No safety concern at typical food uses.")], sub_ingredients=[
        node("unknown", evidence=[])
    ])
    assessed = assess_ingredient_tree([original])
    assert assessed[0]["safety_assessment"]["status"] == "generally_safe"
    assert assessed[0]["sub_ingredients"][0]["safety_assessment"]["status"] == "insufficient_evidence"
    assert "safety_assessment" not in original


def test_serializer_adds_safety_assessment_and_preserves_existing_fields():
    result = serialize_for_frontend([node(evidence=[
        chunk("function", "Used as a flavor.")
    ], llm_analysis={"function": "Flavoring"}, authoritative_evidence=[])], {"total": 1, "matched": 1})
    item = result["ingredients"][0]
    assert item["safety_assessment"]["status"] == "insufficient_evidence"
    assert item["status"] == "identified"
    assert item["why"] == "Flavoring"
    assert "allergens" in result


def test_serializer_adds_safety_assessments_to_nested_ingredient_results():
    parent = node(
        evidence=[chunk("regulatory", "No safety concern at typical food uses.")],
        sub_ingredients=[node("unknown", evidence=[])],
    )
    result = serialize_for_frontend([parent], {"total": 2, "matched": 1, "unknown": 1})
    assert result["ingredients"][0]["safety_assessment"]["status"] == "generally_safe"
    assert result["ingredients"][0]["sub_ingredients"][0]["safety_assessment"]["status"] == "insufficient_evidence"


def test_frontend_renders_safety_and_dynamic_values_without_html_interpolation():
    from pathlib import Path

    html = Path("frontend/index.html").read_text()
    assert "const renderIngredient = (item, index" in html
    assert "data.name) || 'Unknown ingredient'" in html
    assert "data.safety_assessment" in html
    assert "data.sub_ingredients.forEach" in html
    assert "makeExternalLink" in html
    assert "containsList" in html and "mayContainList" in html
    assert "Status counts include nested sub-ingredients" in html
    assert "Top-level label entries" in html
    assert "insertAdjacentHTML" not in html
    assert "innerHTML" not in html


def test_frontend_safety_badge_maps_all_statuses_and_falls_back_conservatively():
    from pathlib import Path

    html = Path("frontend/index.html").read_text()
    assert "generally_safe: 'Generally safe'" in html
    assert "potential_concern: 'Potential concern'" in html
    assert "insufficient_evidence: 'Insufficient evidence'" in html
    assert "assessment.status : 'insufficient_evidence'" in html
    assert "safety-${safety.status}" in html
    assert "Insufficient evidence” means the available evidence does not support a conclusion" in html


def test_static_allergen_and_identification_labels_remain_present():
    from pathlib import Path

    html = Path("frontend/index.html").read_text()
    assert "Contains" in html
    assert "May contain" in html
    assert "Identified" in html
    assert "Needs a closer look" in html
    assert "Not identified" in html

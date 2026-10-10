"""
LabelLens Phase B — Semantic Normalization & Ingredient Structure Test Suite.
Tests Cases A through J as defined in Phase B requirements.
"""

from pathlib import Path
import pytest

from src.extraction.extractor import (
    parse_ingredients,
    clean_ocr_text,
    detect_ingredient_section,
    extract_allergens,
)
from src.normalization.normalizer import (
    load_ingredient_dictionary,
    normalize_ingredient,
    normalize_all,
)
from src.frontend.serializer import serialize_for_frontend
from src.pipeline.pipeline import run_pre_llm_pipeline


def test_case_a_nested_hierarchy_wheat_solids():
    """Case A: Wheat Solids parent with nested Wheat Flour (Atta) and Refined Wheat Flour (Maida).
    Must preserve hierarchy, not flatten into independent top-level items.
    """
    text = "Wheat Solids (62.6%) {Wheat Flour (Atta) (31.3%), Refined Wheat Flour (Maida) (31.3%)}"
    parsed = parse_ingredients(text)
    assert len(parsed) == 1
    wheat_solids = parsed[0]
    assert wheat_solids["name"] == "Wheat Solids"
    assert len(wheat_solids["sub_ingredients"]) == 2
    names = [sub["name"] for sub in wheat_solids["sub_ingredients"]]
    assert "Wheat Flour" in names
    assert "Refined Wheat Flour" in names


def test_case_b_vernacular_alias_recognition():
    """Case B: Atta and Maida recognized as vernacular aliases/common names,
    not detached sub-ingredients or separate independent items.
    """
    text = "Wheat Solids (62.6%) {Wheat Flour (Atta) (31.3%), Refined Wheat Flour (Maida) (31.3%)}"
    parsed = parse_ingredients(text)
    subs = parsed[0]["sub_ingredients"]

    # Wheat Flour
    wf = next(s for s in subs if s["name"] == "Wheat Flour")
    assert wf["aliases_detected"] == ["Atta"]
    assert len(wf["sub_ingredients"]) == 0

    # Refined Wheat Flour
    rwf = next(s for s in subs if s["name"] == "Refined Wheat Flour")
    assert rwf["aliases_detected"] == ["Maida"]
    assert len(rwf["sub_ingredients"]) == 0

    # Normalization check
    dict_path = Path("data/ingredients.json")
    dictionary = load_ingredient_dictionary(str(dict_path))
    normalized = normalize_all(parsed, dictionary)
    norm_subs = normalized[0]["sub_ingredients"]
    norm_wf = next(s for s in norm_subs if "Wheat Flour" in s["canonical_name"])
    assert norm_wf["aliases_detected"] == ["Atta"]
    norm_rwf = next(s for s in norm_subs if "Refined Wheat Flour" in s["canonical_name"])
    assert norm_rwf["aliases_detected"] == ["Maida"]


def test_case_c_ins_150d_strict_resolution():
    """Case C: Colour (INS 150d) must resolve strictly to Caramel IV (Category: Colour),
    never Tartrazine or other codes.
    """
    dict_path = Path("data/ingredients.json")
    dictionary = load_ingredient_dictionary(str(dict_path))

    item = {
        "name": "Colour (INS 150d)",
        "raw": "Colour (INS 150d)",
        "code": "INS 150d",
        "aliases_detected": ["INS 150d"],
        "sub_ingredients": [],
    }
    result = normalize_ingredient(item, dictionary)

    assert result["status"] == "matched"
    assert result["canonical_name"] == "Caramel IV"
    assert result["code"] == "INS 150d"
    assert result["similarity_score"] == 100.0
    assert result["canonical_name"] != "Tartrazine"


def test_case_d_ins_320_strict_resolution():
    """Case D: Antioxidant (INS 320) must resolve strictly to Butylated Hydroxyanisole (BHA),
    never Lactic Acid or Ascorbic Acid.
    """
    dict_path = Path("data/ingredients.json")
    dictionary = load_ingredient_dictionary(str(dict_path))

    item = {
        "name": "Antioxidant (INS 320)",
        "raw": "Antioxidant (INS 320)",
        "code": "INS 320",
        "aliases_detected": ["INS 320"],
        "sub_ingredients": [],
    }
    result = normalize_ingredient(item, dictionary)

    assert result["status"] == "matched"
    assert result["canonical_name"] == "Butylated Hydroxyanisole"
    assert result["code"] == "INS 320"
    assert result["similarity_score"] == 100.0
    assert result["canonical_name"] not in ("Lactic Acid", "Ascorbic Acid")


def test_punctuation_and_spacing_variants_match_exactly():
    dictionary = load_ingredient_dictionary("data/ingredients.json")
    result = normalize_ingredient("CITRIC-ACID", dictionary)
    assert result["canonical_name"] == "Citric Acid"
    assert result["match_type"] == "exact_canonical"
    assert normalize_ingredient("E 260", dictionary)["canonical_name"] == "Acetic Acid"


def test_ocr_typo_can_fuzzy_match_when_candidate_is_unambiguous():
    dictionary = load_ingredient_dictionary("data/ingredients.json")
    result = normalize_ingredient("Citric Acic", dictionary)
    assert result["canonical_name"] == "Citric Acid"
    assert result["match_type"] == "fuzzy"


def test_common_ocr_digit_substitution_inside_name_is_corrected_conservatively():
    dictionary = load_ingredient_dictionary("data/ingredients.json")
    result = normalize_ingredient("Citric Ac1d", dictionary)
    assert result["canonical_name"] == "Citric Acid"
    assert result["match_type"] == "fuzzy"


def test_ins_codes_remain_strict_for_existing_cases():
    dictionary = load_ingredient_dictionary("data/ingredients.json")
    for code, expected in [("INS 150d", "Caramel IV"), ("INS 320", "Butylated Hydroxyanisole")]:
        result = normalize_ingredient(code, dictionary)
        assert result["canonical_name"] == expected
        assert result["match_type"] == "exact_alias"


def test_multi_code_parent_does_not_duplicate_first_additive():
    dictionary = load_ingredient_dictionary("data/ingredients.json")
    parsed = parse_ingredients("(INS 150d, INS 102)")
    normalized = normalize_all(parsed, dictionary)
    assert normalized[0]["status"] == "unknown"
    assert [child["canonical_name"] for child in normalized[0]["sub_ingredients"]] == [
        "Caramel IV", "Tartrazine"
    ]


def test_water_is_recognized_as_water():
    dictionary = load_ingredient_dictionary("data/ingredients.json")
    result = normalize_all(parse_ingredients("WATER"), dictionary)[0]
    assert result["canonical_name"] == "Water"
    assert result["status"] == "matched"


def test_case_e_edible_vegetable_oil_palmolein():
    """Case E: Edible Vegetable Oil (Palmolein) preserved with Palmolein
    represented as descriptor/alias/sub-ingredient.
    """
    dict_path = Path("data/ingredients.json")
    dictionary = load_ingredient_dictionary(str(dict_path))

    text = "Edible Vegetable Oil (Palmolein)"
    parsed = parse_ingredients(text)
    assert len(parsed) == 1
    oil = parsed[0]
    assert "Edible Vegetable Oil" in oil["name"]
    assert "palmolein" in [a.lower() for a in oil["aliases_detected"]]

    normalized = normalize_all(parsed, dictionary)
    norm_oil = normalized[0]
    assert norm_oil["canonical_name"] == "Edible Vegetable Oil"
    assert "palmolein" in [a.lower() for a in norm_oil["aliases_detected"]]


def test_case_f_generic_categories_remain_ambiguous():
    """Case F: Generic ingredient categories (Vitamins, Minerals, Cereal Extract,
    Added Flavours, Permitted Natural Colour) must remain ambiguous,
    never fuzzy-matched to specific unrelated compounds (e.g. Vitamins must NOT become Ascorbic Acid).
    """
    dict_path = Path("data/ingredients.json")
    dictionary = load_ingredient_dictionary(str(dict_path))

    generic_samples = [
        "Vitamins",
        "Minerals",
        "Cereal Extract",
        "Added Flavours",
        "Permitted Natural Colour",
        "Colour",
        "Antioxidant",
    ]

    for g in generic_samples:
        item = {"name": g, "raw": g, "sub_ingredients": []}
        result = normalize_ingredient(item, dictionary)
        assert result["status"] == "ambiguous", (
            f"Expected '{g}' to be ambiguous, got {result['status']} (canonical: {result.get('canonical_name')})"
        )
        assert result["canonical_name"] != "Ascorbic Acid"
        assert result["canonical_name"] != "Citric Acid"
        assert result["canonical_name"] != "Tartrazine"


def test_case_g_allergen_extraction():
    """Case G: Allergen declarations (CONTAINS GLUTEN. MAY CONTAIN TRACES OF TREENUT.)
    must be extracted separately into a structured allergen dictionary
    and excluded from the ingredient list.
    """
    ocr_text = (
        "INGREDIENTS: Wheat Solids, Sugar, Cocoa Solids, Salt\n"
        "CONTAINS GLUTEN. MAY CONTAIN TRACES OF TREENUT.\n"
        "STORAGE INSTRUCTIONS: Store in a cool dry place."
    )
    allergens = extract_allergens(ocr_text)
    assert "Gluten" in allergens["contains"]
    assert any("tree" in a.lower() or "nut" in a.lower() for a in allergens["may_contain"])

    section = detect_ingredient_section(ocr_text)
    cleaned = clean_ocr_text(section)
    parsed = parse_ingredients(cleaned)
    names = [p["name"].lower() for p in parsed]
    assert not any("contains gluten" in n for n in names)
    assert not any("may contain" in n for n in names)


def test_case_h_serializer_preserves_hierarchy():
    """Case H: Hierarchy and aliases preserved through serialize_for_frontend()
    without flattening, including allergen block.
    """
    phase4_mock = [
        {
            "ingredient": "Wheat Solids",
            "canonical_name": "Wheat Solids",
            "phase2_status": "matched",
            "raw_name": "Wheat Solids (62.6%)",
            "code": None,
            "aliases_detected": [],
            "sub_ingredients": [
                {
                    "ingredient": "Wheat Flour",
                    "canonical_name": "Wheat Flour",
                    "phase2_status": "matched",
                    "raw_name": "Wheat Flour (Atta) (31.3%)",
                    "code": None,
                    "aliases_detected": ["Atta"],
                    "sub_ingredients": [],
                    "llm_analysis": {"category": "Flour", "function": "Structure"},
                },
                {
                    "ingredient": "Refined Wheat Flour",
                    "canonical_name": "Refined Wheat Flour",
                    "phase2_status": "matched",
                    "raw_name": "Refined Wheat Flour (Maida) (31.3%)",
                    "code": None,
                    "aliases_detected": ["Maida"],
                    "sub_ingredients": [],
                    "llm_analysis": {"category": "Flour", "function": "Structure"},
                },
            ],
            "llm_analysis": {"category": "Grain", "function": "Base"},
        },
        {
            "ingredient": "Caramel IV",
            "canonical_name": "Caramel IV",
            "phase2_status": "matched",
            "raw_name": "Colour (INS 150d)",
            "code": "INS 150d",
            "aliases_detected": ["INS 150d"],
            "sub_ingredients": [],
            "llm_analysis": {"category": "Food Colour", "function": "Colouring agent"},
        },
    ]
    summary_mock = {
        "total": 3,
        "matched": 3,
        "ambiguous": 0,
        "unknown": 0,
        "allergens": {"contains": ["Gluten"], "may_contain": ["Tree nuts"]},
    }

    serialized = serialize_for_frontend(phase4_mock, summary_mock)

    assert len(serialized["ingredients"]) == 2
    ws = serialized["ingredients"][0]
    assert ws["name"] == "Wheat Solids"
    assert len(ws["sub_ingredients"]) == 2

    wf = ws["sub_ingredients"][0]
    assert wf["name"] == "Wheat Flour"
    assert wf["aliases"] == ["Atta"]

    caramel = serialized["ingredients"][1]
    assert caramel["code"] == "INS 150d"

    assert "Gluten" in serialized["allergens"]["contains"]
    assert "Tree nuts" in serialized["allergens"]["may_contain"]


def test_case_i_e2e_pre_llm_pipeline_label3():
    """Case I: Pre-LLM pipeline on label3.jpeg produces correct hierarchical structure,
    codes, aliases, and allergens.
    """
    label3_path = Path("label3.jpeg")
    if not label3_path.exists():
        pytest.skip("label3.jpeg not found in project root")

    pre_llm = run_pre_llm_pipeline(str(label3_path))

    # Allergens
    allergens = pre_llm["phase2"]["summary"].get("allergens", {})
    assert "Gluten" in allergens.get("contains", [])
    assert any("tree" in a.lower() or "nut" in a.lower() for a in allergens.get("may_contain", []))

    # Find Wheat Solids
    p2_items = pre_llm["phase2"]["ingredients"]
    ws = next((item for item in p2_items if "Wheat Solids" in item["canonical_name"]), None)
    assert ws is not None
    assert len(ws["sub_ingredients"]) == 2

    # Wheat Flour with Atta alias
    wf = next((sub for sub in ws["sub_ingredients"] if "Wheat Flour" in sub["canonical_name"]), None)
    assert wf is not None
    assert "Atta" in wf["aliases_detected"]

    # Refined Wheat Flour with Maida alias
    rwf = next((sub for sub in ws["sub_ingredients"] if "Refined Wheat Flour" in sub["canonical_name"]), None)
    assert rwf is not None
    assert "Maida" in rwf["aliases_detected"]

    # Colour (INS 150d) -> Caramel IV
    colour_item = next((item for item in p2_items if item.get("code") == "INS 150d"), None)
    assert colour_item is not None
    assert colour_item["canonical_name"] == "Caramel IV"
    assert colour_item["status"] == "matched"

    # Antioxidant (INS 320) -> Butylated Hydroxyanisole
    antiox_item = next((item for item in p2_items if item.get("code") == "INS 320"), None)
    assert antiox_item is not None
    assert antiox_item["canonical_name"] == "Butylated Hydroxyanisole"
    assert antiox_item["status"] == "matched"


def test_case_j_label1_regression_safety():
    """Case J: Existing label1 text continues to produce its expected ingredients
    and passes without regressions.
    """
    from src.pipeline.ingredient_pipeline import run_pipeline
    text = "Ingredients: Corn Meal, Edible Vegetable Oil, Seasoning (Maltodextrin, Iodised Salt, Spices & Condiments, Sugar, Milk Solids (Whey Powder, Milk Powder, Cheese Powder), Salt Substitute (Potassium Chloride), Acidity Regulators (270, 330, 260), Flavour (Natural and Nature Identical Flavouring Substances), Anticaking Agent (551))"
    p2_items = run_pipeline(text)["ingredients"]
    all_names = []

    def collect(items):
        for it in items:
            all_names.append(it.get("canonical_name", "").lower())
            all_names.append(it.get("raw_name", it.get("raw", "")).lower())
            collect(it.get("sub_ingredients", []))

    collect(p2_items)

    # Expected label1 ingredients
    assert any("corn meal" in n for n in all_names)
    assert any("sugar" in n for n in all_names)
    assert any("oil" in n for n in all_names)
    assert any("maltodextrin" in n for n in all_names)


def test_label1_ocr_artifact_normalization():
    from src.normalization.normalizer import normalize_ingredient
    from src.normalization.normalizer import load_ingredient_dictionary
    dictionary = load_ingredient_dictionary("data/ingredients.json")

    # Com Med -> Corn Meal
    res1 = normalize_ingredient("Com Med", dictionary)
    assert res1["canonical_name"] == "Corn Meal"
    assert res1["status"] == "matched"

    # Edible Vegetable Sega ct Hs Oil) -> Edible Vegetable Oil
    res2 = normalize_ingredient("Edible Vegetable Sega ct Hs Oil)", dictionary)
    assert res2["canonical_name"] == "Edible Vegetable Oil"
    assert res2["status"] == "ambiguous"

    # Potassium sate T Fara Chloride -> Potassium Chloride
    res3 = normalize_ingredient("Potassium sate T Fara Chloride", dictionary)
    assert res3["canonical_name"] == "Potassium Chloride"
    assert res3["status"] == "matched"

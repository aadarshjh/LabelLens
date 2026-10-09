"""
test_phase2.py — Tests for Phase 2: Ingredient Extraction & Normalization
===========================================================================

Covers:
  - Normal comma-separated ingredients
  - Multiline OCR text
  - Compound ingredients with parentheses
  - Nested brackets
  - INS/E-number alias resolution
  - OCR spelling errors (fuzzy matching)
  - Unknown ingredients
  - Ambiguous/generic ingredients
  - Missing Ingredients heading
  - Stop at Nutrition Information / other sections
  - Real OCR output from Phase 1

Run with:
    pytest tests/ -v
"""

import os
import sys
import pytest

# Add src/ to path so we can import the modules
from src.extraction.extractor import (
    detect_ingredient_section,
    clean_ocr_text,
    parse_ingredients,
    _split_at_top_level_commas,
)
from src.normalization.normalizer import (
    load_ingredient_dictionary,
    normalize_ingredient,
    normalize_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dictionary():
    """Load the real ingredient dictionary."""
    dict_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "ingredients.json"
    )
    return load_ingredient_dictionary(dict_path)


# ===========================================================================
# EXTRACTOR TESTS
# ===========================================================================

class TestSectionDetection:
    """Test ingredient section detection from OCR text."""

    def test_basic_ingredients_heading(self):
        text = "Product Name\nIngredients: Sugar, Salt, Flour\nNet Weight: 100g"
        result = detect_ingredient_section(text)
        assert result is not None
        assert "Sugar" in result
        assert "Salt" in result
        assert "Net Weight" not in result

    def test_ingredients_with_space_before_colon(self):
        """OCR sometimes adds a space before the colon."""
        text = "Ingredients : Sugar, Salt, Flour"
        result = detect_ingredient_section(text)
        assert result is not None
        assert "Sugar" in result

    def test_multiline_ingredients(self):
        """Ingredient lists often wrap across multiple lines."""
        text = (
            "PRODUCT NAME\n"
            "Ingredients: Sugar, Salt,\n"
            "Flour, Butter, Milk\n"
            "Net Wt: 200g\n"
        )
        result = detect_ingredient_section(text)
        assert result is not None
        assert "Sugar" in result
        assert "Milk" in result
        # Stop marker should exclude Net Wt line
        assert "200g" not in result

    def test_stop_at_nutrition_facts(self):
        text = (
            "Ingredients: Sugar, Salt\n"
            "Nutrition Facts\n"
            "Calories: 100\n"
        )
        result = detect_ingredient_section(text)
        assert "Sugar" in result
        assert "Calories" not in result

    def test_stop_at_ocr_corrupted_nutrition_heading_and_numbering_footnote(self):
        text = (
            "Ingredients: Sugar, INS 150d\n"
            "(INS - INTERNATIONAL NUMBERING SYSTEM)\n"
            "SUTOITIONAL INFORMATION (TYPICAL VALUES)\nCalories: 100"
        )
        result = detect_ingredient_section(text)
        assert "Sugar" in result
        assert "INTERNATIONAL NUMBERING SYSTEM" not in result
        assert "SUTOITIONAL INFORMATION" not in result

    def test_stop_at_batch_number(self):
        text = (
            "Ingredients: Sugar, Salt\n"
            "Batch No.: 12345\n"
        )
        result = detect_ingredient_section(text)
        assert "Sugar" in result
        assert "12345" not in result

    def test_stop_at_product_of(self):
        text = (
            "Ingredients: Sugar, Salt\n"
            "PRODUCT OF INDIA\n"
        )
        result = detect_ingredient_section(text)
        assert "Sugar" in result
        assert "INDIA" not in result

    def test_missing_ingredients_heading(self):
        """When there's no 'Ingredients:' heading, return None."""
        text = "Sugar, Salt, Flour, Butter"
        result = detect_ingredient_section(text)
        assert result is None

    def test_empty_text(self):
        assert detect_ingredient_section("") is None
        assert detect_ingredient_section(None) is None

    def test_stop_at_allergen_information(self):
        text = (
            "Ingredients: Sugar, Salt\n"
            "Allergen Information: Contains milk\n"
        )
        result = detect_ingredient_section(text)
        assert "Sugar" in result
        assert "Contains milk" not in result

    def test_case_insensitive_start_marker(self):
        text = "INGREDIENTS: Sugar, Salt"
        result = detect_ingredient_section(text)
        assert result is not None
        assert "Sugar" in result


class TestTextCleaning:
    """Test OCR artifact removal."""

    def test_pipe_artifacts(self):
        """OCR often produces stray pipe characters from table borders."""
        text = "=| Milk Powder, | Cheese |_| Powder"
        result = clean_ocr_text(text)
        assert "=" not in result
        assert "|" not in result
        assert "Milk Powder" in result

    def test_stray_quotes(self):
        text = '"Seasoning (Sugar, Salt)"'
        result = clean_ocr_text(text)
        assert result.startswith("Seasoning")
        assert '"' not in result

    def test_asterisk_removal(self):
        text = "*Seasoning (Sugar, *Salt)"
        result = clean_ocr_text(text)
        assert "*" not in result
        assert "Seasoning" in result

    def test_duplicate_whitespace(self):
        text = "Sugar,   Salt,    Flour"
        result = clean_ocr_text(text)
        assert "   " not in result
        assert "Sugar, Salt, Flour" == result

    def test_tilde_brace_artifact(self):
        text = "~} PRODUCT OF INDIA"
        result = clean_ocr_text(text)
        assert "~" not in result

    def test_preserves_parentheses(self):
        text = "Seasoning (Sugar, Salt)"
        result = clean_ocr_text(text)
        assert "(" in result and ")" in result

    def test_preserves_ampersand(self):
        text = "Spices & Condiments"
        result = clean_ocr_text(text)
        assert "&" in result

    def test_preserves_spelling(self):
        """Cleaning must NOT autocorrect OCR misspellings."""
        text = "Corn Med, Mik Solids"
        result = clean_ocr_text(text)
        assert "Corn Med" in result   # NOT "Corn Meal"
        assert "Mik Solids" in result  # NOT "Milk Solids"


class TestBracketAwareParsing:
    """Test bracket-aware ingredient parsing."""

    def test_simple_comma_separated(self):
        result = parse_ingredients("Sugar, Salt, Flour")
        names = [i["name"] for i in result]
        assert names == ["Sugar", "Salt", "Flour"]

    def test_compound_ingredient(self):
        """Compound ingredient with sub-ingredients in parentheses."""
        result = parse_ingredients("Seasoning (Sugar, Salt, Spices)")
        assert len(result) == 1
        assert result[0]["name"] == "Seasoning"
        sub_names = [s["name"] for s in result[0]["sub_ingredients"]]
        assert sub_names == ["Sugar", "Salt", "Spices"]

    def test_nested_brackets(self):
        """Nested compound ingredients."""
        text = "Seasoning (Sugar, Milk Solids (Whey, Cheese Powder))"
        result = parse_ingredients(text)
        assert len(result) == 1
        assert result[0]["name"] == "Seasoning"

        subs = result[0]["sub_ingredients"]
        assert len(subs) == 2
        assert subs[0]["name"] == "Sugar"
        assert subs[1]["name"] == "Milk Solids"

        # Nested sub-ingredients
        nested = subs[1]["sub_ingredients"]
        nested_names = [n["name"] for n in nested]
        assert nested_names == ["Whey", "Cheese Powder"]

    def test_mixed_simple_and_compound(self):
        text = "Corn Meal, Vegetable Oil (Palmolein Oil), Sugar"
        result = parse_ingredients(text)
        assert len(result) == 3
        assert result[0]["name"] == "Corn Meal"
        assert result[1]["name"] == "Vegetable Oil"
        assert result[1]["sub_ingredients"][0]["name"] == "Palmolein Oil"
        assert result[2]["name"] == "Sugar"

    def test_numeric_codes_in_parens(self):
        """INS codes like Acidity Regulators (270, 330, 260)."""
        text = "Acidity Regulators (270, 330, 260)"
        result = parse_ingredients(text)
        assert len(result) == 1
        assert result[0]["name"] == "Acidity Regulators"
        sub_names = [s["name"] for s in result[0]["sub_ingredients"]]
        assert "270" in sub_names
        assert "330" in sub_names
        assert "260" in sub_names

    def test_unbalanced_brackets(self):
        """OCR may produce unbalanced brackets — parser should not crash."""
        text = "Seasoning (Sugar, Salt"  # Missing closing paren
        result = parse_ingredients(text)
        assert len(result) == 1
        assert result[0]["name"] == "Seasoning"
        assert len(result[0]["sub_ingredients"]) == 2

    def test_empty_input(self):
        assert parse_ingredients("") == []
        assert parse_ingredients(None) == []

    def test_split_top_level_commas(self):
        """Verify comma splitting respects bracket depth."""
        result = _split_at_top_level_commas("A, B (C, D), E")
        stripped = [s.strip() for s in result]
        assert stripped == ["A", "B (C, D)", "E"]


# ===========================================================================
# NORMALIZER TESTS
# ===========================================================================

class TestExactMatching:
    """Test exact canonical and alias matching."""

    def test_exact_canonical(self, dictionary):
        result = normalize_ingredient("Sugar", dictionary)
        assert result["status"] == "matched"
        assert result["match_type"] == "exact_canonical"
        assert result["normalized"] == "Sugar"
        assert result["similarity_score"] == 100

    def test_exact_canonical_case_insensitive(self, dictionary):
        result = normalize_ingredient("sugar", dictionary)
        assert result["status"] == "matched"
        assert result["match_type"] == "exact_canonical"

    def test_exact_alias(self, dictionary):
        """MSG should resolve to Monosodium Glutamate via alias."""
        result = normalize_ingredient("MSG", dictionary)
        assert result["status"] == "matched"
        assert result["match_type"] == "exact_alias"
        assert result["normalized"] == "Monosodium Glutamate"

    def test_ins_code(self, dictionary):
        """INS 621 should resolve to Monosodium Glutamate."""
        result = normalize_ingredient("INS 621", dictionary)
        assert result["status"] == "matched"
        assert result["match_type"] == "exact_alias"
        assert result["normalized"] == "Monosodium Glutamate"

    def test_e_code(self, dictionary):
        """E330 should resolve to Citric Acid."""
        result = normalize_ingredient("E330", dictionary)
        assert result["status"] == "matched"
        assert result["match_type"] == "exact_alias"
        assert result["normalized"] == "Citric Acid"

    def test_bare_number_code(self, dictionary):
        """Plain '330' should resolve to Citric Acid via alias."""
        result = normalize_ingredient("330", dictionary)
        assert result["status"] == "matched"
        assert result["match_type"] == "exact_alias"
        assert result["normalized"] == "Citric Acid"

    def test_palmolein_alias(self, dictionary):
        result = normalize_ingredient("Palmolein Oil", dictionary)
        assert result["status"] == "matched"
        assert result["normalized"] == "Palm Olein Oil"


class TestFuzzyMatching:
    """Test fuzzy matching for OCR errors."""

    def test_ocr_error_corn_med(self, dictionary):
        """'Corn Med' is a common OCR misread of 'Corn Meal'."""
        result = normalize_ingredient("Corn Med", dictionary)
        # Should fuzzy match to Corn Meal
        assert result["status"] == "matched"
        assert result["match_type"] == "fuzzy"
        assert result["normalized"] == "Corn Meal"
        assert result["similarity_score"] >= 80

    def test_ocr_error_mik_solids(self, dictionary):
        """'Mik Solids' is OCR misread of 'Milk Solids'."""
        result = normalize_ingredient("Mik Solids", dictionary)
        assert result["normalized"] == "Milk Solids"
        assert result["match_type"] == "fuzzy"

    def test_exact_match_beats_fuzzy(self, dictionary):
        """Exact match must always take priority over fuzzy."""
        result = normalize_ingredient("Sugar", dictionary)
        assert result["match_type"] == "exact_canonical"
        # Should NOT be "fuzzy" even though fuzzy would also match

    def test_low_similarity_rejected(self, dictionary):
        """Very dissimilar strings should not match."""
        result = normalize_ingredient("Xylophonium", dictionary,
                                       fuzzy_threshold=80)
        assert result["status"] == "unknown"


class TestAmbiguousIngredients:
    """Test that generic ingredients are flagged as 'ambiguous'."""

    def test_vegetable_oil_ambiguous(self, dictionary):
        result = normalize_ingredient("Edible Vegetable Oil", dictionary)
        assert result["status"] == "ambiguous"
        assert result["normalized"] == "Edible Vegetable Oil"

    def test_spices_ambiguous(self, dictionary):
        result = normalize_ingredient("Spices", dictionary)
        assert result["status"] == "ambiguous"

    def test_seasoning_ambiguous(self, dictionary):
        result = normalize_ingredient("Seasoning", dictionary)
        assert result["status"] == "ambiguous"

    def test_flavouring_substances_ambiguous(self, dictionary):
        result = normalize_ingredient("Flavouring Substances", dictionary)
        assert result["status"] == "ambiguous"


class TestUnknownIngredients:
    """Test that truly unknown ingredients are reported correctly."""

    def test_unknown_ingredient(self, dictionary):
        result = normalize_ingredient("Dilithium Crystal", dictionary)
        assert result["status"] == "unknown"
        assert result["match_type"] is None
        assert result["similarity_score"] is None

    def test_empty_name(self, dictionary):
        result = normalize_ingredient("", dictionary)
        assert result["status"] == "unknown"

    def test_ocr_noise(self, dictionary):
        """Pure OCR noise like '2.96 02.' should be unknown."""
        result = normalize_ingredient("2.96 02.", dictionary)
        assert result["status"] == "unknown"


class TestBulkNormalization:
    """Test normalize_all with parsed ingredient trees."""

    def test_simple_list(self, dictionary):
        parsed = [
            {"name": "Sugar", "raw": "Sugar", "sub_ingredients": []},
            {"name": "Salt", "raw": "Salt", "sub_ingredients": []},
        ]
        result = normalize_all(parsed, dictionary)
        assert len(result) == 2
        # Salt should match Sodium Chloride via alias
        salt = result[1]
        assert salt["normalized"] in ("Sodium Chloride", "Iodised Salt")

    def test_compound_with_sub_ingredients(self, dictionary):
        parsed = [
            {
                "name": "Acidity Regulators",
                "raw": "Acidity Regulators (270, 330)",
                "sub_ingredients": [
                    {"name": "270", "raw": "270", "sub_ingredients": []},
                    {"name": "330", "raw": "330", "sub_ingredients": []},
                ],
            },
        ]
        result = normalize_all(parsed, dictionary)
        assert len(result) == 1

        parent = result[0]
        assert parent["status"] == "ambiguous"  # Acidity Regulator is generic

        subs = parent["sub_ingredients"]
        assert len(subs) == 2
        assert subs[0]["normalized"] == "Lactic Acid"      # 270
        assert subs[1]["normalized"] == "Citric Acid"       # 330


# ===========================================================================
# INTEGRATION TESTS (full pipeline on realistic OCR text)
# ===========================================================================

class TestFullPipeline:
    """End-to-end tests using realistic OCR text."""

    def test_clean_label(self, dictionary):
        """Test with a clean, well-formatted ingredient list."""
        text = (
            "PRODUCT NAME\n"
            "Ingredients: Corn Meal, Sugar, Maltodextrin, "
            "Cheese Powder, Salt\n"
            "Net Weight: 100g\n"
        )
        section = detect_ingredient_section(text)
        assert section is not None

        cleaned = clean_ocr_text(section)
        parsed = parse_ingredients(cleaned)
        normalized = normalize_all(parsed, dictionary)

        names = [n["normalized"] for n in normalized]
        assert "Corn Meal" in names
        assert "Sugar" in names
        assert "Maltodextrin" in names
        assert "Cheese Powder" in names

    def test_real_kurkure_ocr(self, dictionary):
        """Test with the actual OCR output from Phase 1 (Kurkure label)."""
        ocr_text = (
            "KURKURE CHATPATA CHEESE\n"
            "\n"
            "Ingredients : Corn Med, Edible Vegetable\n"
            "Oi (Palmolein Oil), \"Seasoning (Maltodextrin,\n"
            "=| Mik Solids (Whey Powder, Milk Powder, |\n"
            "Cheese Powder), Salt Substitute (Potassium\n"
            "Chloride), Acidity Regulators (270, 330, 260.\n"
            "296), Favour Enhancer (635), Favour Natural\n"
            "and Nature Identical Ravouring Subsatances |_|\n"
            "\n"
            "(2.96 02.)\n"
            "=| Batch No.: N2010522\n"
            "~} PRODUCT OF INDIA\n"
        )

        # Section detection should find ingredients and stop at Batch No
        section = detect_ingredient_section(ocr_text)
        assert section is not None
        assert "Corn Med" in section
        assert "N2010522" not in section

        # Cleaning should remove OCR artifacts
        cleaned = clean_ocr_text(section)
        assert "|" not in cleaned
        assert "=" not in cleaned

        # Parsing should handle brackets correctly
        parsed = parse_ingredients(cleaned)
        assert len(parsed) >= 2  # At least Corn Med and Edible Vegetable Oi

        # Normalize and check key matches
        normalized = normalize_all(parsed, dictionary)

        # Collect all ingredient names (flat, including sub-ingredients)
        all_names = _flatten_names(normalized)

        # These should be found (exact or fuzzy)
        assert any("Corn Meal" in n for n in all_names)  # fuzzy from "Corn Med"
        assert any("Maltodextrin" in n for n in all_names)

    def test_with_ins_codes(self, dictionary):
        """Test INS code resolution in compound ingredients."""
        text = "Ingredients: Flavour Enhancer (635), Anticaking Agent (551)"
        section = detect_ingredient_section(text)
        cleaned = clean_ocr_text(section)
        parsed = parse_ingredients(cleaned)
        normalized = normalize_all(parsed, dictionary)

        all_names = _flatten_names(normalized)
        # 635 should resolve to Disodium 5'-Ribonucleotides
        assert any("Disodium" in n for n in all_names)
        # 551 should resolve to Silicon Dioxide
        assert any("Silicon Dioxide" in n for n in all_names)

    def test_bare_ingredients_heading(self):
        """Test heading 'INGREDIENTS' on its own line without colon or dash."""
        text = "BRAND NAME\nINGREDIENTS\nWheat Solids, Sugar, Cocoa Solids\nNet Wt: 50g"
        section = detect_ingredient_section(text)
        assert section is not None
        assert "Wheat Solids" in section
        assert "Sugar" in section
        assert "Net Wt" not in section

    def test_nested_brackets_and_percentage_filtering(self):
        """Test curly brackets, pure percentage filtering, and vernacular aliases."""
        text = "Wheat Solids (62.6%) {Wheat Flour (Atta) (31.3%), Refined Wheat Flour (Maida) (31.3%)}"
        parsed = parse_ingredients(text)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "Wheat Solids"
        subs = parsed[0]["sub_ingredients"]
        assert len(subs) == 2
        assert subs[0]["name"] == "Wheat Flour"
        assert subs[0]["aliases_detected"] == ["Atta"]
        assert len(subs[0]["sub_ingredients"]) == 0
        assert subs[1]["name"] == "Refined Wheat Flour"
        assert subs[1]["aliases_detected"] == ["Maida"]
        assert len(subs[1]["sub_ingredients"]) == 0

    def test_statutory_flavour_and_colour_declaration(self):
        """Test statutory declaration splitting on '&'."""
        text = "CONTAINS PERMITTED NATURAL COLOUR & ADDED FLAVOURS (NATURE IDENTICAL & ARTIFICIAL CREAM FLAVOURING SUBSTANCES)"
        parsed = parse_ingredients(text)
        assert len(parsed) == 2
        assert "PERMITTED NATURAL COLOUR" in parsed[0]["name"].upper()
        assert "ADDED FLAVOURS" in parsed[1]["name"].upper()
        assert len(parsed[1]["sub_ingredients"]) == 1
        assert "CREAM FLAVOURING" in parsed[1]["sub_ingredients"][0]["name"].upper()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_names(ingredients):
    """Recursively collect all normalized names from the ingredient tree."""
    names = []
    for ing in ingredients:
        names.append(ing["normalized"])
        if ing.get("sub_ingredients"):
            names.extend(_flatten_names(ing["sub_ingredients"]))
    return names

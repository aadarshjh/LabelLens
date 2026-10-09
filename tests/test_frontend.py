import pytest
from src.frontend.serializer import serialize_for_frontend

def test_serializer_consumer_schema():
    """
    Tests that the internal data maps correctly to the consumer schema
    without exposing backend terminology or structures.
    """
    phase2_summary = {"total": 3, "matched": 1, "ambiguous": 1, "unknown": 1}
    phase4_results = [
        {
            "ingredient": "Malic Acid",
            "phase2_status": "matched",
            "llm_analysis": {
                "category": "Organic acid",
                "function": "Acidity regulator",
                "summary": "Imparts sour taste.",
                "regulatory_information": "Generally recognized as safe.",
                "uncertainties": []
            },
            "authoritative_evidence": [
                {
                    "content": "Malic acid is safe...",
                    "source": {"organization": "FDA", "url": "https://fda.gov/..."}
                }
            ],
            "tool_calls": [{"name": "search", "result": "..."}]
        },
        {
            "ingredient": "Unknown Extract",
            "phase2_status": "unknown",
            "llm_analysis": {
                "category": "",
                "function": "",
                "summary": "Could not be verified.",
                "regulatory_information": "",
                "uncertainties": ["Exact composition is unknown."]
            },
            "authoritative_evidence": [],
            "tool_calls": []
        }
    ]

    output = serialize_for_frontend(phase4_results, phase2_summary, "Test Product")

    # 1. Check Product Name
    assert output["product_name"] == "Test Product"

    # 2. Check Summary Mapping
    assert output["summary"]["total"] == 3
    assert output["summary"]["identified"] == 1
    assert output["summary"]["review"] == 1
    assert output["summary"]["unclear"] == 1

    # 3. Check Ingredients list
    assert len(output["ingredients"]) == 2

    # Check first item (identified)
    ing1 = output["ingredients"][0]
    assert ing1["status"] == "identified"
    assert ing1["what"] == "Organic acid • Imparts sour taste."
    assert ing1["why"] == "Acidity regulator"
    assert ing1["know"] == "Generally recognized as safe."
    assert ing1["source"]["organization"] == "FDA"
    assert "tool_calls" not in ing1  # Internal terminology exclusion
    assert "authoritative_evidence" not in ing1
    assert "llm_analysis" not in ing1

    # Check second item (unknown)
    ing2 = output["ingredients"][1]
    assert ing2["status"] == "unclear"
    assert "could not be read reliably" in ing2["know"]
    assert ing2["source"] is None


@pytest.mark.parametrize("fragment", [
    "eal 458 ledsed Sat",
    "WALT",
    "FR: STABILIZER (INS 440)",
])
def test_unclear_ocr_does_not_expose_unverified_code_or_analysis(fragment):
    output = serialize_for_frontend([{
        "ingredient": fragment,
        "canonical_name": fragment,
        "raw_name": fragment,
        "phase2_status": "unknown",
        "code": "458" if "458" in fragment else "INS 440",
        "aliases_detected": ["458"],
        "llm_analysis": {"summary": "Detailed analysis is temporarily unavailable.",
                         "function": "Not available."},
        "authoritative_evidence": [],
    }], {"total": 1, "unknown": 1})

    item = output["ingredients"][0]
    assert item["code"] is None
    assert item["aliases"] == []
    assert item["what"] == "The ingredient text could not be read reliably. Please check the original packaging."
    assert item["why"] == item["know"] == item["what"]
    assert item["source"] is None


@pytest.mark.parametrize("code", ["INS 150d", "INS 320"])
def test_serializer_preserves_verified_additive_codes(code):
    output = serialize_for_frontend([{
        "ingredient": "Recognized additive",
        "phase2_status": "matched",
        "code": code,
    }], {"total": 1, "matched": 1})
    assert output["ingredients"][0]["code"] == code

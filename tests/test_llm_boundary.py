from unittest.mock import MagicMock, patch

import pytest

from src.llm.llm_client import LLMClient, LLMUnavailableError
from src.llm.prompt_builder import build_analysis_prompt
from src.llm.validator import validate_llm_analysis


def evidence_item(status="matched", evidence=True):
    return {
        "ingredient": "Citric Acid",
        "raw_name": "Citric Acid",
        "phase2_status": status,
        "phase2_similarity": 100,
        "evidence": ([{"text": "Category:\nAcidulant\n\nDescription:\nFood acid.", "score": 1.0,
                       "source": {"organization": "Example", "title": "KB", "url": "https://example.test"}}] if evidence else []),
    }


def valid_response(status="matched"):
    return {"ingredient": "Citric Acid", "identification": {"status": status, "confidence": "High", "explanation": "Exact match."}, "summary": "Food acid.", "uncertainties": [], "category": "Acidulant", "function": "Acid", "regulatory_information": None, "sources": ["Example"]}


def test_prompt_contains_identity_evidence_and_provenance():
    prompt = build_analysis_prompt(evidence_item())
    assert "Phase 2 identification status: matched" in prompt
    assert "Example - KB" in prompt


def test_prompt_reports_missing_evidence():
    assert "No retrieved evidence available" in build_analysis_prompt(evidence_item(evidence=False))


def test_missing_api_key_is_explicitly_unavailable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        LLMClient().generate_analysis("test")


def test_client_does_not_claim_mock_mode():
    assert LLMClient(api_key=None).is_mock() is False


def test_groq_client_parses_structured_json():
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ingredient": "Citric Acid"}'
    mock_response = MagicMock(choices=[mock_choice])
    with patch("groq.Groq") as client:
        client.return_value.chat.completions.create.return_value = mock_response
        assert LLMClient(api_key="test-key").generate_analysis("test")["ingredient"] == "Citric Acid"


def test_groq_failure_is_wrapped():
    with patch("groq.Groq") as client:
        client.return_value.chat.completions.create.side_effect = RuntimeError("rate limit")
        with pytest.raises(LLMUnavailableError, match="Groq request failed"):
            LLMClient(api_key="test-key").generate_analysis("test")


def test_validator_preserves_ambiguous_status():
    result = validate_llm_analysis("ambiguous", evidence_item("ambiguous")["evidence"], valid_response("matched"))
    assert result["identification"]["status"] == "ambiguous"


def test_validator_preserves_unknown_status():
    result = validate_llm_analysis("unknown", evidence_item("unknown")["evidence"], valid_response("ambiguous"))
    assert result["identification"]["status"] == "unknown"


def test_validator_strips_claims_when_evidence_is_missing():
    result = validate_llm_analysis("matched", [], valid_response())
    assert result["summary"] == "Not available from retrieved evidence."
    assert result["sources"] == []


def test_validator_returns_safe_unavailable_result_for_provider_error():
    result = validate_llm_analysis("matched", [], {"error": "Groq unavailable"})
    assert result["identification"]["status"] == "unknown"
    assert "Validation Error" in result["uncertainties"][0]


# --- New Groq Batch and Robustness Tests ---


def test_groq_batch_request_creation():
    from src.llm.prompt_builder import build_batch_analysis_prompt
    items = [
        {"id": "ing-0", "ingredient": "Corn Meal", "raw_name": "Corn Med", "phase2_status": "matched", "phase2_similarity": 95, "evidence": [{"text": "Ground corn.", "source": {"organization": "FDA"}}]},
        {"id": "ing-1", "ingredient": "Mystery", "raw_name": "Mystery", "phase2_status": "unknown", "phase2_similarity": "N/A", "evidence": []},
    ]
    prompt = build_batch_analysis_prompt(items)
    assert "=== INGREDIENT #1 ===" in prompt
    assert "Ingredient ID: ing-0" in prompt
    assert "Ingredient ID: ing-1" in prompt
    assert "Identification Status: unknown" in prompt
    assert "FDA" in prompt
    assert "No retrieved evidence available" in prompt


def test_groq_batch_client_parses_structured_json():
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ingredients": [{"ingredient_id": "ing-0", "ingredient": "Corn Meal"}]}'
    mock_response = MagicMock(choices=[mock_choice])
    with patch("groq.Groq") as client:
        client.return_value.chat.completions.create.return_value = mock_response
        res = LLMClient(api_key="test-key").generate_batch_analysis("test")
        assert len(res["ingredients"]) == 1
        assert res["ingredients"][0]["ingredient_id"] == "ing-0"


def test_validate_batch_llm_analysis_success():
    from src.llm.validator import validate_batch_llm_analysis
    expected = [
        {"id": "ing-0", "ingredient": "Corn Meal", "phase2_status": "matched", "evidence": [{"text": "Ground corn", "source": {"organization": "FDA"}}]}
    ]
    batch_output = {
        "ingredients": [
            {
                "ingredient_id": "ing-0",
                "ingredient": "Corn Meal",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Identified"},
                "category": "Cereal",
                "function": "Base",
                "summary": "Milled corn.",
                "regulatory_information": "GRAS",
                "uncertainties": [],
                "sources": ["FDA"]
            }
        ]
    }
    validated_map, warnings = validate_batch_llm_analysis(expected, batch_output)
    assert "ing-0" in validated_map
    assert validated_map["ing-0"]["identification"]["status"] == "matched"
    assert validated_map["ing-0"]["category"] == "Cereal"
    assert len(warnings) == 0


def test_invalid_batch_item_does_not_erase_valid_maltodextrin():
    from src.llm.validator import validate_batch_llm_analysis

    evidence = [{"text": "Maltodextrin is a starch hydrolysate.", "source": {"organization": "FDA"}}]
    expected = [
        {"id": "ing-seasoning", "ingredient": "Seasoning", "phase2_status": "ambiguous", "evidence": []},
        {"id": "ing-maltodextrin", "ingredient": "Maltodextrin", "phase2_status": "matched", "evidence": evidence},
    ]
    batch_output = {"ingredients": [
        {
            "ingredient_id": "ing-seasoning",
            "identification": {"status": "ambiguous", "confidence": "Low", "explanation": "Composite."},
            "category": None, "function": None,
            # Missing summary intentionally makes only this item invalid.
            "regulatory_information": None, "uncertainties": [], "sources": [],
        },
        {
            "ingredient_id": "ing-maltodextrin", "ingredient": "Maltodextrin",
            "identification": {"status": "matched", "confidence": "High", "explanation": "Matched."},
            "category": "Thickener", "function": "Adds bulk.", "summary": "Starch hydrolysate.",
            "regulatory_information": None, "uncertainties": [], "sources": ["FDA"],
        },
    ]}

    validated, warnings = validate_batch_llm_analysis(expected, batch_output)
    assert validated["ing-seasoning"]["summary"] == "Detailed analysis is temporarily unavailable."
    assert validated["ing-maltodextrin"]["summary"] == "Starch hydrolysate."
    assert validated["ing-maltodextrin"]["identification"]["status"] == "matched"
    assert "ing-maltodextrin" in validated


def test_malformed_batch_output_falls_back_per_expected_item():
    from src.llm.validator import validate_batch_llm_analysis

    expected = [evidence_item() | {"id": "ing-0"}]
    validated, warnings = validate_batch_llm_analysis(expected, "not-json")
    assert validated["ing-0"]["summary"] == "Detailed analysis is temporarily unavailable."
    assert warnings and "not a dictionary" in warnings[0]


def test_provider_rate_limit_is_reported_as_unavailable_analysis():
    from src.llm.validator import validate_batch_llm_analysis

    expected = [{"id": "ing-seasoning", "ingredient": "Seasoning", "phase2_status": "ambiguous", "evidence": []}]
    response = {"error": "Groq request failed: Error code: 429 - tokens per day limit reached."}
    validated, warnings = validate_batch_llm_analysis(expected, response)
    assert validated["ing-seasoning"]["summary"] == "Detailed analysis is temporarily unavailable."
    assert warnings and "provider reported error" in warnings[0]


def test_validate_batch_llm_analysis_detects_dropped_ingredient():
    from src.llm.validator import validate_batch_llm_analysis
    expected = [
        {"id": "ing-0", "ingredient": "Corn Meal", "phase2_status": "matched", "evidence": []},
        {"id": "ing-1", "ingredient": "Salt", "phase2_status": "matched", "evidence": []}
    ]
    batch_output = {
        "ingredients": [
            {
                "ingredient_id": "ing-0",
                "ingredient": "Corn Meal",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Identified"},
                "category": None, "function": None, "summary": "Corn", "regulatory_information": None,
                "uncertainties": [], "sources": []
            }
        ]
    }
    validated_map, warnings = validate_batch_llm_analysis(expected, batch_output)
    assert "ing-1" in validated_map
    assert validated_map["ing-1"]["identification"]["status"] == "matched"
    assert "omitted by LLM" in validated_map["ing-1"]["uncertainties"][0]
    assert any("omitted by LLM" in w for w in warnings)


def test_validate_batch_llm_analysis_unexpected_ingredient_warning():
    from src.llm.validator import validate_batch_llm_analysis
    expected = [
        {"id": "ing-0", "ingredient": "Corn Meal", "phase2_status": "matched", "evidence": []}
    ]
    batch_output = {
        "ingredients": [
            {
                "ingredient_id": "ing-0",
                "ingredient": "Corn Meal",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Identified"},
                "category": None, "function": None, "summary": "Corn", "regulatory_information": None,
                "uncertainties": [], "sources": []
            },
            {
                "ingredient_id": "ing-fake",
                "ingredient": "Hallucinated Item",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Fake"},
                "category": None, "function": None, "summary": "Fake", "regulatory_information": None,
                "uncertainties": [], "sources": []
            }
        ]
    }
    validated_map, warnings = validate_batch_llm_analysis(expected, batch_output)
    assert any("hallucinated extra ingredient" in w for w in warnings)


def test_validate_batch_llm_analysis_preserves_ambiguous_and_unknown():
    from src.llm.validator import validate_batch_llm_analysis
    expected = [
        {"id": "ing-0", "ingredient": "Ambiguous Ing", "phase2_status": "ambiguous", "evidence": [{"text": "ev"}]},
        {"id": "ing-1", "ingredient": "Unknown Ing", "phase2_status": "unknown", "evidence": []}
    ]
    # LLM tries to claim both are matched
    batch_output = {
        "ingredients": [
            {
                "ingredient_id": "ing-0",
                "ingredient": "Ambiguous Ing",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Upgraded"},
                "category": None, "function": None, "summary": "Summary", "regulatory_information": None,
                "uncertainties": [], "sources": []
            },
            {
                "ingredient_id": "ing-1",
                "ingredient": "Unknown Ing",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Upgraded"},
                "category": "Cat", "function": "Func", "summary": "Summary", "regulatory_information": "Reg",
                "uncertainties": [], "sources": ["FakeSource"]
            }
        ]
    }
    validated_map, _ = validate_batch_llm_analysis(expected, batch_output)
    assert validated_map["ing-0"]["identification"]["status"] == "ambiguous"
    assert validated_map["ing-1"]["identification"]["status"] == "unknown"
    # ing-1 had no evidence, so claims must be stripped
    assert validated_map["ing-1"]["sources"] == []
    assert validated_map["ing-1"]["category"] == "Not available from retrieved evidence."


def test_groq_strict_json_schema_format():
    """Verify BATCH_ANALYSIS_SCHEMA uses anyOf for nullable fields required by strict structured outputs."""
    from src.llm.llm_client import BATCH_ANALYSIS_SCHEMA

    props = BATCH_ANALYSIS_SCHEMA["json_schema"]["schema"]["properties"]["ingredients"]["items"]["properties"]

    # In strict mode, we cannot use {"type": ["string", "null"]}. It must be {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "anyOf" in props["category"]
    assert "type" not in props["category"] or isinstance(props["category"].get("type"), str)

    ident_props = props["identification"]["properties"]
    assert "anyOf" in ident_props["confidence"]

def test_validate_batch_llm_analysis_missing_fields_but_valid_json():
    from src.llm.validator import validate_batch_llm_analysis
    expected = [
        {"id": "ing-0", "ingredient": "Salt", "phase2_status": "matched", "evidence": []}
    ]
    # Simulate a JSON that's technically valid JSON but missing an expected field (e.g. 'summary')
    # to ensure the validator catches it and falls back gracefully.
    batch_output = {
        "ingredients": [
            {
                "ingredient_id": "ing-0",
                "ingredient": "Salt",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Ok"},
                "category": None,
                "function": None,
                # "summary" is intentionally omitted to trigger a fallback
                "regulatory_information": None,
                "uncertainties": [],
                "sources": []
            }
        ]
    }
    validated_map, warnings = validate_batch_llm_analysis(expected, batch_output)

    # Should fall back to safety defaults
    assert "ing-0" in validated_map
    assert validated_map["ing-0"]["summary"] == "Detailed analysis is temporarily unavailable."
    assert any("Missing required field" in u for u in validated_map["ing-0"]["uncertainties"])

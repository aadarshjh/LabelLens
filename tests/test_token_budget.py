import pytest
from src.llm.prompt_builder import build_batch_analysis_prompt

def test_build_batch_analysis_prompt_protects_budget():
    """Verify that the batch analysis prompt builds correctly and limits evidence text length."""
    phase3_items = [
        {
            "id": "ing-0",
            "ingredient": "Long Ingredient",
            "phase2_status": "matched",
            "evidence": [
                {
                    "text": "A" * 1000,  # Extremely long evidence
                    "score": 0.99,
                    "source": {"organization": "Test", "title": "Doc", "url": "http"}
                }
            ]
        }
    ]
    prompt = build_batch_analysis_prompt(phase3_items)

    # Even with 1000 'A's in evidence, it should truncate down (max_evidence_chars=220)
    assert len(prompt) < 3000
    assert "A" * 300 not in prompt
    assert "A" * 200 in prompt


def test_build_batch_analysis_prompt_parameters_contract():
    """Verify that safe_budget and auto_compact parameters are accepted and backward-compatible."""
    items = [{"id": "ing-1", "ingredient": "Salt", "phase2_status": "matched", "evidence": []}]
    p1 = build_batch_analysis_prompt(items)
    p2 = build_batch_analysis_prompt(items, product_name="Test Product", safe_budget=5000, auto_compact=False)
    assert "Salt" in p1 and "Salt" in p2
    assert "Test Product" in p2


def test_build_batch_analysis_prompt_deduplication_and_compactness():
    """Verify that redundant OCR names and duplicate aliases are omitted from the prompt."""
    items = [
        {
            "id": "ing-0",
            "ingredient": "Sugar",
            "raw_name": "Sugar",  # Identical to ingredient
            "code": None,
            "aliases_detected": ["Sugar", "Sucrose"],  # "Sugar" is redundant
            "phase2_status": "matched",
            "evidence": [
                {
                    "text": "Refined sucrose from sugar cane.",
                    "score": 0.98,
                    "source": {"organization": "FDA", "title": "GRAS"}
                }
            ],
        },
        {
            "id": "ing-1",
            "ingredient": "Caramel IV",
            "raw_name": "Caramel Color (INS 150d)",  # Differs from ingredient
            "code": "INS 150d",
            "aliases_detected": ["INS 150d", "E150d"],  # "INS 150d" is redundant with code
            "phase2_status": "matched",
            "evidence": [],
        },
    ]
    prompt = build_batch_analysis_prompt(items, product_name="Beverage")

    # Item 0: Identical raw name should not have a duplicate "Raw OCR Name: Sugar"
    assert "Raw OCR Name: Sugar" not in prompt
    assert "Normalized Name: Sugar" in prompt
    # Alias "Sugar" stripped, "Sucrose" preserved
    assert "Sucrose" in prompt

    # Item 1: Differing raw name must be included
    assert "Raw OCR Name: Caramel Color (INS 150d)" in prompt
    assert "Additive Code: INS 150d" in prompt
    # Redundant code alias stripped, unique alias preserved
    assert "E150d" in prompt

    # Retrieval score should be omitted to save tokens
    assert "(Score:" not in prompt
    # Provenance source must still be preserved
    assert "[Source: FDA - GRAS]" in prompt


def test_calculate_output_token_budget_adaptive_scaling():
    """Verify adaptive output token calculation across item counts and item complexity."""
    from src.llm.token_budget import calculate_output_token_budget

    # 1 item: should not reserve excessive 1500 tokens, should use adaptive minimum
    b_single = calculate_output_token_budget(1)
    assert 500 <= b_single <= 700

    # 5 items: scales smoothly
    b_five = calculate_output_token_budget(5)
    assert b_single < b_five < 2000

    # 14 items (Label 1 scale): stays within safe bounds, below default 4800
    b_fourteen = calculate_output_token_budget(14)
    assert 3000 <= b_fourteen <= 4000

    # Complexity awareness: items with evidence receive more budget than empty items
    items_rich = [{"ingredient": f"Ing-{i}", "evidence": [{"text": "Sample"}]} for i in range(5)]
    items_empty = [{"ingredient": f"Ing-{i}", "evidence": []} for i in range(5)]
    b_rich = calculate_output_token_budget(5, items=items_rich)
    b_empty = calculate_output_token_budget(5, items=items_empty)
    assert b_rich > b_empty

    # Positional backward compatibility: calculate_output_token_budget(count, max_budget)
    b_capped = calculate_output_token_budget(14, 2500)
    assert b_capped == 2500


def test_check_request_budget_enforces_limits():
    """Verify request budget checking against safe limits."""
    from src.llm.token_budget import check_request_budget

    is_safe, total, msg = check_request_budget("Short prompt", "System", 1000, safe_budget=8000)
    assert is_safe is True
    assert msg == "OK"

    is_safe_exceeded, total_exceeded, msg_exceeded = check_request_budget(
        "A" * 30000, "System", 4000, safe_budget=8000
    )
    assert is_safe_exceeded is False
    assert "exceeds safe budget" in msg_exceeded


def test_batch_analysis_truncation_detection_and_retry():
    """Verify that truncation (finish_reason=length) triggers a bounded retry with expanded budget."""
    from unittest.mock import MagicMock, patch
    from src.llm.llm_client import LLMClient

    # First completion is truncated; second completion succeeds
    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = '{"ingredients": [{"ingredient_id": "ing-0"'
    mock_choice_1.finish_reason = "length"

    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = '{"ingredients": [{"ingredient_id": "ing-0", "ingredient": "Salt", "identification": {"status": "matched", "confidence": "High", "explanation": "Pure"}, "category": "Mineral", "function": "Flavor", "summary": "Salt.", "regulatory_information": "GRAS", "uncertainties": [], "sources": ["FDA"]}]}'
    mock_choice_2.finish_reason = "stop"

    mock_response_1 = MagicMock(choices=[mock_choice_1], usage=MagicMock(prompt_tokens=100, completion_tokens=500, total_tokens=600))
    mock_response_2 = MagicMock(choices=[mock_choice_2], usage=MagicMock(prompt_tokens=100, completion_tokens=800, total_tokens=900))

    client = LLMClient(api_key="test-key")
    with patch("groq.Groq") as mock_groq_class:
        mock_groq = mock_groq_class.return_value
        mock_groq.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

        result = client.generate_batch_analysis("=== INGREDIENT #1 ===\nIngredient ID: ing-0", max_output_tokens=600)
        assert len(result["ingredients"]) == 1
        assert result["ingredients"][0]["ingredient_id"] == "ing-0"

        # Verify retry was called with expanded token budget (600 + 800 = 1400)
        assert mock_groq.chat.completions.create.call_count == 2
        first_call_kwargs = mock_groq.chat.completions.create.call_args_list[0][1]
        second_call_kwargs = mock_groq.chat.completions.create.call_args_list[1][1]
        assert first_call_kwargs["max_tokens"] == 600
        assert second_call_kwargs["max_tokens"] == 1400


def test_batch_analysis_truncation_bounded_to_single_retry():
    """Verify that truncation is bounded to exactly 1 retry before raising LLMUnavailableError."""
    from unittest.mock import MagicMock, patch
    from src.llm.llm_client import LLMClient, LLMUnavailableError

    # Both first call and retry return truncated responses
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ingredients": ['
    mock_choice.finish_reason = "length"
    mock_response = MagicMock(choices=[mock_choice], usage=MagicMock(prompt_tokens=100, completion_tokens=500, total_tokens=600))

    client = LLMClient(api_key="test-key")
    with patch("groq.Groq") as mock_groq_class:
        mock_groq = mock_groq_class.return_value
        mock_groq.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMUnavailableError, match="truncated by token limit"):
            client.generate_batch_analysis("=== INGREDIENT #1 ===\nIngredient ID: ing-0", max_output_tokens=600)

        # Must not loop infinitely: exactly 2 calls (initial + 1 retry)
        assert mock_groq.chat.completions.create.call_count == 2


def test_single_analysis_truncation_raises_unavailable():
    """Verify that single analysis truncation also raises LLMUnavailableError."""
    from unittest.mock import MagicMock, patch
    from src.llm.llm_client import LLMClient, LLMUnavailableError

    mock_choice = MagicMock()
    mock_choice.message.content = '{"ingredient": "Salt"'
    mock_choice.finish_reason = "length"
    mock_response = MagicMock(choices=[mock_choice], usage=MagicMock(total_tokens=2048))

    client = LLMClient(api_key="test-key")
    with patch("groq.Groq") as mock_groq_class:
        mock_groq = mock_groq_class.return_value
        mock_groq.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMUnavailableError, match="truncated by token limit"):
            client.generate_analysis("Analyze salt")

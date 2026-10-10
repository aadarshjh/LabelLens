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

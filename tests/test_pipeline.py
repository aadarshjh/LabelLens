import pytest
from unittest.mock import patch
from src.pipeline.pipeline import run_end_to_end_pipeline
from src.llm.llm_client import LLMClient

@pytest.fixture
def mock_pipeline_dependencies():
    with patch('src.pipeline.pipeline.load_image') as m_load, \
         patch('src.pipeline.pipeline.get_all_strategies') as m_strat, \
         patch('src.pipeline.pipeline.run_ocr_all_psms') as m_ocr, \
         patch('src.pipeline.pipeline.find_best_result') as m_best, \
         patch('src.pipeline.pipeline.run_ingredient_pipeline') as m_p2, \
         patch('src.pipeline.pipeline.IngredientRetriever') as m_ret_cls:

        m_best.return_value = {"text": "some text", "strategy": "mock", "confidence": 99.9}
        m_p2.return_value = {"summary": {"total": 1}, "ingredients": []} # Ignored by retriever anyway
        m_ret = m_ret_cls.return_value

        yield m_ret

@patch.object(LLMClient, 'generate_analysis')
def test_known_ingredient_no_llm(mock_llm, mock_pipeline_dependencies):
    # Phase 3 returns a matched ingredient with evidence
    evidence = [{
        "text": "Ingredient:\nCitric Acid\nCategory:\nAcidity Regulator\nDescription:\nA weak acid.\nFunction:\nAdds sour taste.",
        "content": "Ingredient:\nCitric Acid\nCategory:\nAcidity Regulator\nDescription:\nA weak acid.\nFunction:\nAdds sour taste.",
        "score": 0.99,
        "source": {"organization": "FDA"}
    }]
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [{
        "ingredient": "Citric Acid",
        "raw_name": "citricacid",
        "phase2_status": "matched",
        "evidence": evidence,
        "sub_ingredients": []
    }]

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key")

    assert len(results) == 1
    assert results[0]["used_llm"] is False
    assert mock_llm.call_count == 0
    assert results[0]["llm_analysis"]["category"] == "Acidity Regulator"
    assert results[0]["llm_analysis"]["sources"] == ["FDA"]
    assert results[0]["llm_analysis"]["identification"]["status"] == "matched"

@patch.object(LLMClient, 'generate_analysis')
def test_ambiguous_ingredient_invokes_llm(mock_llm, mock_pipeline_dependencies):
    # Phase 3 returns ambiguous status
    evidence = [{
        "content": "Ingredient: Citric Acid\nCategory: Acidity Regulator\nDescription: A weak acid.\nFunction: Adds sour taste.",
        "score": 0.5,
        "source": {"organization": "FDA"}
    }]
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [{
        "ingredient": "Citric Acid",
        "raw_name": "citric?",
        "phase2_status": "ambiguous",
        "evidence": evidence,
        "sub_ingredients": []
    }]

    mock_llm.return_value = {
        "ingredient": "Citric Acid",
        "identification": {"status": "ambiguous", "confidence": "Medium", "explanation": "LLM explanation"},
        "category": "Acidity Regulator",
        "function": "Adds sour taste",
        "summary": "LLM Summary",
        "regulatory_information": "FDA",
        "uncertainties": [],
        "sources": ["FDA"]
    }

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key")

    assert len(results) == 1
    assert results[0]["used_llm"] is True
    assert mock_llm.call_count == 1
    assert results[0]["llm_analysis"]["identification"]["status"] == "ambiguous"

@patch.object(LLMClient, 'generate_analysis')
def test_unknown_ingredient_preserves_uncertainty(mock_llm, mock_pipeline_dependencies):
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [{
        "ingredient": "Unknown",
        "raw_name": "mystery",
        "phase2_status": "unknown",
        "evidence": [],
        "sub_ingredients": []
    }]

    mock_llm.return_value = {
        "ingredient": "Unknown",
        "identification": {"status": "unknown", "confidence": "Low", "explanation": "Cannot identify"},
        "category": "Unknown",
        "function": "Unknown",
        "summary": "Unknown summary",
        "regulatory_information": "Unknown",
        "uncertainties": ["Unknown ingredient"],
        "sources": []
    }

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key")

    assert len(results) == 1
    assert results[0]["used_llm"] is True
    assert results[0]["phase2_status"] == "unknown"
    assert results[0]["llm_analysis"]["identification"]["status"] == "unknown"

@patch.object(LLMClient, 'generate_analysis')
def test_provider_failure_produces_unavailable(mock_llm, mock_pipeline_dependencies):
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [{
        "ingredient": "Citric Acid",
        "raw_name": "citricacid",
        "phase2_status": "ambiguous",
        "evidence": [],
        "sub_ingredients": []
    }]

    # Simulate LLM failure/timeout
    mock_llm.side_effect = Exception("OpenRouter Timeout")

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key")

    assert len(results) == 1
    assert results[0]["used_llm"] is True
    # Validator handles fallback
    assert results[0]["llm_analysis"]["identification"]["explanation"] == "Validation failed."
    assert "OpenRouter Timeout" in results[0]["llm_analysis"]["uncertainties"][0]

@patch.object(LLMClient, 'generate_analysis')
def test_no_evidence_never_fabricates(mock_llm, mock_pipeline_dependencies):
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [{
        "ingredient": "Guar Gum",
        "raw_name": "guargum",
        "phase2_status": "matched",
        "evidence": [], # No evidence
        "sub_ingredients": []
    }]

    mock_llm.return_value = {
        "ingredient": "Guar Gum",
        "identification": {"status": "matched", "confidence": "High", "explanation": "LLM explanation"},
        "category": "Fake Category", # LLM hallucinated
        "function": "Fake Function",
        "summary": "Fake Summary",
        "regulatory_information": "Fake Reg",
        "uncertainties": [],
        "sources": ["Fake Source"] # LLM hallucinated a source
    }

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key")

    assert len(results) == 1
    assert results[0]["used_llm"] is True

    # Validator should have stripped sources because retrieved_evidence was empty
    assert len(results[0]["llm_analysis"]["sources"]) == 0
    assert "No authoritative evidence found in Knowledge Base." in results[0]["llm_analysis"]["uncertainties"][0]


@patch.object(LLMClient, 'generate_batch_analysis')
def test_batch_pipeline_single_llm_request(mock_batch_llm, mock_pipeline_dependencies):
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [
        {
            "ingredient": "Corn Meal",
            "raw_name": "cornmeal",
            "phase2_status": "matched",
            "evidence": [{"text": "Ground corn", "source": {"organization": "FDA"}}],
            "sub_ingredients": [
                {
                    "ingredient": "Salt",
                    "raw_name": "salt",
                    "phase2_status": "matched",
                    "evidence": [],
                    "sub_ingredients": []
                }
            ]
        }
    ]
    mock_batch_llm.return_value = {
        "ingredients": [
            {
                "ingredient_id": "ing-0",
                "ingredient": "Corn Meal",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Identified"},
                "category": "Cereal", "function": "Base", "summary": "Milled corn", "regulatory_information": None,
                "uncertainties": [], "sources": ["FDA"]
            },
            {
                "ingredient_id": "ing-0-sub-0",
                "ingredient": "Salt",
                "identification": {"status": "matched", "confidence": "High", "explanation": "Identified"},
                "category": None, "function": None, "summary": None, "regulatory_information": None,
                "uncertainties": [], "sources": []
            }
        ]
    }

    results, summary = run_end_to_end_pipeline("dummy.jpg", "mock_key", batch=True)

    assert mock_batch_llm.call_count == 1
    assert len(results) == 1
    assert results[0]["ingredient"] == "Corn Meal"
    assert results[0]["llm_analysis"]["category"] == "Cereal"
    assert len(results[0]["sub_ingredients"]) == 1
    assert results[0]["sub_ingredients"][0]["ingredient"] == "Salt"


@patch.object(LLMClient, 'generate_batch_analysis')
def test_batch_pipeline_failure_safely_handled(mock_batch_llm, mock_pipeline_dependencies):
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [
        {
            "ingredient": "Corn Meal",
            "raw_name": "cornmeal",
            "phase2_status": "matched",
            "evidence": [],
            "sub_ingredients": []
        }
    ]
    mock_batch_llm.side_effect = RuntimeError("Groq connection failure")

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key", batch=True)

    assert mock_batch_llm.call_count == 1
    assert len(results) == 1
    assert "Groq connection failure" in results[0]["llm_analysis"]["uncertainties"][0]


@patch.object(LLMClient, 'generate_batch_analysis')
def test_batch_rate_limit_uses_only_available_retrieved_evidence(mock_batch_llm, mock_pipeline_dependencies):
    seasoning_evidence = [{
        "text": "Ingredient:\nSeasoning\nCategory:\nSeasoning / Flavouring Blend\nDescription:\nA composite blend of spices and condiments.\nFunction:\nProvides savory flavor.\n",
        "source": {"organization": "FSSAI", "title": "Seasoning reference", "url": "https://example.test/seasoning"},
    }]
    maltodextrin_evidence = [{
        "text": "Ingredient:\nMaltodextrin\nCategory:\nThickener/Bulking Agent\nDescription:\nA starch hydrolysate.\nFunction:\nProvides bulk.\n",
        "source": {"organization": "FDA", "title": "Maltodextrin reference", "url": "https://example.test/maltodextrin"},
    }]
    mock_pipeline_dependencies.retrieve_for_phase2.return_value = [
        {"ingredient": "Seasoning", "raw_name": "Seasoning", "phase2_status": "ambiguous", "evidence": seasoning_evidence, "sub_ingredients": []},
        {"ingredient": "Maltodextrin", "raw_name": "Maltodextrin", "phase2_status": "matched", "evidence": maltodextrin_evidence, "sub_ingredients": []},
        {"ingredient": "eal 458 ledsed Sat", "raw_name": "eal 458 ledsed Sat", "phase2_status": "unknown", "evidence": [], "sub_ingredients": []},
    ]
    mock_batch_llm.side_effect = RuntimeError("RateLimitError: tokens per day limit reached")

    results, _ = run_end_to_end_pipeline("dummy.jpg", "mock_key", batch=True)
    by_name = {item["ingredient"]: item for item in results}

    assert by_name["Seasoning"]["llm_analysis"]["category"] == "Seasoning / Flavouring Blend"
    assert by_name["Seasoning"]["llm_analysis"]["function"] == "Provides savory flavor."
    assert by_name["Seasoning"]["llm_analysis"]["identification"]["status"] == "ambiguous"
    assert by_name["Seasoning"]["llm_analysis"]["sources"] == ["FSSAI"]
    assert "not available" not in by_name["Seasoning"]["llm_analysis"]["summary"].lower()
    assert by_name["Maltodextrin"]["llm_analysis"]["summary"] == "A starch hydrolysate."
    assert by_name["eal 458 ledsed Sat"]["llm_analysis"]["summary"] == "Detailed analysis is temporarily unavailable."
    assert by_name["eal 458 ledsed Sat"]["llm_analysis"]["sources"] == []


def test_different_uploaded_image_does_not_reuse_label1():
    """Verify that a distinct label image (label3.jpeg) produces its own ingredients and not label1 demo data."""
    from pathlib import Path
    from src.pipeline.pipeline import run_pre_llm_pipeline
    from src.frontend.serializer import serialize_for_frontend

    label3_path = Path("label3.jpeg")
    if not label3_path.exists():
        pytest.skip("label3.jpeg not found in project root")

    pre_llm = run_pre_llm_pipeline(str(label3_path))
    p2_ingredients = pre_llm["phase2"]["ingredients"]
    assert len(p2_ingredients) > 0

    all_names = []
    def collect(items):
        for it in items:
            all_names.append(it.get("normalized", ""))
            all_names.append(it.get("raw_name", it.get("raw", "")))
            collect(it.get("sub_ingredients", []))
    collect(p2_ingredients)
    all_names_joined = " ".join(all_names).lower()

    # Must contain distinctive text from label3.jpeg
    assert any("wheat" in n for n in all_names_joined.split())
    assert any("cocoa" in n for n in all_names_joined.split())

    # Must NOT contain label1 demo items
    assert "corn meal" not in all_names_joined
    assert "whey powder" not in all_names_joined
    assert "monosodium glutamate" not in all_names_joined
    assert "potassium chloride" not in all_names_joined
    assert "citric acid" not in all_names_joined

    # Serializer summary counts must be non-zero and equal to total ingredient count including nested sub-ingredients
    serialized = serialize_for_frontend(pre_llm["phase3"], pre_llm["phase2"]["summary"])
    def count_all(items):
        cnt = len(items)
        for it in items:
            cnt += count_all(it.get("sub_ingredients", []))
        return cnt
    assert serialized["summary"]["total"] == count_all(serialized["ingredients"])
    assert serialized["summary"]["total"] > 0
    assert serialized["summary"]["total"] == (
        serialized["summary"]["identified"] +
        serialized["summary"]["review"] +
        serialized["summary"]["unclear"]
    )


def test_small_label2_retries_ocr_at_higher_resolution():
    """A small but readable ingredient panel should not become an empty scan."""
    from pathlib import Path
    from src.pipeline.pipeline import run_pre_llm_pipeline

    label2_path = Path("label2.jpeg")
    if not label2_path.exists():
        pytest.skip("label2.jpeg not found in project root")

    result = run_pre_llm_pipeline(str(label2_path))

    assert result["ocr"]["strategy"].startswith("upscale_2x_")
    assert result["phase2"]["ingredient_section"]
    assert result["phase2"]["ingredients"]
    assert any(
        item.get("normalized", "").casefold() == "sugar"
        for item in result["phase2"]["ingredients"]
    )

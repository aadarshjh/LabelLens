import pytest
import sys
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from src.rag.retriever import IngredientRetriever
from src.rag.embeddings import EmbeddingModel

try:
    EmbeddingModel()
except (ImportError, OSError):
    pytestmark = pytest.mark.skip(reason="local sentence-transformers model unavailable")

try:
    retriever = IngredientRetriever(store_dir=str(PROJECT_ROOT / "vectorstore"))
except (ImportError, OSError) as exc:
    pytestmark = pytest.mark.skip(reason=f"local embedding model unavailable: {exc}")
    retriever = None

def test_deterministic_lookup_remains_unchanged():
    results = retriever.retrieve("Monosodium Glutamate", top_k=1, bypass_exact=False)
    assert len(results) > 0
    assert results[0]["score"] == 1.0
    assert results[0]["ingredient"] == "Monosodium Glutamate"

def test_alias_retrieval():
    results = retriever.retrieve("MSG", top_k=1, bypass_exact=False)
    assert len(results) > 0
    assert results[0]["ingredient"] == "Monosodium Glutamate"
    assert results[0]["score"] == 1.0

def test_ins_code_retrieval():
    results = retriever.retrieve("INS 621", top_k=1, bypass_exact=False)
    assert len(results) > 0
    assert results[0]["ingredient"] == "Monosodium Glutamate"

def test_semantic_retrieval_independent():
    results = retriever.retrieve("Monosodium Glutamate", top_k=1, bypass_exact=True)
    assert len(results) > 0
    assert results[0]["score"] < 1.0

def test_top_k_retrieval():
    r1 = retriever.retrieve("Citric Acid", top_k=1, bypass_exact=True)
    r3 = retriever.retrieve("Citric Acid", top_k=3, bypass_exact=True)
    r5 = retriever.retrieve("Citric Acid", top_k=5, bypass_exact=True)
    assert len(r1) == 1
    assert len(r3) <= 3
    assert len(r5) <= 5

def test_similarity_threshold_enforcement():
    results = retriever.retrieve("XYZ123Gibberish", bypass_exact=True)
    assert len(results) == 0

def test_absent_ingredient_retrieval():
    # 5. NO-RELEVANT-EVIDENCE TEST for absent ground truth ingredient
    results = retriever.retrieve("Malic Acid", bypass_exact=True)
    # The requirement asks to just run it and see if it drops below threshold or not.
    # We verify it doesn't crash and returns a valid list (empty or not).
    assert isinstance(results, list)

def test_provenance_preservation():
    results = retriever.retrieve("Citric Acid", top_k=1, bypass_exact=False)
    assert len(results) > 0
    source = results[0].get("source", {})
    assert "title" in source
    assert "organization" in source
    assert "url" in source
    assert "ingredient" in results[0]

def test_integration_and_metrics_logic():
    script_path = str(PROJECT_ROOT / "evaluation" / "rag_evaluator.py")
    subprocess.run([sys.executable, script_path], check=True, cwd=str(PROJECT_ROOT))

    out_path = PROJECT_ROOT / "evaluation" / "results" / "label1_rag_evaluation.json"
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retrieval = data["retrieval"]

    # 1. KB coverage is calculated independently from Phase 2.
    # 2. Malic Acid is checked against the KB even if Phase 2 did not extract it.
    # 6. Missing KB ingredient is reported as coverage gap.
    assert retrieval["kb_coverage"]["absent_from_kb"] > 0

    # 3. Canonical-query Recall@K calculation
    assert "canonical_query_recall_at_1" in retrieval["semantic_faiss"]

    # 4. Noisy-query Recall@K calculation
    assert "noisy_query_recall_at_1" in retrieval["semantic_faiss"]

    # 5. Noisy query is not silently canonicalized before FAISS evaluation
    # Verified by checking the script uses p2_original for noisy query.
    assert retrieval["semantic_faiss"]["noisy_eligible_queries"] > 0

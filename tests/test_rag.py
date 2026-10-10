"""
test_phase3.py — Comprehensive Test Suite for Phase 3
======================================================

Tests cover:
  1. Document loading
  2. Metadata loading
  3. Chunking correctness
  4. Section parsing
  5. Metadata propagation
  6. Chunk content integrity
  7. Embedding generation
  8. Vector store operations (build/save/load/search)
  9. Retriever single-ingredient queries
  10. Retriever INS/E-code queries
  11. Phase 2 integration and status preservation
  12. Edge cases
"""

import json
import os
import sys
import shutil
import tempfile

import pytest
import numpy as np

# Add src to path
from src.rag.document_loader import Document, load_documents, load_metadata
from src.rag.chunker import Chunk, chunk_document, chunk_all_documents, _parse_sections
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import IngredientRetriever

try:
    EmbeddingModel()
except (ImportError, OSError):
    pytestmark = pytest.mark.skip(reason="local sentence-transformers model unavailable")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_dir():
    """Root of the phase3 project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def documents_dir(project_dir):
    """Path to data/documents/."""
    return os.path.join(project_dir, "data", "knowledge_base", "documents")


@pytest.fixture(scope="session")
def metadata_path(project_dir):
    """Path to metadata.json."""
    return os.path.join(project_dir, "data", "knowledge_base", "metadata.json")


@pytest.fixture(scope="session")
def loaded_metadata(metadata_path):
    """Load the actual metadata.json."""
    return load_metadata(metadata_path)


@pytest.fixture(scope="session")
def loaded_documents(documents_dir, metadata_path):
    """Load all knowledge base documents."""
    return load_documents(documents_dir, metadata_path)


@pytest.fixture(scope="session")
def all_chunks(loaded_documents):
    """Chunk all loaded documents."""
    return chunk_all_documents(loaded_documents)


@pytest.fixture(scope="session")
def embedding_model():
    """Load the embedding model (shared across tests for speed)."""
    return EmbeddingModel()


@pytest.fixture(scope="session")
def temp_store_dir():
    """Create a temporary directory for vector store tests."""
    tmpdir = tempfile.mkdtemp(prefix="phase3_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def built_store(all_chunks, embedding_model, temp_store_dir):
    """Build and save a vector store for testing."""
    store = VectorStore(dimension=embedding_model.dimension)
    store.build_index(all_chunks, embedding_model)
    store.save(temp_store_dir)
    return store


# ============================================================================
# 1. Document Loading Tests
# ============================================================================

class TestDocumentLoading:
    """Test that documents are loaded correctly from disk."""

    def test_documents_loaded(self, loaded_documents):
        """At least 10 documents should be loaded."""
        assert len(loaded_documents) >= 10, (
            f"Expected at least 10 documents, got {len(loaded_documents)}"
        )

    def test_document_has_text(self, loaded_documents):
        """Every document should have non-empty text."""
        for doc in loaded_documents:
            assert doc.text.strip(), f"Empty text in {doc.filename}"

    def test_document_has_filename(self, loaded_documents):
        """Every document should have a filename."""
        for doc in loaded_documents:
            assert doc.filename.endswith(".txt"), (
                f"Expected .txt file, got {doc.filename}"
            )

    def test_specific_documents_present(self, loaded_documents):
        """Key documents should be present."""
        filenames = [doc.filename for doc in loaded_documents]
        expected = [
            "monosodium_glutamate.txt",
            "citric_acid.txt",
            "maltodextrin.txt",
            "palm_olein_oil.txt",
        ]
        for name in expected:
            assert name in filenames, f"Missing document: {name}"


# ============================================================================
# 2. Metadata Loading Tests
# ============================================================================

class TestMetadataLoading:
    """Test that metadata.json is loaded correctly."""

    def test_metadata_loaded(self, loaded_metadata):
        """Metadata should be a non-empty dict."""
        assert isinstance(loaded_metadata, dict)
        assert len(loaded_metadata) >= 10

    def test_metadata_has_required_fields(self, loaded_metadata):
        """Each entry should have ingredient, aliases, category, source."""
        for filename, meta in loaded_metadata.items():
            assert "ingredient" in meta, f"{filename}: missing 'ingredient'"
            assert "aliases" in meta, f"{filename}: missing 'aliases'"
            assert "category" in meta, f"{filename}: missing 'category'"
            assert "source" in meta, f"{filename}: missing 'source'"

    def test_metadata_aliases_are_lists(self, loaded_metadata):
        """Aliases should be a list of strings."""
        for filename, meta in loaded_metadata.items():
            assert isinstance(meta["aliases"], list), (
                f"{filename}: aliases should be a list"
            )
            for alias in meta["aliases"]:
                assert isinstance(alias, str)

    def test_metadata_matches_documents(self, loaded_documents, loaded_metadata):
        """Every document file should have a metadata entry."""
        for doc in loaded_documents:
            assert doc.filename in loaded_metadata, (
                f"No metadata entry for {doc.filename}"
            )


# ============================================================================
# 3. Chunking Correctness Tests
# ============================================================================

class TestChunking:
    """Test the section-based chunking strategy."""

    def test_chunks_created(self, all_chunks):
        """Chunks should be created from documents."""
        assert len(all_chunks) > 0

    def test_expected_sections(self, loaded_documents):
        """Each document should produce 4 chunks (identity, function,
        regulatory, assessment)."""
        for doc in loaded_documents:
            chunks = chunk_document(doc)
            section_labels = [c.metadata["section"] for c in chunks]
            # All 4 sections should be present
            for expected in ["identity", "function", "regulatory", "assessment"]:
                assert expected in section_labels, (
                    f"{doc.filename}: missing '{expected}' chunk, "
                    f"got: {section_labels}"
                )

    def test_chunks_have_text(self, all_chunks):
        """Every chunk should have non-empty text."""
        for chunk in all_chunks:
            assert chunk.text.strip(), (
                f"Empty chunk: {chunk.metadata.get('ingredient')} / "
                f"{chunk.metadata.get('section')}"
            )


# ============================================================================
# 4. Section Parsing Tests
# ============================================================================

class TestSectionParsing:
    """Test the raw section parser."""

    def test_parse_basic_document(self):
        """Test parsing a simple document structure."""
        text = (
            "Ingredient:\nTest Ingredient\n\n"
            "Aliases:\nAlias1\nAlias2\n\n"
            "Category:\nTest Category\n\n"
            "Function:\nTest function description\n\n"
            "Description:\nTest description\n\n"
            "Regulatory Information:\nTest regulatory info\n\n"
            "Assessment Notes:\nTest notes\n\n"
            "Sources:\n- Org: Test"
        )
        sections = _parse_sections(text)

        assert "ingredient" in sections
        assert sections["ingredient"] == "Test Ingredient"
        assert "function" in sections
        assert "regulatory information" in sections

    def test_parse_multiline_content(self):
        """Sections can span multiple lines."""
        text = (
            "Ingredient:\nTest\n\n"
            "Description:\nLine 1\nLine 2\nLine 3\n\n"
            "Function:\nFunc"
        )
        sections = _parse_sections(text)
        assert "Line 1" in sections["description"]
        assert "Line 2" in sections["description"]
        assert "Line 3" in sections["description"]


# ============================================================================
# 5. Metadata Propagation Tests
# ============================================================================

class TestMetadataPropagation:
    """Verify metadata flows from documents to chunks."""

    def test_chunk_has_ingredient_name(self, all_chunks):
        """Every chunk should know its ingredient name."""
        for chunk in all_chunks:
            assert "ingredient" in chunk.metadata
            assert chunk.metadata["ingredient"], "Empty ingredient name"

    def test_chunk_has_section_label(self, all_chunks):
        """Every chunk should have a section label."""
        for chunk in all_chunks:
            assert "section" in chunk.metadata
            assert chunk.metadata["section"] in [
                "identity", "function", "regulatory", "assessment", "full"
            ]

    def test_chunk_has_filename(self, all_chunks):
        """Every chunk should trace back to a source file."""
        for chunk in all_chunks:
            assert "filename" in chunk.metadata
            assert chunk.metadata["filename"].endswith(".txt")

    def test_chunk_has_source(self, all_chunks):
        """Every chunk should have source provenance."""
        for chunk in all_chunks:
            assert "source" in chunk.metadata


# ============================================================================
# 6. Chunk Content Integrity Tests
# ============================================================================

class TestChunkContent:
    """Test that chunk text content is accurate."""

    def test_identity_chunk_has_ingredient_name(self, loaded_documents):
        """The identity chunk should contain the ingredient name."""
        for doc in loaded_documents:
            chunks = chunk_document(doc)
            identity = [c for c in chunks if c.metadata["section"] == "identity"]
            assert identity, f"No identity chunk for {doc.filename}"
            # The ingredient name should be in the identity text
            ingredient = doc.metadata.get("ingredient", "")
            if ingredient:
                assert ingredient in identity[0].text, (
                    f"Ingredient '{ingredient}' not in identity chunk"
                )

    def test_regulatory_chunk_mentions_regulation(self, loaded_documents):
        """The regulatory chunk should mention a regulatory body."""
        regulatory_terms = ["FDA", "GRAS", "FSSAI", "EFSA", "JECFA",
                            "CFR", "Codex", "WHO", "GMP"]
        for doc in loaded_documents:
            chunks = chunk_document(doc)
            reg = [c for c in chunks if c.metadata["section"] == "regulatory"]
            assert reg, f"No regulatory chunk for {doc.filename}"
            text = reg[0].text.upper()
            found = any(term.upper() in text for term in regulatory_terms)
            assert found, (
                f"No regulatory term found in {doc.filename}'s regulatory chunk"
            )


# ============================================================================
# 7. Embedding Generation Tests
# ============================================================================

class TestEmbeddings:
    """Test embedding model operations."""

    def test_model_dimension(self, embedding_model):
        """Model should produce 384-dimensional embeddings."""
        assert embedding_model.dimension == 384

    def test_single_embedding(self, embedding_model):
        """Single text should produce correct shape."""
        result = embedding_model.encode_single("test text")
        assert result.shape == (1, 384)

    def test_batch_embedding(self, embedding_model):
        """Batch of texts should produce correct shape."""
        texts = ["text 1", "text 2", "text 3"]
        result = embedding_model.encode(texts)
        assert result.shape == (3, 384)

    def test_empty_input(self, embedding_model):
        """Empty input should return empty array."""
        result = embedding_model.encode([])
        assert result.shape == (0, 384)

    def test_embeddings_normalized(self, embedding_model):
        """Embeddings should be L2-normalized (unit vectors)."""
        result = embedding_model.encode(["test text"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01, f"Norm = {norm}, expected ~1.0"

    def test_similar_texts_close(self, embedding_model):
        """Semantically similar texts should have high cosine similarity."""
        e1 = embedding_model.encode(["Citric acid is an acidity regulator"])
        e2 = embedding_model.encode(["E330 regulates pH in food"])
        e3 = embedding_model.encode(["The cat sat on the mat"])

        # Cosine similarity (inner product for normalized vectors)
        sim_related = float(np.dot(e1[0], e2[0]))
        sim_unrelated = float(np.dot(e1[0], e3[0]))

        assert sim_related > sim_unrelated, (
            f"Related texts ({sim_related:.4f}) should be more similar "
            f"than unrelated ({sim_unrelated:.4f})"
        )


# ============================================================================
# 8. Vector Store Tests
# ============================================================================

class TestVectorStore:
    """Test FAISS vector store build/save/load/search."""

    def test_build_index(self, built_store, all_chunks):
        """Index size should match number of chunks."""
        assert built_store.size == len(all_chunks)

    def test_search_returns_results(self, built_store, embedding_model):
        """Search should return results."""
        query = embedding_model.encode_single("Monosodium Glutamate flavour enhancer")
        results = built_store.search(query, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_result_format(self, built_store, embedding_model):
        """Each result should have required fields."""
        query = embedding_model.encode_single("citric acid")
        results = built_store.search(query, top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "text" in r
        assert "score" in r
        assert "ingredient" in r
        assert "section" in r
        assert "source" in r

    def test_save_and_load(self, built_store, temp_store_dir, all_chunks):
        """Saved index should load with same size and data."""
        loaded = VectorStore.load(temp_store_dir)
        assert loaded.size == built_store.size
        assert len(loaded.chunk_metadata) == len(all_chunks)

    def test_search_scores_in_range(self, built_store, embedding_model):
        """Cosine similarity scores should be between -1 and 1."""
        query = embedding_model.encode_single("palm oil")
        results = built_store.search(query, top_k=5)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_top_k_ordering(self, built_store, embedding_model):
        """Results should be ordered by descending score."""
        query = embedding_model.encode_single("sugar sweetener")
        results = built_store.search(query, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"


# ============================================================================
# 9. Retriever Single-Ingredient Tests
# ============================================================================

class TestRetrieverSingle:
    """Test retriever with single ingredient queries."""

    @pytest.fixture(scope="class")
    def retriever(self, temp_store_dir):
        """Create a retriever using the test index."""
        return IngredientRetriever(store_dir=temp_store_dir)

    def test_retrieve_canonical_name(self, retriever):
        """Retrieve using canonical ingredient name."""
        results = retriever.retrieve("Monosodium Glutamate")
        assert len(results) > 0
        # Top result should be about MSG
        assert "Monosodium Glutamate" in results[0]["ingredient"]

    def test_retrieve_citric_acid(self, retriever):
        """Retrieve citric acid information."""
        results = retriever.retrieve("Citric Acid")
        assert len(results) > 0
        top_ingredients = [r["ingredient"] for r in results]
        assert "Citric Acid" in top_ingredients

    def test_retrieve_returns_source(self, retriever):
        """Results should include source provenance."""
        results = retriever.retrieve("Maltodextrin")
        assert len(results) > 0
        source = results[0].get("source", {})
        assert source, "No source info in result"

    def test_retrieve_empty_query(self, retriever):
        """Empty query should return empty results."""
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_top_k(self, retriever):
        """Top-k parameter should limit results."""
        results = retriever.retrieve("acid", top_k=2)
        assert len(results) <= 2


# ============================================================================
# 10. Retriever INS/E-Code Tests
# ============================================================================

class TestRetrieverCodes:
    """Test retriever with INS and E-number queries using exact matching."""

    @pytest.fixture(scope="class")
    def retriever(self, temp_store_dir):
        return IngredientRetriever(store_dir=temp_store_dir)

    def test_retrieve_exact_alias_msg(self, retriever):
        """Retrieve using alias (MSG) -> Monosodium Glutamate."""
        results = retriever.retrieve("MSG")
        assert len(results) > 0
        assert results[0]["ingredient"] == "Monosodium Glutamate"
        assert results[0]["score"] == 1.0  # Exact match

    def test_retrieve_ins_621(self, retriever):
        """Retrieve using INS code (INS 621) -> Monosodium Glutamate."""
        results = retriever.retrieve("INS 621")
        assert len(results) > 0
        assert results[0]["ingredient"] == "Monosodium Glutamate"
        assert results[0]["score"] == 1.0  # Exact match

    def test_retrieve_e621(self, retriever):
        """Retrieve using E code (E621) -> Monosodium Glutamate."""
        results = retriever.retrieve("E621")
        assert len(results) > 0
        assert results[0]["ingredient"] == "Monosodium Glutamate"
        assert results[0]["score"] == 1.0  # Exact match

    def test_retrieve_ins_330(self, retriever):
        """Retrieve using INS 330 -> Citric Acid."""
        results = retriever.retrieve("INS 330")
        assert len(results) > 0
        assert results[0]["ingredient"] == "Citric Acid"
        assert results[0]["score"] == 1.0

    def test_retrieve_ins_270(self, retriever):
        """Retrieve using INS 270 -> Lactic Acid."""
        results = retriever.retrieve("INS 270")
        assert len(results) > 0
        assert results[0]["ingredient"] == "Lactic Acid"
        assert results[0]["score"] == 1.0

    def test_label3_additive_codes_have_curated_evidence(self, retriever, built_store):
        """Label 3's INS 150d and INS 320 aliases resolve to sourced KB entries."""
        for code, expected in (
            ("INS 150d", "Caramel IV"),
            ("INS 320", "Butylated Hydroxyanisole"),
        ):
            results = retriever.retrieve(code)
            assert results
            assert results[0]["ingredient"] == expected
            assert results[0]["source"]["organization"].startswith("Food Safety")

    def test_malic_acid_ins_296_retrieves_identity_scoped_provenance(self, retriever, built_store):
        results = retriever.retrieve("INS 296", top_k=10)
        assert results
        assert {item["ingredient"] for item in results} == {"DL-Malic Acid"}
        assert {item["section"] for item in results} == {"identity", "function", "regulatory", "assessment"}
        regulatory = next(item for item in results if item["section"] == "regulatory")
        assert regulatory["jurisdiction"].startswith("India;")
        assert regulatory["source"]["document_date"] == "2022-01-31"
        assert {source["source_type"] for source in regulatory["sources"]} >= {
            "official_regulatory_compendium", "scientific_evaluation", "call_for_data_not_final_opinion"
        }

    def test_malic_retrieval_preserves_conservative_assessment_and_source_details(self, retriever, built_store):
        from src.normalization.normalizer import load_ingredient_dictionary, normalize_ingredient
        from src.safety.assessment import assess_ingredient_safety

        normalized = normalize_ingredient("INS 296", load_ingredient_dictionary())
        retrieved = retriever.retrieve_for_phase2({"ingredients": [{
            "name": "INS 296",
            "normalized": normalized["canonical_name"],
            "canonical_name": normalized["canonical_name"],
            "status": normalized["status"],
            "match_type": normalized["match_type"],
            "similarity_score": normalized["similarity_score"],
            "sub_ingredients": [],
        }]})[0]
        assessment = assess_ingredient_safety(retrieved)
        assert len(retrieved["safety_evidence"]) == 4
        assert assessment["status"] == "insufficient_evidence"
        assert assessment["diagnostic_reason"] == "insufficient_applicable_evidence"
        assert any(
            ref.get("jurisdiction") == "India"
            and ref.get("document_date") == "2022-01-31"
            for ref in assessment["evidence"]
        )
        assert any("JECFA database chemical record 5163" in ref["title"] for ref in assessment["evidence"])

    def test_phase2_discards_evidence_for_a_different_ingredient(self, retriever, built_store, monkeypatch):
        """A semantic near-match cannot ground claims about another ingredient."""
        monkeypatch.setattr(
            retriever,
            "retrieve",
            lambda _query, top_k=3: [{"ingredient": "Acetic Acid", "score": 0.72}],
        )

        results = retriever.retrieve_for_phase2({
            "ingredients": [{"name": "DL-Malic Acid", "status": "matched"}]
        })

        assert results[0]["evidence"] == []
        assert results[0]["retrieval_status"] == "no_relevant_evidence"

    def test_conflicting_alias_metadata_is_not_assigned_to_either_ingredient(self):
        """An alias collision cannot route evidence to whichever chunk came last."""
        from types import SimpleNamespace
        retriever = object.__new__(IngredientRetriever)
        retriever.vector_store = SimpleNamespace(chunk_metadata=[
            {"ingredient": "Additive Alpha", "aliases": ["Shared Code"]},
            {"ingredient": "Additive Beta", "aliases": ["Shared Code"]},
        ])
        retriever._canonical_map = {
            "additivealpha": "Additive Alpha",
            "additivebeta": "Additive Beta",
        }
        retriever._alias_map = retriever._build_alias_map()
        assert retriever._resolve_exact_identity("Shared Code") is None

    def test_unrelated_query_threshold(self, retriever):
        """Unrelated query should fall below threshold and return empty."""
        results = retriever.retrieve("Random completely unrelated non-food text")
        # Assuming the threshold is working, this should return nothing
        assert len(results) == 0

    def test_empty_retrieval_status(self, retriever):
        """Phase 2 processing should mark no_relevant_evidence for empty results."""
        phase2_data = {
            "ingredients": [{"name": "Random completely unrelated non-food text", "status": "unknown"}]
        }
        res = retriever.retrieve_for_phase2(phase2_data)
        assert len(res) == 1
        assert res[0]["evidence"] == []
        assert res[0]["retrieval_status"] == "no_relevant_evidence"


# ============================================================================
# 11. Phase 2 Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Test processing Phase 2 results with status preservation."""

    @pytest.fixture(scope="class")
    def retriever(self, temp_store_dir):
        return IngredientRetriever(store_dir=temp_store_dir)

    @pytest.fixture(scope="class")
    def sample_phase2_results(self):
        """Create sample Phase 2 results matching actual format."""
        return {
            "raw_ocr_text": "Test OCR text",
            "ingredient_section": "Test section",
            "cleaned_section": "Test cleaned",
            "ingredients": [
                {
                    "raw": "Corn Med",
                    "name": "Corn Med",
                    "normalized": "Corn Meal",
                    "status": "matched",
                    "match_type": "fuzzy",
                    "similarity_score": 82.4,
                    "sub_ingredients": [],
                },
                {
                    "raw": "Seasoning",
                    "name": "Seasoning",
                    "normalized": "Seasoning",
                    "status": "ambiguous",
                    "match_type": "exact_alias",
                    "similarity_score": None,
                    "sub_ingredients": [
                        {
                            "raw": "MSG",
                            "name": "MSG",
                            "normalized": "Monosodium Glutamate",
                            "status": "matched",
                            "match_type": "exact_alias",
                            "similarity_score": None,
                            "sub_ingredients": [],
                        }
                    ],
                },
                {
                    "raw": "260. 296",
                    "name": "260. 296",
                    "normalized": "260. 296",
                    "status": "unknown",
                    "match_type": None,
                    "similarity_score": None,
                    "sub_ingredients": [],
                },
            ],
            "summary": {"matched": 1, "ambiguous": 1, "unknown": 1, "total": 3},
        }

    def test_phase2_processing(self, retriever, sample_phase2_results):
        """Phase 2 results should be processed without errors."""
        results = retriever.retrieve_for_phase2(sample_phase2_results)
        assert len(results) == 3  # 3 top-level ingredients

    def test_phase2_status_preserved(self, retriever, sample_phase2_results):
        """Phase 2 statuses (matched/ambiguous/unknown) must be preserved."""
        results = retriever.retrieve_for_phase2(sample_phase2_results)
        statuses = {r["ingredient"]: r["phase2_status"] for r in results}
        assert statuses["Corn Meal"] == "matched"
        assert statuses["Seasoning"] == "ambiguous"
        assert statuses["260. 296"] == "unknown"

    def test_phase2_sub_ingredients(self, retriever, sample_phase2_results):
        """Sub-ingredients should be recursively processed."""
        results = retriever.retrieve_for_phase2(sample_phase2_results)
        seasoning = [r for r in results if r["ingredient"] == "Seasoning"][0]
        assert len(seasoning["sub_ingredients"]) == 1
        sub = seasoning["sub_ingredients"][0]
        assert sub["ingredient"] == "Monosodium Glutamate"
        assert sub["phase2_status"] == "matched"

    def test_phase2_evidence_attached(self, retriever, sample_phase2_results):
        """Each ingredient should have evidence (even if low-scoring)."""
        results = retriever.retrieve_for_phase2(sample_phase2_results)
        for r in results:
            assert "evidence" in r
            # matched ingredients should have some evidence
            if r["phase2_status"] == "matched":
                assert len(r["evidence"]) > 0

    def test_ambiguous_not_elevated(self, retriever, sample_phase2_results):
        """Ambiguous status should NOT be converted to matched."""
        results = retriever.retrieve_for_phase2(sample_phase2_results)
        ambiguous = [r for r in results if r["phase2_status"] == "ambiguous"]
        assert len(ambiguous) == 1
        assert ambiguous[0]["phase2_status"] == "ambiguous"


# ============================================================================
# 12. Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_document_class_repr(self):
        """Document repr should work."""
        doc = Document("test.txt", "test text", {"ingredient": "Test"})
        assert "Test" in repr(doc)

    def test_chunk_class_repr(self):
        """Chunk repr should work."""
        chunk = Chunk("test text", {"ingredient": "Test", "section": "identity"})
        assert "Test" in repr(chunk)
        assert "identity" in repr(chunk)

    def test_empty_vector_store_search(self):
        """Searching an empty store should return empty results."""
        store = VectorStore(dimension=384)
        query = np.random.randn(1, 384).astype(np.float32)
        results = store.search(query, top_k=3)
        assert results == []

    def test_top_k_larger_than_index(self, built_store, embedding_model):
        """top_k larger than index size should not crash."""
        query = embedding_model.encode_single("test")
        results = built_store.search(query, top_k=10000)
        assert len(results) == built_store.size

    def test_chunk_malformed_document(self):
        """A document with no recognized sections should produce one chunk."""
        doc = Document("bad.txt", "Just some random text with no sections",
                       {"ingredient": "Unknown"})
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["section"] == "full"

    def test_load_nonexistent_metadata(self, tmp_path):
        """Missing metadata file should return empty dict."""
        result = load_metadata(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_vector_store_load_nonexistent(self, tmp_path):
        """Loading from nonexistent directory should raise."""
        with pytest.raises(FileNotFoundError):
            VectorStore.load(str(tmp_path / "nonexistent"))

"""Tests for knowledge-base lookup tools in src/llm/tools.py."""

import pytest
from src.llm.tools import lookup_ingredient, lookup_regulatory_info, lookup_source
from src.llm import tools


def test_successful_ingredient_lookup():
    """lookup_ingredient returns structured data for an existing knowledge-base ingredient."""
    res = lookup_ingredient("Citric Acid")
    assert res["success"] is True
    assert res["found"] is True
    assert res["canonical_name"] == "Citric Acid"
    assert "Acidity Regulator" in res.get("category", "")
    assert res.get("function") is not None


def test_ingredient_lookup_by_alias():
    """lookup_ingredient resolves aliases and additive codes."""
    res = lookup_ingredient("INS 330")
    assert res["success"] is True
    assert res["found"] is True
    assert res["canonical_name"] == "Citric Acid"


def test_missing_ingredient_lookup():
    """lookup_ingredient returns found=False for an unknown ingredient without raising."""
    res = lookup_ingredient("Unknown Chemical Nonexistent 999")
    assert res["success"] is True
    assert res["found"] is False
    assert "not found in the curated Phase 3 knowledge base" in res["message"]


def test_lookup_with_missing_metadata_file(monkeypatch, tmp_path):
    """When metadata.json does not exist, lookup returns a clean found=False result."""
    nonexistent_meta = str(tmp_path / "nonexistent_metadata.json")
    monkeypatch.setattr(tools, "METADATA_PATH", nonexistent_meta)
    res = lookup_ingredient("Citric Acid")
    assert res["success"] is True
    assert res["found"] is False


def test_lookup_with_missing_document_file(monkeypatch, tmp_path):
    """When a document text file is missing from disk, sections are handled gracefully."""
    monkeypatch.setattr(tools, "DOCS_DIR", str(tmp_path / "empty_docs"))
    res = lookup_ingredient("Citric Acid")
    # Finding filename succeeds from metadata, but reading empty document returns None sections
    assert res["success"] is True
    assert res["found"] is True
    assert res["canonical_name"] == "Citric Acid"
    assert res["category"] is None
    assert res["function"] is None


def test_successful_regulatory_lookup():
    """lookup_regulatory_info returns regulatory information and sources for known ingredient."""
    res = lookup_regulatory_info("Citric Acid")
    assert res["success"] is True
    assert res["found"] is True
    assert res["canonical_name"] == "Citric Acid"
    assert "sources" in res
    assert len(res["sources"]) > 0


def test_missing_regulatory_lookup():
    """lookup_regulatory_info handles unknown ingredient gracefully."""
    res = lookup_regulatory_info("Nonexistent Additive 12345")
    assert res["success"] is True
    assert res["found"] is False
    assert "not found in the curated knowledge base" in res["message"]


def test_lookup_source_success_and_failure():
    """lookup_source searches metadata organizations and titles."""
    success_res = lookup_source("FSSAI")
    assert success_res["success"] is True
    assert success_res["found"] is True
    assert len(success_res["results"]) > 0

    fail_res = lookup_source("Nonexistent Agency XZ123")
    assert fail_res["success"] is True
    assert fail_res["found"] is False
    assert "not found" in fail_res["message"]

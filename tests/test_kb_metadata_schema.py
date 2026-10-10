"""Lightweight schema validation for the knowledge-base metadata.

Uses only the standard library.  Validates the required fields and source
provenance of every entry in ``data/knowledge_base/metadata.json`` without
rejecting valid legacy entries that use the older single-source format.
"""

import json
import os
import pathlib

import pytest

KB_DIR = pathlib.Path("data/knowledge_base")
METADATA_PATH = KB_DIR / "metadata.json"
DOCUMENT_DIR = KB_DIR / "documents"


@pytest.fixture(scope="module")
def metadata():
    with open(METADATA_PATH) as fh:
        return json.load(fh)


# ------------------------------------------------------------------
# Structural checks
# ------------------------------------------------------------------

def test_metadata_is_a_non_empty_dict(metadata):
    assert isinstance(metadata, dict) and metadata


def test_every_key_maps_to_a_document_file(metadata):
    on_disk = set(os.listdir(DOCUMENT_DIR))
    for key in metadata:
        assert key in on_disk, f"metadata key {key!r} has no file on disk"


def test_every_document_file_has_a_metadata_entry(metadata):
    on_disk = set(os.listdir(DOCUMENT_DIR))
    for fname in on_disk:
        assert fname in metadata, f"file {fname!r} has no metadata entry"


# ------------------------------------------------------------------
# Per-entry required fields
# ------------------------------------------------------------------

REQUIRED_FIELDS = ("ingredient", "aliases", "category", "source")


@pytest.fixture(scope="module")
def entry_names(metadata):
    return list(metadata.keys())


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_required_fields_present(metadata, field):
    for fname, entry in metadata.items():
        assert field in entry, f"{fname}: missing required field {field!r}"


def test_ingredient_is_nonempty_string(metadata):
    for fname, entry in metadata.items():
        val = entry.get("ingredient")
        assert isinstance(val, str) and val.strip(), f"{fname}: ingredient must be a non-empty string"


def test_aliases_is_a_list_of_strings(metadata):
    for fname, entry in metadata.items():
        aliases = entry.get("aliases")
        assert isinstance(aliases, list), f"{fname}: aliases must be a list"
        for alias in aliases:
            assert isinstance(alias, str) and alias.strip(), f"{fname}: each alias must be a non-empty string"


def test_category_is_nonempty_string(metadata):
    for fname, entry in metadata.items():
        val = entry.get("category")
        assert isinstance(val, str) and val.strip(), f"{fname}: category must be a non-empty string"


# ------------------------------------------------------------------
# Source provenance (single-source format — required for all entries)
# ------------------------------------------------------------------

def test_source_has_required_fields(metadata):
    """Every entry must have a ``source`` dict with at least ``url``."""
    for fname, entry in metadata.items():
        src = entry.get("source")
        assert isinstance(src, dict), f"{fname}: source must be a dict"
        assert isinstance(src.get("url"), str) and src["url"].strip(), (
            f"{fname}: source.url must be a non-empty string"
        )


def test_source_has_organization_or_title(metadata):
    """At least one identifying field beyond the URL."""
    for fname, entry in metadata.items():
        src = entry.get("source", {})
        has_org = isinstance(src.get("organization"), str) and src["organization"].strip()
        has_title = isinstance(src.get("title"), str) and src["title"].strip()
        assert has_org or has_title, f"{fname}: source must have organization or title"


# ------------------------------------------------------------------
# Multi-source provenance (optional — new enhanced entries only)
# ------------------------------------------------------------------

def test_sources_array_entries_are_valid_when_present(metadata):
    """If an entry has ``sources``, each element must have ``organization`` and ``url``."""
    for fname, entry in metadata.items():
        sources = entry.get("sources")
        if sources is None:
            continue  # legacy entries without sources[] are valid
        assert isinstance(sources, list) and sources, f"{fname}: sources must be a non-empty list when present"
        for i, src in enumerate(sources):
            assert isinstance(src, dict), f"{fname}: sources[{i}] must be a dict"
            assert isinstance(src.get("organization"), str) and src["organization"].strip(), (
                f"{fname}: sources[{i}].organization must be a non-empty string"
            )
            assert isinstance(src.get("url"), str) and src["url"].strip(), (
                f"{fname}: sources[{i}].url must be a non-empty string"
            )


def test_primary_source_is_included_in_sources_when_both_present(metadata):
    """The primary ``source.url`` should appear in the ``sources`` array when both exist."""
    for fname, entry in metadata.items():
        sources = entry.get("sources")
        if sources is None:
            continue
        primary_url = entry.get("source", {}).get("url", "")
        source_urls = {s.get("url") for s in sources if isinstance(s, dict)}
        assert primary_url in source_urls, (
            f"{fname}: primary source URL should appear in sources array"
        )


# ------------------------------------------------------------------
# Optional enrichment fields (validated only when present)
# ------------------------------------------------------------------

def test_jurisdiction_is_string_when_present(metadata):
    for fname, entry in metadata.items():
        val = entry.get("jurisdiction")
        if val is None:
            continue
        assert isinstance(val, str) and val.strip(), f"{fname}: jurisdiction must be a non-empty string"


def test_codes_are_strings_when_present(metadata):
    for fname, entry in metadata.items():
        codes = entry.get("codes")
        if codes is None:
            continue
        assert isinstance(codes, list), f"{fname}: codes must be a list"
        for code in codes:
            assert isinstance(code, str) and code.strip(), f"{fname}: each code must be a non-empty string"

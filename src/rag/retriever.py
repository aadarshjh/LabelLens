"""
retriever.py — Ingredient Knowledge Retrieval
===============================================

This module is the main interface for retrieving ingredient
knowledge from the vector store.

WHAT IT DOES:
  Given an ingredient name (or alias/code), it:
    1. Generates an embedding for the query.
    2. Searches the FAISS index for the most similar chunks.
    3. Returns the matching text with scores and full metadata.

WHY A SEPARATE RETRIEVER?
  The retriever adds query-level logic on top of the raw vector
  store search:
    - Query formatting (prepending "Ingredient:" for better matching)
    - Phase 2 integration (processing a batch of normalized ingredients)
    - Result formatting with source provenance

  This keeps the vector store generic (it just finds similar vectors)
  while the retriever handles ingredient-specific retrieval patterns.

PRESERVING UNCERTAINTY:
  When Phase 2 marks an ingredient as "ambiguous" or "unknown",
  the retriever preserves that status in the results. It does NOT
  convert uncertainty into false confidence.
"""

import json
import os
import re
import sys

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


def _identity_key(value):
    """Match harmless punctuation/spacing variants without fuzzy identity."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class IngredientRetriever:
    """
    Retrieves ingredient knowledge from the FAISS vector store.

    Usage:
        retriever = IngredientRetriever(similarity_threshold=0.45)
        results = retriever.retrieve("Monosodium Glutamate")
        results = retriever.retrieve("INS 621")
        results = retriever.retrieve("MSG", top_k=5)
    """

    def __init__(self, store_dir=None, model_name=None, similarity_threshold=0.60):
        """
        Initialize the retriever by loading the embedding model
        and the pre-built FAISS index.

        Args:
            store_dir:  Path to the vectorstore/ directory.
            model_name: Embedding model name (default: all-MiniLM-L6-v2).
            similarity_threshold: Minimum cosine similarity score (0.0 to 1.0)
                                  for a result to be considered relevant.
                                  (Empirical threshold: 0.60)
        """

        self.vector_store = VectorStore.load(store_dir)
        self.model_name = model_name
        self.embedding_model = None
        self.similarity_threshold = similarity_threshold
        self._canonical_map = {
            _identity_key(meta["ingredient"]): meta["ingredient"]
            for meta in self.vector_store.chunk_metadata
            if isinstance(meta.get("ingredient"), str) and meta["ingredient"].strip()
        }
        self._alias_map = self._build_alias_map()

    def _build_alias_map(self):
        """
        Builds a lightweight mapping of lowercase aliases and codes
        to canonical ingredient names using the loaded chunk metadata.
        """
        alias_map = {}
        ambiguous_aliases = set()
        for meta in self.vector_store.chunk_metadata:
            canonical = meta.get("ingredient")
            if canonical:
                for alias in meta.get("aliases", []):
                    key = _identity_key(alias)
                    if not key or key in ambiguous_aliases or key in self._canonical_map:
                        continue
                    previous = alias_map.get(key)
                    if previous is not None and previous != canonical:
                        alias_map.pop(key, None)
                        ambiguous_aliases.add(key)
                    else:
                        alias_map[key] = canonical
        return alias_map

    def _resolve_exact_identity(self, query):
        key = _identity_key(query)
        return self._canonical_map.get(key) or self._alias_map.get(key)

    def retrieve(self, query, top_k=3, bypass_exact=False):
        """
        Retrieve the most relevant knowledge chunks for a query.

        The query can be:
          - A canonical ingredient name: "Monosodium Glutamate"
          - An alias: "MSG"
          - An INS/E code: "INS 621", "E621"
          - A natural language question: "What is citric acid used for?"

        Args:
            query:  The search query string.
            top_k:  Number of results to return (default: 3).

        Returns:
            List of result dicts. If no evidence passes the threshold,
            returns an empty list.
        """
        if not query or not query.strip():
            return []

        # 1. Exact canonical/alias/code lookup
        exact_match_canonical = self._resolve_exact_identity(query)

        if exact_match_canonical and not bypass_exact:
            # Retrieve exact chunks directly
            exact_chunks = []
            for meta in self.vector_store.chunk_metadata:
                if meta.get("ingredient") == exact_match_canonical:
                    # Construct a result dict mirroring VectorStore format
                    exact_chunks.append({
                        "text": meta.get("text", f"Ingredient: {meta.get('ingredient')}..."),
                        "score": 1.0,  # Exact match implies absolute relevance
                        "ingredient": meta.get("ingredient"),
                        "section": meta.get("section"),
                        "source": meta.get("source", {}),
                        "sources": meta.get("sources", []),
                        "jurisdiction": meta.get("jurisdiction", ""),
                        "aliases": meta.get("aliases", []),
                        "category": meta.get("category", ""),
                        "filename": meta.get("filename", "")
                    })

            # Sort by section just for consistency (e.g. identity first)
            exact_chunks.sort(key=lambda x: x["section"] != "identity")
            return exact_chunks[:top_k]

        # 2. Semantic embedding search with FAISS (fallback)
        # Protect generic categories from matching unrelated specific chemicals
        from src.normalization.normalizer import PROTECTED_GENERIC_TERMS
        if query.strip().lower() in PROTECTED_GENERIC_TERMS:
            return []

        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel(self.model_name)
        formatted_query = self._format_query(query)
        query_embedding = self.embedding_model.encode_single(formatted_query)

        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Apply configurable threshold to filter out unrelated results
        filtered_results = [r for r in results if r["score"] >= self.similarity_threshold]

        return filtered_results

    def retrieve_for_phase2(self, phase2_results, top_k=2):
        """
        Retrieve knowledge for all ingredients from Phase 2 output.

        Processes the Phase 2 results JSON and retrieves evidence
        for each ingredient, preserving Phase 2's confidence and
        status information.

        Args:
            phase2_results: Dict from Phase 2's extraction_results.json
                            (or the output of run_pipeline()).
            top_k:          Results per ingredient.

        Returns:
            List of dicts. If no relevant evidence is found above threshold,
            includes {"retrieval_status": "no_relevant_evidence"}.
        """
        ingredients = phase2_results.get("ingredients", [])
        return self._process_ingredients(ingredients, top_k)

    def _process_ingredients(self, ingredients, top_k):
        """Recursively process ingredients and their sub-ingredients."""
        results = []

        for ing in ingredients:
            name = ing.get("normalized", ing.get("name", ""))
            status = ing.get("status", "unknown")

            # Retrieve evidence for this ingredient
            evidence = self.retrieve(name, top_k=top_k)
            # Semantic similarity can return a different ingredient (for
            # example, Malic Acid may retrieve Acetic Acid). Never pass such
            # evidence to the explanation model as if it supported the target.
            query_name = name.strip() if isinstance(name, str) else ""
            expected_ingredient = self._resolve_exact_identity(query_name) or query_name
            evidence = [
                chunk for chunk in evidence
                if isinstance(chunk.get("ingredient"), str)
                and chunk["ingredient"].strip().casefold() == expected_ingredient.casefold()
            ]

            # Keep the existing top-k evidence unchanged for explanation generation.
            # Safety assessment needs regulatory and assessment sections, which
            # may fall beyond that top-k window for exact matches.
            safety_evidence = []
            exact_canonical = self._resolve_exact_identity(query_name)
            if status == "matched" and exact_canonical:
                safety_evidence = self.retrieve(
                    name, top_k=len(self.vector_store.chunk_metadata)
                )
                safety_evidence = [
                    chunk for chunk in safety_evidence
                    if isinstance(chunk.get("ingredient"), str)
                    and chunk["ingredient"].strip().casefold() == exact_canonical.casefold()
                ]

            entry = {
                "ingredient": name,
                "raw_name": ing.get("raw_name", ing.get("raw", name)),
                "canonical_name": ing.get("canonical_name", name),
                "code": ing.get("code"),
                "aliases_detected": ing.get("aliases_detected", []),
                "phase2_status": status,
                "phase2_match_type": ing.get("match_type"),
                "phase2_similarity": ing.get("similarity_score"),
                "evidence": evidence,
                "safety_evidence": safety_evidence,
            }

            # Add retrieval status if no evidence passed the threshold
            if not evidence:
                entry["retrieval_status"] = "no_relevant_evidence"

            # Recursively process sub-ingredients
            sub_ingredients = ing.get("sub_ingredients", [])
            if sub_ingredients:
                entry["sub_ingredients"] = self._process_ingredients(
                    sub_ingredients, top_k
                )
            else:
                entry["sub_ingredients"] = []

            results.append(entry)

        return results

    def _format_query(self, query):
        """
        Format the query for better semantic matching.

        We prepend "Ingredient:" to help the model understand
        that we're searching for ingredient information, which
        improves matching against our document chunks.
        """
        query = query.strip()

        # If the query is very short (likely a code like "635" or "MSG"),
        # or it looks like an INS/E code, add context to improve retrieval
        query_upper = query.upper()
        if len(query) <= 5 or query_upper.startswith("INS ") or query_upper.startswith("E") and query[1:].isdigit():
            code = query_upper.replace("INS ", "").strip()
            return f"Food additive code {code}"

        return f"Ingredient: {query}"

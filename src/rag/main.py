"""
main.py — Knowledge Base + RAG Retrieval
============================================================

This is the main entry point for RAG. It supports two modes:

  1. BUILD MODE (--build):
     Load documents → chunk → embed → build FAISS index → save

  2. QUERY MODE (--ingredient or --input):
     Load index → embed query → search → display results

USAGE:
  # Build the index (run once, or after adding new documents):
  python -m src.rag.main --build

  # Query a single ingredient:
  python -m src.rag.main --ingredient "Monosodium Glutamate"
  python -m src.rag.main --ingredient "MSG"
  python -m src.rag.main --ingredient "INS 621"

  # Retrieve evidence for all normalized ingredients:
  python -m src.rag.main --input artifacts/reference_runs/extraction/extraction_results.json

  # Both build and query:
  python -m src.rag.main --build --ingredient "Citric Acid"

DESIGN:
  RAG is RETRIEVAL ONLY. It does not generate LLM answers,
  provide medical advice, or create safety scores. It returns
  factual evidence with source metadata for the explanation boundary.
"""

import json
import os
import sys
import argparse

from src.config import ARTIFACTS_DIR, VECTORSTORE_DIR
from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_all_documents
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import IngredientRetriever


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = str(ARTIFACTS_DIR.parent)
OUTPUT_DIR = str(VECTORSTORE_DIR)

# Default path to a retained normalized reference payload
REFERENCE_INPUT_PATH = str(ARTIFACTS_DIR / "reference_runs" / "extraction" / "extraction_results.json")


# ---------------------------------------------------------------------------
# Build mode
# ---------------------------------------------------------------------------

def build_index():
    """
    Build the FAISS index from knowledge base documents.

    Steps:
      1. Load all documents from data/documents/
      2. Chunk each document into semantic sections
      3. Load the embedding model
      4. Embed all chunks
      5. Build the FAISS index
      6. Save the index + metadata to vectorstore/
    """
    print("\n" + "=" * 64)
    print("  PHASE 3: BUILDING KNOWLEDGE BASE INDEX")
    print("=" * 64)

    # --- Step 1: Load documents ---
    print("\n  Step 1: Loading documents...")
    documents = load_documents()
    print(f"  ✓ Loaded {len(documents)} documents")
    for doc in documents:
        print(f"    - {doc}")

    # --- Step 2: Chunk documents ---
    print("\n  Step 2: Chunking documents...")
    chunks = chunk_all_documents(documents)
    print(f"  ✓ Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"    - {chunk}")

    # --- Step 3: Load embedding model ---
    print("\n  Step 3: Loading embedding model...")
    embedding_model = EmbeddingModel()

    # --- Step 4: Build FAISS index ---
    print("\n  Step 4: Building FAISS index...")
    store = VectorStore(dimension=embedding_model.dimension)
    store.build_index(chunks, embedding_model)

    # --- Step 5: Save ---
    print("\n  Step 5: Saving index...")
    store.save()

    print("\n" + "=" * 64)
    print(f"  ✓ INDEX BUILT SUCCESSFULLY")
    print(f"    Documents: {len(documents)}")
    print(f"    Chunks:    {len(chunks)}")
    print(f"    Vectors:   {store.size}")
    print("=" * 64 + "\n")

    return store


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

def query_ingredient(ingredient_name, top_k=3):
    """
    Query the knowledge base for a single ingredient.

    Returns and prints the retrieved evidence.
    """
    print("\n" + "=" * 64)
    print("  PHASE 3: INGREDIENT KNOWLEDGE RETRIEVAL")
    print("=" * 64)

    print(f"\n  Query: {ingredient_name}")
    print(f"  Top K: {top_k}")

    # Load retriever
    retriever = IngredientRetriever()

    # Retrieve
    results = retriever.retrieve(ingredient_name, top_k=top_k)

    # Display
    _print_results(results, ingredient_name)

    return results


def query_input(input_path, top_k=3):
    """
    Retrieve evidence for all normalized ingredients from a JSON payload.
    """
    print("\n" + "=" * 64)
    print("  BULK KNOWLEDGE RETRIEVAL")
    print("=" * 64)

    # Load normalized ingredient results
    if not os.path.isfile(input_path):
        print(f"\n  ERROR: normalized ingredient input not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        normalized_results = json.load(f)

    print(f"\n  Input file: {os.path.basename(input_path)}")

    summary = normalized_results.get("summary", {})
    print(f"  Input summary: {summary.get('total', 0)} ingredients "
          f"({summary.get('matched', 0)} matched, "
          f"{summary.get('ambiguous', 0)} ambiguous, "
          f"{summary.get('unknown', 0)} unknown)")

    # Load retriever
    retriever = IngredientRetriever()

    # Retrieve evidence for all ingredients
    results = retriever.retrieve_for_phase2(normalized_results, top_k=top_k)

    # Display
    _print_phase2_results(results)

    # Save results
    output_path = os.path.join(PROJECT_DIR, "output")
    os.makedirs(output_path, exist_ok=True)
    results_path = os.path.join(output_path, "retrieval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_results(results, query_name):
    """Print retrieval results for a single ingredient query."""
    print(f"\n  ── Retrieved Evidence ──\n")

    if not results:
        print("  (no results found)")
        return

    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['ingredient']}")
        print(f"      Score:    {r['score']:.4f}")
        print(f"      Section:  {r['section']}")
        print(f"      Category: {r['category']}")

        # Source info
        source = r.get("source", {})
        if source:
            print(f"      Source:   {source.get('organization', 'N/A')}")
            if source.get("title"):
                print(f"      Title:    {source['title']}")

        # Truncated text preview
        text = r["text"]
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n      Text:")
        for line in preview.split("\n"):
            print(f"        {line}")
        print()

    print("=" * 64 + "\n")


def _print_phase2_results(results, indent=0):
    """Print Phase 2 integration results with status preservation."""
    prefix = "  " + "    " * indent
    for r in results:
        status_icon = {
            "matched": "✓",
            "ambiguous": "~",
            "unknown": "?",
        }.get(r["phase2_status"], " ")

        print(f"{prefix}{status_icon} {r['ingredient']} "
              f"[Phase 2: {r['phase2_status']}]")

        evidence = r.get("evidence", [])
        if evidence:
            best = evidence[0]
            print(f"{prefix}    → Best evidence: {best['ingredient']} "
                  f"(score: {best['score']:.4f}, "
                  f"section: {best['section']})")
            source = best.get("source", {})
            if source.get("organization"):
                print(f"{prefix}    → Source: {source['organization']}")
        else:
            print(f"{prefix}    → No evidence found in knowledge base")

        # Recursively display sub-ingredients
        sub = r.get("sub_ingredients", [])
        if sub:
            _print_phase2_results(sub, indent=indent + 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Knowledge Base + RAG Retrieval Pipeline"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build the FAISS index from knowledge base documents."
    )
    parser.add_argument(
        "--ingredient", type=str, default=None,
        help="Retrieve evidence for a single ingredient."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to normalized ingredient results JSON for bulk retrieval."
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of results to retrieve (default: 3)."
    )
    parser.add_argument(
        "--json-output", action="store_true",
        help="Output results as JSON."
    )

    args = parser.parse_args()

    # Build mode
    if args.build:
        build_index()

    # Query mode — single ingredient
    if args.ingredient:
        results = query_ingredient(args.ingredient, top_k=args.top_k)
        if args.json_output:
            print(json.dumps(results, indent=2, ensure_ascii=False))

    # Query mode — normalized ingredient integration
    if args.input:
        results = query_input(args.input, top_k=args.top_k)
        if args.json_output:
            print(json.dumps(results, indent=2, ensure_ascii=False))

    # Auto-detect retained normalized input if no mode specified
    if not args.build and not args.ingredient and not args.input:
        input_path = os.path.normpath(REFERENCE_INPUT_PATH)
        if os.path.isfile(input_path):
            print(f"  Auto-detected normalized input: {input_path}",
                  file=sys.stderr)
            results = query_input(input_path, top_k=args.top_k)
        else:
            print("\n  No mode specified. Usage:")
            print("    python -m src.rag.main --build")
            print('    python -m src.rag.main --ingredient "Monosodium Glutamate"')
            print("    python -m src.rag.main --input path/to/normalized_results.json")
            sys.exit(1)


if __name__ == "__main__":
    main()

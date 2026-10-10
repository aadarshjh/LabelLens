"""
vector_store.py — FAISS Vector Store for Ingredient Knowledge
==============================================================

This module manages the FAISS index that stores and searches
embedding vectors.

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library for finding
  the most similar vectors in a large collection. Think of it as
  a "search engine for meaning" — instead of matching keywords,
  it finds text that has the most similar meaning to your query.

HOW IT WORKS:
  1. BUILD:  We embed all chunks and add them to a FAISS index.
  2. SAVE:   The index + metadata are saved to disk.
  3. LOAD:   On startup, we load the pre-built index (no need to
             re-embed every time).
  4. SEARCH: Given a query embedding, FAISS finds the k nearest
             vectors (most similar chunks) in milliseconds.

WHY INNER PRODUCT (IndexFlatIP)?
  Since our embeddings are L2-normalized (unit vectors), the inner
  product equals cosine similarity. Cosine similarity measures the
  "angle" between two vectors — vectors pointing in the same
  direction (similar meaning) have similarity close to 1.0.

METADATA MAPPING:
  FAISS only stores numerical vectors, not text or metadata. We
  maintain a parallel list of chunk metadata that maps each FAISS
  index position to its corresponding chunk text and source info:

    FAISS index position 0  →  chunk_metadata[0]
    FAISS index position 1  →  chunk_metadata[1]
    ...

  This mapping is saved alongside the FAISS index so we never
  lose track of where each vector came from.
"""

import os
import json
import numpy as np
import faiss
from src.config import VECTORSTORE_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_STORE_DIR = str(VECTORSTORE_DIR)
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    FAISS-based vector store with metadata tracking.

    Maintains a 1:1 mapping between FAISS vector positions and
    chunk metadata (text, ingredient name, source, etc.).
    """

    def __init__(self, dimension=384):
        """
        Initialize an empty vector store.

        Args:
            dimension: Size of embedding vectors (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        # IndexFlatIP = brute-force inner product search
        # With normalized vectors, inner product = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        # Parallel list: chunk_metadata[i] corresponds to index position i
        self.chunk_metadata = []

    @property
    def size(self):
        """Number of vectors in the index."""
        return self.index.ntotal

    def build_index(self, chunks, embedding_model):
        """
        Build the FAISS index from a list of chunks.

        Steps:
          1. Extract text from each chunk.
          2. Embed all texts using the embedding model.
          3. Add the embedding vectors to the FAISS index.
          4. Store the chunk metadata in the parallel list.

        Args:
            chunks:          List of Chunk objects (from chunker).
            embedding_model: EmbeddingModel instance.
        """
        if not chunks:
            print("  ⚠ No chunks to index.")
            return

        print(f"  Indexing {len(chunks)} chunks...")

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        embeddings = embedding_model.encode(texts, show_progress=True)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata (text + all metadata from each chunk)
        self.chunk_metadata = []
        for chunk in chunks:
            entry = {
                "text": chunk.text,
                **chunk.metadata,
            }
            self.chunk_metadata.append(entry)

        print(f"  ✓ Index built: {self.size} vectors of dimension {self.dimension}")

    def search(self, query_embedding, top_k=3):
        """
        Search the index for the most similar chunks.

        Args:
            query_embedding: numpy array of shape (1, dimension).
            top_k:           Number of results to return.

        Returns:
            List of result dicts, each containing:
              - text:       the chunk text
              - score:      cosine similarity score (0 to 1)
              - ingredient: canonical ingredient name
              - section:    which section of the document
              - source:     provenance info (organization, title, URL)
              - aliases:    known aliases
              - category:   ingredient category
        """
        if self.size == 0:
            return []

        # Clamp top_k to the number of available vectors
        actual_k = min(top_k, self.size)

        # FAISS search returns (distances, indices)
        # distances = similarity scores (since we use IndexFlatIP)
        # indices = positions in the index
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for i in range(actual_k):
            idx = indices[0][i]
            score = float(scores[0][i])

            if idx < 0 or idx >= len(self.chunk_metadata):
                continue  # Safety check

            meta = self.chunk_metadata[idx]

            results.append({
                "text": meta.get("text", ""),
                "score": round(score, 4),
                "ingredient": meta.get("ingredient", "Unknown"),
                "section": meta.get("section", ""),
                "source": meta.get("source", {}),
                "sources": meta.get("sources", []),
                "jurisdiction": meta.get("jurisdiction", ""),
                "aliases": meta.get("aliases", []),
                "category": meta.get("category", ""),
                "filename": meta.get("filename", ""),
            })

        return results

    def save(self, store_dir=None):
        """
        Save the FAISS index and metadata to disk.

        Creates two files:
          - index.faiss:    the FAISS vector index
          - metadata.json:  the parallel chunk metadata

        Args:
            store_dir: Directory to save to (default: vectorstore/).
        """
        store_dir = store_dir or DEFAULT_STORE_DIR
        os.makedirs(store_dir, exist_ok=True)

        index_path = os.path.join(store_dir, INDEX_FILENAME)
        metadata_path = os.path.join(store_dir, METADATA_FILENAME)

        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"  ✓ FAISS index saved: {index_path}")

        # Save metadata as JSON
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metadata saved: {metadata_path}")

    @classmethod
    def load(cls, store_dir=None):
        """
        Load a previously saved FAISS index and metadata from disk.

        Args:
            store_dir: Directory to load from (default: vectorstore/).

        Returns:
            VectorStore instance with the loaded index and metadata.
        """
        store_dir = store_dir or DEFAULT_STORE_DIR

        index_path = os.path.join(store_dir, INDEX_FILENAME)
        metadata_path = os.path.join(store_dir, METADATA_FILENAME)

        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                f"Run the index builder first: python -m src.rag.main --build"
            )

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Run the index builder first: python -m src.rag.main --build"
            )

        # Load FAISS index
        index = faiss.read_index(index_path)
        dimension = index.d

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunk_metadata = json.load(f)

        # Create VectorStore instance
        store = cls(dimension=dimension)
        store.index = index
        store.chunk_metadata = chunk_metadata

        print(f"  ✓ Index loaded: {store.size} vectors of dimension {dimension}")

        return store

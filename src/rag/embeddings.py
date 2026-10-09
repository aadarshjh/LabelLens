"""
embeddings.py — Embedding Generation Using Sentence Transformers
=================================================================

This module converts text chunks into numerical vectors (embeddings)
that capture the semantic meaning of the text.

WHAT IS AN EMBEDDING?
  An embedding is a list of numbers (a vector) that represents the
  "meaning" of a piece of text. Similar texts produce similar vectors.
  For example:
    - "Citric Acid is an acidity regulator" → [0.12, -0.45, 0.87, ...]
    - "E330 regulates pH in food" → [0.14, -0.42, 0.85, ...]
  These two vectors would be very close together because the texts
  have similar meaning, even though the words are different.

WHY all-MiniLM-L6-v2?
  - Small (80 MB) — runs quickly on a laptop without a GPU.
  - Good quality — ranks well on semantic similarity benchmarks.
  - 384-dimensional embeddings — compact but informative.
  - Open source and free — no API key or cloud service needed.
  - Widely used in tutorials — easy to find help and examples.

HOW IT WORKS:
  1. Load the pre-trained model (downloaded once, cached locally).
  2. Pass text strings to the model.
  3. Get back numpy arrays (one per text) that capture meaning.
  4. These arrays are what FAISS stores and searches.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The embedding model to use. This is downloaded automatically on first run.
# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Wrapper around a Sentence Transformers model for generating embeddings.

    Usage:
        model = EmbeddingModel()
        vectors = model.encode(["Citric Acid is an acidity regulator"])
        # vectors.shape == (1, 384)
    """

    def __init__(self, model_name=None):
        """
        Load the embedding model.

        Args:
            model_name: HuggingFace model name/path. Defaults to
                        all-MiniLM-L6-v2.
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        print(f"  Loading embedding model: {self.model_name}")
        # Retrieval must not make hidden network calls at runtime. The model is
        # installed/downloaded during setup and then loaded from the local cache.
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        self.dimension = self.model.get_embedding_dimension()
        print(f"  ✓ Model loaded (dimension: {self.dimension})")

    def encode(self, texts, show_progress=False):
        """
        Convert a list of text strings into embedding vectors.

        Args:
            texts:          List of strings to embed.
            show_progress:  Show a progress bar (useful for large batches).

        Returns:
            numpy array of shape (len(texts), dimension).
            Each row is the embedding vector for the corresponding text.
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        return embeddings.astype(np.float32)

    def encode_single(self, text):
        """
        Embed a single text string.

        Convenience method for embedding one query at a time.

        Returns:
            numpy array of shape (1, dimension).
        """
        return self.encode([text])

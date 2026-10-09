"""
document_loader.py — Load Knowledge Base Documents and Metadata
================================================================

This module reads ingredient knowledge documents from disk and
pairs them with their metadata.

WHAT IT DOES:
  1. Reads all .txt files from the data/documents/ directory.
  2. Loads the companion metadata.json file.
  3. Returns a list of Document objects, each containing:
     - filename:  the source file name (e.g., "monosodium_glutamate.txt")
     - text:      the full text content of the document
     - metadata:  dict with ingredient name, aliases, category, sources

WHY SEPARATE DOCUMENTS?
  Each ingredient has its own .txt file so that:
  - Adding a new ingredient is as simple as dropping in a new file.
  - Each document can be independently reviewed and sourced.
  - The chunker can split each document separately.
"""

import json
import os
import glob
from src.config import DOCUMENTS_DIR, KNOWLEDGE_BASE_METADATA_PATH


# ---------------------------------------------------------------------------
# Document data structure
# ---------------------------------------------------------------------------

class Document:
    """
    A knowledge base document with its associated metadata.

    Attributes:
        filename:  Name of the source file (e.g., "citric_acid.txt")
        text:      Full text content of the document
        metadata:  Dict with ingredient info, aliases, category, source
    """

    def __init__(self, filename, text, metadata=None):
        self.filename = filename
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        ingredient = self.metadata.get("ingredient", self.filename)
        return f"Document('{ingredient}', {len(self.text)} chars)"


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_metadata(metadata_path=None):
    """
    Load the metadata.json file that describes each document.

    Args:
        metadata_path: Path to metadata.json. If None, uses the
                       default location at data/knowledge_base/metadata.json.

    Returns:
        Dict mapping filename → metadata dict.
    """
    if metadata_path is None:
        metadata_path = KNOWLEDGE_BASE_METADATA_PATH

    metadata_path = os.path.normpath(metadata_path)

    if not os.path.isfile(metadata_path):
        print(f"  ⚠ Metadata file not found: {metadata_path}")
        print(f"    Documents will be loaded without metadata.")
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_documents(documents_dir=None, metadata_path=None):
    """
    Load all ingredient documents from the documents/ directory.

    Each .txt file becomes a Document with its corresponding
    metadata from metadata.json.

    Args:
        documents_dir: Path to the documents/ directory.
        metadata_path: Path to metadata.json.

    Returns:
        List of Document objects.
    """
    if documents_dir is None:
        documents_dir = DOCUMENTS_DIR

    documents_dir = os.path.normpath(documents_dir)

    if not os.path.isdir(documents_dir):
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}\n"
            f"Make sure data/knowledge_base/documents/ exists and contains .txt files."
        )

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Find all .txt files
    txt_files = sorted(glob.glob(os.path.join(documents_dir, "*.txt")))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in: {documents_dir}\n"
            f"Add ingredient knowledge documents to this directory."
        )

    documents = []
    for filepath in txt_files:
        filename = os.path.basename(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Get metadata for this file (if available)
        file_metadata = metadata.get(filename, {})

        documents.append(Document(
            filename=filename,
            text=text,
            metadata=file_metadata,
        ))

    return documents

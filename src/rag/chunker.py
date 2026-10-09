"""
chunker.py — Document Chunking for RAG Retrieval
==================================================

This module splits knowledge base documents into smaller chunks
suitable for embedding and retrieval.

WHY CHUNK?
  Embedding models have a limited context window (typically 256-512
  tokens for sentence-transformers). Feeding an entire document as
  one embedding would:
    1. Exceed the model's context window, losing information.
    2. Create overly broad embeddings that dilute specific information.

  By splitting documents into focused chunks (e.g., "Function",
  "Regulatory Info"), each chunk gets a precise embedding that
  can be matched to specific user queries.

CHUNKING STRATEGY:
  We use a SECTION-BASED strategy that splits on the document's
  own structure (the section headings like "Function:", "Description:",
  etc.) rather than arbitrary character counts.

  This preserves semantic meaning within each chunk — a chunk about
  "Regulatory Information" contains ONLY regulatory information,
  not a random slice of mixed content.

METADATA PROPAGATION:
  Every chunk inherits the metadata of its parent document PLUS
  gets a section label (e.g., "identity", "function", "regulatory").
  This is critical because Phase 4 needs to show sources for
  every piece of retrieved information.
"""


# ---------------------------------------------------------------------------
# Chunk data structure
# ---------------------------------------------------------------------------

class Chunk:
    """
    A text chunk from a knowledge base document, with full metadata.

    Attributes:
        text:       The chunk text content
        metadata:   Dict containing:
                    - ingredient: canonical ingredient name
                    - filename: source document filename
                    - section: which section this chunk is from
                    - source: organization/title/URL provenance
                    - aliases: list of known aliases
                    - category: ingredient category
    """

    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        ingredient = self.metadata.get("ingredient", "unknown")
        section = self.metadata.get("section", "unknown")
        return f"Chunk('{ingredient}/{section}', {len(self.text)} chars)"


# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

# The sections we recognize in our document format and how we group them
# into chunks. Each tuple is (section_label, list_of_heading_markers).
#
# Section label: a short, consistent label for the chunk
# Heading markers: the text that starts each section in the document

SECTION_GROUPS = [
    ("identity", ["Ingredient:", "Aliases:", "Category:"]),
    ("function", ["Function:", "Description:"]),
    ("regulatory", ["Regulatory Information:"]),
    ("assessment", ["Assessment Notes:", "Sources:"]),
]


# ---------------------------------------------------------------------------
# Chunking functions
# ---------------------------------------------------------------------------

def chunk_document(document):
    """
    Split a Document into Chunks based on section headings.

    Strategy:
      1. Parse the document text into named sections.
      2. Group related sections into chunks (e.g., "Ingredient" +
         "Aliases" + "Category" → one "identity" chunk).
      3. Attach the document's metadata to each chunk.

    Args:
        document: A Document object (from document_loader).

    Returns:
        List of Chunk objects, each with text and metadata.
    """
    # Parse the raw text into sections
    sections = _parse_sections(document.text)

    # Group sections into chunks
    chunks = []
    for group_label, headings in SECTION_GROUPS:
        # Collect text from all sections that belong to this group
        group_text_parts = []
        for heading in headings:
            # Find the section key (heading without colon, lowercased)
            section_key = heading.rstrip(":").strip().lower()
            if section_key in sections:
                group_text_parts.append(
                    f"{heading}\n{sections[section_key]}"
                )

        if group_text_parts:
            chunk_text = "\n\n".join(group_text_parts)

            # Build chunk metadata from the document metadata
            chunk_metadata = _build_chunk_metadata(
                document, group_label
            )

            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))

    # If no sections were found (malformed document), create a single
    # chunk from the entire text
    if not chunks and document.text.strip():
        chunk_metadata = _build_chunk_metadata(document, "full")
        chunks.append(Chunk(text=document.text, metadata=chunk_metadata))

    return chunks


def chunk_all_documents(documents):
    """
    Chunk all documents and return a flat list of chunks.

    Args:
        documents: List of Document objects.

    Returns:
        List of Chunk objects from all documents.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

# Known section headings in our document format
KNOWN_HEADINGS = [
    "Ingredient:",
    "Aliases:",
    "Category:",
    "Function:",
    "Description:",
    "Regulatory Information:",
    "Assessment Notes:",
    "Sources:",
]


def _parse_sections(text):
    """
    Parse a document's text into a dict of section_name → content.

    We scan line by line, looking for known section headings.
    Everything between one heading and the next belongs to that section.

    Returns:
        Dict mapping lowercase section names to their text content.
        Example: {"ingredient": "Monosodium Glutamate", "function": "..."}
    """
    sections = {}
    current_heading = None
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Check if this line is a section heading
        matched_heading = None
        for heading in KNOWN_HEADINGS:
            if stripped.startswith(heading) or stripped == heading.rstrip(":"):
                matched_heading = heading
                break

        if matched_heading:
            # Save the previous section
            if current_heading is not None:
                section_key = current_heading.rstrip(":").strip().lower()
                sections[section_key] = "\n".join(current_lines).strip()

            current_heading = matched_heading
            current_lines = []

            # Check if there's content on the same line as the heading
            after_heading = stripped[len(matched_heading):].strip()
            if after_heading:
                current_lines.append(after_heading)
        else:
            if current_heading is not None:
                current_lines.append(line.rstrip())

    # Don't forget the last section
    if current_heading is not None:
        section_key = current_heading.rstrip(":").strip().lower()
        sections[section_key] = "\n".join(current_lines).strip()

    return sections


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _build_chunk_metadata(document, section_label):
    """
    Build the metadata dict for a chunk, inheriting from the document.

    Every chunk gets:
      - ingredient:  canonical name
      - filename:    source document filename
      - section:     which section group this chunk represents
      - aliases:     known aliases (for matching)
      - category:    ingredient category
      - source:      provenance info (organization, title, URL)
    """
    doc_meta = document.metadata

    return {
        "ingredient": doc_meta.get("ingredient", _infer_ingredient(document)),
        "filename": document.filename,
        "section": section_label,
        "aliases": doc_meta.get("aliases", []),
        "category": doc_meta.get("category", ""),
        "source": doc_meta.get("source", {}),
    }


def _infer_ingredient(document):
    """
    Try to infer the ingredient name from the document text
    when metadata is missing.

    Looks for the "Ingredient:" line in the document.
    Falls back to the filename.
    """
    for line in document.text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Ingredient:"):
            name = stripped[len("Ingredient:"):].strip()
            if name:
                return name

    # Fallback: derive from filename
    name = document.filename.replace(".txt", "").replace("_", " ").title()
    return name

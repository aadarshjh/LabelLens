import json
from src.config import KNOWLEDGE_BASE_METADATA_PATH, DOCUMENTS_DIR

METADATA_PATH = str(KNOWLEDGE_BASE_METADATA_PATH)
DOCS_DIR = str(DOCUMENTS_DIR)

def _load_metadata() -> dict:
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

def _find_file_for_ingredient(ingredient_name: str) -> str:
    metadata = _load_metadata()
    target = ingredient_name.lower().strip()
    for filename, data in metadata.items():
        canonical = data.get("ingredient", "").lower()
        aliases = [a.lower() for a in data.get("aliases", [])]
        if target == canonical or target in aliases:
            return filename
    return None

def _read_document_sections(filename: str) -> dict:
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        content = f.read()

    sections = {}
    current_section = None
    current_text = []

    for line in content.split('\n'):
        line = line.strip()
        if line.endswith(':') and len(line) < 30:
            if current_section:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = line[:-1].lower()
            current_text = []
        elif line:
            current_text.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_text).strip()
    return sections

def lookup_ingredient(ingredient_name: str) -> dict:
    """
    Search the Phase 3 knowledge base for an ingredient's identity, category, and function.
    Returns a structured dictionary.
    """
    filename = _find_file_for_ingredient(ingredient_name)
    if not filename:
        return {
            "success": True,
            "found": False,
            "message": f"Ingredient '{ingredient_name}' not found in the curated Phase 3 knowledge base."
        }

    sections = _read_document_sections(filename)
    metadata = _load_metadata().get(filename, {})

    return {
        "success": True,
        "found": True,
        "canonical_name": metadata.get("ingredient"),
        "category": sections.get("category"),
        "function": sections.get("function"),
        "description": sections.get("description")
    }

def lookup_regulatory_info(ingredient_name: str) -> dict:
    """
    Retrieve regulatory information and assessment notes from the Phase 3 knowledge base.
    Returns a structured dictionary.
    """
    filename = _find_file_for_ingredient(ingredient_name)
    if not filename:
        return {
            "success": True,
            "found": False,
            "message": f"Regulatory information for '{ingredient_name}' not found in the curated knowledge base."
        }

    sections = _read_document_sections(filename)
    metadata = _load_metadata().get(filename, {})

    has_info = any(k in sections for k in ["regulatory information", "regulatory status", "assessment notes"])

    sources = []
    if "source" in metadata:
        sources.append(metadata["source"])

    if not has_info:
        return {
            "success": True,
            "found": True,
            "canonical_name": metadata.get("ingredient"),
            "message": "Ingredient found, but no regulatory sections are available in the document.",
            "sources": sources
        }

    return {
        "success": True,
        "found": True,
        "canonical_name": metadata.get("ingredient"),
        "regulatory_information": sections.get("regulatory information"),
        "regulatory_status": sections.get("regulatory status"),
        "assessment_notes": sections.get("assessment notes"),
        "sources": sources
    }

def lookup_source(source_identifier: str) -> dict:
    """
    Retrieve source metadata (URL, organization, title) by searching the Phase 3 metadata.
    Returns a structured dictionary of matches.
    """
    metadata = _load_metadata()
    target = source_identifier.lower().strip()

    results = []
    for filename, data in metadata.items():
        src = data.get("source", {})
        if not src:
            continue

        org = src.get("organization", "").lower()
        title = src.get("title", "").lower()

        if target in org or target in title:
            match = src.copy()
            match["associated_ingredient"] = data.get("ingredient")
            results.append(match)

    if not results:
        return {
            "success": True,
            "found": False,
            "message": f"Source matching '{source_identifier}' not found in the curated knowledge base."
        }

    return {
        "success": True,
        "found": True,
        "results": results
    }

"""
main.py — Ingredient Extraction & Normalization
===================================================================

This is the main entry point for Phase 2. It takes raw OCR text
(from Phase 1 or a sample string) and produces structured, normalized
ingredient data.

PIPELINE:
  Raw OCR text
    → Section detection (find "Ingredients:" ... stop at "Batch No" etc.)
    → Text cleaning (remove OCR artifacts)
    → Bracket-aware parsing (handle compound ingredients)
    → Normalization (exact match → alias match → fuzzy match)
    → Structured JSON output

USAGE:
  # With an OCR output file:
  python -m src.pipeline.ingredient_pipeline --file path/to/ocr.txt

  # With inline text:
  python src/main.py --text "Ingredients: Sugar, Salt, MSG"

  # Auto-detect Phase 1 output:
  python src/main.py

INTEGRATION:
  Phase 2 is independently testable with sample strings. When used
  with Phase 1, it reads the best_ocr_result.txt file that Phase 1
  saves to its output/ directory.
"""

import json
import os
import sys
import argparse

from src.config import ARTIFACTS_DIR, INGREDIENTS_PATH
from src.extraction.extractor import (
    detect_ingredient_section,
    clean_ocr_text,
    parse_ingredients,
)
from src.normalization.normalizer import load_ingredient_dictionary, normalize_all


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = str(ARTIFACTS_DIR.parent)
OUTPUT_DIR = str(ARTIFACTS_DIR / "extraction")
DEFAULT_DICT_PATH = str(INGREDIENTS_PATH)

# Default path to a retained OCR reference result
REFERENCE_OCR_PATH = str(ARTIFACTS_DIR / "reference_runs" / "output" / "best_ocr_result.txt")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def get_ocr_text(args):
    """
    Get the raw OCR text from one of three sources:
      1. --text argument (inline string)
      2. --file argument (path to text file)
      3. Auto-detect Phase 1 output file

    Returns the raw OCR text string.
    """
    if args.text:
        return args.text

    if args.file:
        file_path = args.file
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            print(f"\n  ERROR: File not found: {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # Auto-detect: look for Phase 1 output
    reference_path = os.path.normpath(REFERENCE_OCR_PATH)
    if os.path.isfile(reference_path):
        print(f"  Auto-detected OCR reference: {reference_path}", file=sys.stderr)
        with open(reference_path, "r", encoding="utf-8") as f:
            return f.read()

    print("\n  ERROR: No input text provided.")
    print("  Usage:")
    print('    python src/main.py --text "Ingredients: Sugar, Salt"')
    print("    python src/main.py --file path/to/ocr_output.txt")
    print("    python src/main.py   (auto-detects Phase 1 output)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_ocr_text, dictionary_path=None, fuzzy_threshold=80):
    """
    Run the full Phase 2 pipeline on raw OCR text.

    This function is the main API — it can be called from Phase 1
    integration, from tests, or from the CLI.

    Args:
        raw_ocr_text:    The raw OCR text (may contain product name,
                         nutrition facts, etc. — not just ingredients).
        dictionary_path: Path to ingredients.json (or None for default).
        fuzzy_threshold: Minimum fuzzy match score (0-100).

    Returns:
        Dict with:
          - raw_ocr_text:       the original input text
          - ingredient_section: extracted ingredient portion
          - cleaned_section:    after OCR artifact removal
          - ingredients:        list of normalized ingredient dicts
          - summary:            counts of matched/ambiguous/unknown
    """
    # --- Step 1: Load ingredient dictionary ---
    dict_path = dictionary_path or DEFAULT_DICT_PATH
    dictionary = load_ingredient_dictionary(dict_path)

    # --- Step 2: Detect ingredient section ---
    ingredient_section = detect_ingredient_section(raw_ocr_text)

    if ingredient_section is None:
        return {
            "raw_ocr_text": raw_ocr_text,
            "ingredient_section": None,
            "cleaned_section": None,
            "ingredients": [],
            "summary": {"matched": 0, "ambiguous": 0, "unknown": 0, "total": 0},
            "error": "No ingredient section detected. "
                     "Could not find a heading like 'Ingredients:' in the text.",
        }

    # --- Step 2b: Extract allergen declarations ---
    from src.extraction.extractor import extract_allergens
    allergens = extract_allergens(raw_ocr_text)

    # --- Step 3: Clean OCR artifacts ---
    cleaned = clean_ocr_text(ingredient_section)

    # --- Step 4: Parse ingredients (bracket-aware) ---
    parsed = parse_ingredients(cleaned)

    # --- Step 5: Normalize against dictionary ---
    normalized = normalize_all(parsed, dictionary, fuzzy_threshold)

    # --- Step 6: Build summary ---
    summary = _count_statuses(normalized)
    summary["allergens"] = allergens

    return {
        "raw_ocr_text": raw_ocr_text,
        "ingredient_section": ingredient_section,
        "cleaned_section": cleaned,
        "ingredients": normalized,
        "summary": summary,
        "allergens": allergens,
    }


def _count_statuses(ingredients):
    """
    Count matched/ambiguous/unknown across all ingredients
    (including sub-ingredients, recursively).
    """
    counts = {"matched": 0, "ambiguous": 0, "unknown": 0, "total": 0}

    for ing in ingredients:
        counts[ing["status"]] = counts.get(ing["status"], 0) + 1
        counts["total"] += 1
        # Count sub-ingredients too
        if ing.get("sub_ingredients"):
            sub_counts = _count_statuses(ing["sub_ingredients"])
            for key in counts:
                counts[key] += sub_counts[key]

    return counts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(result):
    """
    Print a human-readable summary of the pipeline results.
    """
    print("\n" + "=" * 64)
    print("  PHASE 2: INGREDIENT EXTRACTION & NORMALIZATION")
    print("=" * 64)

    # Ingredient section
    print("\n  ── Detected Ingredient Section ──")
    if result.get("ingredient_section"):
        print(f"  {result['ingredient_section'][:200]}")
        if len(result["ingredient_section"]) > 200:
            print(f"  ... ({len(result['ingredient_section'])} chars total)")
    else:
        print("  (none detected)")

    # Cleaned section
    print("\n  ── Cleaned Text ──")
    if result.get("cleaned_section"):
        print(f"  {result['cleaned_section'][:200]}")
        if len(result["cleaned_section"]) > 200:
            print(f"  ... ({len(result['cleaned_section'])} chars total)")
    else:
        print("  (none)")

    # Parsed + normalized ingredients
    print("\n  ── Normalized Ingredients ──\n")
    if not result.get("ingredients"):
        print("  (no ingredients found)")
    else:
        _print_ingredients(result["ingredients"], indent=2)

    # Summary
    summary = result.get("summary", {})
    print("\n  ── Summary ──")
    print(f"    Total ingredients: {summary.get('total', 0)}")
    print(f"    ✓ Matched:        {summary.get('matched', 0)}")
    print(f"    ~ Ambiguous:      {summary.get('ambiguous', 0)}")
    print(f"    ? Unknown:        {summary.get('unknown', 0)}")

    # Errors
    if result.get("error"):
        print(f"\n  ⚠ {result['error']}")

    print("\n" + "=" * 64 + "\n")


def _print_ingredients(ingredients, indent=2):
    """Recursively print ingredients with indentation for sub-ingredients."""
    prefix = " " * indent
    for ing in ingredients:
        # Status icon
        icon = {"matched": "✓", "ambiguous": "~", "unknown": "?"}.get(
            ing["status"], " "
        )

        # Match info
        match_info = ""
        if ing["match_type"]:
            match_info = f" [{ing['match_type']}"
            if ing["similarity_score"] is not None:
                match_info += f", {ing['similarity_score']}%"
            match_info += "]"

        # Name vs normalized
        if ing["normalized"] != ing["name"]:
            print(f"{prefix}{icon} {ing['name']} → {ing['normalized']}{match_info}")
        else:
            print(f"{prefix}{icon} {ing['name']}{match_info}")

        # Sub-ingredients
        if ing.get("sub_ingredients"):
            _print_ingredients(ing["sub_ingredients"], indent=indent + 4)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract and normalize food ingredients from OCR text."
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Raw OCR text string (inline)."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a text file containing raw OCR output."
    )
    parser.add_argument(
        "--dict", type=str, default=None,
        help="Path to ingredient dictionary JSON (default: data/ingredients.json)."
    )
    parser.add_argument(
        "--threshold", type=int, default=80,
        help="Fuzzy match threshold 0-100 (default: 80)."
    )
    parser.add_argument(
        "--json-output", action="store_true",
        help="Output results as JSON (for machine consumption)."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save JSON results to output/extraction_results.json."
    )

    args = parser.parse_args()

    # Get input text
    raw_text = get_ocr_text(args)

    # Run pipeline
    result = run_pipeline(
        raw_text,
        dictionary_path=args.dict,
        fuzzy_threshold=args.threshold,
    )

    # Output
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_results(result)

    # Save to output/ (always for convenience)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "extraction_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if not args.json_output:
        print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()

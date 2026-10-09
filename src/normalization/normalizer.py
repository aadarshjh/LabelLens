"""
normalizer.py — Ingredient Normalization and Matching
======================================================

This module normalizes extracted ingredient names against a curated
dictionary (ingredients.json).

MATCHING STRATEGY (in strict priority order):
  1. EXACT CANONICAL — raw name matches a dictionary canonical_name
  2. EXACT ALIAS     — raw name matches an alias or code entry
  3. FUZZY MATCH     — RapidFuzz similarity above the threshold
  4. UNKNOWN         — no match found above threshold

WHY THIS ORDER?
  Deterministic exact matches are always preferred over probabilistic
  fuzzy matches. Fuzzy matching is only a fallback for OCR spelling
  errors (e.g., "Mik Solids" → "Milk Solids"). We never want a fuzzy
  match to override a perfect exact match.

AMBIGUITY:
  Some ingredients are legitimately nonspecific ("Vegetable Oil",
  "Spices"). These are in the dictionary with is_generic=true.
  When matched, they receive status "ambiguous" instead of "matched"
  to signal that the name is real but does not identify a specific
  substance.

FUZZY SCORING NOTE:
  The similarity_score (0-100) from RapidFuzz measures string
  similarity, NOT scientific confidence or probability. A score of
  85 means "these strings are 85% similar", not "85% chance this
  is correct". Low-confidence fuzzy matches are reported transparently
  so the user can decide.
"""

import json
import os
from rapidfuzz import fuzz, process
from src.config import INGREDIENTS_PATH


# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------

def load_ingredient_dictionary(path=None):
    """
    Load the ingredient dictionary from a JSON file.

    Args:
        path: Path to ingredients.json. If None, uses the default
              location at data/ingredients.json.

    Returns:
        List of ingredient entry dicts, each with:
          canonical_name, aliases, codes, is_generic
    """
    if path is None:
        # Default path relative to this source file
        path = INGREDIENTS_PATH

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Ingredient dictionary not found: {path}\n"
            f"Make sure data/ingredients.json exists."
        )

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Filter out comment-only entries (entries without canonical_name)
    return [e for e in entries if "canonical_name" in e]


# ---------------------------------------------------------------------------
# Matching functions (in priority order)
# ---------------------------------------------------------------------------

def _exact_canonical_match(name, dictionary):
    """
    Try EXACT match against canonical names (case-insensitive).

    This is the highest-priority match. If the raw ingredient name
    exactly matches a canonical name, we use it immediately.

    Returns:
        (entry, "exact_canonical") or (None, None)
    """
    name_lower = name.lower().strip()
    for entry in dictionary:
        if entry["canonical_name"].lower() == name_lower:
            return entry, "exact_canonical"
    return None, None


import re

PROTECTED_GENERIC_TERMS = {
    "vitamins", "minerals", "cereal extract", "cereal extracts",
    "added flavours", "added flavors", "flavours", "flavors",
    "permitted natural colour", "permitted natural color",
    "colour", "color", "antioxidant", "antioxidants",
    "edible vegetable oil", "vegetable oil", "spices",
    "spices and condiments", "flavouring substances", "milk solids",
    "seasoning", "acidity regulator", "flavour enhancer", "anticaking agent"
}


def _extract_ins_or_e_code(text):
    """
    Extract an INS or E number code if present in the text.
    E.g. 'Colour (INS 150d)' -> 'INS 150d', 'INS 320' -> 'INS 320', 'E320' -> 'E320'.
    """
    if not text:
        return None
    m = re.search(r"\b(?:INS|E)\s*(\d{2,4}[a-z]?)\b", text, re.IGNORECASE)
    if m:
        prefix = "INS" if "ins" in m.group(0).lower() else "E"
        return f"{prefix} {m.group(1)}"
    m_num = re.search(r"\b(\d{3,4}[a-z]?)\b", text)
    if m_num:
        return m_num.group(1)
    return None


def _exact_canonical_match(name, dictionary):
    """
    Try EXACT match against canonical names (case-insensitive).

    This is the highest-priority match. If the raw ingredient name
    exactly matches a canonical name, we use it immediately.

    Returns:
        (entry, "exact_canonical") or (None, None)
    """
    name_lower = name.lower().strip()
    for entry in dictionary:
        if entry["canonical_name"].lower() == name_lower:
            return entry, "exact_canonical"
    return None, None


def _exact_alias_match(name, dictionary):
    """
    Try EXACT match against all aliases and codes (case-insensitive).

    Second priority. Catches INS/E-number codes and common alternate
    names (e.g., "MSG" → "Monosodium Glutamate").

    Returns:
        (entry, "exact_alias") or (None, None)
    """
    name_lower = name.lower().strip()
    for entry in dictionary:
        all_aliases = entry.get("aliases", []) + entry.get("codes", [])
        for alias in all_aliases:
            if alias.lower() == name_lower:
                return entry, "exact_alias"
    return None, None


def _exact_code_match(code, dictionary):
    """
    Try strict match against dictionary codes and aliases using an extracted code.

    Guarantees strict INS/E-number resolution (e.g. 'INS 150d' -> 'Caramel IV').
    """
    if not code:
        return None, None
    code_clean = re.sub(r"\s+", " ", code.strip().lower())
    num_part = re.sub(r"^(?:ins|e)\s*", "", code_clean)

    for entry in dictionary:
        all_codes = entry.get("codes", []) + entry.get("aliases", [])
        for c in all_codes:
            c_clean = re.sub(r"\s+", " ", c.strip().lower())
            c_num = re.sub(r"^(?:ins|e)\s*", "", c_clean)
            if code_clean == c_clean or num_part == c_num:
                return entry, "exact_alias"
    return None, None


def _fuzzy_match(name, dictionary, threshold=80):
    """
    Try FUZZY match using RapidFuzz's token_sort_ratio.

    Third priority — only used when exact matching fails. Helps with
    OCR spelling errors like "Mik Solids" → "Milk Solids".

    Guards:
      - Never fuzzy-matches an INS/E code against a different code number.
      - Never fuzzy-matches generic terms (e.g. 'Vitamins') to specific chemicals.
      - Normalizes case to ensure OCR case anomalies (e.g. 'lodized Salt') match.
    """
    # Guard against fuzzy-matching numeric / INS codes
    if re.search(r"\b(?:INS|E)?\s*\d{2,4}[a-z]?\b", name, re.IGNORECASE):
        return None, None, 0

    clean_name = name.lower().strip()
    if clean_name in PROTECTED_GENERIC_TERMS:
        return None, None, 0

    # Build candidate list with case normalization
    candidates = []
    for i, entry in enumerate(dictionary):
        # Exclude pure numeric codes and INS codes from fuzzy target strings
        # to avoid numeric cross-matching
        targets = [entry["canonical_name"]] + entry.get("aliases", [])
        for t in targets:
            if not re.match(r"^(?:INS|E)?\s*\d{2,4}[a-z]?$", t.strip(), re.IGNORECASE):
                candidates.append((t.lower(), i))

    if not candidates:
        return None, None, 0

    candidate_strings = [c[0] for c in candidates]

    result = process.extractOne(
        clean_name,
        candidate_strings,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )

    if result is None:
        return None, None, 0

    matched_string, score, matched_index = result
    entry_index = candidates[matched_index][1]
    return dictionary[entry_index], "fuzzy", score


# ---------------------------------------------------------------------------
# Single ingredient normalization
# ---------------------------------------------------------------------------

def normalize_ingredient(name, dictionary, fuzzy_threshold=80):
    """
    Normalize a single ingredient name against the dictionary.
    Supports either string name or parsed ingredient dict with metadata.

    Tries matching in strict priority order:
      1. Exact canonical name
      2. Exact alias/code
      3. Strict INS/E code matching (if INS/E code present)
      4. Fuzzy match (with generic & code safety guards)
      5. Unknown
    """
    raw_dict = None
    explicit_code = None
    explicit_aliases = []
    if isinstance(name, dict):
        raw_dict = name
        explicit_code = raw_dict.get("code")
        explicit_aliases = list(raw_dict.get("aliases_detected", []))
        name = raw_dict.get("name", "")

        # A code list attached to a parent label (for example, a category
        # followed by two additive codes) is not itself one ingredient. Leave
        # that parent unresolved and normalize its parsed children separately.
        if raw_dict.get("sub_ingredients") and len(re.findall(
            r"\b(?:INS|E)\s*\d{2,4}[a-z]?\b", name, re.IGNORECASE
        )) > 1:
            return {
                "normalized": name.strip(" ()[]{}"),
                "canonical_name": name.strip(" ()[]{}"),
                "status": "unknown",
                "match_type": None,
                "similarity_score": None,
                "code": None,
                "aliases_detected": explicit_aliases,
            }

    if not name or not isinstance(name, str) or not name.strip():
        return {
            "normalized": name,
            "canonical_name": name,
            "status": "unknown",
            "match_type": None,
            "similarity_score": None,
            "code": explicit_code,
            "aliases_detected": explicit_aliases,
        }

    clean_name = name.strip()

    # --- Priority 1: Exact canonical match ---
    entry, match_type = _exact_canonical_match(clean_name, dictionary)
    if entry:
        res = _build_result(entry, match_type, score=100)
        if explicit_code and not res.get("code"):
            res["code"] = explicit_code
        if explicit_aliases:
            res["aliases_detected"] = list(dict.fromkeys(res.get("aliases_detected", []) + explicit_aliases))
        return res

    # --- Priority 2: Exact alias/code match ---
    entry, match_type = _exact_alias_match(clean_name, dictionary)
    if entry:
        res = _build_result(entry, match_type, score=100)
        if explicit_code and not res.get("code"):
            res["code"] = explicit_code
        if explicit_aliases:
            res["aliases_detected"] = list(dict.fromkeys(res.get("aliases_detected", []) + explicit_aliases))
        return res

    # --- Priority 2.5: Extracted or Explicit INS/E Code match ---
    detected_code = explicit_code or _extract_ins_or_e_code(clean_name)
    if detected_code:
        entry, match_type = _exact_code_match(detected_code, dictionary)
        if entry:
            res = _build_result(entry, match_type, score=100, detected_code=detected_code)
            if explicit_aliases:
                res["aliases_detected"] = list(dict.fromkeys(res.get("aliases_detected", []) + explicit_aliases))
            return res
        # If input has an explicit INS/E code not in dictionary, do NOT allow fuzzy fallback to a different code!
        return {
            "normalized": clean_name,
            "canonical_name": clean_name,
            "status": "unknown",
            "match_type": None,
            "similarity_score": None,
            "code": detected_code,
            "aliases_detected": explicit_aliases,
        }

    # --- Priority 3: Fuzzy match (with guards) ---
    if clean_name.lower() in PROTECTED_GENERIC_TERMS:
        return {
            "normalized": clean_name,
            "canonical_name": clean_name,
            "status": "ambiguous",
            "match_type": None,
            "similarity_score": None,
            "code": None,
            "aliases_detected": explicit_aliases,
        }

    entry, match_type, score = _fuzzy_match(
        clean_name, dictionary, threshold=fuzzy_threshold
    )
    if entry:
        res = _build_result(entry, match_type, score=score)
        if explicit_aliases:
            res["aliases_detected"] = list(dict.fromkeys(res.get("aliases_detected", []) + explicit_aliases))
        return res

    # --- Priority 4: Unknown ---
    return {
        "normalized": clean_name,
        "canonical_name": clean_name,
        "status": "unknown",
        "match_type": None,
        "similarity_score": None,
        "code": None,
        "aliases_detected": explicit_aliases,
    }


def _build_result(entry, match_type, score, detected_code=None):
    """
    Build a normalization result dict from a matched dictionary entry.

    If the entry is marked is_generic, the status is "ambiguous"
    (the name is recognized but nonspecific). Otherwise "matched".
    """
    is_generic = entry.get("is_generic", False)
    code = detected_code or (entry.get("codes", [None])[0] if entry.get("codes") else None)
    return {
        "normalized": entry["canonical_name"],
        "canonical_name": entry["canonical_name"],
        "status": "ambiguous" if is_generic else "matched",
        "match_type": match_type,
        "similarity_score": round(score, 1),
        "code": code,
    }


# ---------------------------------------------------------------------------
# Bulk normalization (processes parsed ingredient tree)
# ---------------------------------------------------------------------------

def normalize_all(parsed_ingredients, dictionary, fuzzy_threshold=80):
    """
    Normalize a list of parsed ingredients (including sub-ingredients).

    Recursively normalizes each ingredient and its sub-ingredients.
    Preserves parent-child structure, aliases, INS codes, and raw OCR wording.

    Args:
        parsed_ingredients: List of ingredient dicts from extractor.parse_ingredients()
        dictionary:         Loaded ingredient dictionary
        fuzzy_threshold:    Minimum fuzzy similarity score

    Returns:
        List of normalized ingredient dicts, each containing:
          - raw:              original text from OCR
          - name:             extracted ingredient name
          - raw_name:         original text from OCR
          - canonical_name:   canonical dictionary name
          - normalized:       canonical name if matched
          - status:           matched / ambiguous / unknown
          - match_type:       how it was matched (or null)
          - similarity_score: match confidence (or null)
          - code:             INS/E code if detected
          - aliases_detected: list of detected aliases
          - sub_ingredients:  recursively normalized sub-ingredients
    """
    results = []

    for ingredient in parsed_ingredients:
        ing_name = ingredient["name"] if isinstance(ingredient, dict) else ingredient
        raw_text = ingredient.get("raw", ing_name) if isinstance(ingredient, dict) else ing_name
        detected_aliases = list(ingredient.get("aliases_detected", [])) if isinstance(ingredient, dict) else []
        existing_code = ingredient.get("code") if isinstance(ingredient, dict) else None

        # Normalize this ingredient's name
        norm = normalize_ingredient(
            ingredient if isinstance(ingredient, dict) else ing_name,
            dictionary,
            fuzzy_threshold,
        )

        # Merge code and aliases
        code = norm.get("code") or existing_code
        if code and code not in detected_aliases:
            detected_aliases.append(code)

        # Recursively normalize sub-ingredients
        sub_results = []
        if isinstance(ingredient, dict) and ingredient.get("sub_ingredients"):
            sub_results = normalize_all(
                ingredient["sub_ingredients"], dictionary, fuzzy_threshold
            )

        results.append({
            "raw": raw_text,
            "name": ing_name,
            "raw_name": raw_text,
            "canonical_name": norm["normalized"],
            "normalized": norm["normalized"],
            "status": norm["status"],
            "match_type": norm["match_type"],
            "similarity_score": norm["similarity_score"],
            "code": code,
            "aliases_detected": detected_aliases,
            "sub_ingredients": sub_results,
        })

    return results

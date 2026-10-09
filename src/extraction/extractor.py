"""
extractor.py — Ingredient Section Detection and Parsing
=========================================================

This module handles three tasks:
  1. SECTION DETECTION: Find where the ingredient list starts/stops
     in the raw OCR text, using configurable marker keywords.
  2. TEXT CLEANING: Remove OCR formatting artifacts (stray pipes,
     duplicate spaces, etc.) WITHOUT autocorrecting spelling.
  3. BRACKET-AWARE PARSING: Split the ingredient string on commas
     while respecting parenthesized sub-ingredient groups.

DESIGN PRINCIPLES:
  - Markers are configurable lists, not hardcoded strings scattered
    throughout the code.
  - Cleaning removes noise but NEVER silently "fixes" words —
    that's the normalizer's job (with fuzzy matching + transparency).
  - The parser correctly handles compound ingredients like:
      Seasoning (Sugar, Salt, Milk Solids (Whey, Cheese Powder))
    by tracking bracket depth so inner commas don't split outer items.
"""

import re


# ---------------------------------------------------------------------------
# Configurable markers — edit these lists to support more label formats
# ---------------------------------------------------------------------------

# Headings that indicate the START of an ingredient list.
# The extractor looks for any of these at the beginning of a line
# or as a distinct prefix within a line.
INGREDIENT_START_MARKERS = [
    "ingredients list:",
    "ingredients list -",
    "ingredients list",
    "ingredients:",
    "ingredients :",
    "ingredients -",
    "ingredients",
    "ingredient:",
    "ingredient :",
    "ingredient -",
    "ingredient",
    "contains:",
]

# Headings or keywords that indicate the END of the ingredient section.
# When a line contains any of these, the extractor stops collecting.
INGREDIENT_STOP_MARKERS = [
    "nutrition facts",
    "nutritional information",
    "international numbering system",
    "nutritional value",
    "nutrient information",
    "allergen information",
    "allergen advice",
    "allergy advice",
    "contains allergen",
    "may contain",
    "contains gluten",
    "contains wheat",
    "contains milk",
    "contains soy",
    "contains treenut",
    "contains peanut",
    "net weight",
    "net wt",
    "batch no",
    "lot no",
    "manufactured by",
    "marketed by",
    "distributed by",
    "packed by",
    "best before",
    "use by",
    "expiry date",
    "mfg date",
    "product of",
    "made in",
    "store in",
    "storage:",
    "directions:",
    "serving size",
    "servings per",
    "suggested serving",
    "serving suggestion",
    "enlarged to show",
    "fssai",
    "lic. no",
    "lic no",
    "license no",
    "customer care",
    "consumer care",
    "contact us",
    "contact customer",
    "barcode",
    "scan barcode",
    "store in a cool",
    "store away",
    "do not buy",
    "consume within",
    "mkt. by",
    "mfg. by",
    "pkd. by",
    "per 100ml",
    "per 100g",
    "per serve",
    "approximate values",
    "typical values",
]

# Tolerant regex for ingredient headings on curved packaging / OCR typos
START_HEADING_REGEX = re.compile(
    r"(?:^|[\s])(ingredients?|contains)\s*[:;\-]",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def detect_ingredient_section(text, start_markers=None, stop_markers=None):
    """
    Find and extract the ingredient section from raw OCR text.

    Strategy:
      1. Scan line-by-line for a start marker (e.g., "Ingredients:")
         or tolerant OCR regex variation.
      2. Once found, take the text AFTER the marker on that line.
      3. Continue collecting subsequent lines.
      4. Stop when a line matches a stop marker (e.g., "Batch No").

    Args:
        text:          Raw OCR text (may be multi-line).
        start_markers: Custom start marker list (or uses default).
        stop_markers:  Custom stop marker list (or uses default).

    Returns:
        The ingredient section text, or None if no start marker found.
    """
    if not text or not text.strip():
        return None

    start_markers = start_markers or INGREDIENT_START_MARKERS
    stop_markers = stop_markers or INGREDIENT_STOP_MARKERS

    lines = text.split("\n")
    collecting = False
    section_lines = []

    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if not collecting:
            # 1. Look for configured exact start markers first
            found = False
            for marker in start_markers:
                pattern = r'(?:^|[\s•*|~#\-\[\(])' + re.escape(marker) + r'(?:$|[\s:\-—\.\)\]])'
                match = re.search(pattern, line_lower)
                if match:
                    marker_end = match.end()
                    after_marker = line_clean[marker_end:].strip().lstrip(":-—, \t").strip()
                    if after_marker:
                        section_lines.append(after_marker)
                    collecting = True
                    found = True
                    break

            # 2. Tolerant heading regex fallback for curved labels / OCR distortions
            if not found:
                match = START_HEADING_REGEX.search(line_clean)
                if match:
                    marker_end = match.end()
                    after_marker = line_clean[marker_end:].strip().lstrip(":-—, \t").strip()
                    if after_marker:
                        section_lines.append(after_marker)
                    collecting = True
        else:
            # Check if this line signals the END of ingredients
            if _line_matches_stop_marker(line_lower, stop_markers):
                break
            # Skip completely empty lines
            if line_clean:
                section_lines.append(line_clean)

    if not section_lines:
        return None

    # Join lines with a single space (ingredient lists wrap across lines)
    return " ".join(section_lines)


def _line_matches_stop_marker(line_lower, stop_markers):
    """
    Check if a line contains any stop marker keyword.

    We check the lowercased, stripped line against each marker.
    """
    for marker in stop_markers:
        if marker in line_lower:
            return True
    # Tesseract commonly substitutes the first few letters in the standard
    # "Nutritional Information" heading. Match the distinctive suffix only
    # when it is followed by the second heading word.
    return bool(re.search(r"\b\w*tional\s+information\b", line_lower))


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_ocr_text(text):
    """
    Clean common OCR formatting artifacts from ingredient text.

    REMOVES:
      - Common OCR noise characters: =|, |_|, ~}, stray pipes
      - Stray quotation marks (often misread from * or other symbols)
      - Leading asterisks/bullets (allergen markers on original labels)
      - Duplicate/excessive whitespace
      - Non-printable control characters

    PRESERVES:
      - Parentheses ()  — needed for compound ingredient parsing
      - Commas          — needed for ingredient splitting
      - Ampersands &    — "Spices & Condiments"
      - Hyphens -       — "Anti-caking", "DL-Malic"
      - Periods .       — "No.", decimal numbers
      - Apostrophes '   — "5'-Ribonucleotides"
      - All word spelling (even misspelled) — fuzzy matcher handles this

    DOES NOT autocorrect words. OCR misspellings like "Mik" for "Milk"
    are preserved and handled later by the fuzzy normalizer.
    """
    if not text:
        return ""

    # Remove multi-character OCR artifacts (order: longest first)
    artifacts = ["|_|", "=|", "~}", "—|", "-|", "|}",  "{|"]
    for artifact in artifacts:
        text = text.replace(artifact, " ")

    # Remove remaining stray pipe characters (OCR noise from table borders)
    text = text.replace("|", " ")

    # Remove stray quotation marks (often misread * or formatting)
    text = text.replace("\u201c", "").replace("\u201d", "")   # smart quotes
    text = text.replace('"', "")

    # Remove leading asterisks (allergen markers on labels)
    text = re.sub(r"\*+", "", text)

    # Remove non-printable / unusual control characters
    # Keep ASCII printable range + common accented letters
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Bracket-aware ingredient parsing
# ---------------------------------------------------------------------------

def parse_ingredients(ingredient_text):
    """
    Parse a cleaned ingredient string into a structured list.

    This is the core parser. It handles compound ingredients like:
        Seasoning (Sugar, Salt, Milk Solids (Whey, Cheese Powder))
    and complex brackets with percentage annotations:
        Wheat Solids (62.6%) {Wheat Flour (Atta) (31.3%), Refined Wheat Flour (Maida) (31.3%)}

    by tracking bracket depth: delimiters inside (), [], {} are sub-ingredient
    separators, not top-level splits.

    Args:
        ingredient_text: Cleaned ingredient string (single line).

    Returns:
        List of ingredient dicts, each with:
          - name: the ingredient name (text before parentheses/brackets)
          - raw:  the full raw text including parenthesized content
          - sub_ingredients: list of parsed sub-ingredient dicts
                             (recursively structured the same way)
    """
    if not ingredient_text or not ingredient_text.strip():
        return []

    # Step 1: Split at top level (bracket depth == 0)
    raw_items = _split_at_top_level_commas(ingredient_text)

    # Step 2: Parse each item (may contain sub-ingredients in brackets)
    ingredients = []
    for item in raw_items:
        item = item.strip()
        if not item:
            continue
        parsed = _parse_single_ingredient(item)
        if parsed:
            if isinstance(parsed, list):
                ingredients.extend(parsed)
            else:
                ingredients.append(parsed)

    return ingredients


def _split_at_top_level_commas(text):
    """
    Split text on delimiters (commas, semicolons, and sentence-ending periods)
    that are NOT inside any brackets (), [], or {}.

    Tracks bracket depth across (), [], and {}. Only splits when depth == 0.
    Does NOT split on decimal numbers (e.g. 62.6%) or abbreviations (e.g. No.).
    """
    items = []
    current = []
    depth = 0
    n = len(text)
    i = 0

    while i < n:
        char = text[i]
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)   # Graceful: don't go negative
            current.append(char)
        elif depth == 0 and (char in ",;" or (char == "." and i + 1 < n and (text[i + 1].isspace() or text[i + 1].isupper()))):
            is_decimal = (char == "." and i > 0 and text[i - 1].isdigit() and i + 1 < n and text[i + 1].isdigit())
            is_abbr = (char == "." and i >= 2 and text[i - 2:i].lower() == "no")
            if is_decimal or is_abbr:
                current.append(char)
            else:
                seg = "".join(current).strip()
                if seg:
                    items.append(seg)
                current = []
        elif depth > 0 and char in ",;":
            # Heuristic recovery for dropped closing brackets in OCR:
            # If a delimiter is followed by a known top-level additive/ingredient category,
            # reset depth so the rest of the list isn't swallowed into nested sub-ingredients.
            remaining = text[i + 1:].lstrip()
            if re.match(r"^(?:acidity regulator|flavour|flavor|colour|color|preservative|antioxidant|stabilis|stabiliz|emulsifi|sweetener|thickener|vitamins|minerals|added flavour|permitted|edible vegetable oil|sugar|water|carbonated water)\b", remaining, re.I):
                depth = 0
                seg = "".join(current).strip()
                if seg:
                    items.append(seg)
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
        i += 1

    # Don't forget the last segment
    if current:
        seg = "".join(current).strip()
        if seg:
            items.append(seg)

    return items


def _is_percentage_text(text):
    """Check if the text represents only a percentage specification (e.g. '62.6%', '5.4%')."""
    clean = text.strip().strip("()[]{}")
    return bool(re.match(r"^\d+(?:\.\d+)?\s*%$", clean.strip()))


def _find_top_level_bracket_spans(text):
    """Find all top-level bracket spans across (), [], and {}."""
    spans = []
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    opening = set(pairs.values())
    first_unclosed = -1

    for i, char in enumerate(text):
        if char in opening:
            if not stack:
                first_unclosed = i
            stack.append(char)
        elif char in pairs:
            if stack and stack[-1] == pairs[char]:
                stack.pop()
                if not stack:
                    spans.append((first_unclosed, i + 1))
                    first_unclosed = -1
            elif stack:
                stack.pop()
    if stack and first_unclosed != -1:
        spans.append((first_unclosed, len(text)))
    return spans


KNOWN_SYNONYMS = {
    "atta", "maida", "besan", "suji", "sooji", "rava", "dalia",
    "all purpose flour", "whole wheat flour"
}
INS_CODE_RE = re.compile(r"^(?:INS|E)?\s*\d{2,4}[a-z]?$", re.IGNORECASE)


def _parse_single_ingredient(text):
    """
    Parse one ingredient string, possibly containing sub-ingredients, aliases, or INS codes.

    Handles:
      - Multiple bracket types: (), [], {}
      - Percentage filtering: e.g. (62.6%) is ignored as a percentage spec
      - Vernacular synonyms/aliases in parentheses: e.g. Wheat Flour (Atta) -> Atta is alias
      - INS additive codes: e.g. Colour (INS 150d) -> INS 150d is code/alias
      - Vegetable oil descriptors: e.g. Edible Vegetable Oil (Palmolein) -> Palmolein is descriptor/sub-ingredient
      - Sub-ingredient groups: e.g. {Wheat Flour (Atta), Refined Wheat Flour (Maida)}
      - Statutory declaration clauses: e.g. CONTAINS PERMITTED NATURAL COLOUR & ADDED FLAVOURS (...)
    """
    text = text.strip()
    if not text:
        return None

    # Handle statutory declarations like "CONTAINS PERMITTED NATURAL COLOUR & ADDED FLAVOURS (...)"
    statutory_match = re.match(r"^contains\s+(permitted|added|natural|artificial|synthetic)\b", text, re.IGNORECASE)
    if statutory_match:
        trimmed = text[len("contains"):].strip().lstrip(":-").strip()
        # Split on top-level '&' between statutory clauses if present
        and_parts = []
        depth = 0
        cur = []
        for ch in trimmed:
            if ch in "([{":
                depth += 1
                cur.append(ch)
            elif ch in ")]}":
                depth = max(0, depth - 1)
                cur.append(ch)
            elif depth == 0 and ch == "&":
                seg = "".join(cur).strip()
                if seg:
                    and_parts.append(seg)
                cur = []
            else:
                cur.append(ch)
        if cur:
            seg = "".join(cur).strip()
            if seg:
                and_parts.append(seg)
        if len(and_parts) > 1:
            res = []
            for part in and_parts:
                p = _parse_single_ingredient(part)
                if p:
                    if isinstance(p, list):
                        res.extend(p)
                    else:
                        res.append(p)
            return res

    spans = _find_top_level_bracket_spans(text)
    if not spans:
        name = _clean_ingredient_name(text)
        if not name:
            return None
        return {
            "name": name,
            "raw": text,
            "sub_ingredients": [],
            "aliases_detected": [],
            "code": None,
        }

    # Extract base name before first bracket span
    first_start = spans[0][0]
    base_name = _clean_ingredient_name(text[:first_start])

    sub_ingredients = []
    aliases_detected = []
    detected_code = None

    for start, end in spans:
        if end <= len(text) and text[end - 1] in ")]}":
            inner = text[start + 1:end - 1].strip()
        else:
            inner = text[start + 1:end].rstrip(")]}").strip()
        if _is_percentage_text(inner):
            continue

        inner_clean = inner.strip()
        # Check if single INS code (e.g. INS 150d, INS 320)
        if INS_CODE_RE.match(inner_clean):
            detected_code = inner_clean
            aliases_detected.append(inner_clean)
            continue

        # Check if known vernacular alias (e.g. Atta, Maida)
        if inner_clean.lower() in KNOWN_SYNONYMS:
            aliases_detected.append(inner_clean)
            continue

        # Check if descriptor for vegetable oil (e.g. Palmolein)
        if "oil" in base_name.lower() and inner_clean.lower() in (
            "palmolein", "palmolein oil", "palm olein", "sunflower", "mustard", "soybean", "groundnut"
        ):
            aliases_detected.append(inner_clean)
            sub_items = parse_ingredients(inner)
            sub_ingredients.extend(sub_items)
            continue

        # General case: parse as sub-ingredients
        sub_items = parse_ingredients(inner)
        sub_ingredients.extend(sub_items)

    item_name = base_name if base_name else _clean_ingredient_name(text)
    if detected_code:
        item_name = f"{base_name} ({detected_code})" if base_name else detected_code

    return {
        "name": item_name,
        "raw": text,
        "sub_ingredients": sub_ingredients,
        "aliases_detected": aliases_detected,
        "code": detected_code,
    }


def _clean_ingredient_name(name):
    """
    Light cleaning on an individual ingredient name.

    Removes trailing/leading whitespace and punctuation noise
    but does NOT autocorrect spelling.
    """
    if not name:
        return ""
    # Strip whitespace and common trailing punctuation noise
    name = name.strip().strip(".,;:-").strip()
    return name


def extract_allergens(text: str) -> dict[str, list[str]]:
    """
    Extract statutory allergen declarations from OCR text.

    Separates 'CONTAINS ...' and 'MAY CONTAIN ...' declarations from regular
    ingredients and normalizes allergen names.

    Args:
        text: Raw OCR text.

    Returns:
        Dict with 'contains' and 'may_contain' lists of allergen strings.
    """
    if not text:
        return {"contains": [], "may_contain": []}

    contains_list: list[str] = []
    may_contain_list: list[str] = []

    allergen_map = {
        "gluten": "Gluten",
        "wheat": "Wheat",
        "milk": "Milk",
        "dairy": "Dairy",
        "soy": "Soy",
        "soya": "Soy",
        "soybean": "Soy",
        "peanut": "Peanuts",
        "peanuts": "Peanuts",
        "treenut": "Tree nuts",
        "treenuts": "Tree nuts",
        "tree nut": "Tree nuts",
        "tree nuts": "Tree nuts",
        "nut": "Nuts",
        "nuts": "Nuts",
        "almond": "Almonds",
        "cashew": "Cashews",
        "walnut": "Walnuts",
        "sesame": "Sesame",
        "mustard": "Mustard",
        "egg": "Eggs",
        "eggs": "Eggs",
        "fish": "Fish",
        "crustacean": "Crustaceans",
        "crustaceans": "Crustaceans",
        "shellfish": "Shellfish",
        "sulphite": "Sulphites",
        "sulphites": "Sulphites",
        "sulfite": "Sulfites",
        "sulfites": "Sulfites",
    }

    # 1. Search for 'MAY CONTAIN' statements
    may_matches = re.finditer(
        r"\bmay\s+contain(?:\s+traces\s+of)?\s+([^.\n;]+)", text, re.IGNORECASE
    )
    for m in may_matches:
        raw_items = re.split(r"[,/&]|(?:\band\b)", m.group(1), flags=re.IGNORECASE)
        for item in raw_items:
            clean = item.strip().strip(".:").lower()
            if not clean:
                continue
            matched = False
            for k, std in allergen_map.items():
                if re.search(r"\b" + re.escape(k) + r"\b", clean):
                    if std not in may_contain_list:
                        may_contain_list.append(std)
                    matched = True
                    break
            if not matched and len(clean) > 2:
                formatted = clean.title()
                if formatted not in may_contain_list:
                    may_contain_list.append(formatted)

    # 2. Search for 'CONTAINS' statements (excluding additive declarations)
    additive_prefixes = (
        "permitted", "added", "natural", "artificial", "synthetic",
        "nature identical", "caffeine", "non-nutritive"
    )
    contains_matches = re.finditer(
        r"(?:^|[\n.;])\s*(?:allergen(?:\s+advice|\s+information)?\s*[:\-—]?\s*)?contains\s+([^.\n;]+)",
        text,
        re.IGNORECASE,
    )
    for m in contains_matches:
        stmt = m.group(1).strip()
        if any(stmt.lower().startswith(p) for p in additive_prefixes):
            continue
        raw_items = re.split(r"[,/&]|(?:\band\b)", stmt, flags=re.IGNORECASE)
        for item in raw_items:
            clean = item.strip().strip(".:").lower()
            if not clean or clean.startswith("traces"):
                continue
            matched = False
            for k, std in allergen_map.items():
                if re.search(r"\b" + re.escape(k) + r"\b", clean):
                    if std not in contains_list:
                        contains_list.append(std)
                    matched = True
                    break
            if not matched and len(clean) > 2:
                formatted = clean.title()
                if formatted not in contains_list:
                    contains_list.append(formatted)

    return {"contains": contains_list, "may_contain": may_contain_list}

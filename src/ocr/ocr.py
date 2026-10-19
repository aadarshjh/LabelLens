"""
ocr.py — OCR Engine for Food Label Text Extraction
====================================================

This module wraps Tesseract OCR and provides:
  1. Text extraction from preprocessed images
  2. Confidence scoring using Tesseract's word-level confidence data
  3. Testing multiple PSM (Page Segmentation Mode) options
  4. Ranking results to find the best OCR output

WHY CONFIDENCE SCORING?
  Tesseract assigns a confidence score (0-100) to each word it detects.
  By averaging these scores, we get a rough measure of how "sure"
  Tesseract is about its reading. Higher confidence generally means
  more accurate text. This lets us objectively compare preprocessing
  strategies instead of guessing.

WHY MULTIPLE PSMs?
  Tesseract's Page Segmentation Mode tells it what kind of text layout
  to expect. A food ingredient list is typically a paragraph/block of
  text, but some labels may be single columns or have unusual layouts.
  Testing a few reasonable PSMs helps find the best interpretation.
"""

import pytesseract
from pytesseract import Output
import shutil
import sys
import os


# ---------------------------------------------------------------------------
# Tesseract availability check
# ---------------------------------------------------------------------------

def _find_tesseract_cmd():
    custom_path = pytesseract.pytesseract.tesseract_cmd
    common_paths = [
        custom_path,
        "tesseract",                           # On system PATH
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.environ.get("TESSERACT_PATH", ""),
        "/opt/homebrew/bin/tesseract",          # Homebrew Apple Silicon
        "/usr/local/bin/tesseract",             # Homebrew Intel Mac
    ]

    for path in common_paths:
        if path and (shutil.which(path) or os.path.isfile(path)):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                version = pytesseract.get_tesseract_version()
                return path, version
            except Exception:
                continue
    return None, None


# Attempt auto-detection on import
_found_cmd, _ = _find_tesseract_cmd()
if _found_cmd:
    pytesseract.pytesseract.tesseract_cmd = _found_cmd


def check_tesseract():
    """
    Verify that Tesseract OCR is installed and accessible.

    Returns the path to the Tesseract binary if found.
    Exits with a helpful error message if not found.
    """
    path, version = _find_tesseract_cmd()
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"  ✓ Tesseract found: {path} (version {version})")
        return path

    # If we get here, Tesseract was not found
    print("\n" + "=" * 60)
    print("  ERROR: Tesseract OCR is NOT installed or not found!")
    print("=" * 60)
    print("\n  To install on Windows, run:")
    print("    winget install UB-Mannheim.TesseractOCR")
    print("\n  To install on macOS, run:")
    print("    brew install tesseract")
    print("\n  After installing, make sure it's on your PATH:")
    print("    which tesseract  (or 'where.exe tesseract' on Windows)")
    print("    tesseract --version")
    print("=" * 60 + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# PSM modes relevant to food labels
# ---------------------------------------------------------------------------

FOOD_LABEL_PSM_MODES = [
    (6,  "single_block"),       # Most common for ingredient paragraphs
    (3,  "fully_automatic"),    # Let Tesseract decide
]


# ---------------------------------------------------------------------------
# Core OCR functions
# ---------------------------------------------------------------------------

def extract_text(image, psm=6):
    """
    Run Tesseract OCR on an image and return the extracted text.

    Args:
        image: numpy array (BGR or grayscale)
        psm:   Page Segmentation Mode (int)

    Returns:
        Extracted text as a string
    """
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()


def get_confidence(image, psm=6):
    """
    Get word-level confidence data from Tesseract.

    Tesseract's image_to_data() returns detailed information about
    every word it detects, including a confidence score (0-100).

    We filter out "empty" detections (non-text areas) and calculate
    the average confidence across all detected words.

    Returns:
        (average_confidence, word_count, word_details)
        - average_confidence: float 0-100, or 0 if no words detected
        - word_count: number of words Tesseract found
        - word_details: list of (word, confidence) tuples
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(image, config=config,
                                      output_type=Output.DICT)

    word_details = []
    for i, word in enumerate(data["text"]):
        # Skip empty detections (whitespace, headers, etc.)
        word = word.strip()
        if not word:
            continue

        conf = int(data["conf"][i])
        # Tesseract uses -1 for "no confidence" on empty detections
        if conf < 0:
            continue

        word_details.append((word, conf))

    if not word_details:
        return 0.0, 0, []

    avg_conf = sum(conf for _, conf in word_details) / len(word_details)
    return avg_conf, len(word_details), word_details


def run_ocr_single(image, strategy_name, psm=6):
    """
    Run OCR on a single preprocessed image with a specific PSM mode.

    Returns a dict with all results:
        {
            "strategy": strategy name,
            "psm": PSM mode used,
            "psm_label": human-readable PSM label,
            "text": extracted text,
            "confidence": average confidence (0-100),
            "word_count": number of words found,
            "word_details": list of (word, confidence) tuples,
        }
    """
    text = extract_text(image, psm=psm)
    confidence, word_count, word_details = get_confidence(image, psm=psm)

    # Find the human-readable PSM label
    psm_label = "unknown"
    for mode, label in FOOD_LABEL_PSM_MODES:
        if mode == psm:
            psm_label = label
            break

    return {
        "strategy": strategy_name,
        "psm": psm,
        "psm_label": psm_label,
        "text": text,
        "confidence": confidence,
        "word_count": word_count,
        "word_details": word_details,
    }


def run_ocr_all_psms(image, strategy_name):
    """
    Run OCR on one image with ALL candidate PSM modes.

    This helps find the best PSM for a given preprocessing strategy.

    Returns a list of result dicts (one per PSM mode).
    """
    results = []
    for psm, label in FOOD_LABEL_PSM_MODES:
        try:
            result = run_ocr_single(image, strategy_name, psm=psm)
            results.append(result)
        except Exception as e:
            # Some PSM modes may fail on certain images — that's OK
            print(f"    ⚠ PSM {psm} ({label}) failed: {e}")
    return results


def score_ocr_result(result):
    """
    Multi-factor scoring function for food label OCR candidates.
    Balances raw OCR confidence, text completeness, presence of ingredient headings,
    and density of food/chemical ingredient keywords.
    Prevents tiny 1-2 word snippets with spuriously high confidence from beating
    full, informative ingredient sections.
    """
    text = result.get("text", "")
    word_count = result.get("word_count", 0)
    confidence = result.get("confidence", 0.0)

    # Penalize tiny snippets (< 5 words) that cannot be an ingredient list
    if word_count < 5:
        base_score = confidence * 0.25
    elif word_count < 10:
        base_score = confidence * 0.6
    else:
        base_score = confidence

    score = base_score

    # Check for ingredient section or ingredient markers
    try:
        from src.extraction.extractor import detect_ingredient_section
        sec = detect_ingredient_section(text)
        if sec:
            score += 35.0
            score += min(15.0, len(sec) * 0.05)
    except Exception:
        pass

    # Word count substantiveness (up to 15 points)
    score += min(15.0, word_count * 0.2)

    return (score, confidence, word_count)


def find_best_result(all_results):
    """
    Find the best OCR result based on content completeness and confidence score.

    Selection criteria:
      1. Candidates with a detected ingredient section are prioritized.
      2. Multi-factor score (presence of ingredient section, length, completeness)
      3. Average confidence score
      4. Total word count
    """
    if not all_results:
        return None

    # Filter out results with no text
    valid = [r for r in all_results if r["text"] and r["word_count"] > 0]

    if not valid:
        return None

    try:
        from src.extraction.extractor import detect_ingredient_section
        with_section = [r for r in valid if detect_ingredient_section(r.get("text", ""))]
        if with_section:
            with_section.sort(key=score_ocr_result, reverse=True)
            return with_section[0]
    except Exception:
        pass

    # Sort by multi-factor score (descending)
    valid.sort(key=score_ocr_result, reverse=True)

    return valid[0]

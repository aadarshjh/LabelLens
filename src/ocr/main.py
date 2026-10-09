"""
main.py — Food Label OCR Pipeline (Phase 1)
=============================================

This is the main entry point for the food label OCR pipeline.

WHAT IT DOES:
  1. Loads a food label image
  2. Applies multiple preprocessing strategies
  3. Runs Tesseract OCR on each preprocessed version
  4. Tests multiple Page Segmentation Modes (PSMs) for each
  5. Calculates confidence scores for every combination
  6. Saves all intermediate images to output/ for inspection
  7. Saves a detailed comparison report
  8. Prints the BEST result based on confidence scoring

USAGE:
  python src/main.py                          # uses first image in images/
  python src/main.py images/label1.jpeg       # specify an image path

DESIGN PHILOSOPHY:
  We do NOT assume any preprocessing is better than the original.
  The original image is always tested as the baseline. The confidence
  score objectively picks the winner.
"""

import cv2
import os
import sys
import glob
import time

from src.config import ARTIFACTS_DIR, IMAGES_DIR
from src.ocr.preprocessing import get_all_strategies
from src.ocr.ocr import check_tesseract, run_ocr_all_psms, find_best_result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory paths are resolved from the repository configuration.
PROJECT_DIR = str(ARTIFACTS_DIR.parent)
IMAGES_DIR = str(IMAGES_DIR)
OUTPUT_DIR = str(ARTIFACTS_DIR / "ocr")


# ---------------------------------------------------------------------------
# Image loading with error handling
# ---------------------------------------------------------------------------

def find_image(image_path=None):
    """
    Find and validate the input image.

    If no path is given, automatically picks the first JPEG/PNG
    in the images/ directory.

    Returns the absolute path to a valid image file.
    Exits with a helpful error if anything goes wrong.
    """
    if image_path:
        # User provided a specific path
        if not os.path.isabs(image_path):
            # Try relative to project dir first, then current dir
            candidates = [
                os.path.join(PROJECT_DIR, image_path),
                os.path.abspath(image_path),
            ]
        else:
            candidates = [image_path]

        for path in candidates:
            if os.path.isfile(path):
                return path

        print(f"\n  ERROR: Image not found: {image_path}")
        print(f"  Searched in:")
        for path in candidates:
            print(f"    - {path}")
        sys.exit(1)
    else:
        # Auto-detect: find the first image in images/
        if not os.path.isdir(IMAGES_DIR):
            print(f"\n  ERROR: Images directory not found: {IMAGES_DIR}")
            print(f"  Create it and add a food label image:")
            print(f"    mkdir -p {IMAGES_DIR}")
            sys.exit(1)

        # Look for common image formats
        patterns = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(os.path.join(IMAGES_DIR, pattern)))

        if not images:
            print(f"\n  ERROR: No images found in {IMAGES_DIR}")
            print(f"  Add a food label image (JPEG or PNG) to the images/ folder.")
            sys.exit(1)

        # Use the first image found
        return sorted(images)[0]


def load_image(image_path, max_dim=2400):
    """
    Load an image using OpenCV with validation.

    Returns the image as a numpy array (BGR format).
    Exits with an error if the image can't be read.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"\n  ERROR: Could not read image: {image_path}")
        print(f"  The file may be corrupted or in an unsupported format.")
        print(f"  Supported formats: JPEG (.jpg/.jpeg), PNG (.png)")
        sys.exit(1)

    height, width = image.shape[:2]
    print(f"  ✓ Image loaded: {os.path.basename(image_path)}")
    print(f"    Size: {width} × {height} pixels")
    print(f"    Channels: {image.shape[2] if len(image.shape) == 3 else 1}")

    if max_dim and max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        h_new, w_new = image.shape[:2]
        print(f"    Rescaled for OCR: {w_new} × {h_new} pixels")

    return image


# ---------------------------------------------------------------------------
# Save intermediate images for inspection
# ---------------------------------------------------------------------------

def save_intermediate_image(image, strategy_name, output_dir):
    """
    Save a preprocessed image to the output directory.

    This lets you visually inspect what each preprocessing technique
    did to the image and understand why OCR might work better/worse.
    """
    filename = f"preprocessed_{strategy_name}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_report(all_results, best_result, image_path, output_dir):
    """
    Save a detailed text report comparing all OCR results.

    The report includes every strategy × PSM combination, sorted by
    confidence score. This makes it easy to see which approaches work
    best and worst for your specific image.
    """
    report_path = os.path.join(output_dir, "ocr_comparison_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  FOOD LABEL OCR — COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Image: {os.path.basename(image_path)}\n")
        f.write(f"  Date:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Total strategies tested: "
                f"{len(set(r['strategy'] for r in all_results))}\n")
        f.write(f"  Total OCR runs: {len(all_results)}\n\n")

        # Sort all results by confidence (descending)
        sorted_results = sorted(all_results,
                                key=lambda r: (r["confidence"], r["word_count"]),
                                reverse=True)

        f.write("-" * 70 + "\n")
        f.write(f"  {'Rank':<5} {'Strategy':<25} {'PSM':<20} "
                f"{'Confidence':>10} {'Words':>6}\n")
        f.write("-" * 70 + "\n")

        for i, r in enumerate(sorted_results, 1):
            marker = " ★" if r is best_result else "  "
            f.write(f"{marker}{i:<4} {r['strategy']:<25} "
                    f"PSM {r['psm']} ({r['psm_label']:<14}) "
                    f"{r['confidence']:>8.1f}% {r['word_count']:>6}\n")

        f.write("\n" + "=" * 70 + "\n\n")

        # Write full text for each strategy (best PSM only)
        strategies_seen = set()
        for r in sorted_results:
            if r["strategy"] in strategies_seen:
                continue
            strategies_seen.add(r["strategy"])

            f.write(f"--- {r['strategy']} (PSM {r['psm']}, "
                    f"Confidence: {r['confidence']:.1f}%) ---\n\n")
            f.write(r["text"] if r["text"] else "(no text detected)")
            f.write("\n\n")

    return report_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """
    Main pipeline execution.

    Steps:
      1. Check Tesseract installation
      2. Find and load the input image
      3. For each preprocessing strategy:
         a. Apply preprocessing
         b. Save the intermediate image
         c. Run OCR with multiple PSM modes
         d. Record all results
      4. Find the best result by confidence
      5. Save the comparison report
      6. Print the best result
    """
    print("\n" + "=" * 60)
    print("  FOOD LABEL OCR PIPELINE — Phase 1")
    print("=" * 60 + "\n")

    # --- Step 1: Check Tesseract ---
    print("Step 1: Checking Tesseract OCR installation...")
    check_tesseract()

    # --- Step 2: Find and load image ---
    print("\nStep 2: Loading image...")
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    image_path = find_image(image_arg)
    image = load_image(image_path)

    # --- Step 3: Prepare output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: {OUTPUT_DIR}")

    # --- Step 4: Run preprocessing + OCR ---
    print("\nStep 3: Running preprocessing strategies + OCR...\n")
    strategies = get_all_strategies()
    all_results = []

    for i, (name, preprocess_fn) in enumerate(strategies, 1):
        print(f"  [{i:>2}/{len(strategies)}] {name}...", end="", flush=True)

        try:
            # Apply preprocessing
            processed = preprocess_fn(image)

            # Save intermediate image for visual inspection
            save_intermediate_image(processed, name, OUTPUT_DIR)

            # Run OCR with all PSM modes for this preprocessed image
            results = run_ocr_all_psms(processed, name)
            all_results.extend(results)

            # Show the best confidence for this strategy
            best_for_strategy = find_best_result(results)
            if best_for_strategy:
                print(f"  confidence: {best_for_strategy['confidence']:.1f}%  "
                      f"(PSM {best_for_strategy['psm']}, "
                      f"{best_for_strategy['word_count']} words)")
            else:
                print(f"  (no text detected)")

        except Exception as e:
            print(f"  ERROR: {e}")

    # --- Step 5: Find overall best result ---
    print("\n" + "-" * 60)
    print("Step 4: Analyzing results...\n")

    best = find_best_result(all_results)

    if not best:
        print("  ERROR: No text was detected by any strategy!")
        print("  Check that the image contains readable text.")
        sys.exit(1)

    # --- Step 6: Save report ---
    report_path = save_report(all_results, best, image_path, OUTPUT_DIR)
    print(f"  ✓ Comparison report saved: {report_path}")
    print(f"  ✓ Preprocessed images saved in: {OUTPUT_DIR}")

    # --- Save best OCR text for Phase 2 integration ---
    best_text_path = os.path.join(OUTPUT_DIR, "best_ocr_result.txt")
    with open(best_text_path, "w") as f:
        f.write(best["text"])
    print(f"  ✓ Best OCR text saved: {best_text_path}")

    # --- Step 7: Print best result ---
    print("\n")
    print("=" * 60)
    print("  ★ BEST OCR RESULT")
    print("=" * 60)
    print(f"  Method:     {best['strategy']}")
    print(f"  PSM:        {best['psm']} ({best['psm_label']})")
    print(f"  Confidence: {best['confidence']:.1f}%")
    print(f"  Words:      {best['word_count']}")
    print("=" * 60)
    print()
    print("  Extracted Text:")
    print("  " + "-" * 56)
    # Indent each line of the extracted text for clean display
    for line in best["text"].split("\n"):
        if line.strip():
            print(f"  {line}")
    print("  " + "-" * 56)
    print()

    # --- Step 8: Show runner-up for comparison ---
    # Find the best result from a DIFFERENT strategy
    runner_ups = [r for r in all_results
                  if r["strategy"] != best["strategy"]
                  and r["text"] and r["word_count"] > 0]
    if runner_ups:
        runner_ups.sort(key=lambda r: (r["confidence"], r["word_count"]),
                        reverse=True)
        runner_up = runner_ups[0]
        print(f"  Runner-up: {runner_up['strategy']} "
              f"(PSM {runner_up['psm']}, "
              f"confidence: {runner_up['confidence']:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

"""
Phase 1 Test Suite — OCR Pipeline Unit Tests
=============================================

Tests image loading, preprocessing strategies, OCR execution,
best-result selection, and error handling.

Does NOT depend on external APIs.
Uses Tesseract directly (must be installed) and the project's label1.jpeg.
"""

import pytest
import numpy as np
import cv2
import os
import sys
import shutil

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract executable unavailable")

# Import the consolidated OCR package from the repository root.
from src.ocr.preprocessing import get_all_strategies
from src.ocr.ocr import (
    run_ocr_all_psms,
    run_ocr_single,
    find_best_result,
    get_confidence,
    extract_text,
    FOOD_LABEL_PSM_MODES,
    check_tesseract,
)
from src.ocr.main import load_image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LABEL1_PATH = os.path.join(PROJECT_ROOT, 'data', 'images', 'label1.jpeg')


@pytest.fixture
def sample_image():
    """Load label1.jpeg if available, else create a synthetic test image."""
    if os.path.isfile(LABEL1_PATH):
        return cv2.imread(LABEL1_PATH)
    # Fallback: create a white image with black text-like features
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Ingredients: Sugar, Salt", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


@pytest.fixture
def blank_image():
    """A completely blank white image — OCR should find nothing meaningful."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 255


# ---------------------------------------------------------------------------
# Image Loading Tests
# ---------------------------------------------------------------------------

class TestImageLoading:
    def test_load_valid_image(self):
        """load_image returns a numpy array for a valid path."""
        if not os.path.isfile(LABEL1_PATH):
            pytest.skip("label1.jpeg not found")
        image = load_image(LABEL1_PATH)
        assert isinstance(image, np.ndarray)
        assert image.ndim >= 2
        assert image.shape[0] > 0 and image.shape[1] > 0

    def test_load_invalid_image_raises(self, tmp_path):
        """load_image raises ValueError for a corrupt/nonexistent file."""
        fake_path = str(tmp_path / "nonexistent.jpg")
        with pytest.raises(ValueError):
            load_image(fake_path)

    def test_load_corrupt_image_raises(self, tmp_path):
        """load_image raises ValueError for a file that exists but is not an image."""
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_text("this is not an image")
        with pytest.raises(ValueError):
            load_image(str(corrupt_file))


# ---------------------------------------------------------------------------
# Preprocessing Strategy Tests
# ---------------------------------------------------------------------------

class TestPreprocessingStrategies:
    def test_strategies_return_nonempty_list(self):
        """get_all_strategies returns a list with at least the original + some techniques."""
        strategies = get_all_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) == 3

    def test_original_is_first_candidate(self):
        """The 'original' (unprocessed) image must be the first strategy."""
        strategies = get_all_strategies()
        assert strategies[0][0] == "original"

    def test_each_strategy_returns_valid_image(self, sample_image):
        """Every strategy must return a valid numpy array without crashing."""
        strategies = get_all_strategies()
        for name, fn in strategies:
            result = fn(sample_image)
            assert isinstance(result, np.ndarray), f"Strategy '{name}' did not return ndarray"
            assert result.size > 0, f"Strategy '{name}' returned empty image"

    def test_original_strategy_preserves_image(self, sample_image):
        """The original strategy returns a copy of the input, not a reference."""
        strategies = get_all_strategies()
        original_fn = strategies[0][1]
        result = original_fn(sample_image)
        assert result is not sample_image
        np.testing.assert_array_equal(result, sample_image)

    def test_strategy_names_are_unique(self):
        """All strategy names must be unique."""
        strategies = get_all_strategies()
        names = [name for name, _ in strategies]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# OCR Execution Tests
# ---------------------------------------------------------------------------

class TestOCRExecution:
    def test_tesseract_is_available(self):
        """check_tesseract should succeed if Tesseract is installed."""
        try:
            path = check_tesseract()
            assert path is not None
        except SystemExit:
            pytest.skip("Tesseract not installed")

    def test_extract_text_returns_string(self, sample_image):
        """extract_text returns a string."""
        text = extract_text(sample_image, psm=6)
        assert isinstance(text, str)

    def test_get_confidence_returns_tuple(self, sample_image):
        """get_confidence returns (avg_conf, word_count, word_details)."""
        result = get_confidence(sample_image, psm=6)
        assert isinstance(result, tuple)
        assert len(result) == 3
        avg_conf, word_count, details = result
        assert isinstance(avg_conf, float)
        assert isinstance(word_count, int)
        assert isinstance(details, list)

    def test_psm_modes_defined(self):
        """FOOD_LABEL_PSM_MODES should contain multiple PSM options."""
        assert len(FOOD_LABEL_PSM_MODES) == 2
        for psm, label in FOOD_LABEL_PSM_MODES:
            assert isinstance(psm, int)
            assert isinstance(label, str)

    def test_run_ocr_all_psms_returns_results(self, sample_image):
        """run_ocr_all_psms returns a list of result dicts."""
        results = run_ocr_all_psms(sample_image, "test_strategy")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "confidence" in r
            assert "strategy" in r
            assert r["strategy"] == "test_strategy"

    def test_run_ocr_single_result_shape(self, sample_image):
        """run_ocr_single returns a dict with all required keys."""
        result = run_ocr_single(sample_image, "test", psm=6)
        required_keys = {"strategy", "psm", "psm_label", "text", "confidence",
                         "word_count", "word_details"}
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Best Result Selection Tests
# ---------------------------------------------------------------------------

class TestBestResultSelection:
    def test_find_best_with_valid_results(self):
        """find_best_result picks the highest confidence result."""
        results = [
            {"text": "hello", "confidence": 50.0, "word_count": 1},
            {"text": "ingredients: sugar", "confidence": 90.0, "word_count": 2},
            {"text": "ing", "confidence": 70.0, "word_count": 1},
        ]
        best = find_best_result(results)
        assert best is not None
        assert best["confidence"] == 90.0

    def test_find_best_tiebreak_by_word_count(self):
        """On tied confidence, prefer the result with more words."""
        results = [
            {"text": "hello", "confidence": 80.0, "word_count": 1},
            {"text": "hello world foo", "confidence": 80.0, "word_count": 3},
        ]
        best = find_best_result(results)
        assert best["word_count"] == 3

    def test_find_best_empty_list(self):
        """find_best_result returns None for an empty list."""
        assert find_best_result([]) is None

    def test_find_best_all_empty_text(self):
        """find_best_result returns None when all results have empty text."""
        results = [
            {"text": "", "confidence": 0.0, "word_count": 0},
            {"text": "", "confidence": 0.0, "word_count": 0},
        ]
        assert find_best_result(results) is None

    def test_find_best_filters_zero_word_count(self):
        """Results with word_count=0 are excluded even if confidence > 0."""
        results = [
            {"text": "x", "confidence": 10.0, "word_count": 0},
            {"text": "ingredients", "confidence": 5.0, "word_count": 1},
        ]
        best = find_best_result(results)
        assert best is not None
        assert best["text"] == "ingredients"


# ---------------------------------------------------------------------------
# Blank Image / Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_blank_image_low_or_no_confidence(self, blank_image):
        """A blank image should produce zero or very low confidence."""
        conf, count, details = get_confidence(blank_image, psm=6)
        # A completely blank image should have no words
        assert count == 0 or conf < 20.0

    def test_full_pipeline_on_real_image(self, sample_image):
        """Run the full strategy+PSM loop and verify find_best_result works."""
        strategies = get_all_strategies()
        all_results = []
        for name, fn in strategies:
            try:
                processed = fn(sample_image)
                results = run_ocr_all_psms(processed, name)
                all_results.extend(results)
            except Exception:
                continue
        best = find_best_result(all_results)
        # With a real label image, we expect a valid result
        if os.path.isfile(LABEL1_PATH):
            assert best is not None
            assert best["confidence"] > 0
            assert len(best["text"]) > 0

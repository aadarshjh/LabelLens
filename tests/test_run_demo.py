"""Tests for run_demo.py CLI behavior and error handling."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import run_demo


def test_missing_default_image_prints_helpful_message_and_exits(monkeypatch, tmp_path, capsys):
    """When no image is supplied and default sample image does not exist, exit 1 cleanly."""
    empty_images_dir = tmp_path / "images"
    empty_images_dir.mkdir()
    monkeypatch.setattr(run_demo, "IMAGES_DIR", empty_images_dir)
    monkeypatch.setattr(sys, "argv", ["run_demo.py"])

    with pytest.raises(SystemExit) as exc_info:
        run_demo.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR: Default demo image" in captured.err
    assert "python run_demo.py <path-to-image> [--with-llm]" in captured.err


def test_explicit_missing_image_prints_error_and_exits(monkeypatch, tmp_path, capsys):
    """When an explicit non-existent image path is given, exit 1 cleanly."""
    missing_file = tmp_path / "nonexistent.jpg"
    monkeypatch.setattr(sys, "argv", ["run_demo.py", str(missing_file)])

    with pytest.raises(SystemExit) as exc_info:
        run_demo.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR: Image file not found" in captured.err
    assert str(missing_file) in captured.err
    assert "python run_demo.py <path-to-image> [--with-llm]" in captured.err


def test_default_image_runs_pipeline_when_present(monkeypatch, tmp_path, capsys):
    """When default image exists and no arg is passed, it uses the default image."""
    dummy_images_dir = tmp_path / "images"
    dummy_images_dir.mkdir()
    dummy_label = dummy_images_dir / "label1.jpeg"
    dummy_label.write_bytes(b"dummy")

    monkeypatch.setattr(run_demo, "IMAGES_DIR", dummy_images_dir)
    monkeypatch.setattr(sys, "argv", ["run_demo.py"])

    mock_result = {
        "ocr": {"strategy": "test", "psm": 6, "confidence": 95.0},
        "phase2": {"summary": {"total": 1, "matched": 1, "ambiguous": 0, "unknown": 0}},
        "llm_inputs": [{"evidence": [{"source": "test_src"}]}],
    }

    with patch.object(run_demo, "run_pre_llm_pipeline", return_value=mock_result) as mock_pipeline:
        run_demo.main()
        mock_pipeline.assert_called_once_with(str(dummy_label))

    captured = capsys.readouterr()
    assert '"structured_inputs": 1' in captured.out


def test_unreadable_image_handled_gracefully(monkeypatch, tmp_path, capsys):
    """When image is present but unreadable, pipeline ValueError is caught with clean exit 1."""
    corrupt_image = tmp_path / "corrupt.jpg"
    corrupt_image.write_bytes(b"bad_data")

    monkeypatch.setattr(sys, "argv", ["run_demo.py", str(corrupt_image)])

    with patch.object(run_demo, "run_pre_llm_pipeline", side_effect=ValueError("Corrupt image data")):
        with pytest.raises(SystemExit) as exc_info:
            run_demo.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR: Failed to process image" in captured.err
    assert "Corrupt image data" in captured.err

from pathlib import Path


def test_single_page_renders_api_values_safely_and_separates_allergen_lists():
    html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "insertAdjacentHTML" not in html
    assert "innerHTML" not in html
    assert "asText(data.name) || 'Unknown ingredient'" in html
    assert "response.product_name" in html
    assert "Product name not identified" in html
    assert "allergen-contains" in html and "allergen-may-contain" in html
    assert "scan_history" not in html


def test_compact_history_storage_contract():
    html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "saveCompactScanHistory" in html
    assert "MAX_HISTORY_ENTRIES = 20" in html
    assert "labellens_recent_scans" in html
    # Ensure bloated raw fields are NOT stored in history
    assert "phase4_results" not in html
    assert "authoritative_evidence" not in html
    assert "tool_calls" not in html
    # Storage error handling is present and non-blocking
    assert "Scan history could not be saved to local storage" in html

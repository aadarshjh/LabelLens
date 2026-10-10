from pathlib import Path


def test_single_page_renders_api_values_safely_and_separates_allergen_lists():
    html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text()
    assert "insertAdjacentHTML" not in html
    assert "innerHTML" not in html
    assert "asText(data.name) || 'Unknown ingredient'" in html
    assert "response.product_name" in html
    assert "Product name not identified" in html
    assert "allergen-contains" in html and "allergen-may-contain" in html
    assert "scan_history" not in html

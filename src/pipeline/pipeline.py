"""Composable LabelLens pipeline orchestration."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import INGREDIENTS_PATH, VECTORSTORE_DIR
from src.llm.deterministic import generate_deterministic_analysis
from src.llm.llm_client import LLMClient
from src.llm.prompt_builder import build_analysis_prompt
from src.llm.validator import validate_llm_analysis
from src.ocr.main import load_image
from src.ocr.ocr import find_best_result, run_ocr_all_psms
from src.ocr.preprocessing import get_all_strategies
from src.pipeline.ingredient_pipeline import run_pipeline as run_ingredient_pipeline
from src.rag.retriever import IngredientRetriever
from src.safety.assessment import assess_ingredient_tree


def run_pre_llm_pipeline(image_path: str | Path) -> dict[str, Any]:
    """Run OCR through RAG and build LLM inputs without external API calls."""
    image = load_image(str(image_path))
    all_ocr_results = []
    failures = []
    for name, preprocess_fn in get_all_strategies():
        try:
            all_ocr_results.extend(run_ocr_all_psms(preprocess_fn(image), name))
        except Exception as exc:
            failures.append({"strategy": name, "error": str(exc)})
    from src.audit_logger import STAGE_OCR_EXTRACTION_COMPLETED, STAGE_RAG_COMPLETED, audit_log

    best_ocr = find_best_result(all_ocr_results)
    if not best_ocr:
        raise ValueError("OCR did not extract usable text from this image.")
    phase2 = run_ingredient_pipeline(best_ocr["text"], dictionary_path=str(INGREDIENTS_PATH))

    # Small label photos can contain readable ingredient text that Tesseract
    # misses at the original resolution. Retry only when the first pass found
    # no ingredients, keeping the normal path and its cost unchanged.
    if not phase2.get("ingredients") and isinstance(image, np.ndarray):
        for name, preprocess_fn in get_all_strategies():
            try:
                processed = preprocess_fn(image)
                enlarged_image = cv2.resize(
                    processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                )
                all_ocr_results.extend(
                    run_ocr_all_psms(enlarged_image, f"upscale_2x_{name}")
                )
            except Exception as exc:
                failures.append({"strategy": f"upscale_2x_{name}", "error": str(exc)})
        retry_ocr = find_best_result(all_ocr_results)
        if retry_ocr and retry_ocr["text"] != best_ocr["text"]:
            retry_phase2 = run_ingredient_pipeline(
                retry_ocr["text"], dictionary_path=str(INGREDIENTS_PATH)
            )
            if retry_phase2.get("ingredients"):
                best_ocr = retry_ocr
                phase2 = retry_phase2

    audit_log(
        STAGE_OCR_EXTRACTION_COMPLETED,
        f"OCR strategy={best_ocr.get('strategy')}, confidence={best_ocr.get('confidence', 0.0):.1f}%, extracted {len(phase2.get('ingredients', []))} ingredients"
    )
    phase3 = IngredientRetriever(store_dir=str(VECTORSTORE_DIR)).retrieve_for_phase2(phase2)
    phase3 = assess_ingredient_tree(phase3)
    audit_log(
        STAGE_RAG_COMPLETED,
        f"Retrieved RAG evidence for {len(phase3)} top-level ingredients"
    )
    return {
        "ocr": best_ocr,
        "ocr_strategy_failures": failures,
        "phase2": phase2,
        "phase3": phase3,
        "llm_inputs": _build_llm_inputs(phase3),
    }


def _build_llm_inputs(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs = []
    for node in nodes:
        inputs.append({
            "ingredient": node.get("ingredient", "Unknown"),
            "phase2_status": node.get("phase2_status", "unknown"),
            "evidence": node.get("evidence", []),
            "prompt": build_analysis_prompt(node),
        })
        inputs.extend(_build_llm_inputs(node.get("sub_ingredients", [])))
    return inputs


def _flatten_phase3_nodes(nodes: list[dict[str, Any]], prefix: str = "") -> list[dict[str, Any]]:
    flat = []
    for idx, node in enumerate(nodes):
        item_id = f"ing-{prefix}{idx}"
        flat.append({
            "id": item_id,
            "node": node,
            "ingredient": node.get("ingredient", "Unknown"),
            "raw_name": node.get("raw_name", node.get("ingredient", "")),
            "code": node.get("code"),
            "aliases_detected": node.get("aliases_detected", []),
            "phase2_status": node.get("phase2_status", "unknown"),
            "phase2_similarity": node.get("phase2_similarity", "N/A"),
            "evidence": node.get("evidence", []),
        })
        if node.get("sub_ingredients"):
            flat.extend(_flatten_phase3_nodes(node["sub_ingredients"], prefix=f"{prefix}{idx}-sub-"))
    return flat


def run_batch_end_to_end_pipeline(image_path: str | Path, api_key: str | None = None, correlation_id: str | None = None):
    """Run full pipeline with exactly ONE Groq batch request for all ingredients."""
    from src.llm.prompt_builder import build_batch_analysis_prompt
    from src.llm.validator import validate_batch_llm_analysis
    from src.audit_logger import audit_log, set_correlation_id

    if correlation_id:
        set_correlation_id(correlation_id)

    pre_llm = run_pre_llm_pipeline(image_path)
    flat_items = _flatten_phase3_nodes(pre_llm["phase3"])

    if not flat_items:
        summary = dict(pre_llm["phase2"].get("summary", {}))
        if "allergens" in pre_llm["phase2"]:
            summary["allergens"] = pre_llm["phase2"]["allergens"]
        return [], summary

    client = LLMClient(api_key=api_key)
    product_name = "Scanned Product"
    batch_prompt = build_batch_analysis_prompt(flat_items, product_name=product_name)

    provider_failed = False
    try:
        raw_batch_output = client.generate_batch_analysis(batch_prompt)
    except Exception as exc:
        provider_failed = True
        audit_log("LLM_BATCH_CALL_FAILED", f"client.generate_batch_analysis failed with {type(exc).__name__}: {exc}")
        raw_batch_output = {"error": str(exc)}

    if provider_failed:
        # When Groq is unavailable, retain only explanations that can be
        # assembled directly from evidence already retrieved for that item.
        # This is the same deterministic formatter used by the non-batch path.
        validated_map = {}
        evidence_fallback_count = 0
        for item in flat_items:
            item_id = item["id"]
            status = item.get("phase2_status", "unknown")
            evidence = item.get("evidence", [])
            analysis = None
            if status in {"matched", "ambiguous"}:
                analysis = generate_deterministic_analysis(
                    item.get("ingredient", "Unknown"),
                    item.get("raw_name", ""),
                    evidence,
                )
            if analysis is not None:
                if status != "matched":
                    analysis["identification"] = {
                        "status": status,
                        "confidence": None,
                        "explanation": "Identity remains ambiguous; details are limited to retrieved evidence for this category.",
                    }
                analysis["uncertainties"] = [
                    "Live analysis unavailable; details below are drawn from retrieved evidence."
                ]
                validated_map[item_id] = validate_llm_analysis(
                    status, evidence, analysis
                )
                evidence_fallback_count += 1
            else:
                item_map, _ = validate_batch_llm_analysis([item], raw_batch_output)
                validated_map.update(item_map)
        audit_log(
            "EVIDENCE_BACKED_FALLBACK_SELECTED",
            f"Provider unavailable; used retrieved evidence for {evidence_fallback_count} of {len(flat_items)} items",
        )
    else:
        validated_map, _ = validate_batch_llm_analysis(flat_items, raw_batch_output)

    def rebuild_tree(nodes: list[dict[str, Any]], prefix: str = "") -> list[dict[str, Any]]:
        rebuilt = []
        for idx, node in enumerate(nodes):
            item_id = f"ing-{prefix}{idx}"
            status = node.get("phase2_status", "unknown")
            evidence = node.get("evidence", [])
            analysis = validated_map.get(item_id)
            if analysis is None:
                analysis = validate_llm_analysis(status, evidence, {"error": "Missing from batch response"})

            sub_prefix = f"{prefix}{idx}-sub-"
            rebuilt_node = {
                "ingredient": node.get("ingredient", "Unknown"),
                "raw_name": node.get("raw_name", node.get("ingredient", "")),
                "canonical_name": node.get("canonical_name", node.get("ingredient", "Unknown")),
                "code": node.get("code"),
                "aliases_detected": node.get("aliases_detected", []),
                "phase2_status": status,
                "safety_assessment": node.get("safety_assessment"),
                "llm_analysis": analysis,
                "authoritative_evidence": evidence,
                "tool_calls": [],
                "used_llm": not provider_failed,
                "sub_ingredients": rebuild_tree(node.get("sub_ingredients", []), prefix=sub_prefix),
            }
            rebuilt.append(rebuilt_node)
        return rebuilt

    results = rebuild_tree(pre_llm["phase3"])
    summary = dict(pre_llm["phase2"].get("summary", {}))
    if "allergens" in pre_llm["phase2"]:
        summary["allergens"] = pre_llm["phase2"]["allergens"]
    return results, summary


def run_end_to_end_pipeline(image_path: str | Path, api_key: str | None = None, batch: bool = False, correlation_id: str | None = None):
    """Run full pipeline. If batch=True, executes in exactly ONE Groq batch request."""
    if batch:
        return run_batch_end_to_end_pipeline(image_path, api_key=api_key, correlation_id=correlation_id)

    pre_llm = run_pre_llm_pipeline(image_path)
    if not pre_llm["phase3"]:
        summary = dict(pre_llm["phase2"].get("summary", {}))
        if "allergens" in pre_llm["phase2"]:
            summary["allergens"] = pre_llm["phase2"]["allergens"]
        return [], summary

    client = LLMClient(api_key=api_key)

    def analyze(node: dict[str, Any]) -> dict[str, Any]:
        evidence = node.get("evidence", [])
        status = node.get("phase2_status", "unknown")
        analysis = None
        used_llm = False
        if status == "matched" and evidence:
            analysis = generate_deterministic_analysis(node.get("ingredient", "Unknown"), node.get("raw_name", ""), evidence)
        if analysis is None:
            used_llm = True
            try:
                analysis = client.generate_analysis(build_analysis_prompt(node))
            except Exception as exc:
                analysis = {"error": str(exc)}
        return {
            "ingredient": node.get("ingredient", "Unknown"),
            "raw_name": node.get("raw_name", ""),
            "canonical_name": node.get("canonical_name", node.get("ingredient", "Unknown")),
            "code": node.get("code"),
            "aliases_detected": node.get("aliases_detected", []),
            "phase2_status": status,
            "safety_assessment": node.get("safety_assessment"),
            "llm_analysis": validate_llm_analysis(status, evidence, analysis),
            "authoritative_evidence": evidence,
            "tool_calls": [],
            "used_llm": used_llm,
            "sub_ingredients": [analyze(child) for child in node.get("sub_ingredients", [])],
        }

    summary = dict(pre_llm["phase2"].get("summary", {}))
    if "allergens" in pre_llm["phase2"]:
        summary["allergens"] = pre_llm["phase2"]["allergens"]
    return [analyze(node) for node in pre_llm["phase3"]], summary

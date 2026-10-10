import json
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import IngredientRetriever

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_phase2_ingredients(ingredients_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_list = []
    for item in ingredients_list:
        flat_list.append(item)
        if item.get("sub_ingredients"):
            flat_list.extend(flatten_phase2_ingredients(item.get("sub_ingredients")))
    return flat_list

def verify_provenance(chunk: Dict[str, Any]) -> bool:
    source = chunk.get("source", {})
    return bool(
        source.get("title") and
        source.get("organization") and
        source.get("url") and
        chunk.get("ingredient")
    )

def chunk_types(results_slice):
    return {
        "identity": any(r.get("section") == "identity" for r in results_slice),
        "function": any(r.get("section") == "function" for r in results_slice),
        "regulatory": any(r.get("section") == "regulatory" for r in results_slice),
        "assessment": any(r.get("section") == "assessment" for r in results_slice)
    }

def evaluate_rag(
    ground_truth_path=None,
    phase2_path=None,
    store_dir=None,
    output_path=None,
):
    print("Running Real RAG Evaluation (Phase 5C - Corrections)...")

    gt_path = Path(ground_truth_path) if ground_truth_path else PROJECT_ROOT / "data" / "ground_truth" / "ground_truth.json"
    p2_path = Path(phase2_path) if phase2_path else PROJECT_ROOT / "artifacts" / "reference_runs" / "extraction" / "extraction_results.json"

    gt_data = load_json(gt_path)
    p2_data = load_json(p2_path)

    label1_gt = next(s for s in gt_data["samples"] if s["id"] == "label1")
    gt_ingredients = label1_gt.get("ingredients", [])

    gt_map = {gt.get("name", "").strip().lower(): gt.get("expected_canonical") for gt in gt_ingredients}

    p2_flat = flatten_phase2_ingredients(p2_data.get("ingredients", []))

    vectorstore_dir = str(store_dir or PROJECT_ROOT / "vectorstore")
    retriever = IngredientRetriever(store_dir=vectorstore_dir)

    # KB Coverage Calculation (Independent of Phase 2)
    gt_known_ingredients = 0
    represented_in_kb = 0
    absent_from_kb = 0

    for gt in gt_ingredients:
        expected_canonical = gt.get("expected_canonical")
        if expected_canonical:
            gt_known_ingredients += 1
            if expected_canonical.lower() in retriever._alias_map:
                represented_in_kb += 1
            else:
                absent_from_kb += 1

    # Retrieval Metrics setup
    exact_lookup_eligible = 0
    exact_lookup_success = 0

    canonical_semantic_eligible = 0
    canonical_recall_1 = 0
    canonical_recall_3 = 0
    canonical_recall_5 = 0

    noisy_semantic_eligible = 0
    noisy_recall_1 = 0
    noisy_recall_3 = 0
    noisy_recall_5 = 0

    ambiguous_unknown_excluded = 0
    provenance_retained = 0
    provenance_total = 0

    evaluation_queries = []

    for item in p2_flat:
        p2_original = item.get("raw", item.get("name", ""))
        p2_canonical = item.get("normalized", p2_original)
        p2_status = item.get("status", "unknown")

        name_key = item.get("name", "").strip().lower()
        expected_canonical = gt_map.get(name_key)

        if name_key not in gt_map:
            continue

        if p2_status in ["ambiguous", "unknown"]:
            ambiguous_unknown_excluded += 1
            evaluation_queries.append({
                "query": p2_original,
                "phase2_original_name": p2_original,
                "phase2_canonical_name": p2_canonical,
                "phase2_status": p2_status,
                "expected_canonical": expected_canonical,
                "status": "not_evaluable",
                "reason": "Phase 2 output is ambiguous/unknown"
            })
            continue

        kb_expected = expected_canonical and expected_canonical.lower() in retriever._alias_map

        query_entry = {
            "phase2_original_name": p2_original,
            "phase2_canonical_name": p2_canonical,
            "phase2_status": p2_status,
            "expected_canonical": expected_canonical,
            "kb_expected": kb_expected,
            "status": "evaluated"
        }

        # Exact Lookup
        exact_results = retriever.retrieve(p2_canonical, top_k=1, bypass_exact=False)
        if exact_results and exact_results[0]["score"] == 1.0:
            query_entry["exact_lookup"] = {
                "success": True,
                "returned_canonical": exact_results[0].get("ingredient")
            }
            if kb_expected and exact_results[0].get("ingredient") == expected_canonical:
                exact_lookup_success += 1
        else:
            query_entry["exact_lookup"] = {"success": False}

        if kb_expected:
            exact_lookup_eligible += 1
            canonical_semantic_eligible += 1

        # Canonical Semantic Evaluation
        canonical_results = retriever.retrieve(p2_canonical, top_k=5, bypass_exact=True)
        for res in canonical_results:
            provenance_total += 1
            if verify_provenance(res): provenance_retained += 1

        if kb_expected:
            r1 = any(r.get("ingredient") == expected_canonical for r in canonical_results[:1])
            r3 = any(r.get("ingredient") == expected_canonical for r in canonical_results[:3])
            r5 = any(r.get("ingredient") == expected_canonical for r in canonical_results[:5])

            if r1: canonical_recall_1 += 1
            if r3: canonical_recall_3 += 1
            if r5: canonical_recall_5 += 1

            query_entry["canonical_semantic"] = {
                "query": p2_canonical,
                "recall_1": r1,
                "recall_3": r3,
                "recall_5": r5,
                "top_1": {"chunk_types": chunk_types(canonical_results[:1])},
                "top_3": {"chunk_types": chunk_types(canonical_results[:3])},
                "top_5": {"chunk_types": chunk_types(canonical_results[:5])}
            }

        # Noisy Semantic Evaluation
        is_noisy = p2_original.lower().strip() != expected_canonical.lower().strip() if expected_canonical else False
        if is_noisy and kb_expected:
            noisy_semantic_eligible += 1
            noisy_results = retriever.retrieve(p2_original, top_k=5, bypass_exact=True)
            for res in noisy_results:
                provenance_total += 1
                if verify_provenance(res): provenance_retained += 1

            nr1 = any(r.get("ingredient") == expected_canonical for r in noisy_results[:1])
            nr3 = any(r.get("ingredient") == expected_canonical for r in noisy_results[:3])
            nr5 = any(r.get("ingredient") == expected_canonical for r in noisy_results[:5])

            if nr1: noisy_recall_1 += 1
            if nr3: noisy_recall_3 += 1
            if nr5: noisy_recall_5 += 1

            query_entry["noisy_semantic"] = {
                "query": p2_original,
                "recall_1": nr1,
                "recall_3": nr3,
                "recall_5": nr5,
                "top_1": {"chunk_types": chunk_types(noisy_results[:1])},
                "top_3": {"chunk_types": chunk_types(noisy_results[:3])},
                "top_5": {"chunk_types": chunk_types(noisy_results[:5])}
            }

        evaluation_queries.append(query_entry)

    final_output = {
        "sample_id": "label1",
        "evaluation_type": "real_empirical",
        "retrieval": {
            "kb_coverage": {
                "ground_truth_known_ingredients": gt_known_ingredients,
                "represented_in_kb": represented_in_kb,
                "absent_from_kb": absent_from_kb,
                "coverage_rate": represented_in_kb / gt_known_ingredients if gt_known_ingredients > 0 else 0.0
            },
            "exact_lookup": {
                "eligible_queries": exact_lookup_eligible,
                "success_count": exact_lookup_success,
                "success_rate": exact_lookup_success / exact_lookup_eligible if exact_lookup_eligible > 0 else 0.0
            },
            "semantic_faiss": {
                "canonical_query_recall_at_1": canonical_recall_1 / canonical_semantic_eligible if canonical_semantic_eligible > 0 else 0.0,
                "canonical_query_recall_at_3": canonical_recall_3 / canonical_semantic_eligible if canonical_semantic_eligible > 0 else 0.0,
                "canonical_query_recall_at_5": canonical_recall_5 / canonical_semantic_eligible if canonical_semantic_eligible > 0 else 0.0,

                "noisy_query_recall_at_1": noisy_recall_1 / noisy_semantic_eligible if noisy_semantic_eligible > 0 else 0.0,
                "noisy_query_recall_at_3": noisy_recall_3 / noisy_semantic_eligible if noisy_semantic_eligible > 0 else 0.0,
                "noisy_query_recall_at_5": noisy_recall_5 / noisy_semantic_eligible if noisy_semantic_eligible > 0 else 0.0,

                "canonical_eligible_queries": canonical_semantic_eligible,
                "noisy_eligible_queries": noisy_semantic_eligible
            },
            "provenance": {
                "total_chunks_checked": provenance_total,
                "chunks_retaining_provenance": provenance_retained,
                "preservation_rate": provenance_retained / provenance_total if provenance_total > 0 else 1.0
            },
            "ambiguous_unknown_exclusions": ambiguous_unknown_excluded
        },
        "queries": evaluation_queries,
        "limitations": [
            "FAISS similarity threshold strictly enforces no_relevant_evidence.",
            "Exact lookup and semantic retrieval are evaluated independently to isolate RAG performance.",
            "These metrics evaluate semantic retrieval when the retriever receives clean canonical ingredient names vs noisy OCR queries."
        ]
    }

    out_path = Path(output_path) if output_path else PROJECT_ROOT / "evaluation" / "results" / "label1_rag_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\n==================================================")
    print("These results were calculated from the existing Phase 3 retrieval system")
    print("using actual Phase 2 outputs and the existing knowledge base.")
    print("Phase 5C evaluates the existing Phase 3 retrieval system without modifying")
    print("its knowledge base or retrieval configuration.")
    print("==================================================")
    print(f"KB Coverage Rate (Independent of Phase 2): {final_output['retrieval']['kb_coverage']['coverage_rate']:.2f}")
    print(f"Absent from KB: {absent_from_kb}")
    print(f"Exact Lookup Success Rate: {final_output['retrieval']['exact_lookup']['success_rate']:.2f}")
    print(f"Canonical-Query Recall@1: {final_output['retrieval']['semantic_faiss']['canonical_query_recall_at_1']:.2f}")
    print(f"Noisy-Query Recall@1: {final_output['retrieval']['semantic_faiss']['noisy_query_recall_at_1']:.2f}")
    print(f"Ambiguous/Unknown inputs excluded: {ambiguous_unknown_excluded}")
    print(f"Results saved to {out_path}")
    return final_output

if __name__ == "__main__":
    evaluate_rag()

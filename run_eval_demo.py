"""Quick evaluation demo — run from terminal."""
import json
from src.llm.evaluator import evaluate_answer

# Simulated RAG scenario
question = "What is Citric Acid (E330) and is it safe?"
context_docs = [
    "Citric Acid (E330) is an acidity regulator widely used in food and beverages.",
    "E330 is approved by FDA and EFSA. It is generally recognized as safe (GRAS).",
]
llm_answer = "Citric Acid (E330) is an acidity regulator used in food products. It is approved by FDA and EFSA and is generally recognized as safe."

# Run evaluation
metrics = evaluate_answer(question, llm_answer, context_docs)

print("=" * 50)
print("LabelLens RAG Evaluation Demo")
print("=" * 50)
print(f"\nQuestion:  {question}")
print(f"\nContext:   {len(context_docs)} retrieved documents")
for i, doc in enumerate(context_docs):
    print(f"  [{i+1}] {doc}")
print(f"\nLLM Answer: {llm_answer}")
print(f"\n{'=' * 50}")
print("EVALUATION METRICS")
print("=" * 50)
for metric, score in metrics.items():
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"  {metric:25s} {bar} {score:.3f}")
print(f"\nVerdict: {'✅ Answer is well-grounded' if metrics['faithfulness'] > 0.5 else '⚠️ Answer may contain hallucination'}")
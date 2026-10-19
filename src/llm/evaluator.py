"""Standalone RAG evaluation metrics — no ragas dependency needed."""

from __future__ import annotations
import re
from typing import List, Dict


def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitter."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences that appear (substring) in any context doc."""
    if not contexts:
        return 0.0
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0
    joined_ctx = " ".join(contexts).lower()
    supported = sum(1 for s in sentences if s.lower() in joined_ctx)
    return supported / len(sentences)


def relevancy_score(question: str, answer: str) -> float:
    """Simple keyword-overlap relevancy (Jaccard similarity)."""
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if not q_words or not a_words:
        return 0.0
    return len(q_words & a_words) / len(q_words | a_words)


def context_precision_score(answer: str, contexts: list[str]) -> float:
    """Fraction of context docs that contributed at least one word to the answer."""
    if not contexts:
        return 0.0
    a_words = set(answer.lower().split())
    used = sum(1 for doc in contexts if set(doc.lower().split()) & a_words)
    return used / len(contexts)


def evaluate_answer(
    question: str,
    answer: str,
    contexts: List[str] | None = None,
) -> Dict[str, float]:
    """Run lightweight RAG evaluation metrics."""
    if contexts is None:
        contexts = []
    return {
        "faithfulness": round(faithfulness_score(answer, contexts), 3),
        "answer_relevancy": round(relevancy_score(question, answer), 3),
        "context_precision": round(context_precision_score(answer, contexts), 3),
    }
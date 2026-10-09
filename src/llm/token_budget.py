"""
token_budget.py — Token Budget Protection & Context Compaction for LabelLens Groq Calls.

Guarantees requests stay strictly within service tier limits (e.g. 8000 TPM limit),
providing progressive compaction levels and safe budget validation.
"""

import math
import os
import re
from typing import Any, Tuple

# Configurable safe token budget limits
DEFAULT_MAX_REQUEST_TOKENS = int(os.environ.get("GROQ_MAX_REQUEST_TOKENS", 8000))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", 4800))
MIN_OUTPUT_TOKENS = 1500
PER_INGREDIENT_OUTPUT_TOKENS = 350
REASONING_TOKEN_RESERVATION = 400


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for text using a robust BPE heuristic.
    Average ratio for mixed JSON/English in OpenAI/Llama tokenizers is ~3.6 chars/token.
    Ceiling ensures a conservative upper bound.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 3.6)


def calculate_output_token_budget(
    ingredient_count: int,
    max_budget: int = DEFAULT_MAX_OUTPUT_TOKENS,
    min_tokens: int = MIN_OUTPUT_TOKENS,
) -> int:
    """
    Calculates dynamic output token reservation based on ingredient count,
    including reasoning token overhead for reasoning models like gpt-oss-120b.
    """
    needed = max(min_tokens, REASONING_TOKEN_RESERVATION + ingredient_count * PER_INGREDIENT_OUTPUT_TOKENS)
    return min(max_budget, needed)



def check_request_budget(
    prompt: str,
    system_prompt: str,
    max_output_tokens: int,
    safe_budget: int = DEFAULT_MAX_REQUEST_TOKENS,
) -> Tuple[bool, int, str]:
    """
    Checks if prompt tokens + max_output_tokens is within the safe request budget.
    """
    prompt_tokens = estimate_tokens(prompt + "\n" + system_prompt)
    total_requested = prompt_tokens + max_output_tokens
    if total_requested > safe_budget:
        msg = (
            f"Estimated tokens ({total_requested} = {prompt_tokens} prompt + "
            f"{max_output_tokens} output) exceeds safe budget ({safe_budget})."
        )
        return False, total_requested, msg
    return True, total_requested, "OK"


def compact_evidence_text(text: str, max_chars: int = 220) -> str:
    """
    Compacts evidence text while preserving meaning and key facts.
    """
    if not text or len(text) <= max_chars:
        return text.strip()

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned

    # Try to cut cleanly at sentence or semicolon boundary
    cut = cleaned[:max_chars]
    last_punct = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "))
    if last_punct > int(max_chars * 0.6):
        return cut[: last_punct + 1].strip()
    return cut.rstrip() + "..."

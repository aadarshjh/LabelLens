"""Groq client isolated from OCR, identity, and retrieval stages."""

import json
import os
from typing import Any


class LLMUnavailableError(RuntimeError):
    """Raised when live Groq analysis cannot be attempted."""


# Strict JSON Schema for single ingredient analysis
SINGLE_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "SingleAnalysisResult",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ingredient": {"type": "string"},
                "identification": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "confidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "explanation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": ["status", "confidence", "explanation"],
                    "additionalProperties": False,
                },
                "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "function": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "summary": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "regulatory_information": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "ingredient",
                "identification",
                "category",
                "function",
                "summary",
                "regulatory_information",
                "uncertainties",
                "sources",
            ],
            "additionalProperties": False,
        },
    },
}

# Strict JSON Schema for batch analysis (all ingredients in one request)
BATCH_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "BatchAnalysisResult",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ingredient_id": {"type": "string"},
                            "ingredient": {"type": "string"},
                            "identification": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "confidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "explanation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                },
                                "required": ["status", "confidence", "explanation"],
                                "additionalProperties": False,
                            },
                            "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "function": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "summary": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "regulatory_information": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "uncertainties": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "ingredient_id",
                            "ingredient",
                            "identification",
                            "category",
                            "function",
                            "summary",
                            "regulatory_information",
                            "uncertainties",
                            "sources",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ingredients"],
            "additionalProperties": False,
        },
    },
}


class LLMClient:
    """Groq client boundary with no mock-success fallback."""

    MAX_TOOL_CALLS = 0

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.tool_calls_history: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return bool(self.api_key)

    def is_mock(self) -> bool:
        """Compatibility helper; LabelLens never fabricates mock analyses."""
        return False

    def generate_analysis(self, prompt: str) -> dict[str, Any]:
        """Generates analysis for a single ingredient or arbitrary prompt."""
        from src.audit_logger import (
            STAGE_LLM_CLIENT_INVOKED,
            STAGE_HTTP_REQUEST_SENT,
            STAGE_PROVIDER_RESPONSE_RECEIVED,
            STAGE_JSON_PARSING_COMPLETED,
            audit_log,
        )

        audit_log(STAGE_LLM_CLIENT_INVOKED, f"Single analysis for model={self.model_name}, api_key_configured={bool(self.api_key)}")
        if not self.api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not configured.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMUnavailableError(
                "groq is not installed. Install requirements.txt before enabling Groq."
            ) from exc
        try:
            client = Groq(api_key=self.api_key)
            audit_log(STAGE_HTTP_REQUEST_SENT, f"Target=Groq single completion, model={self.model_name}")
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an analytical assistant for LabelLens. "
                            "Interpret retrieved factual evidence about food ingredients "
                            "and explain it clearly to a consumer. "
                            "Use ONLY the supplied evidence. Do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=SINGLE_ANALYSIS_SCHEMA,
                temperature=0.0,
                max_tokens=2048,
            )
            raw = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            usage = getattr(completion, "usage", None)
            u_info = f"tokens=({usage.total_tokens})" if usage else ""
            audit_log(STAGE_PROVIDER_RESPONSE_RECEIVED, f"Status=200 finish_reason={finish_reason} {u_info}")
            if finish_reason == "length":
                audit_log("TOKEN_LIMIT_TRUNCATION", "Single analysis truncated due to token limit")
                raise LLMUnavailableError("Groq output truncated by token limit.")
        except LLMUnavailableError:
            raise
        except Exception as exc:
            audit_log(STAGE_PROVIDER_RESPONSE_RECEIVED, f"Provider error: {type(exc).__name__}: {exc}")
            raise LLMUnavailableError(f"Groq request failed: {exc}") from exc
        if not raw:
            audit_log(STAGE_JSON_PARSING_COMPLETED, "Failed: Groq returned empty content")
            raise LLMUnavailableError("Groq returned an empty response.")
        try:
            parsed = json.loads(raw)
            audit_log(STAGE_JSON_PARSING_COMPLETED, "Parsed single ingredient JSON successfully")
            return parsed
        except json.JSONDecodeError as exc:
            audit_log(STAGE_JSON_PARSING_COMPLETED, f"Failed: non-JSON output ({exc})")
            raise LLMUnavailableError("Groq returned non-JSON structured output.") from exc

    def generate_batch_analysis(
        self,
        prompt: str,
        max_output_tokens: int | None = None,
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generates batch analysis for all ingredients in ONE Groq request within token budget."""
        from src.audit_logger import (
            STAGE_LLM_CLIENT_INVOKED,
            STAGE_HTTP_REQUEST_SENT,
            STAGE_PROVIDER_RESPONSE_RECEIVED,
            STAGE_JSON_PARSING_COMPLETED,
            audit_log,
        )

        ingredient_count = max(1, prompt.count("=== INGREDIENT #"))
        audit_log(
            STAGE_LLM_CLIENT_INVOKED,
            f"Batch analysis invoked for {ingredient_count} ingredients (model={self.model_name}, api_key_configured={bool(self.api_key)})"
        )

        if not self.api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not configured.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMUnavailableError(
                "groq is not installed. Install requirements.txt before enabling Groq."
            ) from exc

        from src.llm.token_budget import (
            DEFAULT_MAX_OUTPUT_TOKENS,
            calculate_output_token_budget,
            check_request_budget,
        )

        system_content = (
            "You are an analytical assistant for LabelLens. "
            "Explain the supplied food ingredients concisely (1 brief sentence per field) "
            "using ONLY the supplied Phase 2 identities and Phase 3 evidence. "
            "Do not add, remove, rename, or normalize ingredients. "
            "Do not make unsupported medical or safety claims. "
            "If evidence is insufficient, state that it is not available. "
            "Preserve uncertainty."
        )

        if max_output_tokens is None:
            max_output_tokens = calculate_output_token_budget(ingredient_count, items=items)

        is_safe, est_tokens, reason = check_request_budget(prompt, system_content, max_output_tokens)
        if not is_safe:
            audit_log("TOKEN_BUDGET_EXCEEDED", f"Request budget check failed: {reason}")
            raise LLMUnavailableError(f"Groq request exceeds safe token budget: {reason}")

        try:
            client = Groq(api_key=self.api_key)
            extra_kwargs = {}
            if "gpt-oss" in self.model_name.lower():
                extra_kwargs["reasoning_effort"] = "low"

            audit_log(
                STAGE_HTTP_REQUEST_SENT,
                f"Target=Groq batch completion, model={self.model_name}, max_tokens={max_output_tokens}, est_input={est_tokens}"
            )

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                response_format=BATCH_ANALYSIS_SCHEMA,
                temperature=0.0,
                max_tokens=max_output_tokens,
                **extra_kwargs,
            )
            raw = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            usage = getattr(completion, "usage", None)
            usage_str = f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}" if usage else "unknown"
            audit_log(
                STAGE_PROVIDER_RESPONSE_RECEIVED,
                f"HTTP response 200 OK received: finish_reason={finish_reason}, tokens=({usage_str})"
            )

            if finish_reason == "length":
                audit_log(
                    "TOKEN_LIMIT_TRUNCATION",
                    f"Output truncated due to token limit (finish_reason=length, max_tokens={max_output_tokens})"
                )
                expanded_tokens = min(DEFAULT_MAX_OUTPUT_TOKENS, max_output_tokens + 800)
                is_safe_retry, est_retry_tokens, retry_reason = check_request_budget(
                    prompt, system_content, expanded_tokens
                )
                if is_safe_retry and expanded_tokens > max_output_tokens:
                    audit_log(
                        "TOKEN_BUDGET_RETRY",
                        f"Retrying with expanded output budget {max_output_tokens} -> {expanded_tokens}"
                    )
                    audit_log(
                        STAGE_HTTP_REQUEST_SENT,
                        f"Target=Groq batch completion retry, model={self.model_name}, max_tokens={expanded_tokens}, est_input={est_retry_tokens}"
                    )
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt},
                        ],
                        response_format=BATCH_ANALYSIS_SCHEMA,
                        temperature=0.0,
                        max_tokens=expanded_tokens,
                        **extra_kwargs,
                    )
                    raw = completion.choices[0].message.content
                    finish_reason = completion.choices[0].finish_reason
                    retry_usage = getattr(completion, "usage", None)
                    retry_usage_str = f"prompt={retry_usage.prompt_tokens}, completion={retry_usage.completion_tokens}, total={retry_usage.total_tokens}" if retry_usage else "unknown"
                    audit_log(
                        STAGE_PROVIDER_RESPONSE_RECEIVED,
                        f"HTTP retry response 200 OK received: finish_reason={finish_reason}, tokens=({retry_usage_str})"
                    )
                    if finish_reason == "length":
                        audit_log(
                            "TOKEN_LIMIT_TRUNCATION",
                            "Retry truncated; falling back to evidence-backed analysis"
                        )
                        raise LLMUnavailableError("Groq output truncated by token limit.")
                else:
                    audit_log(
                        "TOKEN_LIMIT_TRUNCATION",
                        f"Cannot retry safely ({retry_reason}); falling back to evidence-backed analysis"
                    )
                    raise LLMUnavailableError("Groq output truncated by token limit.")

        except LLMUnavailableError:
            raise
        except Exception as exc:
            audit_log(STAGE_PROVIDER_RESPONSE_RECEIVED, f"Provider request failed: {type(exc).__name__}: {exc}")
            raise LLMUnavailableError(f"Groq request failed: {exc}") from exc

        if not raw:
            audit_log(STAGE_JSON_PARSING_COMPLETED, "Failed: Groq returned empty content")
            raise LLMUnavailableError("Groq returned an empty response.")
        try:
            parsed = json.loads(raw)
            ret_count = len(parsed.get("ingredients", []))
            audit_log(
                STAGE_JSON_PARSING_COMPLETED,
                f"Parsed JSON successfully ({ret_count} ingredients returned)"
            )
            return parsed
        except json.JSONDecodeError as exc:
            audit_log(STAGE_JSON_PARSING_COMPLETED, f"Failed: non-JSON output ({exc})")
            raise LLMUnavailableError("Groq returned non-JSON structured output.") from exc

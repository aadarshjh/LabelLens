"""
audit_logger.py — Boundary audit logger with correlation ID, timestamps, and sanitization.

Logs key lifecycle stages without ever exposing credentials, API keys, or auth headers.
"""

import contextvars
import datetime
import os
import re
import sys
from typing import Any

_correlation_id_ctx = contextvars.ContextVar("correlation_id", default="none")
_recent_logs: list[dict[str, Any]] = []

# Exact boundary stage names required by LabelLens audit
STAGE_REQUEST_RECEIVED = "Request received by /analyze."
STAGE_OCR_EXTRACTION_COMPLETED = "OCR/extraction completed."
STAGE_RAG_COMPLETED = "RAG completed."
STAGE_LLM_CLIENT_INVOKED = "LLM client invoked."
STAGE_HTTP_REQUEST_SENT = "HTTP request sent."
STAGE_PROVIDER_RESPONSE_RECEIVED = "Provider response received."
STAGE_JSON_PARSING_COMPLETED = "JSON parsing completed."
STAGE_VALIDATION_COMPLETED = "Validation completed."
STAGE_SAFETY_FALLBACK_SELECTED = "Safety fallback selected."

# Sanitization regexes to ensure no secrets ever enter logs
SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9_-]+"),
    re.compile(r"AIza[A-Za-z0-9_-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9_.-]+", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token)[\s:=]+['\"]?([A-Za-z0-9_.-]+)['\"]?", re.IGNORECASE),
]


def sanitize_message(msg: str) -> str:
    """Mask any credential, API key, or authorization header."""
    text = str(msg)
    for pat in SENSITIVE_PATTERNS:
        text = pat.sub("[REDACTED_CREDENTIAL]", text)
    return text


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for current context."""
    _correlation_id_ctx.set(corr_id)


def get_correlation_id() -> str:
    """Get correlation ID from current context."""
    return _correlation_id_ctx.get()


def audit_log(boundary: str, details: str = "", correlation_id: str | None = None) -> dict[str, Any]:
    """
    Record and print a sanitized audit event at a pipeline boundary.
    """
    cid = correlation_id or get_correlation_id()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clean_details = sanitize_message(details)

    entry = {
        "timestamp": now_iso,
        "correlation_id": cid,
        "boundary": boundary,
        "details": clean_details,
    }

    _recent_logs.append(entry)
    # Keep last 200 logs
    if len(_recent_logs) > 200:
        _recent_logs.pop(0)

    log_line = f"[{now_iso}] [req={cid}] [{boundary}] {clean_details}".strip()
    print(log_line, file=sys.stderr, flush=True)
    return entry


def get_recent_audit_logs() -> list[dict[str, Any]]:
    """Return in-memory audit logs for inspection / testing."""
    return list(_recent_logs)


def clear_audit_logs() -> None:
    """Clear in-memory audit logs."""
    _recent_logs.clear()

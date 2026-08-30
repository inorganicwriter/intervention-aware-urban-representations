"""Stable schema names and compatibility checks for causal artifacts."""

from __future__ import annotations

from typing import Any

CAUSAL_RESPONSE_LABELS_SCHEMA = "causal_response_labels"
KNOWN_CAUSAL_RESPONSE_LABEL_SCHEMAS = frozenset(
    {
        CAUSAL_RESPONSE_LABELS_SCHEMA,
        "causal_response_labels_v1",
    }
)


def accepts_legacy_version(value: Any, canonical: str) -> bool:
    """Accept a stable name and its explicitly audited historical schemas.

    Compatibility is fail-closed. Unknown future numbers and arbitrary suffixes
    are rejected until they are reviewed and added to a concrete allowlist.
    """
    text = str(value or "")
    known = {
        CAUSAL_RESPONSE_LABELS_SCHEMA: KNOWN_CAUSAL_RESPONSE_LABEL_SCHEMAS,
    }.get(canonical, frozenset({canonical}))
    return text in known

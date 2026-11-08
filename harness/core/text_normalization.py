"""Deterministic text normalization used by identity checks."""

from __future__ import annotations

import re
import unicodedata


def normalize_prompt(text: str) -> str:
    """Normalize Unicode, case, and whitespace for prompt identity checks."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()

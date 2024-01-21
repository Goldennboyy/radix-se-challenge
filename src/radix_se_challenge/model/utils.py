"""Utilisation functions."""

from __future__ import annotations

import re
from functools import lru_cache

from stop_words import get_stop_words


@lru_cache
def get_stopwords() -> list[str]:
    """Get a list of stopwords."""
    return get_stop_words("en")  # type: ignore


def simple_process(text: str) -> str:
    """Text processing."""
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text).strip()
    text = " ".join([t for t in text.split() if t not in get_stopwords()])
    return text

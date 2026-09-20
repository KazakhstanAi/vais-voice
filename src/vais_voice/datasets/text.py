"""Canonical transcript normalization and content-derived identity."""

import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def text_identity(language: str, normalized_text: str) -> str:
    digest = hashlib.sha256(f"{language}\0{normalized_text}".encode()).hexdigest()[:20]
    return f"text_{language}_{digest}"

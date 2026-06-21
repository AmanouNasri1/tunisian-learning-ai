"""Text normalization helpers for retrieval."""

from __future__ import annotations

import re
import unicodedata


_APOSTROPHE_TRANSLATION = str.maketrans({
    "\u0060": "'",
    "\u00b4": "'",
    "\u02bb": "'",
    "\u02bc": "'",
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\uff07": "'",
})
_LIGATURE_TRANSLATION = str.maketrans({
    "\u00c6": "AE",
    "\u00e6": "ae",
    "\u0152": "OE",
    "\u0153": "oe",
})
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def normalize_text(value: object | None) -> str:
    """Normalize user/query text for accent-insensitive matching.

    French Latin diacritics are stripped, apostrophes and whitespace are
    normalized, Arabic combining marks are preserved, and math symbols are left
    alone unless Unicode lowercase affects them.
    """
    if value is None:
        return ""

    text = str(value).translate(_APOSTROPHE_TRANSLATION)
    text = text.translate(_LIGATURE_TRANSLATION)
    text = _strip_latin_diacritics(text)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalized_tokens(value: object | None) -> list[str]:
    """Return normalized search tokens, ignoring surrounding apostrophes."""
    normalized = normalize_text(value)
    return [token.strip("'") for token in _TOKEN_RE.findall(normalized) if token.strip("'")]


def _strip_latin_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []
    last_base_is_latin = False

    for char in decomposed:
        if unicodedata.combining(char):
            if not last_base_is_latin:
                out.append(char)
            continue

        out.append(char)
        last_base_is_latin = _is_latin(char)

    return unicodedata.normalize("NFC", "".join(out))


def _is_latin(char: str) -> bool:
    return unicodedata.name(char, "").startswith("LATIN")

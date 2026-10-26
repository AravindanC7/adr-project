"""Turn a model's reply into a `Response.answer` of the right type for the item."""

from __future__ import annotations

import re

from adr.contracts import Item

# Letters must be capitals, so "the answer is a good one" never reads as option A.
_EXPLICIT = re.compile(r"(?i:answer|option|choice)\s*(?i:is|:|=)?\s*[\(\[]?\s*(\d+|[A-Z])\b")
_LONE = re.compile(r"^\s*[\(\[]?\s*(\d+|[A-Z])\s*[\)\].:]?\s*$")
_INT = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _to_index(token: str, k: int) -> int | None:
    if token.isdigit():
        i = int(token) - 1
    elif len(token) == 1 and token.isalpha():
        i = ord(token.upper()) - ord("A")
    else:
        return None
    return i if 0 <= i < k else None


def parse_option(text: str, options: list[str]) -> int | None:
    """Best guess at the chosen option index (0-based), or None if unclear."""
    k = len(options)
    # 1) "Answer: 2", "option (B)"
    for m in _EXPLICIT.finditer(text):
        i = _to_index(m.group(1), k)
        if i is not None:
            return i
    # 2) a line that is just "2" or "B)"
    for line in text.splitlines():
        m = _LONE.match(line)
        if m:
            i = _to_index(m.group(1), k)
            if i is not None:
                return i
    # 3) the option text itself, longest match first ("Strongly agree" before "Agree")
    low = text.lower()
    for i in sorted(range(k), key=lambda j: -len(options[j])):
        if options[i].lower() in low:
            return i
    # 4) the first in-range integer anywhere
    for m in _INT.finditer(text):
        i = _to_index(m.group(1), k)
        if i is not None:
            return i
    return None


def parse_ranking(text: str, k: int) -> list[int] | None:
    seen: list[int] = []
    for m in _INT.finditer(text):
        i = int(m.group(1)) - 1
        if 0 <= i < k and i not in seen:
            seen.append(i)
    return seen or None


def parse_answer(item: Item, text: str) -> int | list[int] | float | str | None:
    if item.item_type in ("mcq", "likert"):
        assert item.options
        return parse_option(text, item.options)
    if item.item_type == "ranking":
        assert item.options
        return parse_ranking(text, len(item.options))
    if item.item_type == "numeric":
        m = _NUM.search(text)
        return float(m.group(0).replace(",", "")) if m else None
    return text.strip() or None

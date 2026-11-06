"""Load AI2 Reasoning Challenge JSONL files into the shared Item contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adr.contracts import Item


def load_arc(jsonl_path: str | Path, *, subset: str = "challenge") -> list[Item]:
    """Load an ARC JSONL split.

    Supports both the original nested question form
    ``{"question": {"stem": ..., "choices": [...]}}`` and the normalized form with a string
    ``question`` plus a top-level ``choices`` mapping. Choice order is preserved and the answer is
    resolved through its label rather than assuming A-D labels.
    """

    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"ARC JSONL file not found: {path}")
    normalized_subset = subset.strip().lower().replace("arc-", "")
    if normalized_subset not in {"challenge", "easy"}:
        raise ValueError("ARC subset must be 'challenge' or 'easy'")

    items: list[Item] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: ARC record must be an object")

            native_id = _required_string(path, line_number, record, "id")
            item_id = f"arc:{normalized_subset}:{native_id.replace(':', '_')}"
            if item_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate ARC id {native_id!r}")
            seen_ids.add(item_id)

            stem, labels, options = _question_and_choices(path, line_number, record)
            answer_key = _required_string(path, line_number, record, "answerKey")
            try:
                correct_option = labels.index(answer_key)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: answerKey {answer_key!r} is not a choice label"
                ) from exc

            items.append(
                Item(
                    item_id=item_id,
                    source="arc",
                    assessment_type="quiz",
                    item_type="mcq",
                    stem=stem,
                    options=options,
                    correct_option=correct_option,
                    eligible_mechanisms=["decoy"],
                    split="train",
                )
            )

    if not items:
        raise ValueError(f"{path}: no ARC records found")
    return items


def _question_and_choices(
    path: Path, line_number: int, record: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    question = record.get("question")
    if isinstance(question, str):
        stem = question.strip()
        choices = record.get("choices")
    elif isinstance(question, dict):
        stem_value = question.get("stem")
        stem = stem_value.strip() if isinstance(stem_value, str) else ""
        choices = question.get("choices")
    else:
        raise TypeError(f"{path}:{line_number}: question must be a string or object")
    if not stem:
        raise ValueError(f"{path}:{line_number}: question stem is empty")

    if isinstance(choices, dict):
        labels_raw = choices.get("label")
        texts_raw = choices.get("text")
    elif isinstance(choices, list):
        labels_raw = [
            choice.get("label") if isinstance(choice, dict) else None for choice in choices
        ]
        texts_raw = [choice.get("text") if isinstance(choice, dict) else None for choice in choices]
    else:
        raise TypeError(f"{path}:{line_number}: choices must be an object or list")

    labels = _string_list(path, line_number, "choice labels", labels_raw)
    texts = _string_list(path, line_number, "choice texts", texts_raw)
    if len(labels) != len(texts) or len(labels) < 2:
        raise ValueError(
            f"{path}:{line_number}: choices need at least two equally sized labels and texts"
        )
    if len(labels) != len(set(labels)):
        raise ValueError(f"{path}:{line_number}: choice labels contain duplicates")
    return stem, labels, texts


def _required_string(path: Path, line_number: int, record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"{path}:{line_number}: {field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{path}:{line_number}: {field} is empty")
    return normalized


def _string_list(path: Path, line_number: int, field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise TypeError(f"{path}:{line_number}: {field} must be a list of strings")
    normalized = [entry.strip() for entry in value]
    if any(not entry for entry in normalized):
        raise ValueError(f"{path}:{line_number}: {field} contains an empty value")
    return normalized

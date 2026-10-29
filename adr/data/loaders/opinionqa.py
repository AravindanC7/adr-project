"""OpinionQA loader with population-level human answer distributions.

The official OpinionQA analysis code expects one directory per Pew American Trends Panel wave:

``human_resp/American_Trends_Panel_W<wave>/{info,responses,metadata}.csv``

Only ``info.csv`` and ``responses.csv`` are needed for the overall population distribution.
Raw individual responses stay in ``data/raw`` and are never written by this module.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adr.contracts import HumanDistribution, Item

WAVE_DIRECTORY = re.compile(r"American_Trends_Panel_W(?P<wave>\d+)$")


@dataclass(frozen=True)
class _SourceQuestion:
    """Source-normalized row kept separate from the shared Item contract."""

    wave: int
    native_key: str
    stem: str
    displayed_options: tuple[str, ...]


def load_opinionqa(human_resp_dir: str | Path) -> tuple[list[Item], list[HumanDistribution]]:
    """Load all discovered OpinionQA waves in deterministic order.

    Human probabilities use the official respondent weight for each wave and exclude refusal-like
    references from the denominator. ``n_respondents`` remains the unweighted count of people who
    selected one of the displayed options.
    """

    root = Path(human_resp_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"OpinionQA human_resp directory not found: {root}")

    wave_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and WAVE_DIRECTORY.fullmatch(path.name)),
        key=_wave_number,
    )
    if not wave_dirs:
        raise ValueError(f"no American_Trends_Panel_W<digits> directories found below {root}")

    items: list[Item] = []
    distributions: list[HumanDistribution] = []
    seen_item_ids: set[str] = set()

    for wave_dir in wave_dirs:
        wave = _wave_number(wave_dir)
        questions = _read_questions(wave_dir / "info.csv", wave)
        responses, weight_column = _read_responses(wave_dir / "responses.csv", wave)

        for question in questions:
            item_id = _item_id(question)
            if item_id in seen_item_ids:
                raise ValueError(f"duplicate normalized item_id: {item_id}")
            seen_item_ids.add(item_id)

            distribution = _human_distribution(question, responses, weight_column)
            items.append(_to_item(question, distribution.key))
            distributions.append(distribution)

    return items, distributions


def _to_item(question: _SourceQuestion, human_dist_key: str) -> Item:
    """Map source-normalized data to the shared contract in one replaceable adapter."""

    return Item(
        item_id=_item_id(question),
        source="opinionqa",
        assessment_type="survey",
        item_type="mcq",
        stem=question.stem,
        options=list(question.displayed_options),
        correct_option=None,
        eligible_mechanisms=["opinion_trap"],
        human_dist_key=human_dist_key,
        split="train",
    )


def _read_questions(path: Path, wave: int) -> list[_SourceQuestion]:
    rows = _read_csv(path)
    _require_columns(path, rows, {"key", "question", "references", "option_ordinal"})

    questions: list[_SourceQuestion] = []
    for row_number, row in enumerate(rows, start=2):
        native_key = _required_text(path, row_number, row, "key")
        stem = _required_text(path, row_number, row, "question")
        references = _parse_string_list(path, row_number, "references", row["references"])
        ordinals = _parse_list(path, row_number, "option_ordinal", row["option_ordinal"])

        if not ordinals:
            raise ValueError(f"{path}:{row_number}: option_ordinal is empty")
        if len(references) < len(ordinals):
            raise ValueError(
                f"{path}:{row_number}: fewer references than displayed ordinal options"
            )

        options = tuple(references[: len(ordinals)])
        if len(options) < 2:
            raise ValueError(f"{path}:{row_number}: question needs at least two options")
        if len(options) != len(set(options)):
            raise ValueError(f"{path}:{row_number}: displayed options contain duplicates")

        questions.append(
            _SourceQuestion(
                wave=wave,
                native_key=native_key,
                stem=stem,
                displayed_options=options,
            )
        )
    return questions


def _read_responses(path: Path, wave: int) -> tuple[list[dict[str, str]], str]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"{path}: response file has no data rows")

    weight_column = f"WEIGHT_W{wave}"
    if weight_column not in rows[0]:
        raise ValueError(f"{path}: missing required column {weight_column}")
    return rows, weight_column


def _human_distribution(
    question: _SourceQuestion,
    responses: Iterable[Mapping[str, str]],
    weight_column: str,
) -> HumanDistribution:
    weighted_counts = {option: 0.0 for option in question.displayed_options}
    n_respondents = 0

    for row_number, row in enumerate(responses, start=2):
        if question.native_key not in row:
            raise ValueError(
                f"responses.csv:{row_number}: missing question column {question.native_key}"
            )

        answer = (row.get(question.native_key) or "").strip()
        weight_text = (row.get(weight_column) or "").strip()
        if not answer or not weight_text:
            continue

        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(
                f"responses.csv:{row_number}: invalid respondent weight {weight_text!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"responses.csv:{row_number}: respondent weight must be finite and non-negative"
            )

        if answer in weighted_counts:
            weighted_counts[answer] += weight
            n_respondents += 1

    total_weight = sum(weighted_counts.values())
    if n_respondents == 0 or total_weight <= 0:
        raise ValueError(f"{_item_id(question)}: no valid weighted human responses")

    probabilities = [
        weighted_counts[option] / total_weight for option in question.displayed_options
    ]
    return HumanDistribution(
        key=f"{_item_id(question)}:overall",
        source="opinionqa",
        option_probs=probabilities,
        n_respondents=n_respondents,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required OpinionQA file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is missing")
        return [dict(row) for row in reader]


def _require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    if not rows:
        raise ValueError(f"{path}: file has no data rows")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def _required_text(path: Path, row_number: int, row: Mapping[str, str], field: str) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{path}:{row_number}: {field} is empty")
    return value


def _parse_list(path: Path, row_number: int, field: str, value: str) -> list[Any]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number}: invalid {field} literal") from exc
    if not isinstance(parsed, list):
        raise TypeError(f"{path}:{row_number}: {field} must be a list")
    return parsed


def _parse_string_list(path: Path, row_number: int, field: str, value: str) -> list[str]:
    parsed = _parse_list(path, row_number, field, value)
    if not parsed or not all(isinstance(entry, str) and entry.strip() for entry in parsed):
        raise ValueError(f"{path}:{row_number}: {field} must contain non-empty strings")
    return [entry.strip() for entry in parsed]


def _wave_number(path: Path) -> int:
    match = WAVE_DIRECTORY.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not an OpinionQA wave directory: {path}")
    return int(match.group("wave"))


def _item_id(question: _SourceQuestion) -> str:
    safe_key = question.native_key.strip().replace(":", "_")
    return f"opinionqa:w{question.wave}:{safe_key}"

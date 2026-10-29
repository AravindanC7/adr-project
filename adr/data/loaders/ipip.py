"""Load the public IPIP Big-Five 50-item response dataset."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Literal

from adr.contracts import HumanDistribution, Item

_ITEM_LINE = re.compile(r"^([ENACO](?:10|[1-9]))\t(.+?)\s*$")
_OPTIONS = ["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"]


def load_ipip(
    data_path: str | Path,
    codebook_path: str | Path,
    *,
    split: Literal["train", "heldout_survey"] = "train",
) -> tuple[list[Item], list[HumanDistribution]]:
    """Return IPIP items and empirical human answer distributions.

    The Open-Source Psychometrics export uses 1-5 responses and 0 for a missed item. Both
    tab-separated and comma-separated exports are accepted.
    """

    responses_path = Path(data_path)
    items_path = Path(codebook_path)
    if not responses_path.is_file():
        raise FileNotFoundError(f"IPIP response file not found: {responses_path}")
    if not items_path.is_file():
        raise FileNotFoundError(f"IPIP codebook not found: {items_path}")

    item_text = _parse_codebook(items_path)
    counts = {code: [0] * 5 for code in item_text}
    respondent_counts = {code: 0 for code in item_text}

    with responses_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        delimiter = "\t" if "\t" in header else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{responses_path}: missing header")
        missing = sorted(set(item_text).difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{responses_path}: missing IPIP columns {missing}")
        for row_number, row in enumerate(reader, start=2):
            for code in item_text:
                raw = (row.get(code) or "").strip()
                if raw in {"", "0"}:
                    continue
                try:
                    value = int(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{responses_path}:{row_number}: {code} must be an integer from 0 to 5"
                    ) from exc
                if not 1 <= value <= 5:
                    raise ValueError(
                        f"{responses_path}:{row_number}: {code} must be between 0 and 5"
                    )
                counts[code][value - 1] += 1
                respondent_counts[code] += 1

    items: list[Item] = []
    distributions: list[HumanDistribution] = []
    for code, stem in item_text.items():
        n = respondent_counts[code]
        if n == 0:
            raise ValueError(f"{responses_path}: no usable responses for {code}")
        item_id = f"ipip:{code.lower()}"
        dist_key = f"{item_id}:human"
        items.append(
            Item(
                item_id=item_id,
                source="ipip",
                assessment_type="survey",
                item_type="likert",
                stem=stem,
                options=_OPTIONS,
                reverse_keyed_with=f"ipip:{_paired_code(code).lower()}",
                eligible_mechanisms=["opinion_trap"],
                human_dist_key=dist_key,
                split=split,
            )
        )
        distributions.append(
            HumanDistribution(
                key=dist_key,
                source="ipip",
                option_probs=[count / n for count in counts[code]],
                n_respondents=n,
            )
        )
    return items, distributions


def _parse_codebook(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = _ITEM_LINE.match(line)
        if match:
            code, stem = match.groups()
            items[code] = stem
    expected = {f"{trait}{number}" for trait in "ENACO" for number in range(1, 11)}
    missing = sorted(expected.difference(items))
    if missing:
        raise ValueError(f"{path}: codebook is missing IPIP items {missing}")
    return items


def _paired_code(code: str) -> str:
    number = int(code[1:])
    paired = number + 1 if number % 2 else number - 1
    return f"{code[0]}{paired}"

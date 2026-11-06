"""Load a GSS Data Explorer CSV extract using an explicit question manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from adr.contracts import HumanDistribution, Item


def load_gss(
    csv_path: str | Path,
    manifest_path: str | Path,
    *,
    split: Literal["train", "heldout_survey"] = "train",
) -> tuple[list[Item], list[HumanDistribution]]:
    """Convert selected GSS variables and their human frequencies to shared contracts.

    GSS extracts are wide tables whose selected variables differ by project. The companion JSON
    manifest records the exact question wording and ordered value labels from the GSS codebook.
    When a weight column is configured, probabilities are weighted while ``n_respondents`` remains
    the unweighted number of usable response rows.
    """

    data_path = Path(csv_path)
    spec_path = Path(manifest_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"GSS CSV file not found: {data_path}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"GSS manifest file not found: {spec_path}")

    manifest = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{spec_path}: manifest must be a JSON object")
    questions = manifest.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{spec_path}: manifest needs a non-empty questions list")
    weight_column = manifest.get("weight_column")
    if weight_column is not None and not isinstance(weight_column, str):
        raise TypeError(f"{spec_path}: weight_column must be a string or null")

    normalized = [_normalize_question(spec_path, question) for question in questions]
    counts = {question["variable"]: [0.0] * len(question["options"]) for question in normalized}
    respondents = {question["variable"]: 0 for question in normalized}

    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{data_path}: missing CSV header")
        required = {question["variable"] for question in normalized}
        if weight_column:
            required.add(weight_column)
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{data_path}: missing columns {missing}")

        for row_number, row in enumerate(reader, start=2):
            weight = _row_weight(data_path, row_number, row, weight_column)
            for question in normalized:
                value = (row.get(question["variable"]) or "").strip()
                if not value or value in question["missing_values"]:
                    continue
                try:
                    option_index = question["codes"].index(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{data_path}:{row_number}: unexpected value {value!r} "
                        f"for {question['variable']}"
                    ) from exc
                counts[question["variable"]][option_index] += weight
                # Keep the raw usable-person count separate from the weighted probability mass.
                respondents[question["variable"]] += 1

    items: list[Item] = []
    distributions: list[HumanDistribution] = []
    for question in normalized:
        variable = question["variable"]
        total = sum(counts[variable])
        if total <= 0:
            raise ValueError(f"{data_path}: no usable responses for GSS variable {variable}")
        item_id = f"gss:{variable.lower()}"
        dist_key = f"{item_id}:human"
        items.append(
            Item(
                item_id=item_id,
                source="gss",
                assessment_type="survey",
                item_type=question["item_type"],
                stem=question["stem"],
                options=question["options"],
                reverse_keyed_with=question["reverse_keyed_with"],
                eligible_mechanisms=["opinion_trap"],
                human_dist_key=dist_key,
                split=split,
            )
        )
        distributions.append(
            HumanDistribution(
                key=dist_key,
                source="gss",
                option_probs=[value / total for value in counts[variable]],
                n_respondents=respondents[variable],
            )
        )
    return items, distributions


def _normalize_question(path: Path, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: every question must be a JSON object")
    variable = raw.get("variable")
    stem = raw.get("stem")
    options = raw.get("options")
    item_type = raw.get("item_type", "mcq")
    if not isinstance(variable, str) or not variable.strip():
        raise ValueError(f"{path}: question variable must be a non-empty string")
    if not isinstance(stem, str) or not stem.strip():
        raise ValueError(f"{path}: {variable}: stem must be a non-empty string")
    if item_type not in {"mcq", "likert"}:
        raise ValueError(f"{path}: {variable}: item_type must be mcq or likert")
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError(f"{path}: {variable}: options must map at least two codes to labels")
    codes = list(options)
    labels = list(options.values())
    if not all(isinstance(code, str) and code for code in codes):
        raise TypeError(f"{path}: {variable}: option codes must be non-empty strings")
    if not all(isinstance(label, str) and label for label in labels):
        raise TypeError(f"{path}: {variable}: option labels must be non-empty strings")
    missing_values = raw.get("missing_values", [])
    if not isinstance(missing_values, list) or not all(
        isinstance(value, str) for value in missing_values
    ):
        raise TypeError(f"{path}: {variable}: missing_values must be a list of strings")
    reverse = raw.get("reverse_keyed_with")
    if reverse is not None:
        if not isinstance(reverse, str) or not reverse:
            raise TypeError(f"{path}: {variable}: reverse_keyed_with must be a string or null")
        if not reverse.startswith("gss:"):
            reverse = f"gss:{reverse.lower()}"
    return {
        "variable": variable,
        "stem": stem.strip(),
        "item_type": item_type,
        "codes": codes,
        "options": labels,
        "missing_values": set(missing_values),
        "reverse_keyed_with": reverse,
    }


def _row_weight(
    path: Path, row_number: int, row: dict[str, str | None], weight_column: str | None
) -> float:
    if weight_column is None:
        return 1.0
    raw = (row.get(weight_column) or "").strip()
    try:
        weight = float(raw)
    except ValueError as exc:
        raise ValueError(f"{path}:{row_number}: invalid weight {raw!r}") from exc
    if weight <= 0:
        raise ValueError(f"{path}:{row_number}: weight must be positive")
    return weight

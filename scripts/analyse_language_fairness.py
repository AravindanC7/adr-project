"""Audit writing-style false flags by survey-completion language.

The Nature PhD survey records the language in which each respondent completed
the questionnaire.  It does *not* record whether the respondent is a native
English speaker.  This script therefore reports English-interface versus
non-English-interface results and never treats the field as a nativeness label.

Respondents, rather than individual answers, are deterministically split.  The
style profile is fitted on the training respondents and evaluated on held-out
respondents, preventing answers from the same person appearing on both sides.
Only aggregate counts and rates are written; no answer text or respondent rows
leave the source workbook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adr.contracts import Item, RenderedItem, Response
from adr.data.loaders import load_asurre_items
from adr.mechanisms.style import StyleMechanism, fit_style_profile


def _in_evaluation_split(respondent_id: str, *, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{respondent_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def _item_columns(config: dict[str, Any], items: list[Item]) -> list[tuple[Item, int]]:
    by_field = {item.item_id.rsplit(":", 1)[1].casefold(): item for item in items}
    result: list[tuple[Item, int]] = []
    for field, metadata in config["attributes"].items():
        item = by_field.get(str(field).casefold())
        if item is not None:
            result.append((item, int(metadata["column_index"])))
    return result


def _responses_by_person(
    frame: pd.DataFrame,
    config: dict[str, Any],
    items: list[Item],
) -> dict[str, tuple[str, list[tuple[Item, str]]]]:
    start = int(config["data_starts_row"])
    language_column = next(
        index
        for index, value in enumerate(frame.iloc[0])
        if str(value).strip() == "ID.language"
    )
    columns = _item_columns(config, items)
    people: dict[str, tuple[str, list[tuple[Item, str]]]] = {}
    for row_index in range(start, len(frame)):
        language_value = frame.iat[row_index, language_column]
        language = "" if pd.isna(language_value) else str(language_value).strip()
        if not language:
            continue
        answers: list[tuple[Item, str]] = []
        for item, column in columns:
            value = frame.iat[row_index, column]
            if pd.isna(value) or not str(value).strip():
                continue
            answers.append((item, str(value).strip()))
        if answers:
            people[f"row-{row_index}"] = (language, answers)
    return people


def _group(language: str) -> str:
    return "english_interface" if language.casefold() == "english" else "non_english_interface"


def _cluster_interval(
    person_flags: list[list[int]], *, seed: int, draws: int
) -> tuple[float, float]:
    if not person_flags:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rates = np.empty(draws)
    for draw in range(draws):
        sampled = rng.integers(0, len(person_flags), size=len(person_flags))
        values = [flag for index in sampled for flag in person_flags[index]]
        rates[draw] = float(np.mean(values))
    low, high = np.quantile(rates, [0.025, 0.975])
    return float(low), float(high)


def analyse(
    *,
    nature_frame: pd.DataFrame,
    nature_config: dict[str, Any],
    nature_items: list[Item],
    extra_training_samples: list[tuple[Item, str]],
    seed: int = 576,
    evaluation_fraction: float = 0.20,
    bootstrap_draws: int = 2_000,
) -> dict[str, Any]:
    """Fit on training people and return aggregate held-out language results."""

    people = _responses_by_person(nature_frame, nature_config, nature_items)
    training = list(extra_training_samples)
    evaluation: dict[str, tuple[str, list[tuple[Item, str]]]] = {}
    for person_id, record in people.items():
        if _in_evaluation_split(person_id, seed=seed, fraction=evaluation_fraction):
            evaluation[person_id] = record
        else:
            training.extend(record[1])

    mechanism = StyleMechanism(fit_style_profile(training))
    by_group: dict[str, list[list[int]]] = defaultdict(list)
    language_counts: dict[str, int] = defaultdict(int)
    for language, answers in evaluation.values():
        flags: list[int] = []
        for item, text in answers:
            rendered = RenderedItem(
                item_id=item.item_id,
                mechanism=None,
                variant_id=f"{item.item_id}|plain|0",
                visible={"stem": item.stem},
                copy_text=item.stem,
            )
            outcome = mechanism.score(
                rendered,
                Response(
                    session_id="language-fairness",
                    item_id=item.item_id,
                    variant_id=rendered.variant_id,
                    answer=text,
                    raw_text=text,
                ),
                None,
            )
            flags.append(int(outcome == "high"))
        by_group[_group(language)].append(flags)
        language_counts[language] += 1

    groups: dict[str, Any] = {}
    for offset, group_name in enumerate(("english_interface", "non_english_interface")):
        person_flags = by_group.get(group_name, [])
        flat = [flag for flags in person_flags for flag in flags]
        low, high = _cluster_interval(
            person_flags, seed=seed + offset + 1, draws=bootstrap_draws
        )
        groups[group_name] = {
            "people": len(person_flags),
            "responses": len(flat),
            "high_flags": int(sum(flat)),
            "response_false_flag_rate": float(np.mean(flat)) if flat else None,
            "response_false_flag_ci95_clustered": [low, high] if flat else None,
            "people_with_any_high": int(sum(any(flags) for flags in person_flags)),
            "person_any_high_rate": (
                float(np.mean([any(flags) for flags in person_flags])) if person_flags else None
            ),
        }

    return {
        "version": "language-fairness-v1",
        "label_limitation": (
            "ID.language is the survey-completion language, not native-speaker status. "
            "Results must not be described as a direct native/non-native comparison."
        ),
        "design": {
            "split_unit": "respondent row",
            "evaluation_fraction": evaluation_fraction,
            "split_seed": seed,
            "bootstrap_unit": "respondent",
            "bootstrap_draws": bootstrap_draws,
            "flag_outcome": "style=high",
            "extra_training_responses": len(extra_training_samples),
            "profile": asdict(mechanism.profile),
        },
        "groups": groups,
        "evaluation_people_by_completion_language": dict(sorted(language_counts.items())),
    }


def _load_samples(raw_path: Path, config_path: Path) -> list[tuple[Item, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    items = load_asurre_items(config_path)
    frame = pd.read_excel(raw_path, header=None)
    samples: list[tuple[Item, str]] = []
    for item, column in _item_columns(config, items):
        for value in frame.iloc[int(config["data_starts_row"]) :, column]:
            if pd.isna(value) or not str(value).strip():
                continue
            samples.append((item, str(value).strip()))
    return samples


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/asurre/raw"))
    ap.add_argument("--config-dir", type=Path, default=Path("data/raw/asurre/configs"))
    ap.add_argument("--output", type=Path, default=Path("results/language_fairness.json"))
    ap.add_argument("--seed", type=int, default=576)
    ap.add_argument("--bootstrap-draws", type=int, default=2_000)
    return ap


def main() -> None:
    args = parser().parse_args()
    nature_config_path = args.config_dir / "phd2019.json"
    nature_config = json.loads(nature_config_path.read_text(encoding="utf-8"))
    nature_items = load_asurre_items(nature_config_path)
    extra_training = _load_samples(
        args.raw_dir / "springer_2017.xlsx", args.config_dir / "springer_2017.json"
    )
    result = analyse(
        nature_frame=pd.read_excel(args.raw_dir / "phd2019.xlsx", header=None),
        nature_config=nature_config,
        nature_items=nature_items,
        extra_training_samples=extra_training,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()

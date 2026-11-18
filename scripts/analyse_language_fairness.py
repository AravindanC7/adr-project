"""Audit writing-style false flags by survey-completion language.

The Nature PhD survey records the language in which each respondent completed
the questionnaire.  It does *not* record whether the respondent is a native
English speaker.  This script therefore reports English-interface versus
non-English-interface results and never treats the field as a nativeness label.

The analysis uses the one canonical style profile locked in
``configs/data_decisions.json`` so it measures the check used by the real runs.
Because that profile was fitted on Nature responses, this is an in-sample
fairness audit rather than a held-out performance estimate.
Bootstrap resampling is clustered by respondent.  Only aggregate counts, rates
and confidence intervals are written; no answer text or respondent rows leave
the source workbook.
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
from adr.mechanisms.style import StyleMechanism, StyleProfile, load_style_profile


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
) -> tuple[dict[str, tuple[str, list[tuple[Item, str]]]], dict[str, int]]:
    start = int(config["data_starts_row"])
    language_column = next(
        index for index, value in enumerate(frame.iloc[0]) if str(value).strip() == "ID.language"
    )
    columns = _item_columns(config, items)
    people: dict[str, tuple[str, list[tuple[Item, str]]]] = {}
    missing_language = 0
    no_usable_answers = 0
    for row_index in range(start, len(frame)):
        language_value = frame.iat[row_index, language_column]
        language = "" if pd.isna(language_value) else str(language_value).strip()
        if not language:
            missing_language += 1
            continue
        answers: list[tuple[Item, str]] = []
        for item, column in columns:
            value = frame.iat[row_index, column]
            if pd.isna(value) or not str(value).strip():
                continue
            answers.append((item, str(value).strip()))
        if answers:
            people[f"row-{row_index}"] = (language, answers)
        else:
            no_usable_answers += 1
    source_rows = max(len(frame) - start, 0)
    return people, {
        "source_rows": source_rows,
        "included_rows": len(people),
        "dropped_rows": missing_language + no_usable_answers,
        "dropped_missing_completion_language": missing_language,
        "dropped_no_usable_open_text_answers": no_usable_answers,
    }


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


def _cluster_gap_intervals(
    english_flags: list[list[int]],
    non_english_flags: list[list[int]],
    *,
    seed: int,
    draws: int,
) -> dict[str, tuple[float, float]]:
    """CIs for response and person gaps, resampling people within each group."""

    if not english_flags or not non_english_flags:
        missing = (float("nan"), float("nan"))
        return {"response": missing, "person_any": missing}
    rng = np.random.default_rng(seed)
    response_gaps = np.empty(draws)
    person_gaps = np.empty(draws)
    for draw in range(draws):
        english_sample = rng.integers(0, len(english_flags), size=len(english_flags))
        non_english_sample = rng.integers(0, len(non_english_flags), size=len(non_english_flags))
        english = [flag for index in english_sample for flag in english_flags[index]]
        non_english = [flag for index in non_english_sample for flag in non_english_flags[index]]
        response_gaps[draw] = float(np.mean(non_english) - np.mean(english))
        english_any = [any(english_flags[index]) for index in english_sample]
        non_english_any = [any(non_english_flags[index]) for index in non_english_sample]
        person_gaps[draw] = float(np.mean(non_english_any) - np.mean(english_any))
    response_ci = np.quantile(response_gaps, [0.025, 0.975])
    person_ci = np.quantile(person_gaps, [0.025, 0.975])
    return {
        "response": (float(response_ci[0]), float(response_ci[1])),
        "person_any": (float(person_ci[0]), float(person_ci[1])),
    }


def _profile_object_sha256(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    profile = document.get("profile", document)
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_canonical_profile(
    decisions_path: Path, style_profile_path: Path | None = None
) -> tuple[StyleProfile, dict[str, Any]]:
    """Load and verify the style profile selected in data_decisions.json."""

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    choice = decisions.get("style_profile")
    if not isinstance(choice, dict):
        raise TypeError(f"{decisions_path}: missing style_profile decision")
    configured = Path(str(choice["file"]))
    profile_path = style_profile_path or decisions_path.parent.parent / configured
    actual_sha = _profile_object_sha256(profile_path)
    expected_sha = str(choice["profile_sha256"])
    if actual_sha != expected_sha:
        raise ValueError(
            f"{profile_path}: profile sha256 {actual_sha} does not match "
            f"the locked value {expected_sha}"
        )
    profile = load_style_profile(profile_path)
    if "asurre_osmi_2017_18" in profile.sources:
        raise ValueError(f"{profile_path}: canonical profile includes held-out OSMI responses")
    return profile, {
        "decisions_file": str(decisions_path),
        "profile_file": str(profile_path),
        "profile_sha256": actual_sha,
    }


def analyse(
    *,
    nature_frame: pd.DataFrame,
    nature_config: dict[str, Any],
    nature_items: list[Item],
    style_profile: StyleProfile,
    seed: int = 576,
    bootstrap_draws: int = 2_000,
) -> dict[str, Any]:
    """Return aggregate language results under the canonical style profile."""

    people, row_accounting = _responses_by_person(nature_frame, nature_config, nature_items)
    mechanism = StyleMechanism(style_profile)
    by_group: dict[str, list[list[int]]] = defaultdict(list)
    language_counts: dict[str, int] = defaultdict(int)
    for language, answers in people.values():
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
        low, high = _cluster_interval(person_flags, seed=seed + offset + 1, draws=bootstrap_draws)
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

    english_flags = by_group.get("english_interface", [])
    non_english_flags = by_group.get("non_english_interface", [])
    english_flat = [flag for flags in english_flags for flag in flags]
    non_english_flat = [flag for flags in non_english_flags for flag in flags]
    gap_intervals = _cluster_gap_intervals(
        english_flags,
        non_english_flags,
        seed=seed + 10,
        draws=bootstrap_draws,
    )
    gap = (
        float(np.mean(non_english_flat) - np.mean(english_flat))
        if english_flat and non_english_flat
        else None
    )
    person_gap = (
        float(
            np.mean([any(flags) for flags in non_english_flags])
            - np.mean([any(flags) for flags in english_flags])
        )
        if english_flags and non_english_flags
        else None
    )

    return {
        "version": "language-fairness-v2",
        "label_limitation": (
            "ID.language is the survey-completion language, not native-speaker status. "
            "Results must not be described as a direct native/non-native comparison. "
            "This audit is in-sample because Nature responses helped fit the canonical profile."
        ),
        "design": {
            "profile_rule": "canonical profile from configs/data_decisions.json",
            "evaluation_scope": (
                "in-sample: Nature responses helped fit the canonical style profile"
            ),
            "bootstrap_unit": "respondent",
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": seed,
            "flag_outcome": "style=high",
            "profile": asdict(mechanism.profile),
            "row_accounting": row_accounting,
        },
        "groups": groups,
        "gap_non_english_minus_english": {
            "response_false_flag_rate_difference": gap,
            "response_ci95_clustered": (
                list(gap_intervals["response"]) if gap is not None else None
            ),
            "person_any_high_rate_difference": person_gap,
            "person_any_high_ci95_clustered": (
                list(gap_intervals["person_any"]) if person_gap is not None else None
            ),
        },
        "evaluation_people_by_completion_language": dict(sorted(language_counts.items())),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/asurre/raw"))
    ap.add_argument("--config-dir", type=Path, default=Path("data/raw/asurre/configs"))
    ap.add_argument("--decisions", type=Path, default=Path("configs/data_decisions.json"))
    ap.add_argument(
        "--style-profile",
        type=Path,
        help="Override the profile file location; its object hash must still match decisions",
    )
    ap.add_argument("--output", type=Path, default=Path("results/language_fairness.json"))
    ap.add_argument("--seed", type=int, default=576)
    ap.add_argument("--bootstrap-draws", type=int, default=2_000)
    return ap


def main() -> None:
    args = parser().parse_args()
    nature_config_path = args.config_dir / "phd2019.json"
    nature_config = json.loads(nature_config_path.read_text(encoding="utf-8"))
    nature_items = load_asurre_items(nature_config_path)
    profile, profile_audit = load_canonical_profile(args.decisions, args.style_profile)
    result = analyse(
        nature_frame=pd.read_excel(args.raw_dir / "phd2019.xlsx", header=None),
        nature_config=nature_config,
        nature_items=nature_items,
        style_profile=profile,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    result["design"]["canonical_profile"] = profile_audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()

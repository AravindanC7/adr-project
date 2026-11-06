from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adr.data.loaders import load_opinionqa


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_wave(root: Path) -> Path:
    wave = root / "American_Trends_Panel_W26"
    wave.mkdir(parents=True)
    _write_csv(
        wave / "info.csv",
        ["key", "question", "references", "option_ordinal", "survey"],
        [
            {
                "key": "Q1",
                "question": "Which fictional park plan do you prefer?",
                "references": "['More trees', 'More benches', 'Refused']",
                "option_ordinal": "[0, 1]",
                "survey": "Pew_American_Trends_Panel_W26",
            },
            {
                "key": "Q2",
                "question": "How often do you visit the fictional park?",
                "references": "['Often', 'Sometimes', 'Never']",
                "option_ordinal": "[0, 1, 2]",
                "survey": "Pew_American_Trends_Panel_W26",
            },
        ],
    )
    _write_csv(
        wave / "responses.csv",
        ["WEIGHT_W26", "Q1", "Q2"],
        [
            {"WEIGHT_W26": "1", "Q1": "More trees", "Q2": "Often"},
            {"WEIGHT_W26": "2", "Q1": "More benches", "Q2": "Sometimes"},
            {"WEIGHT_W26": "3", "Q1": "Refused", "Q2": "Never"},
            {"WEIGHT_W26": "4", "Q1": "", "Q2": "Never"},
        ],
    )
    return wave


def test_loads_items_and_weighted_overall_distributions(tmp_path: Path) -> None:
    _synthetic_wave(tmp_path)

    items, distributions = load_opinionqa(tmp_path)

    assert len(items) == len(distributions) == 2
    first = items[0]
    assert first.item_id == "opinionqa:w26:Q1"
    assert first.options == ["More trees", "More benches"]
    assert first.eligible_mechanisms == ["opinion_trap"]
    assert first.human_dist_key == distributions[0].key
    assert distributions[0].option_probs == pytest.approx([1 / 3, 2 / 3])
    assert distributions[0].n_respondents == 2


def test_ignores_refusal_and_missing_answers(tmp_path: Path) -> None:
    _synthetic_wave(tmp_path)

    _, distributions = load_opinionqa(tmp_path)

    assert distributions[0].n_respondents == 2
    assert sum(distributions[0].option_probs) == pytest.approx(1.0)


def test_rejects_missing_wave_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no American_Trends_Panel"):
        load_opinionqa(tmp_path)


def test_rejects_invalid_weights(tmp_path: Path) -> None:
    wave = _synthetic_wave(tmp_path)
    _write_csv(
        wave / "responses.csv",
        ["WEIGHT_W26", "Q1", "Q2"],
        [{"WEIGHT_W26": "not-a-number", "Q1": "More trees", "Q2": "Often"}],
    )

    with pytest.raises(ValueError, match="invalid respondent weight"):
        load_opinionqa(tmp_path)


def test_normalizes_case_and_trailing_periods_before_matching(tmp_path: Path) -> None:
    wave = _synthetic_wave(tmp_path)
    _write_csv(
        wave / "responses.csv",
        ["WEIGHT_W26", "Q1", "Q2"],
        [
            {"WEIGHT_W26": "1", "Q1": "More trees", "Q2": "Often"},
            {"WEIGHT_W26": "1", "Q1": "more trees", "Q2": "Often"},
            {"WEIGHT_W26": "1", "Q1": "More benches.", "Q2": "Often"},
            {"WEIGHT_W26": "1", "Q1": "Refused", "Q2": "Often"},
        ],
    )

    _, distributions = load_opinionqa(tmp_path)

    assert distributions[0].n_respondents == 3
    assert distributions[0].option_probs == pytest.approx([2 / 3, 1 / 3])


def test_warns_only_for_answers_unmatched_after_normalization(tmp_path: Path) -> None:
    wave = _synthetic_wave(tmp_path)
    _write_csv(
        wave / "responses.csv",
        ["WEIGHT_W26", "Q1", "Q2"],
        [
            {"WEIGHT_W26": "1", "Q1": "More trees", "Q2": "Often"},
            {"WEIGHT_W26": "1", "Q1": "unknown answer", "Q2": "Often"},
            {"WEIGHT_W26": "1", "Q1": "refused.", "Q2": "Often"},
        ],
    )

    with pytest.warns(RuntimeWarning, match="ignored 1/3.*non-refusal answers"):
        _, distributions = load_opinionqa(tmp_path)

    assert distributions[0].n_respondents == 1

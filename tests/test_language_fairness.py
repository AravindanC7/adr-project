from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from adr.contracts import Item
from adr.mechanisms.style import StyleProfile
from scripts.analyse_language_fairness import analyse, load_canonical_profile


def _item() -> Item:
    return Item(
        item_id="asurre_nature_phd_2019:q1",
        source="asurre_nature_phd_2019",
        assessment_type="survey",
        item_type="open_text",
        stem="Explain.",
        eligible_mechanisms=["style"],
        split="train",
    )


def test_analysis_uses_language_labels_without_claiming_nativeness() -> None:
    rows = [["ID.language", "Q1"], ["Language question", "Prompt"]]
    for index in range(160):
        language = "English" if index % 2 == 0 else "Spanish"
        rows.append([language, f"A complete human response number {index}. It has words."])
    frame = pd.DataFrame(rows)
    config = {
        "data_starts_row": 2,
        "attributes": {"Q1": {"type": "text", "column_index": 1}},
    }
    profile = StyleProfile(
        feature_means=(4.0, 0.8, 5.0, 8.0, 0.02),
        feature_stds=(1.0, 0.2, 1.0, 3.0, 0.05),
        mid_threshold=2.0,
        high_threshold=3.0,
        n_training=32_299,
        sources=("asurre_nature_phd_2019", "asurre_springer_social_2017"),
    )

    result = analyse(
        nature_frame=frame,
        nature_config=config,
        nature_items=[_item()],
        style_profile=profile,
        seed=3,
        bootstrap_draws=50,
    )

    assert "not native-speaker status" in result["label_limitation"]
    assert result["design"]["profile_rule"].startswith("canonical profile")
    assert result["design"]["evaluation_scope"].startswith("in-sample")
    assert result["groups"]["english_interface"]["people"] == 80
    assert result["groups"]["non_english_interface"]["people"] == 80
    gap = result["gap_non_english_minus_english"]
    assert gap["response_ci95_clustered"] is not None
    assert gap["person_any_high_ci95_clustered"] is not None
    assert result["design"]["row_accounting"] == {
        "source_rows": 160,
        "included_rows": 160,
        "dropped_rows": 0,
        "dropped_missing_completion_language": 0,
        "dropped_no_usable_open_text_answers": 0,
    }


def test_canonical_profile_must_match_locked_object_hash(tmp_path) -> None:
    profile_object = {
        "feature_means": [4.0, 0.8, 5.0, 8.0, 0.02],
        "feature_stds": [1.0, 0.2, 1.0, 3.0, 0.05],
        "mid_threshold": 2.0,
        "high_threshold": 3.0,
        "n_training": 100,
        "sources": ["asurre_nature_phd_2019"],
    }
    profile_path = tmp_path / "style_profile.json"
    profile_path.write_text(json.dumps({"profile": profile_object}), encoding="utf-8")
    payload = json.dumps(profile_object, sort_keys=True, separators=(",", ":")).encode()
    expected = hashlib.sha256(payload).hexdigest()
    decisions_path = tmp_path / "configs" / "data_decisions.json"
    decisions_path.parent.mkdir()
    decisions_path.write_text(
        json.dumps(
            {
                "style_profile": {
                    "file": "ignored-when-overridden.json",
                    "profile_sha256": expected,
                }
            }
        ),
        encoding="utf-8",
    )

    profile, audit = load_canonical_profile(decisions_path, profile_path)

    assert profile.n_training == 100
    assert audit["profile_sha256"] == expected


def test_canonical_profile_rejects_heldout_osmi(tmp_path) -> None:
    profile_object = {
        "feature_means": [4.0, 0.8, 5.0, 8.0, 0.02],
        "feature_stds": [1.0, 0.2, 1.0, 3.0, 0.05],
        "mid_threshold": 2.0,
        "high_threshold": 3.0,
        "n_training": 100,
        "sources": ["asurre_nature_phd_2019", "asurre_osmi_2017_18"],
    }
    profile_path = tmp_path / "style_profile.json"
    profile_path.write_text(json.dumps({"profile": profile_object}), encoding="utf-8")
    payload = json.dumps(profile_object, sort_keys=True, separators=(",", ":")).encode()
    decisions_path = tmp_path / "configs" / "data_decisions.json"
    decisions_path.parent.mkdir()
    decisions_path.write_text(
        json.dumps(
            {
                "style_profile": {
                    "file": "ignored-when-overridden.json",
                    "profile_sha256": hashlib.sha256(payload).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="held-out OSMI"):
        load_canonical_profile(decisions_path, profile_path)


def test_gap_ci_resamples_respondents_not_individual_answers() -> None:
    rows = [["ID.language", "Q1"], ["Language question", "Prompt"]]
    for index in range(100):
        language = "English" if index < 50 else "Spanish"
        text = "short" if language == "English" else ("very long " * 40)
        rows.append([language, text])
    profile = StyleProfile(
        feature_means=(2.0, 1.0, 5.0, 1.0, 0.0),
        feature_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        mid_threshold=2.0,
        high_threshold=3.0,
        n_training=100,
        sources=("asurre_nature_phd_2019",),
    )

    result = analyse(
        nature_frame=pd.DataFrame(rows),
        nature_config={
            "data_starts_row": 2,
            "attributes": {"Q1": {"type": "text", "column_index": 1}},
        },
        nature_items=[_item()],
        style_profile=profile,
        bootstrap_draws=50,
    )

    gap = result["gap_non_english_minus_english"]
    assert gap["response_false_flag_rate_difference"] == 1.0
    assert gap["response_ci95_clustered"] == [1.0, 1.0]
    assert gap["person_any_high_rate_difference"] == 1.0
    assert gap["person_any_high_ci95_clustered"] == [1.0, 1.0]


def test_row_accounting_reports_each_drop_reason() -> None:
    frame = pd.DataFrame(
        [
            ["ID.language", "Q1"],
            ["Language question", "Prompt"],
            ["English", "usable answer"],
            [None, "answer without language"],
            ["Spanish", ""],
        ]
    )
    profile = StyleProfile(
        feature_means=(2.0, 1.0, 5.0, 1.0, 0.0),
        feature_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        mid_threshold=2.0,
        high_threshold=3.0,
        n_training=100,
        sources=("asurre_nature_phd_2019",),
    )

    result = analyse(
        nature_frame=frame,
        nature_config={
            "data_starts_row": 2,
            "attributes": {"Q1": {"type": "text", "column_index": 1}},
        },
        nature_items=[_item()],
        style_profile=profile,
        bootstrap_draws=20,
    )

    assert result["design"]["row_accounting"] == {
        "source_rows": 3,
        "included_rows": 1,
        "dropped_rows": 2,
        "dropped_missing_completion_language": 1,
        "dropped_no_usable_open_text_answers": 1,
    }


def test_identical_groups_have_gap_interval_covering_zero() -> None:
    rows = [["ID.language", "Q1"], ["Language question", "Prompt"]]
    texts = ["short", "very long " * 40] * 40
    for language in ("English", "Spanish"):
        rows.extend([language, text] for text in texts)
    profile = StyleProfile(
        feature_means=(2.0, 1.0, 5.0, 1.0, 0.0),
        feature_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        mid_threshold=2.0,
        high_threshold=3.0,
        n_training=100,
        sources=("asurre_nature_phd_2019",),
    )
    result = analyse(
        nature_frame=pd.DataFrame(rows),
        nature_config={
            "data_starts_row": 2,
            "attributes": {"Q1": {"type": "text", "column_index": 1}},
        },
        nature_items=[_item()],
        style_profile=profile,
        seed=41,
        bootstrap_draws=200,
    )

    gap = result["gap_non_english_minus_english"]
    assert gap["response_ci95_clustered"][0] <= 0 <= gap["response_ci95_clustered"][1]
    assert gap["person_any_high_ci95_clustered"][0] <= 0 <= gap["person_any_high_ci95_clustered"][1]


def test_same_seed_produces_identical_output() -> None:
    rows = [["ID.language", "Q1"], ["Language question", "Prompt"]]
    for index in range(80):
        rows.append(
            [
                "English" if index % 2 == 0 else "Spanish",
                "short" if index % 3 else ("long response " * 20),
            ]
        )
    arguments = {
        "nature_frame": pd.DataFrame(rows),
        "nature_config": {
            "data_starts_row": 2,
            "attributes": {"Q1": {"type": "text", "column_index": 1}},
        },
        "nature_items": [_item()],
        "style_profile": StyleProfile(
            feature_means=(2.0, 1.0, 5.0, 1.0, 0.0),
            feature_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            mid_threshold=2.0,
            high_threshold=3.0,
            n_training=100,
            sources=("asurre_nature_phd_2019",),
        ),
        "seed": 72,
        "bootstrap_draws": 100,
    }

    assert analyse(**arguments) == analyse(**arguments)

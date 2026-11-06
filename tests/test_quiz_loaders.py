from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from adr.data.loaders import load_arc, load_mmlu


def test_loads_mmlu_files_in_stable_subject_order(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    for subject, rows in {
        "world_history": [["History question?", "One", "Two", "Three", "Four", "C"]],
        "astronomy": [["Astronomy question?", "A1", "A2", "A3", "A4", "A"]],
    }.items():
        with (split_dir / f"{subject}_test.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)

    items = load_mmlu(tmp_path, split="test")

    assert [item.item_id for item in items] == [
        "mmlu:astronomy:test:1",
        "mmlu:world_history:test:1",
    ]
    assert items[0].correct_option == 0
    assert items[1].correct_option == 2
    assert all(item.eligible_mechanisms == ["decoy"] for item in items)


def test_mmlu_subject_filter_reports_missing_subjects(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    with (split_dir / "astronomy_test.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["Q?", "A", "B", "C", "D", "A"])

    with pytest.raises(ValueError, match="subjects not found"):
        load_mmlu(tmp_path, split="test", subjects={"astronomy", "virology"})


def test_mmlu_rejects_duplicate_options_case_insensitively(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    with (split_dir / "astronomy_test.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["Q?", "Yes", "yes", "No", "Maybe", "A"])

    with pytest.raises(ValueError, match="options must be unique"):
        load_mmlu(tmp_path, split="test")


def test_loads_normalized_arc_record(tmp_path: Path) -> None:
    path = tmp_path / "ARC-Challenge-Train.jsonl"
    record = {
        "id": "Mercury_SC_1",
        "question": "Which option is correct?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["One", "Two", "Three", "Four"]},
        "answerKey": "B",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    items = load_arc(path, subset="ARC-Challenge")

    assert len(items) == 1
    assert items[0].item_id == "arc:challenge:Mercury_SC_1"
    assert items[0].options == ["One", "Two", "Three", "Four"]
    assert items[0].correct_option == 1
    assert items[0].eligible_mechanisms == ["decoy"]


def test_loads_original_nested_arc_record_with_numeric_labels(tmp_path: Path) -> None:
    path = tmp_path / "ARC-Easy-Test.jsonl"
    record = {
        "id": "question-2",
        "question": {
            "stem": "Pick the second answer.",
            "choices": [
                {"label": "1", "text": "First"},
                {"label": "2", "text": "Second"},
            ],
        },
        "answerKey": "2",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    items = load_arc(path, subset="easy")

    assert items[0].correct_option == 1


def test_arc_rejects_unknown_answer_label(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    record = {
        "id": "bad",
        "question": "Q?",
        "choices": {"label": ["A", "B"], "text": ["One", "Two"]},
        "answerKey": "C",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is not a choice label"):
        load_arc(path)

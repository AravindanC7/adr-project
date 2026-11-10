import json
from pathlib import Path

import pytest

from adr.contracts import HumanDistribution, Item, RenderedItem
from adr.contracts.io import read_jsonl, variants_path
from adr.mechanisms.opinion_trap import OpinionTrapMechanism
from scripts.build_runs_v1 import (
    LOCKED_HELDOUT_ASURRE_SOURCE,
    balanced_slice,
    make_bank,
    parser,
    stable_sample,
    write_bundle,
)


def quiz_item(item_id: str) -> Item:
    return Item(
        item_id=item_id,
        source="mmlu",
        assessment_type="quiz",
        item_type="mcq",
        stem=f"Question {item_id}?",
        options=["A", "B", "C"],
        correct_option=0,
        eligible_mechanisms=["decoy"],
    )


def test_balanced_slice_round_robins_groups():
    items = [quiz_item(f"mmlu:a:test:{n}") for n in range(3)] + [
        quiz_item(f"mmlu:b:test:{n}") for n in range(3)
    ]
    selected = balanced_slice(items, 4, key=lambda item: item.item_id.split(":")[1])
    assert [item.item_id for item in selected] == [
        "mmlu:a:test:0",
        "mmlu:b:test:0",
        "mmlu:a:test:1",
        "mmlu:b:test:1",
    ]


def test_stable_sample_is_reproducible_and_keeps_source_order():
    items = [quiz_item(f"mmlu:a:test:{n}") for n in range(20)]
    first = stable_sample(items, 5, seed=29)
    second = stable_sample(items, 5, seed=29)
    assert first == second
    positions = [items.index(item) for item in first]
    assert positions == sorted(positions)


def test_bundle_writes_valid_plain_and_active_variants(tmp_path: Path):
    quiz = make_bank("quiz", "Quiz", "quiz", [quiz_item("mmlu:a:test:1")], seed=29)
    opinion_item = Item(
        item_id="opinionqa:1:q1",
        source="opinionqa",
        assessment_type="survey",
        item_type="mcq",
        stem="Choose one",
        options=["Yes", "No"],
        eligible_mechanisms=["opinion_trap"],
        human_dist_key="opinionqa:1:q1:overall",
    )
    distribution = HumanDistribution(
        key="opinionqa:1:q1:overall",
        source="opinionqa",
        option_probs=[0.9, 0.1],
        n_respondents=100,
    )
    opinion = OpinionTrapMechanism([distribution])
    survey = make_bank("survey", "Survey", "survey", [opinion_item], seed=29, opinion=opinion)
    manifest = write_bundle(
        tmp_path,
        [quiz, survey],
        [distribution],
        seed=29,
        heldout_asurre_source="asurre_osmi_2017_18",
        requested_sizes={"quiz": 1, "survey": 1},
    )

    quiz_variants = read_jsonl(variants_path(tmp_path, "quiz"), RenderedItem)
    survey_variants = read_jsonl(variants_path(tmp_path, "survey"), RenderedItem)
    assert [variant.mechanism for variant in quiz_variants] == [None, "decoy"]
    assert [variant.mechanism for variant in survey_variants] == [None, "opinion_trap"]
    assert manifest["n_unique_items"] == 2
    assert json.loads((tmp_path / "manifest.json").read_text())["seed"] == 29
    quiz_assessment = tmp_path / "assessments" / "quiz.json"
    assert json.loads(quiz_assessment.read_text(encoding="utf-8"))["assessment_id"] == "quiz"


def test_make_bank_rejects_mismatched_assessment_type():
    survey_item = Item(
        item_id="ipip:e1",
        source="ipip",
        assessment_type="survey",
        item_type="likert",
        stem="I talk a lot.",
        options=["No", "Yes"],
    )
    with pytest.raises(ValueError, match="wrong assessment type"):
        make_bank("bad", "Bad", "assignment", [survey_item], seed=1)


def test_parser_uses_locked_asurre_holdout_by_default():
    args = parser().parse_args(["--datasets", "mmlu", "arc", "ipip"])
    assert args.datasets == ["mmlu", "arc", "ipip"]
    assert args.heldout_asurre_source == LOCKED_HELDOUT_ASURRE_SOURCE


def test_data_decisions_match_locked_asurre_default():
    decisions = json.loads(Path("configs/data_decisions.json").read_text(encoding="utf-8"))
    assert decisions["locked_before_fitting"] is True
    assert decisions["asurre"]["heldout_source"] == LOCKED_HELDOUT_ASURRE_SOURCE
    counts = decisions["asurre"]["source_respondent_counts"]
    assert min(counts, key=counts.get) == LOCKED_HELDOUT_ASURRE_SOURCE

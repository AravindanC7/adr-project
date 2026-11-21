import json
from pathlib import Path

import pytest

from adr.contracts import HumanDistribution, Item, RenderedItem
from adr.contracts.io import read_jsonl, variants_path
from adr.mechanisms.opinion_trap import OpinionTrapMechanism
from scripts.build_runs_v1 import (
    HELDOUT_ASURRE_BANK,
    LOCKED_HELDOUT_ASURRE_SOURCE,
    balanced_slice,
    make_asurre_banks,
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


def test_opinion_variants_are_not_limited_to_opinionqa():
    item = Item(
        item_id="gss:happy",
        source="gss",
        assessment_type="survey",
        item_type="mcq",
        stem="How happy are you?",
        options=["Very", "Pretty", "Not too"],
        eligible_mechanisms=["opinion_trap"],
        human_dist_key="gss:happy:human",
    )
    distribution = HumanDistribution(
        key="gss:happy:human",
        source="gss",
        option_probs=[0.3, 0.6, 0.1],
        n_respondents=1000,
    )

    bank = make_bank(
        "gss",
        "GSS",
        "survey",
        [item],
        seed=29,
        opinion=OpinionTrapMechanism([distribution]),
    )

    assert [variant.mechanism for variant in bank.variants] == [None, "opinion_trap"]


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
    assert args.ipip_limit is None
    assert args.gss_limit is None
    assert args.include_heldout_asurre_bank is False
    assert args.heldout_asurre_only is False


def test_heldout_asurre_bank_uses_all_osmi_items_without_training_leakage():
    def open_item(item_id: str, source: str) -> Item:
        return Item(
            item_id=item_id,
            source=source,
            assessment_type="survey",
            item_type="open_text",
            stem=f"Question {item_id}?",
            eligible_mechanisms=["canary", "style"],
        )

    items = [
        open_item("asurre_nature_phd_2019:q1", "asurre_nature_phd_2019"),
        open_item("asurre_osmi_2017_18:q1", "asurre_osmi_2017_18"),
        open_item("asurre_osmi_2017_18:q2", "asurre_osmi_2017_18"),
    ]
    banks = make_asurre_banks(
        items,
        heldout_source=LOCKED_HELDOUT_ASURRE_SOURCE,
        seed=29,
        train_limit=1,
        include_heldout=True,
    )

    heldout = next(bank for bank in banks if bank.assessment.assessment_id == HELDOUT_ASURRE_BANK)
    assert len(heldout.items) == 2
    assert {item.source for item in heldout.items} == {LOCKED_HELDOUT_ASURRE_SOURCE}
    assert {item.split for item in heldout.items} == {"heldout_survey"}
    assert {item.assessment_type for item in heldout.items} == {"assignment"}
    assert [variant.mechanism for variant in heldout.variants] == [
        None,
        "canary",
        None,
        "canary",
    ]


def test_heldout_asurre_bank_refuses_an_audit_override():
    item = Item(
        item_id="asurre_nature_phd_2019:q1",
        source="asurre_nature_phd_2019",
        assessment_type="survey",
        item_type="open_text",
        stem="Question?",
        eligible_mechanisms=["canary", "style"],
    )
    with pytest.raises(ValueError, match="locked OSMI"):
        make_asurre_banks(
            [item],
            heldout_source="asurre_nature_phd_2019",
            seed=29,
            train_limit=1,
            include_heldout=True,
        )


def test_heldout_asurre_only_omits_training_items():
    items = [
        Item(
            item_id="asurre_nature_phd_2019:q1",
            source="asurre_nature_phd_2019",
            assessment_type="survey",
            item_type="open_text",
            stem="Training question?",
            eligible_mechanisms=["canary", "style"],
        ),
        Item(
            item_id="asurre_osmi_2017_18:q1",
            source="asurre_osmi_2017_18",
            assessment_type="survey",
            item_type="open_text",
            stem="Held-out question?",
            eligible_mechanisms=["canary", "style"],
        ),
    ]
    banks = make_asurre_banks(
        items,
        heldout_source=LOCKED_HELDOUT_ASURRE_SOURCE,
        seed=29,
        train_limit=1,
        include_heldout=True,
        include_train=False,
    )

    assert [bank.assessment.assessment_id for bank in banks] == [HELDOUT_ASURRE_BANK]


def test_parser_accepts_survey_smoke_limits():
    args = parser().parse_args(["--ipip-limit", "10", "--gss-limit", "10"])
    assert args.ipip_limit == 10
    assert args.gss_limit == 10


def test_data_decisions_match_locked_asurre_default():
    decisions = json.loads(Path("configs/data_decisions.json").read_text(encoding="utf-8"))
    assert decisions["locked_before_fitting"] is True
    assert decisions["asurre"]["heldout_source"] == LOCKED_HELDOUT_ASURRE_SOURCE
    counts = decisions["asurre"]["source_respondent_counts"]
    assert min(counts, key=counts.get) == LOCKED_HELDOUT_ASURRE_SOURCE

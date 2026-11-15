from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adr.contracts import HumanDistribution, Item, Mechanism, Response
from adr.harness.registry import get_mechanism
from adr.mechanisms.decoy import DecoyMechanism
from adr.mechanisms.opinion_trap import OpinionTrapMechanism


def _quiz() -> Item:
    return Item(
        item_id="mmlu:astronomy:test:1",
        source="mmlu",
        assessment_type="quiz",
        item_type="mcq",
        stem="Which is closest to the Sun?",
        options=["Venus", "Mercury", "Earth", "Mars"],
        correct_option=1,
        eligible_mechanisms=["decoy"],
    )


def _response(item: Item, variant_id: str, answer: int | None) -> Response:
    return Response(
        session_id="test-session",
        item_id=item.item_id,
        variant_id=variant_id,
        answer=answer,
    )


def test_decoy_is_deterministic_wrong_and_scoreable() -> None:
    item = _quiz()
    mechanism = DecoyMechanism()

    first = mechanism.generate(item, seed=7)
    second = mechanism.generate(item, seed=7)

    assert first == second
    assert first.hidden["decoy_option_index"] != item.correct_option
    assert first.hidden["hidden_text"] in first.copy_text
    assert first.hidden["hidden_position"] == "after_stem"
    assert (
        mechanism.score(
            first, _response(item, first.variant_id, first.hidden["decoy_option_index"]), None
        )
        == "picked_decoy"
    )
    assert (
        mechanism.score(first, _response(item, first.variant_id, item.correct_option), None)
        == "picked_correct"
    )


def test_decoy_accepts_test_items_and_rejects_non_integer_answers() -> None:
    item = _quiz().model_copy(update={"assessment_type": "test"})
    mechanism = DecoyMechanism()
    rendered = mechanism.generate(item, seed=2)

    boolean_response = _response(item, rendered.variant_id, None).model_copy(
        update={"answer": True}
    )
    float_response = _response(item, rendered.variant_id, None).model_copy(
        update={"answer": float(rendered.hidden["decoy_option_index"])}
    )

    assert mechanism.applies_to(item)
    assert mechanism.score(rendered, boolean_response, None) == "picked_other"
    assert mechanism.score(rendered, float_response, None) == "picked_other"


def test_decoy_registry_object_satisfies_contract() -> None:
    mechanism = get_mechanism("decoy")
    assert isinstance(mechanism, Mechanism)


def test_opinion_trap_uses_human_probability_threshold() -> None:
    distribution = HumanDistribution(
        key="opinionqa:w26:Q1:overall",
        source="opinionqa",
        option_probs=[0.01, 0.29, 0.70],
        n_respondents=100,
    )
    item = Item(
        item_id="opinionqa:w26:Q1",
        source="opinionqa",
        assessment_type="survey",
        item_type="mcq",
        stem="Choose an option.",
        options=["Rare", "Sometimes", "Common"],
        eligible_mechanisms=["opinion_trap"],
        human_dist_key=distribution.key,
    )
    mechanism = OpinionTrapMechanism([distribution], implausible_max_probability=0.05)
    rendered = mechanism.generate(item, seed=3)

    assert mechanism.score(rendered, _response(item, rendered.variant_id, 0), None) == "implausible"
    assert mechanism.score(rendered, _response(item, rendered.variant_id, 2), None) == "plausible"
    assert (
        mechanism.score(rendered, _response(item, rendered.variant_id, None), None) == "plausible"
    )
    assert mechanism.score(rendered, _response(item, rendered.variant_id, -1), None) == "plausible"
    assert mechanism.score(rendered, _response(item, rendered.variant_id, 3), None) == "plausible"


def test_opinion_trap_rejects_duplicate_distribution_keys() -> None:
    distribution = HumanDistribution(
        key="opinionqa:w26:Q1:overall",
        source="opinionqa",
        option_probs=[0.5, 0.5],
        n_respondents=2,
    )

    with pytest.raises(ValueError, match="must be unique"):
        OpinionTrapMechanism([distribution, distribution])


def test_opinion_trap_can_load_opinionqa_distributions(tmp_path: Path) -> None:
    wave = tmp_path / "American_Trends_Panel_W26"
    wave.mkdir()
    with (wave / "info.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["key", "question", "references", "option_ordinal"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "key": "Q1",
                "question": "Choose.",
                "references": "['A', 'B', 'Refused']",
                "option_ordinal": "[0, 1]",
            }
        )
    with (wave / "responses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["WEIGHT_W26", "Q1"])
        writer.writeheader()
        writer.writerows(
            [
                {"WEIGHT_W26": "1", "Q1": "A"},
                {"WEIGHT_W26": "1", "Q1": "B"},
            ]
        )

    mechanism = OpinionTrapMechanism.from_opinionqa(tmp_path)

    assert "opinionqa:w26:Q1:overall" in mechanism.distributions


def test_opinion_trap_rejects_distribution_length_mismatch() -> None:
    distribution = HumanDistribution(
        key="opinionqa:w26:Q1:overall",
        source="opinionqa",
        option_probs=[0.4, 0.6],
        n_respondents=10,
    )
    item = Item(
        item_id="opinionqa:w26:Q1",
        source="opinionqa",
        assessment_type="survey",
        item_type="mcq",
        stem="Choose.",
        options=["A", "B", "C"],
        eligible_mechanisms=["opinion_trap"],
        human_dist_key=distribution.key,
    )
    mechanism = OpinionTrapMechanism([distribution])

    try:
        mechanism.generate(item, seed=0)
    except ValueError as exc:
        assert "probability count" in str(exc)
    else:
        raise AssertionError("expected distribution length mismatch to fail")


def test_unconfigured_opinion_trap_registry_object_satisfies_contract() -> None:
    mechanism = get_mechanism("opinion_trap")
    assert isinstance(mechanism, Mechanism)
    assert not mechanism.applies_to(
        Item(
            item_id="opinionqa:x",
            source="opinionqa",
            assessment_type="survey",
            item_type="mcq",
            stem="Q?",
            options=["A", "B"],
            eligible_mechanisms=["opinion_trap"],
            human_dist_key="missing",
        )
    )

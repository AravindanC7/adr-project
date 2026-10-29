from __future__ import annotations

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

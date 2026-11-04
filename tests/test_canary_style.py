from __future__ import annotations

import pytest

from adr.contracts import Item, Response
from adr.mechanisms.canary import MECHANISM as CANARY
from adr.mechanisms.style import StyleMechanism, fit_style_profile


def _open_item() -> Item:
    return Item(
        item_id="handwritten:open-1",
        source="handwritten",
        assessment_type="survey",
        item_type="open_text",
        stem="Describe one thing you value in a workplace.",
        eligible_mechanisms=["canary", "style"],
    )


def _response(variant_id: str, text: str) -> Response:
    return Response(
        session_id="session-1",
        item_id="handwritten:open-1",
        variant_id=variant_id,
        answer=text,
        raw_text=text,
    )


def test_canary_is_deterministic_and_present_in_copy_text() -> None:
    first = CANARY.generate(_open_item(), seed=12)
    second = CANARY.generate(_open_item(), seed=12)

    assert first.hidden["canary_token"] == second.hidden["canary_token"]
    assert first.hidden["canary_token"] in first.copy_text
    assert first.hidden["hidden_position"] == "after_stem"
    assert first.hidden["hidden_text"] not in first.visible["stem"]


def test_canary_scores_followed_leaked_and_clean() -> None:
    rendered = CANARY.generate(_open_item(), seed=4)
    token = rendered.hidden["canary_token"]

    assert (
        CANARY.score(rendered, _response(rendered.variant_id, f"My answer includes {token}."), None)
        == "followed_instruction"
    )
    assert (
        CANARY.score(
            rendered, _response(rendered.variant_id, "I noticed a hidden instruction."), None
        )
        == "leaked"
    )
    assert (
        CANARY.score(
            rendered, _response(rendered.variant_id, "Supportive colleagues matter."), None
        )
        == "clean"
    )


def test_canary_rejects_non_open_item() -> None:
    item = Item(
        item_id="handwritten:mcq-1",
        source="handwritten",
        assessment_type="survey",
        item_type="mcq",
        stem="Choose one.",
        options=["A", "B"],
        eligible_mechanisms=["canary"],
    )

    with pytest.raises(ValueError, match="does not apply"):
        CANARY.generate(item, seed=1)


def test_style_profile_requires_training_data() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        fit_style_profile(["too little data"] * 19)


def test_style_signal_is_inactive_until_calibrated() -> None:
    mechanism = StyleMechanism()

    assert not mechanism.applies_to(_open_item())
    with pytest.raises(RuntimeError, match="configured"):
        mechanism.score(
            CANARY.generate(_open_item(), seed=1),
            _response("handwritten:open-1|canary|1", "A response."),
            None,
        )


def test_style_signal_uses_training_percentiles() -> None:
    training = [
        f"I value clear communication and thoughtful teamwork in project number {index}."
        for index in range(30)
    ]
    profile = fit_style_profile(training)
    mechanism = StyleMechanism(profile)
    rendered = CANARY.generate(_open_item(), seed=2)

    ordinary = mechanism.score(
        rendered,
        _response(
            rendered.variant_id,
            "I value clear communication and thoughtful teamwork in project number 31.",
        ),
        None,
    )
    unusual = mechanism.score(
        rendered,
        _response(rendered.variant_id, "WHY?!?!?! x x x x x x x x x x x x x x x x x x x x"),
        None,
    )

    assert ordinary in {"low", "mid"}
    assert unusual == "high"
    assert profile.n_training == 30


def test_style_never_generates_an_active_variant() -> None:
    mechanism = StyleMechanism(fit_style_profile([f"Training answer {i}." for i in range(20)]))

    with pytest.raises(RuntimeError, match="passive"):
        mechanism.generate(_open_item(), seed=0)

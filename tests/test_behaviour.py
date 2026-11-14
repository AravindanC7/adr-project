"""Behaviour check (#25): telemetry -> paste / focus_loss / too_fast / normal, first match wins."""

import pytest

from adr.contracts import OUTCOME_LABELS, Item, NoObservation, RenderedItem, Response, Telemetry
from adr.harness.copy_paste import plain_variant
from adr.harness.registry import get_mechanism
from adr.mechanisms import behaviour as b
from adr.mechanisms.behaviour import MECHANISM, reading_floor_ms

QUESTION = RenderedItem(  # 11 words: reading floor 1000 + 11 * 100 = 2100 ms
    item_id="x:1",
    mechanism=None,
    variant_id="x:1|plain|0",
    visible={
        "stem": "Which of these is the largest planet in our solar system?",
        "options": ["Mercury", "Venus", "Jupiter", "Mars"],
    },
)
CALM = {
    "time_on_item_ms": 8000,
    "paste_events": 0,
    "focus_loss_count": 0,
    "keystroke_count": 0,
    "mean_interkey_ms": None,
    "first_key_ms": None,
    "away_ms": 0,
    "pasted_chars": 0,
    "edit_keys": 0,
}


def score(answer=2, **telemetry):
    response = Response(session_id="s", item_id="x:1", variant_id="x:1|plain|0", answer=answer)
    return MECHANISM.score(QUESTION, response, Telemetry(**{**CALM, **telemetry}))


def test_plugs_into_the_registry_with_the_contract_labels():
    assert get_mechanism("behaviour") is MECHANISM
    assert MECHANISM.outcomes == OUTCOME_LABELS["behaviour"]
    with pytest.raises(TypeError, match="passive"):
        MECHANISM.generate(None, 0)


def test_a_calm_answer_is_normal():
    assert score() == "normal"
    typed = "Jupiter, because it is more than twice as heavy as the rest."
    assert (
        score(typed, keystroke_count=len(typed) + 4, mean_interkey_ms=140, edit_keys=4) == "normal"
    )


def test_reading_floor_counts_passage_and_question_not_options():
    assert reading_floor_ms(QUESTION) == 2100
    assert reading_floor_ms(QUESTION, passage="a b c") == 2400


def test_a_real_items_passage_reaches_the_reading_floor():
    # Variants never carry the passage (the harness and page add it), so score() takes it.
    item = Item(
        item_id="x:2",
        source="handwritten",
        assessment_type="quiz",
        item_type="mcq",
        passage=" ".join(["word"] * 40),  # 40 words: 4000 ms more to read
        stem="What is the passage about?",  # 5 words
        options=["a", "b"],
        correct_option=0,
    )
    rendered = plain_variant(item)
    assert "passage" not in rendered.visible
    assert reading_floor_ms(rendered, item.passage) == 1000 + 45 * 100
    response = Response(session_id="s", item_id="x:2", variant_id=rendered.variant_id, answer=0)
    quick = Telemetry(**{**CALM, "time_on_item_ms": 3000})  # enough for the stem alone
    assert MECHANISM.score(rendered, response, quick) == "normal"
    assert MECHANISM.score(rendered, response, quick, passage=item.passage) == "too_fast"


@pytest.mark.parametrize(
    "telemetry, expected",
    [
        ({"paste_events": 1, "pasted_chars": 120}, "paste"),
        ({"paste_events": 1, "pasted_chars": None}, "paste"),  # length not measured: any paste
        ({"paste_events": 1, "pasted_chars": 7}, "normal"),  # a pasted word is ordinary
        ({"focus_loss_count": 1, "away_ms": 15000}, "focus_loss"),
        ({"focus_loss_count": 1, "away_ms": None}, "normal"),  # length unknown: not proof
        ({"focus_loss_count": 1, "away_ms": 1999}, "normal"),
        ({"focus_loss_count": 1, "away_ms": 2000}, "focus_loss"),
        ({"focus_loss_count": 2, "away_ms": 900}, "normal"),  # brief blurs, straight back
        ({"time_on_item_ms": 2099}, "too_fast"),
        ({"time_on_item_ms": 2100}, "normal"),
    ],
)
def test_each_label(telemetry, expected):
    assert score(**telemetry) == expected


def test_first_match_wins():
    everything = {
        "paste_events": 1,
        "pasted_chars": 300,
        "focus_loss_count": 1,
        "away_ms": 20000,
        "time_on_item_ms": 500,
    }
    assert score(**everything) == "paste"
    assert score(**{**everything, "paste_events": 0, "pasted_chars": 0}) == "focus_loss"
    assert score(time_on_item_ms=500) == "too_fast"


def test_text_that_was_never_typed_counts_as_paste():
    answer = "Jupiter is the largest planet in the solar system by far."
    assert score(answer, keystroke_count=3) == "paste"  # set by a script
    assert (
        score(answer, keystroke_count=len(answer) * 6 // 10) == "normal"
    )  # autocorrect, autocomplete
    assert score("Jupiter", keystroke_count=0) == "normal"  # too short to judge


def test_superhuman_typing_is_too_fast():
    answer = "Jupiter, the gas giant."
    assert score(answer, keystroke_count=len(answer), mean_interkey_ms=8) == "too_fast"
    assert score(answer, keystroke_count=len(answer), mean_interkey_ms=60) == "normal"
    assert score("Jup", keystroke_count=3, mean_interkey_ms=8) == "normal"  # too few keys


def test_no_telemetry_is_no_observation():
    response = Response(session_id="s", item_id="x:1", variant_id="x:1|plain|0", answer=2)
    with pytest.raises(NoObservation):
        MECHANISM.score(QUESTION, response, None)


def test_thresholds_are_named_constants_for_calibration():
    # #42 calibrates these on held-out data; they must stay module-level and documented.
    for name in (
        "MIN_PASTE_CHARS",
        "UNTYPED_MIN_CHARS",
        "UNTYPED_COVERAGE",
        "MIN_AWAY_MS",
        "READ_MS_PER_WORD",
        "ANSWER_MS",
        "MIN_INTERKEY_MS",
        "MIN_KEYS_FOR_SPEED",
    ):
        assert hasattr(b, name)

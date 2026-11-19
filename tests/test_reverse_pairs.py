"""The shared IPIP reverse-keyed pair rule (adr/mechanisms/opinion_trap.py).

The same pair of answers must give the same opinion-trap outcome on the human side (the pilot
cleaner, server/clean.py) and on the AI side (score_reverse_pairs, used for the harness runs).
"""

import itertools

import pytest

from adr.contracts import Item
from adr.mechanisms.opinion_trap import pair_outcome, pair_side, score_reverse_pairs
from server.clean import clean, pair_leaning
from tests.test_clean import BANK, by_check, full, session, setup  # noqa: F401  (fixture)

ITEMS = {d["item_id"]: Item.model_validate(d) for d in BANK["items"]}
# Option indices 0-4 = the 5 agree-scale points (2 = Neutral), plus missing / invalid answers.
ANSWERS = [0, 1, 2, 3, 4, None, "4", 7, -1, 2.0]
PLAUSIBLE = {"ipip:e1": "plausible", "ipip:e2": "plausible"}


@pytest.mark.parametrize("answer", ANSWERS)
def test_pair_side_matches_the_pilot_cleaner(answer):
    assert pair_side(answer, 5) == pair_leaning(answer, 5)


@pytest.mark.parametrize("e1, e2", list(itertools.product(ANSWERS, ANSWERS)))
@pytest.mark.parametrize("e2_first", [False, True])
def test_same_pairs_give_the_same_result_on_both_sides(setup, e1, e2, e2_first):  # noqa: F811
    data, bank = setup
    answers = full(e1=e1, e2=e2)
    if e2_first:  # the pilot judges the SECOND-answered item, so answer order matters
        answers = {k: answers[k] for k in ("arc:q1", "opinionqa:w41:X", "ipip:e2", "ipip:e1",
                                           "handwritten:o1")}
    session(data, "a" * 32, answers)
    human = by_check(clean(data, bank, origin="pilot", run_id="r")[0])

    order = [k for k in answers if k.startswith("ipip:")]
    ai = score_reverse_pairs(ITEMS, [(k, answers[k][0]) for k in order], PLAUSIBLE)
    for item_id in ("ipip:e1", "ipip:e2"):
        assert human.get((item_id, "opinion_trap")) == ai[item_id], (item_id, e1, e2)


@pytest.mark.parametrize("a, b, expected", [
    (4, 0, "plausible"),       # opposite raw sides: consistent, keeps its own outcome
    (0, 4, "plausible"),
    (4, 3, "contradiction"),   # same side
    (0, 1, "contradiction"),
    (2, 4, None),              # neutral -> no observation
    (None, 4, None),           # missing
    ("4", 0, None),            # non-integer
    (5, 0, None),              # out of range
])
def test_pair_rule(a, b, expected):
    assert pair_outcome(a, b, 5, "plausible") == expected


def test_implausible_wins_over_contradiction_but_neutral_still_drops_the_row():
    assert pair_outcome(4, 4, 5, "implausible") == "implausible"
    assert pair_outcome(4, 0, 5, "implausible") == "implausible"
    assert pair_outcome(2, 4, 5, "implausible") is None


def test_only_the_second_answered_item_is_judged_and_unpaired_items_pass_through():
    single = {"ipip:e1": "plausible", "ipip:e2": "plausible", "opinionqa:w41:X": "implausible"}
    out = score_reverse_pairs(
        ITEMS, [("opinionqa:w41:X", 2), ("ipip:e1", 4), ("ipip:e2", 4)], single)
    assert out == {"ipip:e1": "plausible", "ipip:e2": "contradiction",
                   "opinionqa:w41:X": "implausible"}
    # partner never answered in this session -> no pair, single-item outcome kept
    assert score_reverse_pairs(ITEMS, [("ipip:e2", 4)], {"ipip:e2": "plausible"}) == {
        "ipip:e2": "plausible"}


def test_an_answer_outside_single_is_context_only():
    # e1 answered (e.g. on a plain variant) but has no opinion-trap row: it still pairs with e2
    out = score_reverse_pairs(ITEMS, [("ipip:e1", 0), ("ipip:e2", 1)], {"ipip:e2": "plausible"})
    assert out == {"ipip:e2": "contradiction"}

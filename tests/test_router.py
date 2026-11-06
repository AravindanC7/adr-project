import math
import statistics

import pytest

from adr.baselines.simple import RandomCheckRouter, SingleMechanismRouter
from adr.contracts import MapCell, Router, make_variant_id
from adr.router.core import RouterConfig, SPRTRouter
from adr.router.selection import expected_info_gain, false_flag_risk
from adr.router.sprt import capped_delta, wald_thresholds
from adr.router.synthetic import (
    detection_at_fpr,
    simulate_session,
    synthetic_bank,
    synthetic_failure_map,
)


@pytest.fixture(scope="module")
def setup():
    fmap = synthetic_failure_map()
    assessment, items, variants = synthetic_bank()
    return fmap, assessment, items, variants


def test_wald_thresholds():
    a, b = wald_thresholds(0.01, 0.2)
    assert math.isclose(a, math.log(80))
    assert math.isclose(b, math.log(0.2 / 0.99))


def test_capped_delta_limits_one_mechanism():
    assert capped_delta(0.0, 2.0, 3.0) == 2.0
    assert math.isclose(capped_delta(2.0, 2.0, 3.0), 1.0)
    assert capped_delta(3.0, 1.0, 3.0) == 0.0
    assert math.isclose(capped_delta(3.0, -1.0, 3.0), -1.0)
    assert capped_delta(5.0, 1.0, None) == 1.0


def test_info_gain_prefers_strong_checks():
    strong = MapCell.from_counts("decoy", "mcq", {"picked_decoy": 90, "picked_correct": 10},
                                 {"picked_decoy": 2, "picked_correct": 98})
    weak = MapCell.from_counts("decoy", "mcq", {"picked_decoy": 52, "picked_correct": 48},
                               {"picked_decoy": 48, "picked_correct": 52})
    assert expected_info_gain(0.5, strong) > 10 * expected_info_gain(0.5, weak)
    assert expected_info_gain(0.999, strong) < expected_info_gain(0.5, strong)
    assert 0 < false_flag_risk(strong) < 0.1


def test_chernoff_objective_keeps_value_when_confident():
    from adr.router.selection import expected_evidence

    cell = MapCell.from_counts("opinion_trap", "likert",
                               {"implausible": 22, "contradiction": 15, "plausible": 63},
                               {"implausible": 9, "contradiction": 8, "plausible": 83})
    # Information gain collapses near certainty; expected evidence does not.
    assert expected_info_gain(0.97, cell) < 0.2 * expected_info_gain(0.5, cell)
    assert expected_evidence(0.97, cell) > 0.8 * expected_evidence(0.5, cell)


def test_router_satisfies_protocol(setup):
    fmap, _, items, variants = setup
    assert isinstance(SPRTRouter(items, variants, fmap), Router)


def test_first_choice_is_a_strong_check_and_deterministic(setup):
    fmap, assessment, items, variants = setup
    r = SPRTRouter(items, variants, fmap)
    _, c1 = r.start(assessment, "s", seed=1)
    _, c2 = r.start(assessment, "s", seed=1)
    assert c1 == c2
    assert c1.mechanism in {"decoy", "canary"}  # the two strongest synthetic checks


def test_ties_are_broken_at_random_across_seeds(setup):
    fmap, assessment, items, variants = setup
    r = SPRTRouter(items, variants, fmap)
    firsts = {r.start(assessment, f"s{i}", seed=i)[1].item_id for i in range(30)}
    assert len(firsts) > 3  # 8 equivalent decoy items: an attacker can't predict which


def test_update_moves_score_and_status(setup):
    fmap, assessment, items, variants = setup
    r = SPRTRouter(items, variants, fmap)
    state, choice = r.start(assessment, "s", seed=0)
    bad = "picked_decoy" if choice.mechanism == "decoy" else "followed_instruction"
    s2 = r.update(state, choice.variant_id, bad)
    assert s2.score > state.score and state.score == 0.0  # input state not mutated
    assert s2.served == [choice.variant_id]
    assert s2.history[-1][0] == choice.variant_id
    s3 = r.update(state, choice.variant_id, "picked_correct" if choice.mechanism == "decoy" else "clean")
    assert s3.score < 0


@pytest.mark.parametrize("cls", [SPRTRouter, RandomCheckRouter])
def test_outcome_none_means_shown_with_nothing_to_score(setup, cls):
    fmap, assessment, items, variants = setup
    r = cls(items, variants, fmap)
    state, _ = r.start(assessment, "s", seed=0)
    state = r.update(state, make_variant_id("handwritten:mcq0", "decoy"), "picked_decoy")
    trapped = make_variant_id("handwritten:mcq1", "decoy")
    after = r.update(state, trapped, None)                  # NoObservation on a trapped variant
    assert trapped in after.served                          # the real variant, not a plain one
    assert after.score == state.score and after.status == state.status
    assert after.contrib == state.contrib
    assert after.history[-1] == (trapped, "none", 0.0)
    while (choice := r.next(after)) is not None:            # never offered again
        assert choice.item_id != "handwritten:mcq1"
        after = r.update(after, choice.variant_id, None)


def test_review_ends_session_and_cleared_serves_plain(setup):
    fmap, assessment, items, variants = setup
    r = SPRTRouter(items, variants, fmap, RouterConfig(mech_cap_frac=None, beta=0.2))
    state, _ = r.start(assessment, "s", seed=0)
    for i in range(3):
        state = r.update(state, make_variant_id(f"handwritten:mcq{i}", "decoy"), "picked_decoy")
    assert state.status == "review" and r.next(state) is None

    state, _ = r.start(assessment, "h", seed=0)
    for i in range(3):
        state = r.update(state, make_variant_id(f"handwritten:open{i}", "canary"), "clean")
    state = r.update(state, make_variant_id("handwritten:mcq0", "decoy"), "picked_correct")
    assert state.status == "cleared"
    assert r.next(state).mechanism is None


def test_mech_cap_bounds_one_checks_total(setup):
    fmap, assessment, items, variants = setup
    r = SPRTRouter(items, variants, fmap, RouterConfig(mech_cap_frac=0.4))
    assert math.isclose(r.mech_cap, 0.4 * r.A)
    state, _ = r.start(assessment, "s", seed=0)
    for i in range(4):
        state = r.update(state, make_variant_id(f"handwritten:open{i}", "style"), "high")
    assert math.isclose(state.contrib["style"], r.mech_cap)
    assert math.isclose(state.score, r.mech_cap)


def test_mech_cap_is_a_fraction_of_a_and_follows_the_upper_override(setup):
    fmap, assessment, items, variants = setup
    default = SPRTRouter(items, variants, fmap)
    a, _ = wald_thresholds(0.01, 0.05)
    assert math.isclose(default.mech_cap, 0.65 * a)                  # ~2.96 at A = 4.55
    assert math.isclose(0.65 * wald_thresholds(0.01, 0.01)[0], 3.0, abs_tol=0.02)  # old 3.0
    moved = SPRTRouter(items, variants, fmap, RouterConfig(upper=8.0))
    assert moved.A == 8.0 and math.isclose(moved.mech_cap, 0.65 * 8.0)
    assert SPRTRouter(items, variants, fmap, RouterConfig(mech_cap_frac=None)).mech_cap is None
    with pytest.raises(ValueError, match="mech_cap_frac"):
        SPRTRouter(items, variants, fmap, RouterConfig(mech_cap_frac=0.0))


def test_every_item_is_shown_once(setup):
    fmap, assessment, items, variants = setup
    r = SPRTRouter(items, variants, fmap, RouterConfig(alpha=1e-9, beta=1e-9))  # never stops
    res = simulate_session(r, assessment, fmap, "human", seed=3)
    assert res.items_shown == len(assessment.item_ids)
    assert len({v.split("|")[0] for v in res.final.served}) == len(assessment.item_ids)


def test_router_beats_random_and_single_check_on_synthetic_map(setup):
    fmap, assessment, items, variants = setup
    n = 300

    def run(router):
        ai = [simulate_session(router, assessment, fmap, "ai", s) for s in range(n)]
        hu = [simulate_session(router, assessment, fmap, "human", 10_000 + s) for s in range(n)]
        tpr, _, _ = detection_at_fpr([x.final.score for x in ai],
                                     [x.final.score for x in hu], 0.05)
        flagged_h = sum(x.final.status == "review" for x in hu) / n
        checks_h = statistics.mean(x.checks_run for x in hu)
        return tpr, flagged_h, checks_h

    cfg = RouterConfig()
    router = run(SPRTRouter(items, variants, fmap, cfg))
    rnd = run(RandomCheckRouter(items, variants, fmap, cfg))
    weak = run(SingleMechanismRouter(items, variants, fmap, cfg, mechanism="opinion_trap"))
    # Similar detection to checking every item at random, with well under half the checks
    assert router[0] >= rnd[0] - 0.05
    assert router[2] < 0.5 * rnd[2]
    assert router[0] > weak[0] + 0.3
    assert router[1] <= 0.02  # honest people paused for review stays near alpha

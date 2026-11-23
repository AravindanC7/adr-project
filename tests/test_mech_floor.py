"""Each mechanism's (or cap group's) running total is bounded to [B, +cap] (team decision, Nov 18).

+cap < A, so one check alone can't pause a session; the floor is B, read from the router's own
config, so one check alone CAN clear a session but can't push its total below B.
"""

import math
import random
from dataclasses import replace

import pytest

from adr.baselines.policies import make_policy
from adr.contracts import Assessment, FailureMap, Item, MapCell, RenderedItem, make_variant_id
from adr.router.core import RouterConfig, SPRTRouter
from adr.router.sprt import capped_delta, wald_thresholds
from adr.router.synthetic import sample_outcome, synthetic_bank, synthetic_failure_map

# One strong two-outcome decoy cell: llr(picked_decoy) = +ln 9.9 ~ 2.29, llr(picked_correct) =
# ln(10/99 * ... ) ~ -2.29 per check, so a few checks hit either bound.
_DECOY = MapCell.from_counts(
    "decoy", "mcq", {"picked_decoy": 90, "picked_correct": 10}, {"picked_decoy": 9, "picked_correct": 91}
)
_PASSIVE = [
    MapCell.from_counts("behaviour", "mcq", {"paste": 90, "normal": 10}, {"paste": 9, "normal": 91}),
    MapCell.from_counts("style", "mcq", {"high": 90, "low": 10}, {"high": 9, "low": 91}),
]


def _router(cfg: RouterConfig, n: int = 12) -> tuple[SPRTRouter, Assessment]:
    items = {
        f"arc:f{i}": Item(
            item_id=f"arc:f{i}", source="arc", assessment_type="quiz", item_type="mcq",
            stem=f"q{i}", options=["a", "b", "c", "d"], correct_option=0,
            eligible_mechanisms=["decoy"],
        )
        for i in range(n)
    }
    fmap = FailureMap(version="t", cells=[_DECOY, *_PASSIVE])
    router = SPRTRouter(items, {i: [] for i in items}, fmap, cfg)
    assessment = Assessment(
        assessment_id="a", title="t", assessment_type="quiz", item_ids=list(items)
    )
    return router, assessment


def _feed(router, assessment, mech, outcome, k):
    state, _ = router.start(assessment, "s", 0)
    for i in range(k):
        state = router.update(state, make_variant_id(f"arc:f{i}", mech), outcome)
    return state


def test_capped_delta_bounds_to_floor_and_cap():
    assert capped_delta(0.0, -5.0, 3.0, floor=-4.0) == -4.0
    assert math.isclose(capped_delta(-3.5, -1.0, 3.0, floor=-4.0), -0.5)
    assert capped_delta(-4.0, -1.0, 3.0, floor=-4.0) == 0.0
    assert capped_delta(0.0, 5.0, 3.0, floor=-4.0) == 3.0
    assert capped_delta(0.0, -5.0, 3.0) == -3.0          # no floor given: the old symmetric bound
    assert capped_delta(0.0, -5.0, None, floor=-4.0) == -5.0  # no cap: no bound at all


def test_a_one_positive_check_alone_cannot_reach_review():
    router, assessment = _router(RouterConfig())
    state = _feed(router, assessment, "decoy", "picked_decoy", 12)
    assert state.contrib["decoy"] == pytest.approx(router.mech_cap)
    assert state.score == pytest.approx(router.mech_cap) and state.score < router.A
    assert state.status != "review"


def test_b_one_negative_check_alone_can_clear():
    router, assessment = _router(RouterConfig())
    b = wald_thresholds(0.01, 0.05)[1]
    assert router.mech_cap < -router.B  # 2.96 < 2.99: under the old symmetric cap this never cleared
    state = _feed(router, assessment, "decoy", "picked_correct", 2)
    assert state.status == "cleared"
    assert state.score == pytest.approx(b)
    assert router.next(state).mechanism is None  # cleared: no more checks


def test_c_negative_floor_holds_and_follows_calibrated_b():
    router, assessment = _router(replace(RouterConfig(), clear=False))  # keep feeding past B
    state = _feed(router, assessment, "decoy", "picked_correct", 12)
    assert state.contrib["decoy"] == pytest.approx(router.B)
    assert min(state.contrib.values()) >= router.B - 1e-9
    # B from the current config: a calibrated lower threshold moves the floor with it.
    moved, assessment = _router(replace(RouterConfig(), lower=-1.5, clear=False))
    assert moved.mech_floor == -1.5
    state = _feed(moved, assessment, "decoy", "picked_correct", 12)
    assert state.contrib["decoy"] == pytest.approx(-1.5)


def test_noclear_keeps_the_floor_and_only_skips_clearing():
    router, assessment = _router(RouterConfig())
    noclear = make_policy("router_noclear", assessment, router.items, router.variants, router.fmap)
    assert noclear.B == router.B and noclear.mech_floor == router.B
    state = _feed(noclear, assessment, "decoy", "picked_correct", 6)
    assert state.status == "active" and state.contrib["decoy"] == pytest.approx(router.B)


def test_no_cap_ablation_stays_unbounded():
    router, assessment = _router(RouterConfig(mech_cap_frac=None, clear=False))
    assert router.mech_floor is None
    state = _feed(router, assessment, "decoy", "picked_correct", 4)
    assert state.score < router.B - 4


def test_e_group_caps_use_the_same_bounds():
    cfg = RouterConfig(clear=False, mech_group_caps={"passive": ("behaviour", "style")})
    router, assessment = _router(cfg)
    state, _ = router.start(assessment, "s", 0)
    for i in range(6):
        state = router.update(state, make_variant_id(f"arc:f{i}", "behaviour"), "normal")
        state = router.update(state, make_variant_id(f"arc:f{i}", "style"), "low")
    assert state.contrib == {"passive": pytest.approx(router.B)}
    for i in range(6):
        state = router.update(state, make_variant_id(f"arc:f{i}", "behaviour"), "paste")
        state = router.update(state, make_variant_id(f"arc:f{i}", "style"), "high")
    assert state.contrib == {"passive": pytest.approx(router.mech_cap)}
    # and with clearing on, the group alone can clear
    router, assessment = _router(replace(cfg, clear=True))
    state = _feed(router, assessment, "style", "low", 2)
    assert state.status == "cleared"


# --------------------------------------------------------------- (d) stealth AI
def _stealth_statuses(router, assessment, fmap, n: int) -> list[str]:
    """A synthetic AI that answers decoys like a human but is AI-like on every other check."""
    out = []
    for seed in range(n):
        rng = random.Random(seed)
        state, choice = router.start(assessment, f"stealth-{seed}", seed)
        while choice is not None:
            outcome = "none"
            if choice.mechanism:
                cell = fmap.lookup(choice.mechanism, router.items[choice.item_id].item_type)
                kind = "human" if choice.mechanism == "decoy" else "ai"
                outcome = sample_outcome(cell, kind, rng)
            state = router.update(state, choice.variant_id, outcome)
            choice = router.next(state)
        out.append(state.status)
    return out


def test_d_stealth_ai_is_not_cleared_more_than_under_the_symmetric_cap():
    fmap = synthetic_failure_map()
    assessment, items, variants = synthetic_bank(n_mcq=20, n_likert=0, n_open=3)  # decoy-heavy
    n = 400

    def cleared(router) -> float:
        return _stealth_statuses(router, assessment, fmap, n).count("cleared") / n

    floor = SPRTRouter(items, variants, fmap, RouterConfig())
    symmetric = SPRTRouter(items, variants, fmap, RouterConfig())
    symmetric.mech_floor = None  # the old [-cap, +cap] bound
    unbounded = SPRTRouter(items, variants, fmap, RouterConfig(mech_cap_frac=None))
    c_floor, c_sym, c_unb = cleared(floor), cleared(symmetric), cleared(unbounded)
    assert floor.mech_floor == floor.B and symmetric.mech_floor is None
    assert c_floor <= c_sym + 0.02, (c_floor, c_sym)
    assert c_floor < c_unb, (c_floor, c_unb)


def test_trace_cleared_at_counts_only_sessions_that_reached_b_first():
    import numpy as np

    from adr.replay.crossfit import Trace

    run = np.array([0.5, 1.0, -3.0])
    cleared = Trace("s", 0, 1.0, run, np.array([1, 2, 3]), 3, cleared=True)
    assert cleared.cleared_at(2.0)        # paused at 2.0? never: it cleared at B first
    assert not cleared.cleared_at(1.0)    # deployed at 1.0 it pauses before clearing
    assert not Trace("t", 0, 1.0, run, np.array([1, 2, 3]), 3).cleared_at(2.0)


def test_rates_report_ai_sessions_wrongly_cleared():
    import numpy as np

    from adr.replay.crossfit import Trace, rates

    def t(who, score, cleared):
        return Trace(who, 0, score, np.array([score]), np.array([1]), 1, cleared=cleared)

    ai = [[t("ai|a", 0.2, True), t("ai|a", 0.2, False)], [t("ai|b", 3.0, False)]]
    r = rates([t("h", 0.0, True)], ai, [0.0], {"A": 2.0})["A"]
    assert r["ai_cleared_rate"] == pytest.approx(0.25)  # session a: cleared under 1 of 2 maps
    assert r["honest_cleared_rate"] == 1.0


# ---------------------------------------------------------------- clear_policy: reopen | sticky
def _clear_then_style(policy: str, n_high: int = 2):
    """One clean canary-like negative check clears the session; then passive 'high' style
    evidence arrives (as on every open-text answer)."""
    router, assessment = _router(replace(RouterConfig(), clear_policy=policy))
    state, _ = router.start(assessment, "s", 0)
    state = router.update(state, make_variant_id("arc:f0", "decoy"), "picked_correct")
    state = router.update(state, make_variant_id("arc:f1", "decoy"), "picked_correct")
    assert state.status == "cleared"
    for i in range(n_high):
        state = router.update(state, make_variant_id(f"arc:f{2 + i}", "style"), "high")
    return router, state


def test_reopen_is_the_default_and_lets_later_evidence_reopen_a_cleared_session():
    assert RouterConfig().clear_policy == "reopen"
    router, state = _clear_then_style("reopen", n_high=1)
    assert state.score > router.B and state.status == "active"  # reopened: checks again


def test_sticky_keeps_a_cleared_session_cleared():
    router, state = _clear_then_style("sticky", n_high=1)
    assert state.score > router.B and state.status == "cleared"
    assert router.next(state).mechanism is None  # plain items only


def test_sticky_review_still_wins():
    # no cap, so passive evidence alone can climb from B to A = 1.0
    cfg = replace(RouterConfig(), clear_policy="sticky", upper=1.0, mech_cap_frac=None)
    router, assessment = _router(cfg)
    state, _ = router.start(assessment, "s", 0)
    state = router.update(state, make_variant_id("arc:f0", "decoy"), "picked_correct")
    state = router.update(state, make_variant_id("arc:f1", "decoy"), "picked_correct")
    assert state.status == "cleared"
    for i in range(2, 8):
        state = router.update(state, make_variant_id(f"arc:f{i}", "style"), "high")
        if state.status == "review":
            break
    assert state.score >= router.A and state.status == "review"


def test_unknown_clear_policy_is_refused():
    with pytest.raises(ValueError, match="clear_policy"):
        _router(replace(RouterConfig(), clear_policy="maybe"))


def test_trace_ever_cleared_vs_cleared_at_the_end():
    import numpy as np

    from adr.replay.crossfit import Trace

    # cleared at step 1 (score -3.0), reopened by passive evidence, ends active at 0.4
    reopened = Trace("s", 0, 0.4, np.array([-1.0, -3.0, -1.5, 0.4]), np.array([1, 2, 2, 2]), 2,
                     cleared=False, cleared_step=1)
    assert reopened.ever_cleared_at(2.0) and not reopened.cleared_at(2.0)
    assert reopened.ever_cleared_at(0.3)       # cleared before it ever reached 0.3
    # paused before it would clear: not "ever cleared" at that threshold
    late = Trace("t", 0, 1.0, np.array([1.0, -3.0]), np.array([1, 2]), 2, cleared=True,
                 cleared_step=1)
    assert not late.ever_cleared_at(0.5) and late.ever_cleared_at(2.0)
    never = Trace("u", 0, 0.0, np.array([0.0]), np.array([0]), 0)
    assert not never.ever_cleared_at(1.0) and not never.cleared_at(1.0)


def test_replay_records_the_first_cleared_step_even_when_reopened():
    from adr.replay.donors import Donors, Group
    from adr.replay.simulate import replay_session

    class Script:  # passive style: "low" on the first two items (clears), then "high"
        group, session_id, fallbacks, synthetic = Group.human("x"), "d", 0, 0

        def outcome(self, item_id, mechanism):
            assert mechanism == "style"
            return "low" if item_id in ("arc:f0", "arc:f1") else "high"

    assert Donors  # (the real Donors is exercised elsewhere)
    ends = {}
    for policy in ("reopen", "sticky"):
        router, assessment = _router(replace(RouterConfig(), clear_policy=policy), n=6)
        res = replay_session(router, assessment, Script(), passive=("style",), seed=0)
        assert res.cleared_step is not None
        assert res.final.history[res.cleared_step][0] == make_variant_id("arc:f1", "style")
        ends[policy] = res.final.status
    assert ends["sticky"] == "cleared"  # stayed cleared
    assert ends["reopen"] in ("active", "escalated")  # reopened: no longer cleared


# ---------------------------------------------------------------- two-check clearing
def test_neg_floor_frac_default_keeps_the_floor_at_b():
    router, _ = _router(RouterConfig())
    assert RouterConfig().neg_floor_frac == 1.0 and router.mech_floor == router.B


def test_two_check_clearing_one_check_alone_cannot_reach_b_but_two_can():
    cfg = replace(RouterConfig(), neg_floor_frac=0.65)
    router, assessment = _router(cfg)
    assert router.mech_floor == pytest.approx(0.65 * router.B)
    # one check alone, as human-looking as it gets: stuck at 0.65 * B, never cleared
    state = _feed(router, assessment, "decoy", "picked_correct", 12)
    assert state.contrib["decoy"] == pytest.approx(0.65 * router.B)
    assert state.score > router.B and state.status != "cleared"
    # a second check's human-looking evidence takes it past B: cleared
    for i in range(3):
        state = router.update(state, make_variant_id(f"arc:f{i}", "style"), "low")
        if state.status == "cleared":
            break
    assert state.status == "cleared" and state.score <= router.B + 1e-9
    assert state.contrib["style"] >= 0.65 * router.B - 1e-9  # each check within its own floor


def test_two_check_clearing_applies_to_cap_groups_too():
    cfg = replace(
        RouterConfig(),
        neg_floor_frac=0.65,
        clear=False,
        mech_group_caps={"passive": ("behaviour", "style")},
    )
    router, assessment = _router(cfg)
    state, _ = router.start(assessment, "s", 0)
    for i in range(6):  # two passive checks, but one group: one floor
        state = router.update(state, make_variant_id(f"arc:f{i}", "behaviour"), "normal")
        state = router.update(state, make_variant_id(f"arc:f{i}", "style"), "low")
    assert state.contrib == {"passive": pytest.approx(0.65 * router.B)}
    assert state.score > router.B


def test_neg_floor_frac_must_be_positive():
    with pytest.raises(ValueError, match="neg_floor_frac"):
        _router(replace(RouterConfig(), neg_floor_frac=0.0))


# ---------------------------------------------------------------- pre-registered ablation:
# --clear-requires-style-human (configs/data_decisions.json preregistered_ablations)
def test_clear_requires_style_human_is_off_by_default_and_leaves_clearing_unchanged():
    assert RouterConfig().clear_requires_style_human is False
    seqs = [("decoy", "picked_correct")] * 12 + [("style", "high"), ("style", "low")]
    for cfg in (RouterConfig(), replace(RouterConfig(), clear_requires_style_human=False)):
        router, assessment = _router(cfg)
        state, _ = router.start(assessment, "s", 0)
        statuses = []
        for i, (mech, outcome) in enumerate(seqs):
            state = router.update(state, make_variant_id(f"arc:f{i % 12}", mech), outcome)
            statuses.append(state.status)
        assert statuses[1] == "cleared"  # decoy alone clears, with no style evidence (as #51)
        assert "cleared" in statuses


@pytest.mark.parametrize("style_first", [None, "high"])
def test_clearing_is_blocked_while_the_style_total_is_not_below_zero(style_first):
    router, assessment = _router(replace(RouterConfig(), clear_requires_style_human=True))
    state, _ = router.start(assessment, "s", 0)
    if style_first:  # style total > 0
        state = router.update(state, make_variant_id("arc:f0", "style"), style_first)
        assert state.contrib["style"] > 0
    for i in range(12):  # human-looking decoy answers: the score reaches B
        state = router.update(state, make_variant_id(f"arc:f{i}", "decoy"), "picked_correct")
        assert state.status != "cleared"
    assert state.contrib.get("style", 0.0) >= 0
    if style_first is None:
        assert state.score <= router.B + 1e-9  # would have cleared without the rule
    assert state.status in ("active", "escalated")


def test_clearing_is_allowed_once_the_style_total_is_below_zero():
    router, assessment = _router(replace(RouterConfig(), clear_requires_style_human=True))
    state = _feed(router, assessment, "decoy", "picked_correct", 12)
    assert state.score <= router.B + 1e-9 and state.status == "active"  # blocked: no style yet
    state = router.update(state, make_variant_id("arc:f0", "style"), "low")
    assert state.contrib["style"] < 0
    assert state.status == "cleared"


def test_a_session_blocked_from_clearing_keeps_getting_checks():
    """Blocked: the session is not cleared, so next() still offers checks, not only plain items."""
    items = {
        f"arc:f{i}": Item(
            item_id=f"arc:f{i}", source="arc", assessment_type="quiz", item_type="mcq",
            stem=f"q{i}", options=["a", "b", "c", "d"], correct_option=0,
            eligible_mechanisms=["decoy"],
        )
        for i in range(12)
    }
    variants = {
        i: [RenderedItem(item_id=i, mechanism="decoy", variant_id=make_variant_id(i, "decoy"),
                         visible={"stem": "s"}, hidden={"decoy_option_index": 1})]
        for i in items
    }
    fmap = FailureMap(version="t", cells=[_DECOY, *_PASSIVE])
    assessment = Assessment(assessment_id="a", title="t", assessment_type="quiz",
                            item_ids=list(items))
    for flag, expect_check in ((False, False), (True, True)):
        router = SPRTRouter(items, variants, fmap,
                            replace(RouterConfig(), clear_requires_style_human=flag))
        state, choice = router.start(assessment, "s", 0)
        for _ in range(3):
            outcome = "picked_correct" if choice.mechanism == "decoy" else "none"
            state = router.update(state, choice.variant_id, outcome)
            choice = router.next(state)
        assert (state.status == "cleared") is not flag
        assert (choice.mechanism == "decoy") is expect_check  # cleared: plain items only


def test_under_a_cap_group_the_rule_reads_the_groups_shared_total():
    """passive_group: style shares the 'passive' total with behaviour, and the rule reads that
    shared total (cap_key("style")), not style alone (docs/notes/freeze_plan.md (b2))."""
    cfg = replace(RouterConfig(), clear_requires_style_human=True,
                  mech_group_caps={"passive": ("behaviour", "style")})
    router, assessment = _router(cfg)
    assert router.cap_key("style") == "passive"
    state = _feed(router, assessment, "decoy", "picked_correct", 12)  # score at B, blocked
    state = router.update(state, make_variant_id("arc:f0", "behaviour"), "paste")
    state = router.update(state, make_variant_id("arc:f1", "style"), "low")
    # style alone is human-looking, but the shared passive total is not below 0: still blocked
    assert state.contrib["passive"] >= 0 and state.status != "cleared"
    state = router.update(state, make_variant_id("arc:f2", "style"), "low")
    assert state.contrib["passive"] < 0 and state.status == "cleared"

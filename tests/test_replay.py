import random
import statistics

import pandas as pd
import pytest

from adr.contracts import NoObservation, OutcomeRow, make_variant_id, rows_to_frame
from adr.replay.donors import Donors, Group
from adr.replay.metrics import detection_at_fpr, evaluate
from adr.replay.quick_map import quick_failure_map, train_eval_split
from adr.replay.simulate import replay_session
from adr.router.core import SPRTRouter
from adr.router.synthetic import detection_at_fpr as old_detection_at_fpr
from adr.router.synthetic import (
    sample_outcome,
    simulate_session,
    synthetic_bank,
    synthetic_failure_map,
)

GPT = Group.ai("gpt-mini", "copy_paste", "L0")
PILOT = Group.human("pilot")


def _row(session, item_id, mech, outcome, *, kind="ai", model="gpt-mini", heldout=False,
         split="train", item_type="mcq"):
    ai = kind == "ai"
    return OutcomeRow(
        run_id="t", session_id=session, respondent_kind=kind,
        data_origin="harness" if ai else "pilot", model=model if ai else None,
        attack_mode="copy_paste" if ai else None, attacker_level="L0" if ai else None,
        assessment_type="quiz", item_type=item_type, item_id=item_id, split=split,
        heldout_model=heldout if ai else False, mechanism=mech,
        variant_id=make_variant_id(item_id, mech), outcome=outcome, raw_ref="x",
    )


def _table():
    """Two AI sessions with opposite behaviour, so draws show which session was copied."""
    rows = []
    for item in ("x:1", "x:2", "x:3"):
        rows.append(_row("ai-A", item, "decoy", "picked_decoy"))
        rows.append(_row("ai-B", item, "decoy", "picked_correct"))
    rows.append(_row("ai-A", "x:4", "decoy", "picked_other"))   # only session A has x:4
    rows.append(_row("hu-1", "x:1", "decoy", "picked_correct", kind="human"))
    return rows_to_frame(rows)


# ---------- donors ----------
def test_donors_are_deterministic_for_a_fixed_seed():
    a, b = Donors(_table()), Donors(_table().sample(frac=1, random_state=3))  # row order differs
    for seed in range(20):
        da, db = a.respondent(GPT, seed), b.respondent(GPT, seed)
        assert da.session_id == db.session_id
        assert [da.outcome(f"x:{i}", "decoy") for i in (1, 2, 3, 4)] == \
               [db.outcome(f"x:{i}", "decoy") for i in (1, 2, 3, 4)]


def test_a_respondent_copies_one_whole_session():
    donors = Donors(_table())
    assert donors.groups() == [GPT, PILOT]
    assert donors.sessions(GPT) == ["ai-A", "ai-B"]
    seen = set()
    for seed in range(30):
        d = donors.respondent(GPT, seed)
        got = {d.outcome(f"x:{i}", "decoy") for i in (1, 2, 3)}
        expected = "picked_decoy" if d.session_id == "ai-A" else "picked_correct"
        assert got == {expected}  # never a mix of the two sessions
        seen.add(d.session_id)
        assert d.fallbacks == d.synthetic == 0
    assert seen == {"ai-A", "ai-B"}


def test_fallbacks_and_synthetic_draws_are_counted():
    fmap = synthetic_failure_map()
    items = {it_id: it for it_id, it in synthetic_bank()[1].items()}
    items["x:5"] = items["handwritten:mcq0"].model_copy(update={"item_id": "x:5"})
    donors = Donors(_table(), items, fallback_map=fmap)
    d = next(d for s in range(50) if (d := donors.respondent(GPT, s)).session_id == "ai-B")
    assert d.outcome("x:4", "decoy") == "picked_other"     # only session A has it: fallback
    assert (d.fallbacks, d.synthetic) == (1, 0)
    d.outcome("x:5", "decoy")                               # nobody has it: from the map
    assert (d.fallbacks, d.synthetic) == (1, 1)
    with pytest.raises(NoObservation):                      # no row and no map cell
        d.outcome("x:1", "canary")
    assert d.missing == 1


def test_a_group_without_rows_draws_everything_from_the_map():
    fmap = synthetic_failure_map()
    _, items, _ = synthetic_bank()
    donors = Donors(_table().iloc[:0], items, fallback_map=fmap,
                    extra_groups=(Group.human("synthetic"),))
    d = donors.respondent(Group.human("synthetic"), 0)
    assert d.session_id is None
    d.outcome("handwritten:mcq0", "decoy")
    assert d.synthetic == 1


# ---------- replay ----------
class _NoData:
    """A donor with nothing for one mechanism: every request for it raises NoObservation."""

    def __init__(self, inner, blind):
        self.inner, self.blind, self.group = inner, blind, inner.group
        self.session_id, self.fallbacks, self.synthetic = inner.session_id, 0, 0

    def outcome(self, item_id, mechanism):
        if mechanism == self.blind:
            raise NoObservation("nothing to score")
        return self.inner.outcome(item_id, mechanism)


def test_no_observation_is_skipped_and_never_scored():
    fmap = synthetic_failure_map()
    assessment, items, variants = synthetic_bank()
    router = SPRTRouter(items, variants, fmap)
    donor = _NoData(Donors(pd.DataFrame(columns=_table().columns), items, fmap)
                    .respondent(Group.human("x"), 0), blind="behaviour")
    res = replay_session(router, assessment, donor, passive=("behaviour",), seed=1)
    assert 1 <= res.items_shown <= len(assessment.item_ids)  # ended normally, no loop
    assert res.skipped == res.items_shown                  # one skipped behaviour per item
    assert not any("|behaviour|" in v for v, _, _ in res.final.history)

    donor = _NoData(donor.inner, blind="decoy")           # an active check with no data
    res = replay_session(router, assessment, donor, seed=1)
    decoys = [v for v, _, _ in res.final.history if "|decoy|" in v]
    assert res.skipped >= 1 and decoys == []               # no decoy evidence ever recorded
    assert all(d == 0.0 for v, _, d in res.final.history if v.endswith("|plain|0"))


# ---------- quick map ----------
def test_quick_map_refuses_heldout_rows():
    held = rows_to_frame([
        _row("g", "x:1", "decoy", "picked_decoy", model="gemini-flash", heldout=True),
        _row("h", "x:2", "decoy", "picked_decoy", kind="human", split="heldout_survey"),
    ])
    with pytest.raises(ValueError, match="held-out"):
        quick_failure_map(held)
    mixed = pd.concat([held, rows_to_frame([_row("a", "x:1", "decoy", "picked_correct")])])
    fmap = quick_failure_map(mixed)
    pooled = fmap.lookup("decoy", "mcq")
    assert pooled.counts_ai == {"picked_decoy": 0, "picked_correct": 1, "picked_other": 0}
    assert sum(pooled.counts_human.values()) == 0          # the held-out survey row is dropped
    assert "TEMPORARY" in fmap.note and "No human rows" in fmap.note
    fit, ev, same = train_eval_split(mixed)
    assert (len(fit), len(ev), same) == (1, 2, False)
    assert train_eval_split(fit)[2] is True                # nothing held out: same data


# ---------- metrics ----------
class _R:
    def __init__(self, score, status="active", checks=0, decided=None):
        self.final = type("S", (), {"score": score, "status": status})()
        self.checks_run, self.items_to_decision = checks, decided


def test_metrics_match_hand_computed_values():
    assert old_detection_at_fpr is detection_at_fpr       # still importable from synthetic.py
    ai = [_R(5, "review", 2, 3), _R(4, "review", 3, 4), _R(1, "active", 4, None),
          _R(-1, "cleared", 1, 2)]
    hu = [_R(s, "cleared" if s < 0 else "active", 2, 5 if s < 0 else None) for s in
          (-3, -2, -1, 0, 0, 0, 1, 1, 2, 3)]
    m = evaluate(ai, hu, n_boot=200, seed=0)
    # 5% FPR of 10 people allows 0 flags: threshold = max human score 3 -> AI above 3: 5, 4
    assert m["tpr_at_5pct_fpr"] == m["tpr_at_1pct_fpr"] == 0.5
    assert m["ai_paused_for_review"] == 0.5 and m["honest_paused_for_review"] == 0.0
    assert m["checks_per_honest_person"] == 2.0
    # decided: AI 3, 4, 2 and humans 5, 5, 5 (the three cleared) -> median of [2,3,4,5,5,5] = 4.5
    assert m["median_items_to_decision"] == 4.5 and m["share_decided"] == 6 / 14
    lo, hi = m["ci95"]["ai_paused_for_review"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert evaluate(ai, hu, n_boot=200, seed=0) == m        # fixed seed: same intervals
    assert evaluate(ai, [], n_boot=50)["tpr_at_1pct_fpr"] is None


# ---------- sanity: replaying synthetic rows ~ simulating from the map ----------
def test_replaying_synthetic_rows_matches_simulate_session():
    fmap = synthetic_failure_map()
    assessment, items, variants = synthetic_bank()
    rng = random.Random(0)
    rows = []
    for kind in ("ai", "human"):
        for s in range(300):   # each logged session answers every check, like the harness does
            for item_id in assessment.item_ids:
                item = items[item_id]
                for v in variants[item_id]:
                    if v.mechanism is None:
                        continue
                    cell = fmap.lookup(v.mechanism, item.item_type)
                    out = sample_outcome(cell, kind, rng)
                    rows.append(_row(f"{kind}-{s}", item_id, v.mechanism, out, kind=kind,
                                     item_type=item.item_type))
    donors = Donors(rows_to_frame(rows), items)
    router = SPRTRouter(items, variants, fmap)
    n = 300
    rep = {g.kind: [replay_session(router, assessment, donors.respondent(g, s), seed=s)
                    for s in range(n)] for g in donors.groups()}
    sim = {k: [simulate_session(router, assessment, fmap, k, 10_000 + s) for s in range(n)]
           for k in ("ai", "human")}
    assert all(r.fallbacks == r.synthetic == r.skipped == 0 for rs in rep.values() for r in rs)
    def tpr(res):
        return detection_at_fpr([r.final.score for r in res["ai"]],
                                [r.final.score for r in res["human"]], 0.05)[0]

    def checks(res):
        return statistics.mean(r.checks_run for r in res["human"])

    assert abs(tpr(rep) - tpr(sim)) < 0.08
    assert abs(checks(rep) - checks(sim)) < 1.0

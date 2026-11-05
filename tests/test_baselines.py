import pandas as pd
import pytest

from adr.baselines.plans import (
    AllChecksRouter,
    FixedPlanRouter,
    best_single_mechanism,
    multi_check_items,
)
from adr.baselines.policies import COMPARE, make_policy
from adr.baselines.simple import RandomCheckRouter, SingleMechanismRouter
from adr.contracts import (
    Assessment,
    Item,
    OutcomeRow,
    RenderedItem,
    Router,
    make_variant_id,
    rows_to_frame,
)
from adr.replay.compare import compare, comparison_table, write_comparison
from adr.replay.donors import Donors, Group
from adr.replay.quick_map import quick_failure_map
from adr.router.core import RouterConfig, SPRTRouter
from adr.router.selection import expected_evidence, utility
from adr.router.synthetic import synthetic_bank, synthetic_failure_map

FMAP = synthetic_failure_map()
BURDEN = {"decoy": 0.02, "canary": 0.0, "glitch_token": 0.15, "opinion_trap": 5.0}


def _bank():
    """q1 has a decoy, q2 nothing, o1 two checks (canary, then glitch), o2 only glitch, and l1
    an opinion trap so burdensome that it's never worth it at pi = 0.5."""
    spec = [
        ("handwritten:q1", "mcq", ["decoy"]),
        ("handwritten:q2", "mcq", []),
        ("handwritten:o1", "open_text", ["canary", "glitch_token"]),
        ("handwritten:o2", "open_text", ["glitch_token"]),
        ("handwritten:l1", "likert", ["opinion_trap"]),
    ]
    items, variants = {}, {}
    for item_id, item_type, mechs in spec:
        opts = {"mcq": ["A", "B", "C"], "likert": ["1", "2", "3"]}.get(item_type)
        item = Item(item_id=item_id, source="handwritten", assessment_type="quiz",
                    item_type=item_type, stem="Q?", options=opts,
                    correct_option=0 if item_type == "mcq" else None, eligible_mechanisms=mechs)
        items[item_id] = item
        vis = {"stem": "Q?", **({"options": opts} if opts else {})}
        variants[item_id] = [RenderedItem(item_id=item_id, mechanism=None,
                                          variant_id=make_variant_id(item_id, None), visible=vis)]
        variants[item_id] += [RenderedItem(item_id=item_id, mechanism=m,
                                           variant_id=make_variant_id(item_id, m), visible=vis,
                                           burden=BURDEN[m]) for m in mechs]
    assessment = Assessment(assessment_id="hand", title="Hand", assessment_type="quiz",
                            item_ids=list(items))
    return assessment, items, variants


def _walk(router, assessment, outcome_for=lambda mech: None, seed=0):
    """Serve the whole assessment; return [(item_id, mechanism)] in the order served."""
    state, choice = router.start(assessment, f"s{seed}", seed)
    served = []
    while choice is not None:
        served.append((choice.item_id, choice.mechanism))
        out = "none" if choice.mechanism is None else outcome_for(choice.mechanism)
        state = router.update(state, choice.variant_id, out)
        choice = router.next(state)
    return served, state


def test_all_checks_gives_every_item_its_first_eligible_check():
    a, items, variants = _bank()
    served, _ = _walk(AllChecksRouter(items, variants, FMAP), a)
    assert served == [("handwritten:q1", "decoy"), ("handwritten:q2", None),
                      ("handwritten:o1", "canary"), ("handwritten:o2", "glitch_token"),
                      ("handwritten:l1", "opinion_trap")]
    assert multi_check_items(a, items, variants) == ["handwritten:o1"]


def test_fixed_plan_picks_best_utility_at_half_or_plain():
    a, items, variants = _bank()
    cfg = RouterConfig()
    router = FixedPlanRouter(items, variants, FMAP, cfg)

    def u(item_id, mech):
        cell = FMAP.lookup(mech, items[item_id].item_type)
        return utility(0.5, cell, BURDEN[mech], cfg.lambda_burden, cfg.lambda_fp, cfg.objective)

    assert u("handwritten:q1", "decoy") > 0 and u("handwritten:o1", "canary") > 0
    assert u("handwritten:o2", "glitch_token") <= 0             # weak and burdensome: plain
    assert u("handwritten:l1", "opinion_trap") <= 0             # far too burdensome: plain
    served, _ = _walk(router, a)
    assert served == [("handwritten:q1", "decoy"), ("handwritten:q2", None),
                      ("handwritten:o1", "canary"), ("handwritten:o2", None),
                      ("handwritten:l1", None)]
    # the plan never changes with the answers (an AI session just stops early, at review)
    ai_served, honest_served = _walk(router, a, AI.get)[0], _walk(router, a, HUMAN.get)[0]
    assert len(ai_served) < len(honest_served) and honest_served[:len(ai_served)] == ai_served


def test_single_and_random_choose_by_their_definitions():
    a, items, variants = _bank()
    served, _ = _walk(SingleMechanismRouter(items, variants, FMAP, mechanism="glitch_token"), a)
    assert [m for _, m in served] == [None, None, "glitch_token", "glitch_token", None]
    rnd = RandomCheckRouter(items, variants, FMAP)
    picks = {seed: dict(_walk(rnd, a, seed=seed)[0]) for seed in range(30)}
    for p in picks.values():
        assert p["handwritten:q1"] == "decoy" and p["handwritten:q2"] is None
        assert p["handwritten:o1"] in ("canary", "glitch_token")
    assert {p["handwritten:o1"] for p in picks.values()} == {"canary", "glitch_token"}


def test_best_single_sums_expected_evidence_over_items():
    a, items, variants = _bank()
    best, totals = best_single_mechanism(FMAP, a, items, variants)
    ev = {m: expected_evidence(0.5, FMAP.lookup(m, t))
          for m, t in [("decoy", "mcq"), ("canary", "open_text"), ("glitch_token", "open_text"),
                       ("opinion_trap", "likert")]}
    assert totals == pytest.approx({"decoy": ev["decoy"], "canary": ev["canary"],
                                    "glitch_token": 2 * ev["glitch_token"],
                                    "opinion_trap": ev["opinion_trap"]})
    assert best == max(totals, key=totals.get)


def _row(session, item_id, mech, outcome, kind="ai", model="gpt-mini", heldout=False):
    ai = kind == "ai"
    item_type = "mcq" if mech == "decoy" else "open_text"
    return OutcomeRow(run_id="t", session_id=session, respondent_kind=kind,
                      data_origin="harness" if ai else "pilot", model=model if ai else None,
                      attack_mode="copy_paste" if ai else None,
                      attacker_level="L0" if ai else None, assessment_type="quiz",
                      item_type=item_type, item_id=item_id, split="train",
                      heldout_model=heldout if ai else False, mechanism=mech,
                      variant_id=make_variant_id(item_id, mech), outcome=outcome, raw_ref="x")


def test_best_single_never_looks_at_heldout_rows():
    a, items, variants = _bank()
    rows = []
    for s in range(20):
        rows += [_row(f"h{s}", "handwritten:q1", "decoy", "picked_correct", kind="human"),
                 _row(f"h{s}", "handwritten:o1", "canary", "clean", kind="human"),
                 # the train model falls for half the decoys and ignores canaries ...
                 _row(f"a{s}", "handwritten:q1", "decoy",
                      "picked_decoy" if s % 2 else "picked_correct"),
                 _row(f"a{s}", "handwritten:o1", "canary", "clean"),
                 # ... the held-out model always follows the canary, with more rows
                 *[_row(f"g{s}{k}", "handwritten:o1", "canary", "followed_instruction",
                        model="gemini-flash", heldout=True) for k in range(5)]]
    table = rows_to_frame(rows)
    assert best_single_mechanism(quick_failure_map(table), a, items, variants)[0] == "decoy"
    # the held-out rows would have changed the choice: so they really were left out
    leaked = table.assign(heldout_model=False)
    assert best_single_mechanism(quick_failure_map(leaked), a, items, variants)[0] == "canary"


@pytest.mark.parametrize("name", [*COMPARE, "single:decoy"])
def test_every_policy_satisfies_the_router_protocol(name):
    a, items, variants = _bank()
    router = make_policy(name, a, items, variants, FMAP)
    assert isinstance(router, Router)
    state, _ = router.start(a, "s", 0)
    after = router.update(state, make_variant_id("handwritten:q1", "decoy"), None)
    assert make_variant_id("handwritten:q1", "decoy") in after.served
    assert after.score == state.score and after.status == state.status


def _trace(router, assessment, outcome_for):
    """[(mechanism served, score before serving it)] for one whole session."""
    state, choice = router.start(assessment, "s", 0)
    out = []
    while choice is not None:
        out.append((choice.mechanism, state.score, state.status))
        o = "none" if choice.mechanism is None else outcome_for(choice.mechanism)
        state = router.update(state, choice.variant_id, o)
        choice = router.next(state)
    return out, state


HUMAN = {"decoy": "picked_correct", "canary": "clean", "glitch_token": "normal",
         "opinion_trap": "plausible"}
AI = {"decoy": "picked_decoy", "canary": "followed_instruction", "glitch_token": "anomalous",
      "opinion_trap": "implausible"}


@pytest.mark.parametrize("cls", [AllChecksRouter, FixedPlanRouter, RandomCheckRouter])
def test_baselines_never_clear_early_but_still_pause_for_review(cls):
    a, items, variants = _bank()
    cfg = RouterConfig(lower=-0.5, mech_cap=None)   # B reachable after one honest answer
    trace, state = _trace(cls(items, variants, FMAP, cfg), a, HUMAN.get)
    assert all(status != "cleared" for _, _, status in trace) and state.status != "cleared"
    after_b = [m for m, score, _ in trace if score <= cfg.lower]
    assert any(m is not None for m in after_b)      # still checking below B
    router_trace, router_state = _trace(SPRTRouter(items, variants, FMAP, cfg), a, HUMAN.get)
    assert router_state.status == "cleared"
    assert all(m is None for m, _, status in router_trace if status == "cleared")

    served, state = _trace(cls(items, variants, FMAP, cfg), a, AI.get)
    assert state.status == "review" and len(served) < len(a.item_ids)  # stopped at A


def test_compare_on_synthetic_data_router_needs_fewer_checks_than_all_checks(tmp_path):
    assessment, items, variants = synthetic_bank()
    donors = Donors(pd.DataFrame(columns=rows_to_frame([]).columns), items, fallback_map=FMAP,
                    extra_groups=(Group.ai("synthetic", "copy_paste", "L0"),
                                  Group.human("synthetic")))
    policies = {n: make_policy(n, assessment, items, variants, FMAP) for n in COMPARE}
    results = compare(policies, assessment, donors, n=150, seed=0, n_boot=50)
    checks = {n: r["overall"]["checks_per_honest_person"] for n, r in results.items()}
    assert checks["router"] < checks["all_checks"]
    # every synthetic item has a check; only an honest person paused for review sees fewer
    assert 0.9 * len(assessment.item_ids) < checks["all_checks"] <= len(assessment.item_ids)
    table = comparison_table(results)
    assert list(table["policy"]) == list(COMPARE)
    assert {"tpr_at_5pct_fpr", "tpr_at_5pct_fpr_lo", "tpr_at_5pct_fpr_hi"} <= set(table.columns)
    csv_path, md_path = write_comparison(table, tmp_path)
    assert len(pd.read_csv(csv_path)) == len(COMPARE)
    assert md_path.read_text().count("\n| ") == len(COMPARE) + 1  # header + one row per policy

"""Thresholds from replayed honest sessions (adr/replay/crossfit.py), mech_group_caps and
scripts/run_real_router.py.

All data is SYNTHETIC, pilot-shaped rows built here (not the dry-run data: the builder rejects
data_origin "dry_run"). No model calls.
"""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from adr.contracts import (
    Assessment,
    FailureMap,
    Item,
    MapCell,
    OutcomeRow,
    make_variant_id,
    rows_to_frame,
    write_outcomes,
)
from adr.map import build_failure_map
from adr.replay import crossfit as cf
from adr.replay.crossfit import (
    HONEST,
    _index,
    all_maps,
    deploy_config,
    fold_map,
    resampled_donor,
    router_factory,
    score_policy,
    summarize,
    tail_warnings,
    threshold_at,
    threshold_none,
)
from adr.replay.donors import Donors
from adr.replay.folds import assign_session_folds, fold_lists
from adr.replay.simulate import replay_session
from adr.router.core import RouterConfig, SPRTRouter
from adr.router.map_guard import ProvisionalMapError

ROOT = Path(__file__).resolve().parents[1]


def session_folds(sessions, k, seed, stratum=()):
    """The shared assignment (adr/replay/folds.py) as k lists, for these tests."""
    return fold_lists(assign_session_folds(sessions, k, str(seed), list(stratum)), k)


sys.path.insert(0, str(ROOT / "scripts"))
import run_real_router

N_SESSIONS, N_PILOT_ITEMS, N_AI_ITEMS, N_AI_SESSIONS = 20, 8, 40, 5


def _item(item_id: str) -> Item:
    return Item(
        item_id=item_id,
        source="arc",
        assessment_type="quiz",
        item_type="mcq",
        stem=f"Question {item_id}?",
        options=["a", "b", "c", "d"],
        correct_option=1,
        eligible_mechanisms=["decoy"],
    )


PILOT_ITEMS = [_item(f"arc:pilot-{i}") for i in range(N_PILOT_ITEMS)]
AI_ITEMS = [f"mmlu:bank-{i}" for i in range(N_AI_ITEMS)]  # the AI runs used OTHER items


def human_row(session: str, item_id: str, outcome: str) -> OutcomeRow:
    return OutcomeRow(
        run_id="pilot-test",
        session_id=session,
        respondent_kind="human",
        data_origin="pilot",
        assessment_type="quiz",
        item_type="mcq",
        item_id=item_id,
        split="train",
        mechanism="decoy",
        variant_id=make_variant_id(item_id, "decoy", 0),
        outcome=outcome,
        raw_ref="pilot",
    )


def ai_row(session: str, item_id: str, outcome: str) -> OutcomeRow:
    return OutcomeRow(
        run_id="ai-test",
        session_id=session,
        respondent_kind="ai",
        data_origin="harness",
        model="gpt-mini",
        attack_mode="copy_paste",
        attacker_level="L0",
        assessment_type="quiz",
        item_type="mcq",
        item_id=item_id,
        split="train",
        mechanism="decoy",
        variant_id=make_variant_id(item_id, "decoy", 0),
        outcome=outcome,
        raw_ref="x",
        prompt_version="t",
    )


def session_ids() -> list[str]:
    return [f"pilot-test:p{j:02d}" for j in range(N_SESSIONS)]


def human_frame() -> pd.DataFrame:
    rows = []
    for j, p in enumerate(session_ids()):
        for i, item in enumerate(PILOT_ITEMS):
            # most sessions never pick the decoy; some pick it once or twice
            decoy = (j % 4 == 0 and i == 0) or (j % 7 == 0 and i == 1)
            rows.append(human_row(p, item.item_id, "picked_decoy" if decoy else "picked_correct"))
    return rows_to_frame(rows)


def ai_frame() -> pd.DataFrame:
    rows = []
    for s in range(N_AI_SESSIONS):
        for i, item in enumerate(AI_ITEMS):
            outcome = "picked_decoy" if (i + s) % 4 else "picked_correct"  # 75% decoy
            rows.append(ai_row(f"ai-test:gpt-mini:{s}", item, outcome))
    return rows_to_frame(rows)


def bank():
    items = {it.item_id: it for it in PILOT_ITEMS}
    assessment = Assessment(
        assessment_id="pilot-quiz", title="Quiz", assessment_type="quiz", item_ids=list(items)
    )
    from adr.harness.copy_paste import plain_variant
    from adr.harness.registry import get_mechanism

    decoy = get_mechanism("decoy")
    variants = {i: [plain_variant(it), decoy.generate(it, 0)] for i, it in items.items()}
    return assessment, items, variants


def scored_router(human=None, n_synthetic=60, folds=None):
    human = human_frame() if human is None else human
    ai = ai_frame()
    assessment, items, variants = bank()
    folds = folds or session_folds(session_ids(), 4, 0)
    maps = all_maps(ai, human, folds, 20)
    make = router_factory("router", assessment, items, variants)
    sc = score_policy(make, RouterConfig(), assessment, items, human, ai, folds, maps, n_synthetic)
    return sc, make, assessment, items, human, folds, maps


# ---------------------------------------------------------------- folds and leakage
def test_folds_partition_sessions_with_no_overlap():
    ps = session_ids()
    folds = session_folds(ps, 4, seed=0)
    flat = [p for f in folds for p in f]
    assert sorted(flat) == sorted(ps) and len(flat) == len(set(flat))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    assert session_folds(ps, 4, seed=0) == folds
    assert sorted(len(f) for f in session_folds(ps[:7], 3, seed=1)) == [2, 2, 3]
    with pytest.raises(ValueError):
        session_folds(ps[:3], 4, 0)


def test_map_for_fold_k_has_no_human_rows_from_fold_k():
    human, ai = human_frame(), ai_frame()
    held = session_folds(session_ids(), 4, 0)[0]
    fmap = fold_map(ai, human, held, "f0", n_bootstrap=20)
    n_others = int((~human.session_id.isin(held)).sum())
    assert sum(fmap.lookup("decoy", "mcq").counts_human.values()) == n_others


# ---------------------------------------------------------------- out-of-fold scores
def test_out_of_fold_score_uses_a_map_without_that_person():
    """Flip test: change everything one session answered. The map that scores them must not
    move (it never saw them); the maps of the other folds must (they do include them)."""
    folds = session_folds(session_ids(), 4, 0)
    p = folds[1][0]
    flipped = human_frame()
    flipped.loc[flipped.session_id == p, "outcome"] = "picked_decoy"
    base, *_, maps0 = scored_router()
    flip, *_, maps1 = scored_router(human=flipped)
    k = base.real[p].fold
    assert k == 1
    assert maps1[k] == maps0[k]  # p's own map: identical
    assert all(maps1[j] != maps0[j] for j in range(4) if j != k)  # p is in every other map
    assert flip.real[p].score > base.real[p].score  # only p's own answers changed p's score
    for q in folds[1]:
        if q != p:  # same fold, same map, own answers unchanged
            assert flip.real[q].score == base.real[q].score
    assert set(base.real) == set(session_ids())  # every session scored exactly once, out of fold


def test_pausing_at_a_threshold_is_exactly_score_at_or_above_it():
    sc, make, assessment, items, human, folds, maps = scored_router()
    cfg = RouterConfig()
    for a in (0.5, 1.5, 3.0, 4.5):
        for k, fold in enumerate(folds):
            policy = make(maps[k], deploy_config(cfg, a))
            donors = Donors(human[human.session_id.isin(fold)], items)
            for p in fold:
                r = replay_session(policy, assessment, donors.respondent_from(HONEST, p, 0), (), 0)
                assert (r.final.status == "review") == (sc.real[p].score >= a)
                assert r.checks_run == sc.real[p].checks_at(a)  # burden read off the trace


# ---------------------------------------------------------------- resampled honest sessions
def test_resampled_sessions_draw_only_from_held_out_sessions():
    human = human_frame()
    _, items, _ = bank()
    index = _index(human)
    clean = [p for j, p in enumerate(session_ids()) if j % 4 and j % 7]  # never pick the decoy
    donor = resampled_donor(index, clean, items, random.Random(0))
    drawn = [donor.outcome(it.item_id, "decoy") for it in PILOT_ITEMS for _ in range(20)]
    assert set(drawn) == {"picked_correct"}  # others' picked_decoy rows are never used
    assert donor.synthetic == 0  # no map draws
    # an item none of them answered falls back to the same item type, from them only
    other = _item("arc:never-seen")
    donor = resampled_donor(index, clean, {**items, other.item_id: other}, random.Random(0))
    assert donor.outcome(other.item_id, "decoy") == "picked_correct"


def test_each_folds_synthetic_sessions_use_only_that_folds_people(monkeypatch):
    seen = []
    real = cf.resampled_donor

    def spy(index, sessions_, items, rng):
        seen.append(tuple(sorted(set(sessions_))))
        return real(index, sessions_, items, rng)

    monkeypatch.setattr(cf, "resampled_donor", spy)
    sc, *_, folds, _ = scored_router(n_synthetic=5)
    assert set(seen) == {tuple(f) for f in folds}
    assert len(sc.synthetic) == 4 * 5


# ---------------------------------------------------------------- thresholds
def test_threshold_definitions():
    scores = [0.0] * 90 + [1.0] * 6 + [2.0] * 3 + [5.0]
    assert threshold_at(scores, 0.05) == 2.0  # P(s >= 2) = 4% <= 5%; P(s >= 1) = 10%
    assert threshold_at(scores, 0.01) == 5.0  # P(s >= 5) = 1%
    assert threshold_at(scores, 0.001) > 5.0  # above everyone
    assert threshold_none([0.2, 3.0, 1.0]) > 3.0 and threshold_none([0.2, 3.0]) < 3.0001


def test_three_thresholds_their_rates_and_tail_warnings():
    sc, make, assessment, items, human, *_ = scored_router()
    s = summarize(
        sc,
        make,
        RouterConfig(),
        assessment,
        items,
        human,
        (0.01, 0.05),
        n_boot=20,
        n_synthetic_boot=10,
    )
    assert set(s["thresholds"]) == {"A_0.01", "A_0.05", "A_max"}
    r = s["results"]
    assert r["A_max"]["honest_pause_rate"] == 0.0  # pauses none of the real sessions
    assert r["A_0.05"]["resampled_pause_rate"] <= 0.05
    assert r["A_0.01"]["resampled_pause_rate"] <= 0.01
    assert s["thresholds"]["A_0.01"] >= s["thresholds"]["A_0.05"]
    warns = tail_warnings(s)
    assert warns[0] == f"1% tail extrapolated from {N_SESSIONS} sessions"
    assert any("A_max" in w and "fewer than 3 real sessions" in w for w in warns)
    # decoy is the only mechanism: a threshold above the decoy cap pauses nobody (TPR 0); one
    # exactly AT the cap still pauses the sessions that saturate it, so a finding is attached
    # only where the TPR really is 0
    cap = 0.65 * cf.nominal_A(RouterConfig())
    above = [n for n, v in r.items() if v["A"] > cap + 1e-9]
    assert all(r[n]["tpr"] == 0 and r[n]["honest_pause_rate"] == 0 for n in above)
    at_cap = [n for n, v in r.items() if v["A"] >= cap - 1e-9 and v["tpr"] == 0]
    found = cf.cap_findings(s, RouterConfig(), ["decoy"])
    assert set(found) == set(at_cap)
    assert all("finding, not a failure" in t and "0.65A" in t for t in found.values())
    assert cf.cap_findings(s, RouterConfig(), ["decoy", "behaviour"]) == {}
    assert cf.cap_findings(s, RouterConfig(mech_cap_frac=None), ["decoy"]) == {}


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_resamples_sessions_and_regenerates_sessions(monkeypatch):
    sc, make, assessment, items, human, *_ = scored_router()
    sizes, regenerated = [], []
    real_rng, real_syn = np.random.default_rng, cf._synthetic

    class Spy:
        def __init__(self, seed):
            self.rng = real_rng(seed)

        def integers(self, low, high=None, size=None):
            sizes.append(size)
            return self.rng.integers(low, high, size)

    def syn_spy(*a, tag="", **k):
        regenerated.append(tag)
        return real_syn(*a, tag=tag, **k)

    monkeypatch.setattr(cf.np.random, "default_rng", Spy)
    monkeypatch.setattr(cf, "_synthetic", syn_spy)
    s = summarize(
        sc,
        make,
        RouterConfig(),
        assessment,
        items,
        human,
        (0.01, 0.05),
        n_boot=6,
        n_synthetic_boot=4,
    )
    assert set(sizes) == {N_SESSIONS, N_AI_SESSIONS}  # one index per PERSON / AI session
    assert sizes.count(N_SESSIONS) == 6
    assert regenerated and all(t.startswith(":boot") for t in regenerated)  # rebuilt per draw
    assert s["n_sessions"] == N_SESSIONS and s["boot_unit"].startswith("session")


# ---------------------------------------------------------------- exclusions
def test_exclude_sessions_removes_exactly_those(tmp_path):
    human = human_frame()
    f = tmp_path / "low_effort.txt"
    f.write_text("# low-effort sessions\npilot-test:p03\n\npilot-test:p07\nnot-a-session\n")
    kept, missing = run_real_router.exclude_sessions(human, run_real_router.read_exclusions(str(f)))
    assert set(human.session_id) - set(kept.session_id) == {"pilot-test:p03", "pilot-test:p07"}
    assert len(kept) == len(human) - 2 * N_PILOT_ITEMS
    assert missing == ["not-a-session"]


# ---------------------------------------------------------------- group caps
def _passive_map() -> FailureMap:
    return FailureMap(
        version="t",
        cells=[
            MapCell.from_counts(
                "behaviour", "mcq", {"paste": 90, "normal": 10}, {"paste": 1, "normal": 99}
            ),
            MapCell.from_counts("style", "mcq", {"high": 90, "low": 10}, {"high": 1, "low": 99}),
        ],
    )


def _feed(cfg: RouterConfig):
    item = _item("arc:cap-0")
    router = SPRTRouter({item.item_id: item}, {item.item_id: []}, _passive_map(), cfg)
    state, _ = router.start(
        Assessment(assessment_id="a", title="t", assessment_type="quiz", item_ids=[item.item_id]),
        "s",
        0,
    )
    for _ in range(5):
        state = router.update(state, make_variant_id(item.item_id, "behaviour"), "paste")
        state = router.update(state, make_variant_id(item.item_id, "style"), "high")
    return router, state


def test_mech_group_caps_caps_the_group_total():
    router, separate = _feed(RouterConfig(upper=100.0, mech_cap_frac=0.03))  # cap = 3 nats
    assert separate.score == pytest.approx(2 * router.mech_cap)
    router, grouped = _feed(
        RouterConfig(
            upper=100.0, mech_cap_frac=0.03, mech_group_caps={"passive": ("behaviour", "style")}
        )
    )
    assert grouped.score == pytest.approx(router.mech_cap)
    assert grouped.contrib == {"passive": pytest.approx(router.mech_cap)}
    assert router.cap_key("style") == "passive" and router.cap_key("decoy") == "decoy"


def test_mech_group_caps_is_off_by_default_and_validated():
    assert RouterConfig().mech_group_caps is None
    item = _item("arc:cap-1")
    for bad in (
        {"g": ("behaviour",), "h": ("behaviour",)},
        {"g": ("nope",)},
        {"decoy": ("style",)},
        {"g": ()},
    ):
        with pytest.raises(ValueError, match="mech_group_caps"):
            SPRTRouter({item.item_id: item}, {}, _passive_map(), RouterConfig(mech_group_caps=bad))


# ---------------------------------------------------------------- the #25 guard
def test_ai_only_map_is_refused():
    human, ai = human_frame(), ai_frame()
    assessment, items, variants = bank()
    ai_only = build_failure_map(ai, version="v1-ai-only", n_bootstrap=10, ai_only=True)
    folds = session_folds(session_ids(), 4, 0)
    make = router_factory("router", assessment, items, variants)
    with pytest.raises(ProvisionalMapError):
        score_policy(make, RouterConfig(), assessment, items, human, ai, folds, [ai_only] * 4, 5)
    with pytest.raises(ValueError, match="both AI and human"):
        fold_map(ai, human, session_ids(), "all-held-out", n_bootstrap=10)


# ---------------------------------------------------------------- the one command
def _files(tmp_path, pilot=True):
    write_outcomes(
        [OutcomeRow(**r) for r in ai_frame().to_dict("records")], tmp_path / "ai.parquet"
    )
    rows = human_frame() if pilot else ai_frame()
    write_outcomes([OutcomeRow(**r) for r in rows.to_dict("records")], tmp_path / "pilot.parquet")
    b = {
        "flow": {"title": "t", "parts": ["pilot-quiz"]},
        "items": [it.model_dump(exclude_defaults=True) | {"split": "train"} for it in PILOT_ITEMS],
        "assessments": [
            {
                "assessment_id": "pilot-quiz",
                "title": "Quiz",
                "assessment_type": "quiz",
                "item_ids": [i.item_id for i in PILOT_ITEMS],
            }
        ],
    }
    (tmp_path / "bank.json").write_text(json.dumps(b), encoding="utf-8")


def _cmd(tmp_path, *extra):
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_real_router.py"),
        "--ai-outcomes",
        str(tmp_path / "ai.parquet"),
        "--pilot-outcomes",
        str(tmp_path / "pilot.parquet"),
        "--bank",
        str(tmp_path / "bank.json"),
        "--boot",
        "5",
        "--n-synthetic",
        "20",
        "--n-synthetic-boot",
        "5",
        "--betas",
        "0.05,0.2",
        "--calibration-frac",
        "0.5",
        "--results-dir",
        str(tmp_path),
        "--run-id",
        "r",
        *extra,
        *([] if "--low-effort" in extra else ["--no-low-effort"]),
    ]


def test_run_real_router_end_to_end(tmp_path):
    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "BOTH AI and human rows (the only ones used): decoy" in proc.stdout
    assert "decoy only" in proc.stdout
    assert f"1% tail extrapolated from {N_SESSIONS} sessions" in proc.stdout
    assert "fewer than 3 real sessions" in proc.stdout
    out = tmp_path / "real-router" / "r"
    for name in (
        "comparison.csv",
        "comparison.md",
        "thresholds.csv",
        "budget_curve.csv",
        "metrics.json",
        "config.yaml",
    ):
        assert (out / name).exists(), name
    # the plot needs the optional [plot] extra; without matplotlib the script skips it (H2, A2)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert not (out / "budget_curve.png").exists()
    else:
        assert (out / "budget_curve.png").exists()
    th = pd.read_csv(out / "thresholds.csv")
    assert set(th.threshold) == {"A_0.01", "A_0.05", "A_max"}
    assert len(th) == 6 * 3  # 6 policies x 3 thresholds
    assert (th[th.threshold == "A_max"].honest_pause_rate_real == 0).all()
    curve = pd.read_csv(out / "budget_curve.csv")
    assert list(curve[curve.policy == "router"].beta) == [0.05, 0.2]


def test_run_real_router_needs_pilot_rows(tmp_path):
    _files(tmp_path, pilot=False)
    proc = subprocess.run(
        _cmd(tmp_path), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode != 0 and "no pilot rows" in proc.stderr


# ---------------------------------------------------------------- stratified folds
def test_low_effort_sessions_are_spread_across_folds():
    ps = session_ids()
    low = [ps[0], ps[1], ps[2], ps[3]]  # 4 low-effort sessions, 4 folds
    folds = session_folds(ps, 4, seed=0, stratum=low)
    assert [len(set(f) & set(low)) for f in folds] == [1, 1, 1, 1]
    assert sorted(p for f in folds for p in f) == sorted(ps)
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    for seed in range(5):  # 6 low-effort in 4 folds: at most one apart
        f6 = session_folds(ps, 4, seed=seed, stratum=ps[:6])
        counts = [len(set(f) & set(ps[:6])) for f in f6]
        assert sum(counts) == 6 and max(counts) - min(counts) <= 1
        assert max(map(len, f6)) - min(map(len, f6)) <= 1
    # an id that is not a session stops the run (a typo must not drop the stratification)
    with pytest.raises(ValueError, match="not among"):
        session_folds(ps, 4, 0, stratum=["nope"])


# ---------------------------------------------------------------- headline and cap variants
def test_headline_is_tpr_at_5pct_with_1pct_secondary():
    sc, make, assessment, items, human, *_ = scored_router()
    s = summarize(
        sc,
        make,
        RouterConfig(),
        assessment,
        items,
        human,
        (0.01, 0.05),
        n_boot=10,
        n_synthetic_boot=5,
    )
    text = run_real_router.headline_text(s)
    assert text.startswith("HEADLINE (pilot): router TPR at 5% FPR")
    assert text.index("5% FPR") < text.index("Secondary: TPR at 1% FPR")
    assert run_real_router._ordered(s)[:3] == ["A_0.05", "A_0.01", "A_max"]


def test_cap_variants():
    v = run_real_router.CAP_VARIANTS
    assert v["per_mech"] == {}
    assert set(v["passive_group"]["mech_group_caps"]["passive"]) == {"behaviour", "style"}
    assert set(v["open_text_group"]["mech_group_caps"]["open_text"]) == {"canary", "style"}
    assert v["no_cap"] == {"mech_cap_frac": None}
    assert run_real_router.ABLATIONS["no_cap"] == (
        "router[no_cap] (ablation: single check may decide)"
    )
    item = _item("arc:cap-2")
    for change in v.values():  # every variant is a valid router config
        SPRTRouter({item.item_id: item}, {}, _passive_map(), RouterConfig(**change))
    assert (
        SPRTRouter({item.item_id: item}, {}, _passive_map(), RouterConfig(**v["no_cap"])).mech_cap
        is None
    )


def test_run_real_router_low_effort_and_cap_variants(tmp_path):
    _files(tmp_path)
    low = tmp_path / "low_effort_ids.txt"
    low.write_text("\n".join(session_ids()[:4]) + "\n")
    proc = subprocess.run(
        _cmd(
            tmp_path,
            "--low-effort",
            str(low),
            "--cap-variant",
            "per_mech",
            "passive_group",
            "open_text_group",
            "no_cap",
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "low-effort sessions per fold: [1, 1, 1, 1]" in proc.stdout
    assert "one person may span sessions" in proc.stdout
    assert "HEADLINE (pilot): router TPR at 5% FPR" in proc.stdout
    assert "cap variant passive_group has no effect here" in proc.stdout  # decoy only
    out = tmp_path / "real-router" / "r"
    th = pd.read_csv(out / "thresholds.csv")
    ablation = "router[no_cap] (ablation: single check may decide)"
    assert {"router[passive_group]", "router[open_text_group]", ablation} <= set(th.policy)
    md = (out / "comparison.md").read_text()
    assert md.startswith("HEADLINE (pilot): router TPR at 5% FPR")  # the main router only
    assert f"{ablation} is an ablation" in md and "never the headline" in md
    cfg = __import__("yaml").safe_load((out / "config.yaml").read_text())
    assert cfg["cap_variants"] == ["per_mech", "passive_group", "open_text_group", "no_cap"]
    assert cfg["low_effort_per_fold"] == [1, 1, 1, 1]
    assert json.loads((out / "metrics.json").read_text())["headline"].startswith("HEADLINE")


def test_no_cap_can_only_be_an_ablation(tmp_path):
    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path, "--cap-variant", "no_cap", "per_mech"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode != 0 and "ablation only" in proc.stderr


def test_cap_finding_is_printed_next_to_tpr_at_1pct(tmp_path):
    """Decoy only, cap on: the explanation sits next to TPR at 1% exactly when the 1% threshold
    is at or above the cap, in the output and in comparison.md."""
    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = tmp_path / "real-router" / "r"
    th = pd.read_csv(out / "thresholds.csv").set_index(["policy", "threshold"])
    row = th.loc[("router", "A_0.01")]
    at_cap = row["A"] >= 0.65 * cf.nominal_A(RouterConfig()) - 1e-9 and row["tpr"] == 0
    line = next(x for x in proc.stdout.splitlines() if x.startswith("HEADLINE"))
    assert ("finding, not a failure" in line) == at_cap
    first = (out / "comparison.md").read_text().splitlines()[0]
    assert ("finding, not a failure" in first) == at_cap


def test_headline_carries_the_cap_finding_when_the_1pct_threshold_is_at_the_cap():
    cap = 0.65 * cf.nominal_A(RouterConfig())
    summ = {
        "results": {
            "A_0.05": {"A": 2.2, "tpr": 0.8},
            "A_0.01": {"A": cap, "tpr": 0.0},
            "A_max": {"A": cap + 0.01, "tpr": 0.0},
        },
        "ci95": {"A_0.05": {"tpr": (0.6, 0.9)}, "A_0.01": {"tpr": (0.0, 0.0)}, "A_max": {}},
    }
    found = cf.cap_findings(summ, RouterConfig(), ["decoy"])
    assert set(found) == {"A_0.01", "A_max"}
    # at the cap but some session saturated it and paused: TPR > 0, so no "TPR = 0" finding
    summ_hit = {**summ, "results": {**summ["results"], "A_0.01": {"A": cap, "tpr": 1.0}}}
    assert set(cf.cap_findings(summ_hit, RouterConfig(), ["decoy"])) == {"A_max"}
    text = run_real_router.headline_text(summ, found["A_0.01"])
    assert text.startswith("HEADLINE (pilot): router TPR at 5% FPR = 0.80 [0.60, 0.90]")
    assert "Secondary: TPR at 1% FPR = 0.00 [0.00, 0.00]" in text
    assert text.index("1% FPR = 0.00") < text.index("finding, not a failure")
    assert "decoy as the only check" in text and "correctly stops any single check" in text
    assert (
        cf.cap_findings(summ, RouterConfig(**run_real_router.CAP_VARIANTS["no_cap"]), ["decoy"])
        == {}
    )


# ---------------------------------------------------------------- shared fold assignment
def _run_folds(tmp_path, *extra):
    proc = subprocess.run(
        _cmd(tmp_path, *extra),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return proc


def test_run_real_router_folds_are_the_shared_assignment(tmp_path):
    """The script's folds are exactly adr/replay/folds.assign_session_folds with the recorded
    K and seed (what scripts/calibrate_behaviour.py imports), and excluding the low-effort
    sessions keeps everyone else in the same fold."""
    from adr.replay.folds import DEFAULT_K, DEFAULT_SEED

    _files(tmp_path)
    low = tmp_path / "low_effort_ids.txt"
    low.write_text("\n".join(session_ids()[:4]) + "\n")
    proc = _run_folds(tmp_path, "--low-effort", str(low))
    assert proc.returncode == 0, proc.stderr[-3000:]
    cfg = __import__("yaml").safe_load((tmp_path / "real-router" / "r" / "config.yaml").read_text())
    expected = fold_lists(
        assign_session_folds(session_ids(), DEFAULT_K, DEFAULT_SEED, session_ids()[:4]), DEFAULT_K
    )
    assert cfg["folds"] == expected
    assert (cfg["fold_k"], cfg["fold_seed"]) == (DEFAULT_K, DEFAULT_SEED)

    (tmp_path / "x").mkdir()
    _files(tmp_path / "x")
    proc = _run_folds(tmp_path / "x", "--low-effort", str(low), "--exclude-sessions", str(low))
    assert proc.returncode == 0, proc.stderr[-3000:]
    cfg_x = __import__("yaml").safe_load(
        (tmp_path / "x" / "real-router" / "r" / "config.yaml").read_text()
    )
    gone = set(session_ids()[:4])
    assert cfg_x["folds"] == [[s for s in f if s not in gone] for f in expected]


def test_run_real_router_stops_on_an_unknown_low_effort_id(tmp_path):
    _files(tmp_path)
    low = tmp_path / "low_effort_ids.txt"
    low.write_text(session_ids()[0] + "\nd3ba4f10\n")  # a prefix is not a session id
    proc = _run_folds(tmp_path, "--low-effort", str(low))
    assert proc.returncode != 0 and "not among" in proc.stderr


# ---------------------------------------------------------------- review H3: per-group reporting
def _gt(who, score, sources=(0, 0, 0, 0)):
    return cf.Trace(who, 0, score, np.array([score]), np.array([1]), 3, sources=sources)


def test_rates_report_tpr_per_ai_group():
    ai = [
        [_gt("ai:m:copy_paste:L0|a", 5.0)],
        [_gt("ai:m:copy_paste:L0|b", 1.0)],
        [_gt("ai:m:doc_upload:L3|c", 5.0)],
    ]
    r = cf.rates([_gt("h", 0.0)], ai, [0.0], {"A": 2.0})["A"]
    assert r["tpr"] == pytest.approx(2 / 3)
    assert r["tpr[m:copy_paste:L0]"] == 0.5 and r["tpr[m:doc_upload:L3]"] == 1.0
    for g in ("m:copy_paste:L0", "m:doc_upload:L3"):  # per-group AI cleared rates too
        assert r[f"ai_ever_cleared_rate[{g}]"] == 0.0 and r[f"ai_cleared_rate[{g}]"] == 0.0
    one = cf.rates([_gt("h", 0.0)], ai[:2], [0.0], {"A": 2.0})["A"]
    assert not any(k.startswith("tpr[") for k in one)  # one group: pooled TPR only


def test_ai_sources_count_same_type_fallback_and_missing_per_group():
    ai = [
        [_gt("ai:m:copy_paste:L0|a", 1.0, (2, 0, 0, 1)), _gt("ai:m:copy_paste:L0|a", 1.0, (4, 0, 0, 1))],
        [_gt("ai:m:doc_upload:L0|c", 1.0, (0, 3, 0, 0))],
    ]
    src = cf.ai_sources(ai)
    assert src["m:copy_paste:L0"] == {
        "sessions": 1, "checks": 3.0, "same_type": 3.0, "fallback": 0.0, "synthetic": 0.0,
        "missing": 1.0,
    }
    assert src["m:doc_upload:L0"]["fallback"] == 3.0


# ---------------------------------------------------------------- review H1: --final rows
def test_run_real_router_final_uses_only_the_decided_heldout_model(tmp_path):
    from adr.replay.quick_map import decided_heldout_model

    model = decided_heldout_model()
    assert model == "claude-haiku"
    rows = [OutcomeRow(**r) for r in ai_frame().to_dict("records")]
    for name in (model, "gemini-flash"):  # the decided model + an earlier held-out model
        for s in range(3):
            for i, item in enumerate(AI_ITEMS):
                r = ai_row(f"ai-test:{name}:{s}", item, "picked_decoy" if i % 2 else "picked_correct")
                rows.append(r.model_copy(update={"model": name, "heldout_model": True}))
    _files(tmp_path)
    write_outcomes(rows, tmp_path / "ai.parquet")
    proc = subprocess.run(
        _cmd(tmp_path, "--final"), cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    n_stale = 3 * len(AI_ITEMS)
    assert f"--final: dropped {n_stale} held-out AI rows" in proc.stdout
    assert "['gemini-flash']" in proc.stdout
    cfg = yaml.safe_load((tmp_path / "real-router" / "r" / "config.yaml").read_text())
    assert cfg["final"] is True
    assert cfg["n_ai_eval_sessions"] == 3  # the claude-haiku sessions only


# ---------------------------------------------------------------- review A1: exclusion file
def test_exclude_sessions_rejects_a_handoff_style_file(tmp_path):
    _files(tmp_path)
    full = session_ids()[:2]
    # shaped like the Drive handoff note: prose plus short id prefixes, not full ids
    (tmp_path / "handoff.txt").write_text(
        "Low-effort sessions flagged in the pilot review:\n"
        + "".join(f"- {s[:8]} (fast, straight-lined)\n" for s in full),
        encoding="utf-8",
    )
    proc = subprocess.run(
        _cmd(tmp_path, "--exclude-sessions", str(tmp_path / "handoff.txt")),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode != 0
    assert "not full pilot session_ids" in proc.stderr
    # a machine-only file with full ids works and drops exactly those sessions
    (tmp_path / "ids.txt").write_text("# full ids\n" + "\n".join(full) + "\n", encoding="utf-8")
    proc = subprocess.run(
        _cmd(tmp_path, "--exclude-sessions", str(tmp_path / "ids.txt")),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    cfg = yaml.safe_load((tmp_path / "real-router" / "r" / "config.yaml").read_text())
    assert cfg["n_sessions"] == N_SESSIONS - 2


# ---------------------------------------------------------------- leave one check out
def test_run_real_router_drop_mechanisms_leave_one_out(tmp_path):
    _files(tmp_path)

    def style(r: OutcomeRow, outcome: str) -> OutcomeRow:
        return r.model_copy(
            update={
                "mechanism": "style",
                "variant_id": make_variant_id(r.item_id, "style", 0),
                "outcome": outcome,
            }
        )

    ai = [OutcomeRow(**r) for r in ai_frame().to_dict("records")]
    human = [OutcomeRow(**r) for r in human_frame().to_dict("records")]
    ai += [style(r, "high" if i % 3 else "mid") for i, r in enumerate(ai)]
    human += [style(r, "low" if i % 5 else "mid") for i, r in enumerate(human)]
    write_outcomes(ai, tmp_path / "ai.parquet")
    write_outcomes(human, tmp_path / "pilot.parquet")
    proc = subprocess.run(
        _cmd(tmp_path, "--drop-mechanisms", "style,decoy,canary"),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "--drop-mechanisms canary: not among the mechanisms used here" in proc.stdout
    out = tmp_path / "real-router" / "r"
    loo = pd.read_csv(out / "leave_one_out.csv")
    assert list(dict.fromkeys(loo.dropped)) == ["style", "decoy"]  # in the order given
    assert set(loo[loo.dropped == "style"].policy) == {"router", "best_single", "all_checks"}
    # without decoy only the passive style check is left: baselines need an active check
    assert set(loo[loo.dropped == "decoy"].policy) == {"router"}
    assert "only passive checks remain" in proc.stdout
    assert len(loo) == (3 + 1) * 3  # (policies without style + without decoy) x thresholds
    assert {"tpr", "checks_per_honest_session", "honest_pause_rate", "nominal_fpr"} <= set(loo)
    md = (out / "comparison.md").read_text(encoding="utf-8")
    assert "Leave one check out" in md and "nominal_fpr" in md
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["drop_mechanisms"] == ["style", "decoy"]
    comp = pd.read_csv(out / "comparison.csv")
    router = comp[comp.policy == "router"]
    nominal = dict(zip(router.threshold, router.nominal_fpr, strict=True))
    assert nominal == {"A_0.05": 0.05, "A_0.01": 0.01, "A_max": 0.0}


def test_run_real_router_clear_policy_and_ever_cleared_columns(tmp_path):
    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path, "--clear-policy", "sticky", "--cap-variant", "per_mech", "no_cap"),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = tmp_path / "real-router" / "r"
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["router_config"]["clear_policy"] == "sticky"
    comp = pd.read_csv(out / "comparison.csv")
    for c in ("ai_ever_cleared_rate", "ai_cleared_rate", "honest_ever_cleared_rate",
              "honest_cleared_rate"):
        assert c in comp.columns
    # ever cleared >= cleared at the end, for every policy and threshold
    assert (comp.honest_ever_cleared_rate >= comp.honest_cleared_rate - 1e-12).all()
    assert (comp.ai_ever_cleared_rate >= comp.ai_cleared_rate - 1e-12).all()


# ---------------------------------------------------------------- --jobs and --resume
def _two_mechanism_files(tmp_path):
    """The pilot fixture plus passive style rows, so there are leave-one-check-out tasks too."""
    _files(tmp_path)

    def style(r: OutcomeRow, outcome: str) -> OutcomeRow:
        return r.model_copy(
            update={
                "mechanism": "style",
                "variant_id": make_variant_id(r.item_id, "style", 0),
                "outcome": outcome,
            }
        )

    ai = [OutcomeRow(**r) for r in ai_frame().to_dict("records")]
    human = [OutcomeRow(**r) for r in human_frame().to_dict("records")]
    ai += [style(r, "high" if i % 3 else "mid") for i, r in enumerate(ai)]
    human += [style(r, "low" if i % 5 else "mid") for i, r in enumerate(human)]
    write_outcomes(ai, tmp_path / "ai.parquet")
    write_outcomes(human, tmp_path / "pilot.parquet")


TASK_ARGS = ("--cap-variant", "per_mech", "no_cap", "--drop-mechanisms", "style,decoy")
TABLES = ("comparison.csv", "comparison.md", "thresholds.csv", "leave_one_out.csv",
          "budget_curve.csv")


def _run(tmp_path, run_id, *extra):
    cmd = [a if a != "r" else run_id for a in _cmd(tmp_path, *TASK_ARGS, *extra)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return tmp_path / "real-router" / run_id, proc


@pytest.mark.skipif(
    not run_real_router.fork_available(),
    reason="--jobs > 1 needs fork (not on Windows): it falls back to one task at a time",
)
def test_jobs_gives_identical_outputs(tmp_path):
    _two_mechanism_files(tmp_path)
    serial, _ = _run(tmp_path, "serial", "--jobs", "1")
    par, _ = _run(tmp_path, "parallel", "--jobs", "4")
    for name in TABLES:  # same numbers, same row order
        assert (serial / name).read_bytes() == (par / name).read_bytes(), name
    m1 = json.loads((serial / "metrics.json").read_text())
    m4 = json.loads((par / "metrics.json").read_text())
    assert m1["results"] == m4["results"]
    steps1 = sorted(p.name for p in (serial / "steps").glob("*.json"))
    assert steps1 == sorted(p.name for p in (par / "steps").glob("*.json"))
    assert len(steps1) == 6 + 1 + (3 + 1) + 2  # policies, no_cap, loo (style: 3, decoy: 1), betas
    for name in steps1:  # same summary and the same bootstrap seed per task
        a = json.loads((serial / "steps" / name).read_text())
        b = json.loads((par / "steps" / name).read_text())
        assert a["summary"] == b["summary"] and a["seed"] == b["seed"] == 0, name
        assert a["fingerprint"] == b["fingerprint"]  # --jobs is not a setting
    inputs = (par / "inputs.txt").read_text()
    assert "ai_outcomes:" in inputs
    digest = hashlib.sha256((tmp_path / "pilot.parquet").read_bytes()).hexdigest()
    assert f"pilot.parquet  sha256 {digest}" in inputs  # the pilot file's hash is recorded


def test_resume_reruns_only_missing_steps_and_refuses_other_settings(tmp_path):
    _two_mechanism_files(tmp_path)
    out, _ = _run(tmp_path, "r", "--jobs", "2")
    first = {name: (out / name).read_bytes() for name in TABLES}
    steps = out / "steps"
    lost = steps / "loo-style-router.json"
    lost.unlink()  # as if the run crashed during this step
    mtimes = {p.name: p.stat().st_mtime_ns for p in steps.glob("*.json")}
    out, proc = _run(tmp_path, "r", "--resume")
    assert lost.exists()
    assert "resume: policy-router loaded" in proc.stdout
    assert "step done: loo-style-router" in proc.stdout
    for p in steps.glob("*.json"):  # nothing else was recomputed
        if p.name != lost.name:
            assert p.stat().st_mtime_ns == mtimes[p.name], p.name
    for name in TABLES:
        assert (out / name).read_bytes() == first[name], name
    # a step made with other settings is never reused
    cmd = [a if a != "0.05,0.2" else "0.05" for a in _cmd(tmp_path, *TASK_ARGS, "--resume")]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          check=False)
    assert proc.returncode != 0 and "other settings or inputs" in proc.stderr


def test_run_real_router_neg_floor_frac_reaches_every_router_row(tmp_path):
    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path, "--neg-floor-frac", "0.65", "--cap-variant", "per_mech", "no_cap"),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    cfg = yaml.safe_load((tmp_path / "real-router" / "r" / "config.yaml").read_text())
    assert cfg["router_config"]["neg_floor_frac"] == 0.65


def test_run_real_router_clear_requires_style_human_is_off_unless_asked(tmp_path):
    """The pre-registered ablation flag reaches every router row's config; off by default."""
    _files(tmp_path)
    for flag, rid in (((), "off"), (("--clear-requires-style-human",), "on")):
        proc = subprocess.run(
            _cmd(tmp_path, "--cap-variant", "per_mech", *flag, "--run-id", rid),  # last id wins
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        cfg = yaml.safe_load((tmp_path / "real-router" / rid / "config.yaml").read_text())
        assert cfg["router_config"]["clear_requires_style_human"] is bool(flag)


def test_run_real_router_config_records_the_git_state_after_build_parser(tmp_path):
    """After the build_parser() refactor the run config still records the code that ran, as main
    does: git_state() (HEAD + tracked-files dirty flag) read when the run starts."""
    from adr.harness.provenance import git_state

    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    cfg = yaml.safe_load((tmp_path / "real-router" / "r" / "config.yaml").read_text())
    assert {"git_commit", "git_dirty"} <= set(cfg)
    expected = git_state(ROOT)
    assert cfg["git_commit"] == expected["git_commit"]
    assert cfg["git_dirty"] == expected["git_dirty"]
    assert isinstance(cfg["git_dirty"], bool)


def test_jobs_falls_back_to_one_at_a_time_without_fork(tmp_path, monkeypatch, capsys):
    """Windows has no fork: --jobs > 1 must not crash, it runs the tasks serially."""
    monkeypatch.setattr(run_real_router.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    assert not run_real_router.fork_available()
    tasks = {f"t{i}": (lambda i=i: {"value": i}) for i in range(3)}
    done = run_real_router.run_tasks(tasks, tmp_path, jobs=4, resume=False, seed=0)
    assert done == {f"t{i}": {"value": i} for i in range(3)}
    assert "needs the 'fork' start method" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["t0.json", "t1.json", "t2.json"]

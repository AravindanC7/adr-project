"""Person-level cross-fitting (adr/replay/crossfit.py), mech_group_caps, run_real_router.py.

All data is SYNTHETIC, pilot-shaped rows built here (not the dry-run data: the builder rejects
data_origin "dry_run"). No model calls.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from adr.replay.crossfit import (
    DEFAULT_GRID,
    calibrate,
    crossfit,
    fold_map,
    person_folds,
    pooled_metrics,
    router_factory,
)
from adr.replay.donors import Donors, Group
from adr.router.core import RouterConfig, SPRTRouter
from adr.router.map_guard import ProvisionalMapError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_real_router

N_PEOPLE, N_PILOT_ITEMS, N_AI_ITEMS, N_AI_SESSIONS = 20, 8, 40, 5


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


def human_row(person: str, item_id: str, outcome: str) -> OutcomeRow:
    return OutcomeRow(
        run_id="pilot-test",
        session_id=person,
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


def ai_row(session: str, item_id: str, outcome: str, heldout: bool = False) -> OutcomeRow:
    return OutcomeRow(
        run_id="ai-test",
        session_id=session,
        respondent_kind="ai",
        data_origin="harness",
        model="claude-haiku" if heldout else "gpt-mini",
        attack_mode="copy_paste",
        attacker_level="L0",
        assessment_type="quiz",
        item_type="mcq",
        item_id=item_id,
        split="train",
        heldout_model=heldout,
        mechanism="decoy",
        variant_id=make_variant_id(item_id, "decoy", 0),
        outcome=outcome,
        raw_ref="x",
        prompt_version="t",
    )


def people() -> list[str]:
    return [f"pilot-test:p{j:02d}" for j in range(N_PEOPLE)]


def human_frame() -> pd.DataFrame:
    rows = []
    for j, p in enumerate(people()):
        for i, item in enumerate(PILOT_ITEMS):
            # most people never pick the decoy; every 4th person picks it once
            decoy = j % 4 == 0 and i == 0
            rows.append(human_row(p, item.item_id, "picked_decoy" if decoy else "picked_correct"))
    return rows_to_frame(rows)


def ai_frame(heldout: bool = False) -> pd.DataFrame:
    rows = []
    for s in range(N_AI_SESSIONS):
        for i, item in enumerate(AI_ITEMS):
            outcome = "picked_decoy" if (i + s) % 4 else "picked_correct"  # 75% decoy
            rows.append(ai_row(f"ai-test:gpt-mini:{s}", item, outcome, heldout))
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


# ---------------------------------------------------------------- folds
def test_folds_partition_people_with_no_overlap():
    ps = people()
    folds = person_folds(ps, 4, seed=0)
    flat = [p for f in folds for p in f]
    assert sorted(flat) == sorted(ps)  # every person used ...
    assert len(flat) == len(set(flat))  # ... exactly once
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    assert person_folds(ps, 4, seed=0) == folds  # fixed seed
    assert sorted(len(f) for f in person_folds(ps[:7], 3, seed=1)) == [2, 2, 3]
    with pytest.raises(ValueError):
        person_folds(ps[:3], 4, 0)


# ---------------------------------------------------------------- leakage
def test_map_for_fold_k_has_no_human_rows_from_fold_k():
    human, ai = human_frame(), ai_frame()
    folds = person_folds(people(), 4, 0)
    held = folds[0]
    fmap = fold_map(ai, human, held, "f0", n_bootstrap=20)
    cell = fmap.lookup("decoy", "mcq")
    n_others = int((~human.session_id.isin(held)).sum())
    assert sum(cell.counts_human.values()) == n_others  # only the other folds' rows
    # changing fold k's answers cannot change map_k
    flipped = human.copy()
    flipped.loc[flipped.session_id.isin(held), "outcome"] = "picked_decoy"
    assert fold_map(ai, flipped, held, "f0", n_bootstrap=20) == fmap


# ---------------------------------------------------------------- thresholds
def test_threshold_uses_only_fold_k_people_and_is_the_smallest_ok_grid_value():
    human, ai = human_frame(), ai_frame()
    assessment, items, variants = bank()
    folds = person_folds(people(), 4, 0)
    make = router_factory("router", assessment, items, variants, RouterConfig())
    res = crossfit(make, assessment, items, ai, human, ai, folds, alpha=0.01)
    for f, fold in zip(res, folds, strict=True):
        assert sorted(r.donor_session for r in f.honest) == sorted(fold)  # fold k only
        # the same A_k comes out of calibrating on fold k's rows alone
        only_k = human[human.session_id.isin(fold)]
        fmap = fold_map(ai, human, fold, "x", n_bootstrap=200)
        donors = Donors(only_k, items, fallback_map=fmap)
        a, honest, _ = calibrate(make, fmap, assessment, donors, Group.human("pilot"), fold, 0.01)
        assert a == f.threshold
        assert np.mean([r.final.status == "review" for r in honest]) <= 0.01
        i = DEFAULT_GRID.index(a) if a in DEFAULT_GRID else None
        if i:  # the grid value just below pauses too many of fold k
            lower = _replay_rate(make(fmap, DEFAULT_GRID[i - 1]), assessment, donors, fold)
            assert lower > 0.01
        # every AI session is replayed in every fold
        assert len(f.ai) == N_AI_SESSIONS


def _replay_rate(policy, assessment, donors, fold):
    from adr.replay.simulate import replay_session

    g = Group.human("pilot")
    rs = [replay_session(policy, assessment, donors.respondent_from(g, p, 0), (), 0) for p in fold]
    return float(np.mean([r.final.status == "review" for r in rs]))


def test_ai_sessions_draw_from_their_own_same_type_rows():
    _, items, _ = bank()
    donors = Donors(ai_frame(), items, type_match_kinds=("ai",))
    g = donors.groups("ai")[0]
    d = donors.respondent_from(g, "ai-test:gpt-mini:0", seed=0)
    got = [d.outcome(it.item_id, "decoy") for it in PILOT_ITEMS]
    own = ai_frame().query("session_id == 'ai-test:gpt-mini:0'").outcome.tolist()
    assert d.same_type == N_PILOT_ITEMS and d.fallbacks == 0 and d.synthetic == 0
    assert set(got) <= set(own)
    # without replacement: 8 draws from 30 decoys + 10 correct can't exceed those counts
    assert got.count("picked_correct") <= own.count("picked_correct")
    # default Donors (no type matching) are unchanged: no same-type draws
    assert Donors(ai_frame(), items).respondent_from(g, "ai-test:gpt-mini:0", 0)._own_type == {}


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_resamples_people_and_ai_sessions_not_rows():
    human, ai = human_frame(), ai_frame()
    assessment, items, variants = bank()
    folds = person_folds(people(), 4, 0)
    make = router_factory("router", assessment, items, variants, RouterConfig())
    res = crossfit(make, assessment, items, ai, human, ai, folds, alpha=0.05)
    m = pooled_metrics(res, n_boot=300, seed=0)
    assert m["n_people"] == N_PEOPLE  # 20 people, not 160 rows
    assert m["n_ai_sessions"] == N_AI_SESSIONS  # 5 sessions, not 5 x 4 folds
    assert m["boot_unit"].startswith("person")
    assert m["thresholds"] == [f.threshold for f in res]
    assert m["granularity"] == pytest.approx(1 / 5)
    lo, hi = m["ci95"]["checks_per_honest_person"]
    assert lo <= m["checks_per_honest_person"] <= hi


def test_bootstrap_draw_count_is_people(monkeypatch):
    # every bootstrap draw picks one index per PERSON and one per AI SESSION
    sizes = []
    real_rng = np.random.default_rng

    class Spy:
        def __init__(self, seed):
            self.rng = real_rng(seed)

        def integers(self, low, high=None, size=None):
            sizes.append(size)
            return self.rng.integers(low, high, size)

    human, ai = human_frame(), ai_frame()
    assessment, items, variants = bank()
    folds = person_folds(people(), 4, 0)
    res = crossfit(
        router_factory("router", assessment, items, variants, RouterConfig()),
        assessment,
        items,
        ai,
        human,
        ai,
        folds,
        alpha=0.05,
    )
    import adr.replay.crossfit as cf

    monkeypatch.setattr(cf.np.random, "default_rng", Spy)
    pooled_metrics(res, n_boot=10, seed=0)
    assert sizes and set(sizes) == {N_AI_SESSIONS, N_PEOPLE}


# ---------------------------------------------------------------- exclusions
def test_exclude_sessions_removes_exactly_those(tmp_path):
    human = human_frame()
    f = tmp_path / "low_effort.txt"
    f.write_text("# low-effort people\npilot-test:p03\n\npilot-test:p07\nnot-a-session\n")
    excluded = run_real_router.read_exclusions(str(f))
    kept, missing = run_real_router.exclude_people(human, excluded)
    assert set(human.session_id) - set(kept.session_id) == {"pilot-test:p03", "pilot-test:p07"}
    assert len(kept) == len(human) - 2 * N_PILOT_ITEMS  # every row of those two, nothing else
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
    cfg = RouterConfig(upper=100.0, mech_cap_frac=0.03)  # cap = 3 nats
    router, separate = _feed(cfg)
    assert separate.score == pytest.approx(2 * router.mech_cap)  # each mechanism capped alone
    router, grouped = _feed(
        RouterConfig(
            upper=100.0, mech_cap_frac=0.03, mech_group_caps={"passive": ("behaviour", "style")}
        )
    )
    assert grouped.score == pytest.approx(router.mech_cap)  # ONE cap for both
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
    folds = person_folds(people(), 4, 0)
    make = router_factory("router", assessment, items, variants, RouterConfig())
    with pytest.raises(ProvisionalMapError):
        crossfit(make, assessment, items, ai, human, ai, folds, 0.01, maps=[ai_only] * 4)
    with pytest.raises(ValueError, match="both AI and human"):  # strict builder: no human rows
        fold_map(ai, human, people(), "all-held-out", n_bootstrap=10)


# ---------------------------------------------------------------- the one command
def _files(tmp_path):
    write_outcomes(
        [OutcomeRow(**r) for r in ai_frame().to_dict("records")], tmp_path / "ai.parquet"
    )
    write_outcomes(
        [OutcomeRow(**r) for r in human_frame().to_dict("records")], tmp_path / "pilot.parquet"
    )
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
        "50",
        "--betas",
        "0.05,0.2",
        "--calibration-frac",
        "0.5",
        "--results-dir",
        str(tmp_path),
        "--run-id",
        "r",
        *extra,
    ]


def test_run_real_router_end_to_end(tmp_path):
    _files(tmp_path)
    proc = subprocess.run(
        _cmd(tmp_path), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "BOTH AI and human rows (the only ones used): decoy" in proc.stdout
    assert "decoy only" in proc.stdout
    out = tmp_path / "real-router" / "r"
    for name in (
        "comparison.csv",
        "comparison.md",
        "thresholds.csv",
        "budget_curve.csv",
        "budget_curve.png",
        "metrics.json",
        "config.yaml",
    ):
        assert (out / name).exists(), name
    th = pd.read_csv(out / "thresholds.csv")
    assert set(th.policy) == {
        "router",
        "router_noclear",
        "best_single",
        "all_checks",
        "fixed_plan",
        "random",
    }
    assert len(th) == 6 * 2 * 4  # policies x alphas (0.01, 0.05) x folds
    curve = pd.read_csv(out / "budget_curve.csv")
    assert list(curve[curve.policy == "router"].beta) == [0.05, 0.2]


def test_run_real_router_needs_pilot_rows(tmp_path):
    _files(tmp_path)
    write_outcomes(
        [OutcomeRow(**r) for r in ai_frame().to_dict("records")], tmp_path / "pilot.parquet"
    )  # no human rows
    proc = subprocess.run(
        _cmd(tmp_path), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode != 0 and "no pilot rows" in proc.stderr

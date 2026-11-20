"""scripts/error_analysis.py on a SYNTHETIC run: synthetic pilot-shaped honest rows and synthetic
AI rows (decoy + canary; L2 attackers answer the canary cleanly), a real scripts/run_real_router.py
run on them, then the error analysis on its folder. No model calls, no pilot data."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from adr.contracts import Item, OutcomeRow, make_variant_id, write_outcomes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import error_analysis as ea

N_HUMAN, N_AI_PER_GROUP = 24, 8
PILOT_MCQ = [f"arc:pilot-{i}" for i in range(6)]
PILOT_OPEN = [f"handwritten:popen-{i}" for i in range(6)]
AI_MCQ = [f"mmlu:bank-{i}" for i in range(30)]
AI_OPEN = [f"handwritten:open-{i}" for i in range(20)]
GROUPS = [("copy_paste", "L0"), ("copy_paste", "L2"), ("doc_upload", "L0"), ("doc_upload", "L2")]


def _mcq(item_id):
    return Item(
        item_id=item_id,
        source="arc",
        assessment_type="quiz",
        item_type="mcq",
        stem="Q?",
        options=["a", "b", "c", "d"],
        correct_option=1,
        eligible_mechanisms=["decoy"],
    )


def _open(item_id):
    return Item(
        item_id=item_id,
        source="handwritten",
        assessment_type="assignment",
        item_type="open_text",
        stem="Explain.",
        eligible_mechanisms=["canary"],
    )


def _row(**kw):
    base = {"split": "train", "raw_ref": "x", "assessment_type": "quiz"}
    mech = kw["mechanism"]
    if mech == "canary":
        base |= {"assessment_type": "assignment", "item_type": "open_text"}
    else:
        base |= {"item_type": "mcq"}
    base["variant_id"] = make_variant_id(kw["item_id"], mech, 0)
    return OutcomeRow(**(base | kw))


def make_files(tmp: Path) -> None:
    rng = random.Random(7)
    human = []
    for j in range(N_HUMAN):
        sid = f"pilot-syn:p{j:02d}"
        common = {
            "run_id": "pilot-syn",
            "session_id": sid,
            "respondent_kind": "human",
            "data_origin": "pilot",
        }
        for it in PILOT_MCQ:
            o = "picked_decoy" if rng.random() < 0.08 else "picked_correct"
            human.append(_row(**common, item_id=it, mechanism="decoy", outcome=o))
        for it in PILOT_OPEN:
            o = "leaked" if rng.random() < 0.03 else "clean"
            human.append(_row(**common, item_id=it, mechanism="canary", outcome=o))
    ai = []
    for mode, level in GROUPS:
        for bank, items, mech in (("q", AI_MCQ, "decoy"), ("o", AI_OPEN, "canary")):
            run_id = f"syn-{mode}-{level}-{bank}"
            for s in range(N_AI_PER_GROUP):
                sid = f"{run_id}:gpt-mini:{s}" + (":d0" if mode == "doc_upload" else "")
                for it in items:
                    if mech == "decoy":
                        p = 0.8 if level == "L0" else 0.3
                        o = "picked_decoy" if rng.random() < p else "picked_correct"
                    else:
                        o = (
                            ("leaked" if rng.random() < 0.9 else "clean")
                            if level == "L0"
                            else "clean"
                        )
                    ai.append(
                        _row(
                            run_id=run_id,
                            session_id=sid,
                            respondent_kind="ai",
                            data_origin="harness",
                            model="gpt-mini",
                            attack_mode=mode,
                            attacker_level=level,
                            prompt_version="cp-v1",
                            item_id=it,
                            mechanism=mech,
                            outcome=o,
                        )
                    )
    write_outcomes(human, tmp / "pilot.parquet")
    write_outcomes(ai, tmp / "ai.parquet")
    items = [_mcq(i) for i in PILOT_MCQ] + [_open(i) for i in PILOT_OPEN]
    bank = {
        "flow": {"title": "t", "parts": ["pilot-quiz", "pilot-open"]},
        "items": [it.model_dump(exclude_defaults=True) | {"split": "train"} for it in items],
        "assessments": [
            {
                "assessment_id": "pilot-quiz",
                "title": "Quiz",
                "assessment_type": "quiz",
                "item_ids": PILOT_MCQ,
            },
            {
                "assessment_id": "pilot-open",
                "title": "Open",
                "assessment_type": "assignment",
                "item_ids": PILOT_OPEN,
            },
        ],
    }
    (tmp / "bank.json").write_text(json.dumps(bank), encoding="utf-8")


def run_router(tmp: Path) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_real_router.py"),
        "--ai-outcomes",
        str(tmp / "ai.parquet"),
        "--pilot-outcomes",
        str(tmp / "pilot.parquet"),
        "--bank",
        str(tmp / "bank.json"),
        "--boot",
        "5",
        "--n-synthetic",
        "40",
        "--n-synthetic-boot",
        "5",
        "--betas",
        "0.05",
        "--calibration-frac",
        "0.5",
        "--results-dir",
        str(tmp),
        "--run-id",
        "syn",
        "--no-low-effort",
    ]
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return tmp / "real-router" / "syn"


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ea")
    make_files(tmp)
    run = run_router(tmp)
    out = ea.main(
        ["--run", str(run), "--boot", "200", "--out-dir", str(tmp / "ea"), "--threshold", "A_0.05"]
    )
    return tmp, out


# ---------------------------------------------------------------- end to end
def test_outputs_exist(analysed):
    _, out = analysed
    for name in (
        "error_analysis.md",
        "group_outcomes.csv",
        "cleared_breakdown.csv",
        "undecided.csv",
        "honest_pauses.csv",
        "examples.csv",
        "counterfactual.csv",
        "sessions.csv",
        "steps.csv",
        "consistency.csv",
    ):
        assert (out / name).exists(), name
    pytest.importorskip("matplotlib")
    assert (out / "example_trajectories.png").exists()


def test_replay_matches_the_run(analysed):
    _, out = analysed
    cons = pd.read_csv(out / "consistency.csv")
    assert len(cons) > 0
    assert cons["match"].all(), cons


def test_no_session_or_item_ids_in_any_output(analysed):
    _, out = analysed
    for f in out.iterdir():
        if f.suffix in (".csv", ".md", ".json"):
            text = f.read_text(encoding="utf-8")
            for bad in (
                "pilot-syn",
                "syn-copy",
                "syn-doc",
                "composite:",
                "arc:",
                "mmlu:",
                "handwritten:",
            ):
                assert bad not in text, (f.name, bad)
    labels = pd.read_csv(out / "sessions.csv")["label"]
    assert labels.str.match(r"^(Human|AI-L[02]-(cp|up))-#\d+$").all()


def test_group_outcomes_partition_and_counts(analysed):
    _, out = analysed
    g = pd.read_csv(out / "group_outcomes.csv")
    assert set(g["group"]) == {"honest"} | {f"gpt-mini:{m}:{lv}" for m, lv in GROUPS}
    s = g["paused"] + g["cleared"] + g["undecided"]
    assert (s - 1).abs().max() < 1e-9
    ai = g[g.group != "honest"]
    assert (ai["n_sessions"] == N_AI_PER_GROUP).all()
    assert (g["paused_lo"] <= g["paused"] + 1e-12).all()
    assert (g["paused"] <= g["paused_hi"] + 1e-12).all()


def test_l2_is_cleared_by_the_canary(analysed):
    _, out = analysed
    g = pd.read_csv(out / "group_outcomes.csv").set_index("group")
    assert (
        g.loc["gpt-mini:copy_paste:L2", "ever_cleared"]
        > g.loc["gpt-mini:copy_paste:L0", "ever_cleared"]
    )
    cb = pd.read_csv(out / "cleared_breakdown.csv")
    l2 = cb[cb.group == "gpt-mini:copy_paste:L2"]
    assert l2.sessions.sum() > 0
    assert set(l2.clear_mechanism) <= {"canary", "decoy"}


def test_counterfactual_rule_bounds(analysed):
    _, out = analysed
    cf = pd.read_csv(out / "counterfactual.csv")
    rule = cf[cf.method.str.startswith("rule")]
    assert (rule.flip + rule.stay_cleared - rule.wrongly_cleared).abs().max() < 1e-9
    g = pd.read_csv(out / "group_outcomes.csv").set_index("group")
    for _, r in rule.iterrows():
        assert abs(r.wrongly_cleared - g.loc[r.group, "ever_cleared_n"]) < 1e-9


def test_examples_are_short_and_labelled(analysed):
    _, out = analysed
    ex = pd.read_csv(out / "examples.csv")
    assert set(ex.columns) == {"failure", "label", "fold_map", "step", "check", "outcome", "score"}
    assert (ex.groupby("failure")["label"].nunique() <= 4).all()


def test_run_folder_without_config_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="config.yaml"):
        ea.main(["--run", str(tmp_path)])


# ---------------------------------------------------------------- units
def _rec(scores, mechs, outcomes=None, cleared_step=None, cleared_end=False):
    steps, contrib, prev = [], {}, 0.0
    for i, (s, m) in enumerate(zip(scores, mechs, strict=True)):
        d = s - prev
        if m:
            contrib = dict(contrib)
            contrib[m] = contrib.get(m, 0.0) + d
        steps.append(ea.Step(m, (outcomes or ["x"] * len(scores))[i], d, s, i + 1, contrib))
        prev = s
    return ea.Rec("id", "ai", "g", 0, 0.0, steps, len(scores), cleared_end, cleared_step)


def test_outcome_partition_and_ever_cleared():
    r = _rec([-1.0, -3.0, -1.0], ["decoy", "canary", "decoy"], cleared_step=1)
    assert r.outcome_at(4.0) == "undecided"  # reopened: not cleared at the end
    assert r.ever_cleared_at(4.0)
    assert not r.ever_cleared_at(-0.5)  # it was already past this "A" before clearing
    p = _rec([2.0, 5.0], ["decoy", "decoy"])
    assert p.outcome_at(4.0) == "paused" and p.first_cross(4.0) == 1


def test_shapes_and_negative_mechanisms():
    one = _rec([-3.0], ["canary"], cleared_step=0, cleared_end=True)
    assert ea.shape_of(one) == "one mechanism"
    assert ea.n_negative(one.steps[0].contrib) == 1
    rose = _rec([1.0, -3.0], ["decoy", "canary"], cleared_step=1, cleared_end=True)
    assert ea.shape_of(rose) == "rose then fell"
    assert ea.n_negative(rose.steps[1].contrib) == 1  # decoy is +1, canary -4
    down = _rec([-1.0, -3.0], ["decoy", "canary"], cleared_step=1, cleared_end=True)
    assert ea.shape_of(down) == "steady descent"
    assert ea.n_negative(down.steps[1].contrib) == 2


def test_counterfactual_rule_flips_single_mechanism_clearings():
    df = pd.DataFrame(
        [
            {
                "label": "AI-L2-cp-#1",
                "kind": "ai",
                "group": "g",
                "fold": 0,
                "ever_cleared": True,
                "clear_n_neg_mechanisms": 1,
            },
            {
                "label": "AI-L2-cp-#1",
                "kind": "ai",
                "group": "g",
                "fold": 1,
                "ever_cleared": True,
                "clear_n_neg_mechanisms": 2,
            },
            {
                "label": "AI-L2-cp-#2",
                "kind": "ai",
                "group": "g",
                "fold": 0,
                "ever_cleared": False,
                "clear_n_neg_mechanisms": None,
            },
            {
                "label": "AI-L2-cp-#2",
                "kind": "ai",
                "group": "g",
                "fold": 1,
                "ever_cleared": False,
                "clear_n_neg_mechanisms": None,
            },
        ]
    )
    r = ea.counterfactual_rule(df).iloc[0]
    assert r.n_sessions == 2 and r.wrongly_cleared == 1.0 and r.flip == 0.5


def test_labels_hide_ids_and_group_by_prefix():
    recs = [
        ea.Rec(w, k, g, f, 0.0, [], 0, False, None)
        for w, k, g in (
            ("secret-a", "honest", "honest"),
            ("secret-b", "honest", "honest"),
            ("ai|x", "ai", "gpt-mini:copy_paste:L2"),
            ("ai|y", "ai", "gpt-mini:doc_upload:L2g"),
        )
        for f in (0, 1)
    ]
    ea.assign_labels(recs)
    labels = {r.label for r in recs}
    assert labels == {"Human-#1", "Human-#2", "AI-L2-cp-#1", "AI-L2g-up-#1"}
    assert len({(r.who, r.label) for r in recs}) == 4  # same label for all fold traces


def test_router_config_refuses_settings_it_cannot_replay():
    with pytest.raises(SystemExit, match="settings"):
        ea.router_config({"alpha": 0.01, "made_up_setting": 3})
    assert ea.router_config({"alpha": 0.01}).alpha == 0.01


def test_counterfactual_replay_when_the_router_supports_it(analysed):
    """The full re-replay needs RouterConfig.neg_floor_frac (arav/rq2-l2); skipped without it."""
    import dataclasses

    from adr.router.core import RouterConfig

    _, out = analysed
    cf = pd.read_csv(out / "counterfactual.csv")
    rep = cf[cf.method.str.startswith("re-replay")]
    if "neg_floor_frac" not in {f.name for f in dataclasses.fields(RouterConfig)}:
        assert rep.empty
        pytest.skip("RouterConfig has no neg_floor_frac on this branch")
    assert not rep.empty
    rep = rep.fillna(0.0)
    now = sum(rep.get(f"now_{o}", 0.0) for o in ("paused", "cleared", "undecided"))
    assert (now - rep.wrongly_cleared).abs().max() < 1e-9
    info = json.loads((out / "counterfactual_replay.json").read_text(encoding="utf-8"))
    assert info["n_honest"] == N_HUMAN and info["frac"] == 0.65

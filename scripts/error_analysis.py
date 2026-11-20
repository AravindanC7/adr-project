"""Router error analysis for one real-router run (RQ2): where and how the router goes wrong.

    python scripts/error_analysis.py --run results/real-router/<run_id> \\
        [--ai-outcomes ... --pilot-outcomes ... --bank ...]   # only if the paths moved
        [--threshold A_0.01] [--boot 2000] [--examples 4] [--counterfactual-frac 0.65]

The run folder (scripts/run_real_router.py) keeps only aggregates, so this script REPLAYS the
router (the main design, "router" policy) on the same inputs, folds, split, seed and config that
config.yaml records, and keeps every step: check, outcome, score change, running score and the
running total per mechanism. Replay only: no model calls. It then checks that its per-group TPR /
cleared rates match metrics.json (consistency.csv) before anything else is read.

Unit: as in crossfit.py, each honest session has one out-of-fold trace; each AI session has one
trace per fold map, and its rates are the mean over those traces (so every session counts once).
Thresholds are the run's own (config.yaml router_thresholds); the CIs here resample SESSIONS
within each group with the threshold held fixed (the run's CIs also refit the threshold).

Outcome of a trace at threshold A:
  paused     its highest running score reaches A (pausing happens exactly then);
  cleared    not paused and the replay ENDS cleared (score <= B);
  undecided  neither (it ran out of items). `ever_cleared` (cleared before reaching A, maybe
             reopened later) is reported next to these, as in the run's ai_ever_cleared_rate.
For an AI session "cleared" is a wrong clearing and "undecided" a miss; for an honest session
"paused" is a false pause.

Outputs, in results/error_analysis/<run_id>/ (gitignored; no session ids anywhere: sessions are
AI-<level>-<mode>-#k / Human-#k, numbered by a hash of the id; no item ids, no answer text):
  group_outcomes.csv     (a) per group: paused / cleared / undecided / ever cleared, 95% CIs
  cleared_breakdown.csv  (b) wrongly cleared AI: which check cleared, steps, shape, contrib
  undecided.csv          (c) undecided AI: checks shown, distance of the final score to A
  honest_pauses.csv      (d) honest false pauses: the check that pushed the score over A
  examples.csv           (e) 3-4 anonymised trajectories per failure type (outcome + score)
  counterfactual.csv     (f) two-check clearing: rule on the logged trajectories, plus a full
                         re-replay with RouterConfig.neg_floor_frac when the code supports it
  sessions.csv, steps.csv  every trace and every step (anonymised)
  consistency.csv, error_analysis.md, example_trajectories.png/.pdf
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from adr.contracts import ACTIVE_MECHANISMS, PASSIVE_MECHANISMS, parse_variant_id
from adr.replay.composite import (
    add_attacker_variant,
    check_allowed,
    join_ai_sessions,
)
from adr.replay.crossfit import (
    HONEST,
    ai_group,
    all_maps,
    router_factory,
    score_config,
    score_policy,
    thresholds_from,
    trace,
)
from adr.replay.donors import Donors
from adr.replay.quick_map import (
    CALIBRATION_FRAC,
    SPLIT_SEED,
    decided_heldout_model,
    final_eval_rows,
    split_rows,
)
from adr.replay.simulate import replay_session
from adr.router.core import RouterConfig

TOL = 1e-9
TWO_CHECK_FRAC = 0.65  # neg_floor_frac for "two-check clearing" (adr/router/core.py, arav/rq2-l2)
MODE_SHORT = {"copy_paste": "cp", "doc_upload": "up", "browser_agent": "ba"}
OUTCOMES = ("paused", "cleared", "undecided")


# ------------------------------------------------------------------ data classes
@dataclass
class Step:
    mechanism: str | None  # None = plain item
    outcome: str
    delta: float
    score: float  # running score after this update
    checks: int  # active checks served so far
    contrib: dict[str, float]  # running total per mechanism (cap key) after this update


@dataclass
class Rec:
    """One replayed trace (one session under one fold map), with every step kept."""

    who: str  # real id: kept in memory only, never written
    kind: str  # "ai" | "honest"
    group: str  # model:attack_mode:level for AI, "honest" for pilot sessions
    fold: int
    prior: float
    steps: list[Step]
    total_checks: int
    cleared_end: bool
    cleared_step: int | None
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return max([self.prior] + [s.score for s in self.steps])

    @property
    def final_score(self) -> float:
        return self.steps[-1].score if self.steps else self.prior

    def outcome_at(self, a: float) -> str:
        if self.score >= a:
            return "paused"
        return "cleared" if self.cleared_end else "undecided"

    def ever_cleared_at(self, a: float) -> bool:
        if self.cleared_step is None:
            return False
        upto = [self.prior] + [s.score for s in self.steps[: self.cleared_step + 1]]
        return max(upto) < a

    def first_cross(self, a: float) -> int | None:
        for i, s in enumerate(self.steps):
            if s.score >= a:
                return i
        return None


# ------------------------------------------------------------------ the run folder
def load_run(run_dir: Path) -> tuple[dict, dict]:
    conf_path = run_dir / "config.yaml"
    if not conf_path.exists():
        raise SystemExit(
            f"{conf_path} not found: point --run at a results/real-router/<run> folder"
        )
    conf = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return conf, metrics


def router_config(raw: dict) -> RouterConfig:
    """RouterConfig from config.yaml's router_config. A recorded setting this checkout's
    RouterConfig does not have is fine at its default, and an error otherwise (the replay
    would not be the run)."""
    known = {f.name: f for f in dataclasses.fields(RouterConfig)}
    kw, unknown = {}, {}
    for k, v in (raw or {}).items():
        if k in known:
            kw[k] = v
        else:
            unknown[k] = v
    if unknown.get("neg_floor_frac", 1.0) != 1.0 or set(unknown) - {"neg_floor_frac"}:
        raise SystemExit(
            f"the run used router settings this checkout does not have: {sorted(unknown)}. Run "
            "this script on a checkout of the run's git_commit (config.yaml)."
        )
    if kw.get("mech_group_caps"):
        kw["mech_group_caps"] = {g: tuple(m) for g, m in kw["mech_group_caps"].items()}
    return RouterConfig(**kw)


@dataclass
class Inputs:
    assessment: object
    items: dict
    variants: dict
    human: pd.DataFrame
    ai_eval: pd.DataFrame
    folds: list[list[str]]
    maps: list
    passive: tuple[str, ...]
    cfg: RouterConfig
    seed: int
    n_synthetic: int


def rebuild(conf: dict, ai_paths=None, pilot_paths=None, bank=None) -> Inputs:
    """The run's inputs, rebuilt as scripts/run_real_router.py builds them (same steps)."""
    import run_real_router as rrr

    final = bool(conf.get("final"))
    ai = rrr.read_rows(ai_paths or conf["ai_outcomes"])
    ai = ai[ai["respondent_kind"].eq("ai")].reset_index(drop=True)
    if not final:
        check_allowed(ai)
    ai = add_attacker_variant(ai)
    if not str(conf.get("ai_sessions", "composite")).startswith("per-run"):
        ai = join_ai_sessions(ai, allow_heldout=final).rows
    rows = rrr.read_rows(pilot_paths or conf["pilot_outcomes"])
    human = rows[rows["respondent_kind"].eq("human") & rows["data_origin"].eq("pilot")]
    human = human[~human["session_id"].isin(set(conf.get("excluded_sessions") or []))]
    mechanisms = list(conf["mechanisms"])
    found = rrr.usable_mechanisms(ai, human)
    if found != sorted(mechanisms):
        raise SystemExit(f"mechanisms differ from the run: {found} vs {mechanisms}")
    ai = ai[ai["mechanism"].isna() | ai["mechanism"].isin(mechanisms)]
    human = human[human["mechanism"].isna() | human["mechanism"].isin(mechanisms)]
    passive = tuple(conf.get("passive") or [m for m in mechanisms if m in PASSIVE_MECHANISMS])

    bank_path = bank or conf["bank"]
    raw = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    flow = (raw.get("flow") or {}).get("parts") or [a["assessment_id"] for a in raw["assessments"]]
    aid = conf.get("assessment_id")
    aid = None if (not aid or aid == "+".join(flow)) else aid
    assessment, items, variants = rrr.load_bank(bank_path, aid, mechanisms)
    if conf.get("n_items") and len(assessment.item_ids) != conf["n_items"]:
        raise SystemExit(
            f"bank has {len(assessment.item_ids)} items, the run had {conf['n_items']}"
        )

    splits = split_rows(
        ai, conf.get("calibration_frac", CALIBRATION_FRAC), conf.get("split_seed", SPLIT_SEED)
    )
    if final:
        kept = final_eval_rows(splits.heldout, decided_heldout_model())
        ai_eval = kept[kept["respondent_kind"].eq("ai")]
    else:
        ai_eval = splits.calibration
    folds = [list(f) for f in conf["folds"]]
    present = set(human["session_id"])
    if any(p not in present for f in folds for p in f):
        raise SystemExit("the run's folds name pilot sessions that are not in --pilot-outcomes")
    maps = all_maps(splits.train, human, folds)
    return Inputs(
        assessment,
        items,
        variants,
        human,
        ai_eval,
        folds,
        maps,
        passive,
        router_config(conf.get("router_config")),
        int(conf.get("replay_seed", 0)),
        int(conf.get("n_synthetic_per_fold", 500)),
    )


# ------------------------------------------------------------------ replay, keeping every step
def to_rec(result, who: str, kind: str, group: str, fold: int, prior: float, cap_key) -> Rec:
    steps, contrib, checks, score = [], defaultdict(float), 0, prior
    for vid, outcome, delta in result.final.history:
        mech = parse_variant_id(vid)[1]
        if mech in ACTIVE_MECHANISMS:
            checks += 1
        if mech is not None and delta:
            contrib[cap_key(mech)] += delta
        score += delta
        steps.append(Step(mech, outcome, float(delta), float(score), checks, dict(contrib)))
    return Rec(
        who,
        kind,
        group,
        fold,
        prior,
        steps,
        result.checks_run,
        result.final.status == "cleared",
        result.cleared_step,
    )


def replay_all(inp: Inputs, cfg: RouterConfig | None = None) -> list[Rec]:
    """Every honest session out of fold and every AI session under every fold map, exactly as
    crossfit.score_policy replays them (same donors, seeds and scoring config)."""
    cfg = cfg or inp.cfg
    make = router_factory("router", inp.assessment, inp.items, inp.variants)
    scfg = score_config(cfg)
    ai_donors = Donors(inp.ai_eval, inp.items, type_match_kinds=("ai",))
    out: list[Rec] = []
    for k, (sessions, fmap) in enumerate(zip(inp.folds, inp.maps, strict=True)):
        policy = make(fmap, scfg)
        cap_key = getattr(policy, "cap_key", lambda m: m)
        donors = Donors(inp.human[inp.human["session_id"].isin(set(sessions))], inp.items)
        for p in sessions:
            r = replay_session(
                policy,
                inp.assessment,
                donors.respondent_from(HONEST, p, inp.seed),
                inp.passive,
                inp.seed,
            )
            out.append(to_rec(r, p, "honest", "honest", k, cfg.prior_log_odds, cap_key))
        for g in ai_donors.groups("ai"):
            for s in ai_donors.sessions(g):
                r = replay_session(
                    policy,
                    inp.assessment,
                    ai_donors.respondent_from(g, s, inp.seed),
                    inp.passive,
                    inp.seed,
                )
                who = f"{g.label}|{s}"
                grp = ai_group([trace(r, who, k, cfg.prior_log_odds)])
                out.append(to_rec(r, who, "ai", grp, k, cfg.prior_log_odds, cap_key))
    return out


# ------------------------------------------------------------------ anonymised labels
def group_prefix(group: str, multi_model: bool) -> str:
    if group == "honest":
        return "Human"
    model, mode, level = (group.split(":") + ["", "", ""])[:3]
    mode = MODE_SHORT.get(mode, mode)
    return f"AI-{model}-{level}-{mode}" if multi_model else f"AI-{level}-{mode}"


def assign_labels(recs: list[Rec]) -> None:
    """AI-L2-cp-#k / Human-#k, numbered within each prefix by a hash of the id (not by id order),
    the same label for all of a session's fold traces."""
    models = {r.group.split(":")[0] for r in recs if r.kind == "ai"}
    by_prefix: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        by_prefix[group_prefix(r.group, len(models) > 1)].add(r.who)
    names = {}
    for prefix, whos in by_prefix.items():
        ordered = sorted(whos, key=lambda w: hashlib.sha256(w.encode()).hexdigest())
        for i, w in enumerate(ordered, 1):
            names[w] = f"{prefix}-#{i}"
    for r in recs:
        r.label = names[r.who]


# ------------------------------------------------------------------ per trace
def n_negative(contrib: dict[str, float]) -> int:
    return sum(v < -TOL for v in contrib.values())


def shape_of(rec: Rec) -> str:
    """How the score got to B: one mechanism carried all the evidence; or it rose above the
    prior (some AI-looking evidence) and then fell; or it went down steadily."""
    cs = rec.cleared_step
    contrib = rec.steps[cs].contrib
    if sum(abs(v) > TOL for v in contrib.values()) <= 1:
        return "one mechanism"
    peak = max(s.score for s in rec.steps[: cs + 1])
    return "rose then fell" if peak > rec.prior + TOL else "steady descent"


def session_rows(recs: list[Rec], a: float, mechs: list[str]) -> pd.DataFrame:
    rows = []
    for r in recs:
        row = {
            "label": r.label,
            "kind": r.kind,
            "group": r.group,
            "fold": r.fold,
            "outcome": r.outcome_at(a),
            "ever_cleared": r.ever_cleared_at(a),
            "max_score": r.score,
            "final_score": r.final_score,
            "dist_final_to_A": a - r.final_score,
            "dist_max_to_A": a - r.score,
            "checks_shown": r.total_checks,
            "steps": len(r.steps),
        }
        cs = r.cleared_step
        if row["ever_cleared"]:
            st = r.steps[cs]
            row |= {
                "clear_step": cs + 1,
                "clear_checks": st.checks,
                "clear_mechanism": st.mechanism,
                "clear_outcome": st.outcome,
                "clear_delta": st.delta,
                "clear_n_neg_mechanisms": n_negative(st.contrib),
                "clear_shape": shape_of(r),
            }
            row |= {f"clear_contrib_{m}": st.contrib.get(m, 0.0) for m in mechs}
        x = r.first_cross(a)
        if x is not None:
            st = r.steps[x]
            row |= {
                "cross_step": x + 1,
                "cross_checks": st.checks,
                "cross_mechanism": st.mechanism,
                "cross_outcome": st.outcome,
                "cross_delta": st.delta,
            }
            row |= {f"cross_contrib_{m}": st.contrib.get(m, 0.0) for m in mechs}
        end = r.steps[-1].contrib if r.steps else {}
        row |= {f"end_contrib_{m}": end.get(m, 0.0) for m in mechs}
        rows.append(row)
    return pd.DataFrame(rows)


def step_rows(recs: list[Rec], mechs: list[str]) -> pd.DataFrame:
    rows = []
    for r in recs:
        for i, s in enumerate(r.steps, 1):
            row = {
                "label": r.label,
                "kind": r.kind,
                "group": r.group,
                "fold": r.fold,
                "step": i,
                "mechanism": s.mechanism or "plain",
                "outcome": s.outcome,
                "delta": s.delta,
                "score": s.score,
                "checks": s.checks,
            }
            row |= {f"contrib_{m}": s.contrib.get(m, 0.0) for m in mechs}
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (a) group outcomes
def per_session(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: the share of its fold traces in each outcome."""
    d = df.assign(**{o: df["outcome"].eq(o).astype(float) for o in OUTCOMES})
    d["ever_cleared"] = d["ever_cleared"].astype(float)
    return d.groupby(["group", "label"], as_index=False)[[*OUTCOMES, "ever_cleared"]].mean()


def boot_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0 or n_boot <= 0:
        return (math.nan, math.nan)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def group_outcomes(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ps = per_session(df)
    rows = []
    for g, d in ps.groupby("group", sort=True):
        row = {"group": g, "n_sessions": len(d)}
        for col in (*OUTCOMES, "ever_cleared"):
            v = d[col].to_numpy()
            lo, hi = boot_ci(v, n_boot, rng)
            row |= {
                col: float(v.mean()),
                f"{col}_n": float(v.sum()),
                f"{col}_lo": lo,
                f"{col}_hi": hi,
            }
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {"honest": 1}
    return out.sort_values("group", key=lambda s: s.map(lambda g: order.get(g, 0))).reset_index(
        drop=True
    )


# ------------------------------------------------------------------ (b) (c) (d)
def _weights(d: pd.DataFrame) -> pd.Series:
    """1 / (fold traces of that session): every session counts once in the sums."""
    return 1.0 / d.groupby("label")["fold"].transform("count")


def cleared_breakdown(df: pd.DataFrame, mechs: list[str]) -> pd.DataFrame:
    """Wrongly cleared AI traces (ever cleared before A), by group and the check that cleared."""
    ai = df[df["kind"].eq("ai")]
    n_folds = ai.groupby("label")["fold"].transform("count")
    d = ai[ai["ever_cleared"]].copy()
    if d.empty:
        return pd.DataFrame()
    d["w"] = 1.0 / n_folds[d.index]
    rows = []
    for (g, m, o), x in d.groupby(["group", "clear_mechanism", "clear_outcome"], dropna=False):
        w = x["w"]
        row = {
            "group": g,
            "clear_mechanism": m,
            "clear_outcome": o,
            "sessions": float(w.sum()),
            "steps_to_clear_median": float(x["clear_step"].median()),
            "checks_to_clear_median": float(x["clear_checks"].median()),
            "checks_to_clear_min": int(x["clear_checks"].min()),
            "checks_to_clear_max": int(x["clear_checks"].max()),
            "one_mechanism_cleared": float((w * x["clear_n_neg_mechanisms"].le(1)).sum()),
        }
        for s in ("one mechanism", "rose then fell", "steady descent"):
            row[f"shape:{s}"] = float((w * x["clear_shape"].eq(s)).sum())
        for m2 in mechs:
            row[f"end_contrib_{m2}_mean"] = float(np.average(x[f"end_contrib_{m2}"], weights=w))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "sessions"], ascending=[True, False])


def undecided_table(df: pd.DataFrame) -> pd.DataFrame:
    ai = df[df["kind"].eq("ai")]
    n_folds = ai.groupby("label")["fold"].transform("count")
    d = ai[ai["outcome"].eq("undecided")].copy()
    if d.empty:
        return pd.DataFrame()
    d["w"] = 1.0 / n_folds[d.index]
    rows = []
    for g, x in d.groupby("group"):
        q = x["dist_final_to_A"].quantile([0.25, 0.5, 0.75]).to_numpy()
        rows.append(
            {
                "group": g,
                "sessions": float(x["w"].sum()),
                "checks_shown_mean": float(np.average(x["checks_shown"], weights=x["w"])),
                "checks_shown_median": float(x["checks_shown"].median()),
                "final_score_median": float(x["final_score"].median()),
                "dist_final_to_A_q25": q[0],
                "dist_final_to_A_median": q[1],
                "dist_final_to_A_q75": q[2],
                "dist_max_to_A_median": float(x["dist_max_to_A"].median()),
                "within_1_nat_of_A": float((x["w"] * x["dist_max_to_A"].le(1.0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def honest_pauses(df: pd.DataFrame, mechs: list[str]) -> pd.DataFrame:
    d = df[df["kind"].eq("honest") & df["outcome"].eq("paused")]
    cols = [
        "label",
        "cross_step",
        "cross_checks",
        "cross_mechanism",
        "cross_outcome",
        "cross_delta",
        "max_score",
    ] + [f"cross_contrib_{m}" for m in mechs]
    return d[[c for c in cols if c in d]].reset_index(drop=True)


# ------------------------------------------------------------------ (e) examples
def pick_examples(recs: list[Rec], a: float, n: int) -> dict[str, list[Rec]]:
    """Up to n traces per failure type, one fold trace per session (the first in fold order
    showing the failure), sessions in label order; wrongly cleared prefers L2 copy-paste."""

    def first(pred, pool):
        seen, out = set(), []
        for r in sorted(pool, key=lambda r: (_label_key(r.label), r.fold)):
            if r.label not in seen and pred(r):
                seen.add(r.label)
                out.append(r)
        return out

    ai = [r for r in recs if r.kind == "ai"]
    cleared = first(lambda r: r.ever_cleared_at(a) and r.outcome_at(a) != "paused", ai)
    cleared.sort(key=lambda r: (not r.label.startswith("AI-L2-cp"), _label_key(r.label)))
    picks = {
        "AI wrongly cleared": _spread(cleared, n),
        "AI undecided (missed)": _spread(first(lambda r: r.outcome_at(a) == "undecided", ai), n),
        "Honest false pause": first(
            lambda r: r.outcome_at(a) == "paused", [r for r in recs if r.kind == "honest"]
        )[:n],
    }
    return {k: v for k, v in picks.items() if v}


def _label_key(label: str):
    head, _, num = label.rpartition("-#")
    return (head, int(num) if num.isdigit() else 0)


def _spread(recs: list[Rec], n: int) -> list[Rec]:
    """Take from as many groups as possible (round robin over prefixes, in the given order)."""
    by: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        by[r.label.rpartition("-#")[0]].append(r)
    out, queues = [], list(by.values())
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def example_rows(examples: dict[str, list[Rec]]) -> pd.DataFrame:
    rows = []
    for kind, recs in examples.items():
        for r in recs:
            rows.append(
                {
                    "failure": kind,
                    "label": r.label,
                    "fold_map": r.fold,
                    "step": 0,
                    "check": "start",
                    "outcome": "",
                    "score": r.prior,
                }
            )
            for i, s in enumerate(r.steps, 1):
                if s.mechanism is None:
                    continue  # plain items add no evidence
                rows.append(
                    {
                        "failure": kind,
                        "label": r.label,
                        "fold_map": r.fold,
                        "step": i,
                        "check": s.mechanism,
                        "outcome": s.outcome,
                        "score": s.score,
                    }
                )
    return pd.DataFrame(rows)


def trajectory_text(r: Rec) -> str:
    parts = [f"{s.mechanism}:{s.outcome} -> {s.score:+.2f}" for s in r.steps if s.mechanism]
    head = f"{r.label} (fold map {r.fold})" if r.kind == "ai" else r.label
    return f"{head}: start {r.prior:+.2f}; " + "; ".join(parts)


# ------------------------------------------------------------------ (f) counterfactual
def counterfactual_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Two-check rule on the logged trajectories: a clearing counts only if at least two
    mechanisms had a negative total at that step. A cleared trace whose clearing came from one
    mechanism flips to "not cleared at that step" (what it would have done next is not known)."""
    ai = df[df["kind"].eq("ai")]
    n_folds = ai.groupby("label")["fold"].transform("count")
    d = ai[ai["ever_cleared"]].copy()
    rows = []
    for g in sorted(ai["group"].unique()):
        x = d[d["group"].eq(g)]
        w = 1.0 / n_folds[x.index] if len(x) else pd.Series(dtype=float)
        flips = x["clear_n_neg_mechanisms"].lt(2) if len(x) else pd.Series(dtype=bool)
        rows.append(
            {
                "group": g,
                "method": "rule: clearing needs >=2 mechanisms with negative contrib",
                "n_sessions": int(ai[ai["group"].eq(g)]["label"].nunique()),
                "wrongly_cleared": float(w.sum()),
                "flip": float((w * flips).sum()),
                "stay_cleared": float((w * ~flips).sum()),
            }
        )
    return pd.DataFrame(rows)


def counterfactual_replay(
    inp: Inputs, recs: list[Rec], threshold: str, frac: float, alphas: list[float]
) -> tuple[pd.DataFrame, dict] | None:
    """Full re-replay with RouterConfig.neg_floor_frac = frac (two-check clearing), thresholds
    recalibrated from the same honest procedure (crossfit.score_policy; no bootstrap). None when
    this checkout's RouterConfig has no neg_floor_frac."""
    if "neg_floor_frac" not in {f.name for f in dataclasses.fields(RouterConfig)}:
        return None
    cfg2 = replace(inp.cfg, neg_floor_frac=frac)
    make = router_factory("router", inp.assessment, inp.items, inp.variants)
    sc = score_policy(
        make,
        cfg2,
        inp.assessment,
        inp.items,
        inp.human,
        inp.ai_eval,
        inp.folds,
        inp.maps,
        inp.n_synthetic,
        inp.passive,
        inp.seed,
    )
    th2 = thresholds_from(
        [t.score for t in sc.real.values()], [t.score for t in sc.synthetic], alphas
    )
    if threshold not in th2:
        return None
    a2 = th2[threshold]
    new = replay_all(inp, cfg2)
    labels = {r.who: r.label for r in recs}
    for r in new:
        r.label = labels.get(r.who, r.who)
    old = {(r.who, r.fold): r for r in recs}
    traces_per = Counter(r.who for r in recs)
    rows = defaultdict(lambda: defaultdict(float))
    for r in new:
        if r.kind != "ai":
            continue
        o = old[(r.who, r.fold)]
        k = traces_per[r.who]  # this session's fold traces: each session counts once
        g = rows[r.group]
        g["n_sessions_x"] += 1.0 / k
        if o.extra["ever_cleared"]:
            g["wrongly_cleared"] += 1.0 / k
            g[f"now_{r.outcome_at(a2)}"] += 1.0 / k
            g["now_still_ever_cleared"] += r.ever_cleared_at(a2) / k
        for out in OUTCOMES:
            g[f"all_{out}"] += (r.outcome_at(a2) == out) / k
    honest = [r for r in new if r.kind == "honest"]
    table = pd.DataFrame(
        [{"group": g, "method": f"re-replay neg_floor_frac={frac}", **v} for g, v in rows.items()]
    ).rename(columns={"n_sessions_x": "n_sessions"})
    for col in ("wrongly_cleared", *(f"now_{o}" for o in OUTCOMES), "now_still_ever_cleared"):
        if col not in table:
            table[col] = 0.0
    table = table.fillna(0.0)
    info = {
        "frac": frac,
        "A_new": a2,
        "honest_paused_new": int(sum(r.score >= a2 for r in honest)),
        "n_honest": len(honest),
    }
    return table, info


# ------------------------------------------------------------------ consistency with the run
def consistency(df: pd.DataFrame, metrics: dict, threshold: str) -> pd.DataFrame:
    res = (((metrics.get("results") or {}).get("router") or {}).get("results") or {}).get(
        threshold
    ) or {}
    ps = per_session(df[df["kind"].eq("ai")])
    rows = []
    for g, d in ps.groupby("group"):
        for ours, theirs in (
            ("paused", f"tpr[{g}]"),
            ("cleared", f"ai_cleared_rate[{g}]"),
            ("ever_cleared", f"ai_ever_cleared_rate[{g}]"),
        ):
            if theirs in res:
                rows.append({"metric": theirs, "run": res[theirs], "replay": d[ours].mean()})
    honest = df[df["kind"].eq("honest")]
    if "honest_pause_rate" in res and len(honest):
        rows.append(
            {
                "metric": "honest_pause_rate",
                "run": res["honest_pause_rate"],
                "replay": honest["outcome"].eq("paused").mean(),
            }
        )
    out = pd.DataFrame(rows, columns=["metric", "run", "replay"])
    out["match"] = (out["run"] - out["replay"]).abs() < 1e-9
    return out


# ------------------------------------------------------------------ figure
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4 (fixed order)


def plot_examples(examples: dict[str, list[Rec]], a: float, b: float, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("no figure: matplotlib missing (pip install -e '.[plot]')")
        return False
    n = len(examples)
    if not n:
        return False
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), sharey=True, squeeze=False)
    for ax, (kind, recs) in zip(axes[0], examples.items(), strict=True):
        for c, r in zip(COLORS, recs, strict=False):
            # x = evidence step: every scored check outcome in order (plain items skipped)
            ys = [r.prior] + [s.score for s in r.steps if s.mechanism]
            xs = list(range(len(ys)))
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, label=r.label)
        for y, name in ((a, "A (pause)"), (b, "B (clear)")):
            ax.axhline(y, color="#8a8984", lw=1, ls="--")
            ax.annotate(
                name, (0, y), xytext=(2, 3), textcoords="offset points", fontsize=7, color="#52514e"
            )
        ax.axhline(0, color="#d8d7d2", lw=0.8)
        ax.set_title(kind, fontsize=10)
        ax.set_xlabel("evidence step (check outcomes, in order)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axes[0][0].set_ylabel("running score (log-odds AI)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------------ markdown
def _pct(v, lo=None, hi=None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    s = f"{v:.2f}"
    return s if lo is None or math.isnan(lo) else f"{s} [{lo:.2f}, {hi:.2f}]"


def md_table(df: pd.DataFrame, cols: list[str], fmt=None) -> str:
    if df is None or df.empty:
        return "_none_\n"
    cols = [c for c in cols if c in df]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                f"{v:.2f}"
                if isinstance(v, float) and not math.isnan(v)
                else ("-" if isinstance(v, float) else str(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, ctx: dict) -> None:
    g = ctx["groups"]
    rows = [
        "| group | sessions | paused | cleared (wrong for AI) | undecided | ever cleared |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in g.iterrows():
        rows.append(
            f"| {r['group']} | {r['n_sessions']} | "
            + " | ".join(
                f"{r[c + '_n']:.1f} = {_pct(r[c], r[c + '_lo'], r[c + '_hi'])}"
                for c in (*OUTCOMES, "ever_cleared")
            )
            + " |"
        )
    cons = ctx["consistency"]
    cons_line = (
        "no per-group rates in metrics.json to compare with"
        if cons.empty
        else (
            "all match the run"
            if cons["match"].all()
            else "MISMATCH: see consistency.csv; do not report these tables"
        )
    )
    m = ctx["mechs"]
    parts = [
        f"# Router error analysis: {ctx['run_id']}\n",
        (
            f"Threshold {ctx['threshold']} = {ctx['A']:.3f}, B = {ctx['B']:.3f}. Replayed from the "
            f"run's config ({ctx['n_honest']} honest sessions, {ctx['n_ai']} AI sessions, "
            f"{ctx['k']} fold maps). Consistency with metrics.json: **{cons_line}**.\n"
        ),
        (
            "Rates per session = share of its fold traces; counts are session-equivalents "
            "(sum of those shares). 95% CIs: bootstrap over sessions within the group, threshold "
            "fixed.\n"
        ),
        "## (a) Outcomes per group\n",
        "\n".join(rows) + "\n",
        "## (b) Wrongly cleared AI sessions: what cleared them\n",
        md_table(
            ctx["cleared"],
            [
                "group",
                "clear_mechanism",
                "clear_outcome",
                "sessions",
                "checks_to_clear_median",
                "one_mechanism_cleared",
                "shape:one mechanism",
                "shape:rose then fell",
                "shape:steady descent",
            ]
            + [f"end_contrib_{x}_mean" for x in m],
        ),
        "## (c) Undecided AI sessions (missed)\n",
        md_table(
            ctx["undecided"],
            [
                "group",
                "sessions",
                "checks_shown_mean",
                "final_score_median",
                "dist_final_to_A_median",
                "dist_max_to_A_median",
                "within_1_nat_of_A",
            ],
        ),
        "## (d) Honest false pauses: the check that pushed the score over A\n",
        md_table(
            ctx["honest"],
            [
                "label",
                "cross_step",
                "cross_checks",
                "cross_mechanism",
                "cross_outcome",
                "cross_delta",
                "max_score",
            ],
        ),
        "## (e) Example trajectories (check:outcome -> score after it)\n",
    ]
    for kind, recs in ctx["examples"].items():
        parts.append(f"**{kind}**\n")
        parts += [f"- {trajectory_text(r)}" for r in recs]
        parts.append("")
    parts.append(
        "An AI session is replayed under every fold map, so one session can appear under two "
        "failure types (under different maps). Figure: example_trajectories.png\n"
    )
    parts.append("## (f) Counterfactual: two-check clearing\n")
    parts.append(
        md_table(ctx["cf_rule"], ["group", "n_sessions", "wrongly_cleared", "flip", "stay_cleared"])
    )
    parts.append(
        "Rule: on the logged trajectories, a clearing counts only if >= 2 mechanisms had a "
        "negative total at the clearing step. A flip means the router would NOT have cleared "
        "there and would have kept checking; whether it then pauses is shown only by the "
        "re-replay below.\n"
    )
    if ctx["cf_replay"] is not None:
        t, info = ctx["cf_replay"]
        parts.append(
            f"Re-replay with neg_floor_frac = {info['frac']}: recalibrated {ctx['threshold']} = "
            f"{info['A_new']:.3f}; honest sessions paused {info['honest_paused_new']} / "
            f"{info['n_honest']} (point estimate, no bootstrap).\n"
        )
        parts.append(
            md_table(
                t,
                [
                    "group",
                    "n_sessions",
                    "wrongly_cleared",
                    "now_paused",
                    "now_cleared",
                    "now_undecided",
                    "all_paused",
                    "all_cleared",
                    "all_undecided",
                ],
            )
        )
    else:
        parts.append(
            "Re-replay skipped: this checkout's RouterConfig has no neg_floor_frac (it is on "
            "arav/rq2-l2), or --no-counterfactual-replay.\n"
        )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="results/real-router/<run_id> folder")
    ap.add_argument("--ai-outcomes", nargs="+", help="override config.yaml's paths")
    ap.add_argument("--pilot-outcomes", nargs="+", help="override config.yaml's paths")
    ap.add_argument("--bank", help="override config.yaml's bank path")
    ap.add_argument("--threshold", default="A_0.01", help="which run threshold (A_0.01, A_0.05)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap draws over sessions")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--examples", type=int, default=4, help="example trajectories per failure")
    ap.add_argument("--counterfactual-frac", type=float, default=TWO_CHECK_FRAC)
    ap.add_argument(
        "--no-counterfactual-replay",
        action="store_true",
        help="only the rule-based counterfactual (no re-replay, faster)",
    )
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "error_analysis"))
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    conf, metrics = load_run(run_dir)
    run_id = conf.get("run_id") or run_dir.name
    th = conf.get("router_thresholds") or {}
    if args.threshold not in th or not math.isfinite(th[args.threshold]):
        raise SystemExit(f"{args.threshold} not in the run's router thresholds {sorted(th)}")
    a = float(th[args.threshold])

    inp = rebuild(conf, args.ai_outcomes, args.pilot_outcomes, args.bank)
    make = router_factory("router", inp.assessment, inp.items, inp.variants)
    b = make(inp.maps[0], score_config(inp.cfg)).B
    print(f"Replaying the router ({args.threshold} = {a:.3f}, B = {b:.3f})...", flush=True)
    recs = replay_all(inp)
    assign_labels(recs)
    for r in recs:
        r.extra = {"A": a, "ever_cleared": r.ever_cleared_at(a)}
    mechs = sorted({k for r in recs for s in r.steps for k in s.contrib})

    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    sessions = session_rows(recs, a, mechs)
    sessions.to_csv(out / "sessions.csv", index=False)
    step_rows(recs, mechs).to_csv(out / "steps.csv", index=False)
    cons = consistency(sessions, metrics, args.threshold)
    cons.to_csv(out / "consistency.csv", index=False)
    if not cons.empty and not cons["match"].all():
        print(
            "WARNING: the replay does not match metrics.json (consistency.csv): run this on "
            "a checkout of the run's git_commit with the run's input files"
        )
    groups = group_outcomes(sessions, args.boot, args.seed)
    groups.to_csv(out / "group_outcomes.csv", index=False)
    cleared = cleared_breakdown(sessions, mechs)
    cleared.to_csv(out / "cleared_breakdown.csv", index=False)
    undecided = undecided_table(sessions)
    undecided.to_csv(out / "undecided.csv", index=False)
    honest = honest_pauses(sessions, mechs)
    honest.to_csv(out / "honest_pauses.csv", index=False)
    examples = pick_examples(recs, a, args.examples)
    example_rows(examples).to_csv(out / "examples.csv", index=False)
    plot_examples(examples, a, b, out / "example_trajectories.png")
    cf_rule = counterfactual_rule(sessions)
    cf_replay = None
    if not args.no_counterfactual_replay:
        alphas = sorted({float(x) for x in conf.get("alphas") or [0.01, 0.05]})
        cf_replay = counterfactual_replay(
            inp, recs, args.threshold, args.counterfactual_frac, alphas
        )
    cf = [cf_rule] + ([cf_replay[0]] if cf_replay else [])
    pd.concat(cf, ignore_index=True).to_csv(out / "counterfactual.csv", index=False)
    if cf_replay:
        (out / "counterfactual_replay.json").write_text(
            json.dumps(cf_replay[1], indent=1), encoding="utf-8"
        )
    write_markdown(
        out / "error_analysis.md",
        {
            "run_id": run_id,
            "threshold": args.threshold,
            "A": a,
            "B": b,
            "groups": groups,
            "cleared": cleared,
            "undecided": undecided,
            "honest": honest,
            "examples": examples,
            "cf_rule": cf_rule,
            "cf_replay": cf_replay,
            "consistency": cons,
            "mechs": mechs,
            "n_honest": len({r.who for r in recs if r.kind == "honest"}),
            "n_ai": len({r.who for r in recs if r.kind == "ai"}),
            "k": len(inp.folds),
        },
    )
    print(
        f"Saved {out}/: error_analysis.md, group_outcomes.csv, cleared_breakdown.csv, "
        "undecided.csv, honest_pauses.csv, examples.csv, counterfactual.csv, sessions.csv, "
        "steps.csv, consistency.csv, example_trajectories.png"
    )
    return out


if __name__ == "__main__":
    main()

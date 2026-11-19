"""Session-level cross-fitting: the review threshold from replayed honest sessions (step #42).

Owner: Arav. Used by scripts/run_real_router.py once pilot rows exist. Matches the mid-term
report: "threshold from replayed honest sessions, built by resampling the pilot's per-check
outcomes".

Honest sessions are the pilot sessions (data_origin "pilot"), one per pilot session_id. Note: one
person may span sessions (e.g. someone who started over); sessions are the unit throughout, so
the folds and the bootstrap treat such sessions as independent. Sessions are split into K folds
by adr/replay/folds.assign_session_folds, the one fold assignment shared with the behaviour
calibration (fixed K and seed string from configs/data_decisions.json; low-effort sessions
spread across the folds first). For each fold k,
map_k = Archana's builder in STRICT mode on the AI train rows plus the human rows of the OTHER
folds only (the #25 guard refuses provisional maps). Then:

  1. Out-of-fold scores: each real session in fold k is replayed once, from its own rows, with
     map_k: a map that never saw it. Pooled over folds: one honest score per session.
  2. Resampled honest sessions: N synthetic sessions per fold, each check's outcome drawn (with
     replacement) from fold k's sessions only (same item, else same item type), scored with map_k.
     Pooled over folds (K x N).
  3. Thresholds: A_1% and A_5% = the smallest threshold that pauses at most 1% / 5% of the
     resampled pool; A_max = the lowest threshold that pauses none of the real out-of-fold
     sessions. AI sessions are scored with every map_k; TPR(A) = share of (session, fold) scores
     at or above A.

Session score. A session's score is the highest running SPRT score it reaches when replayed with
pausing switched off. Pausing at threshold A then happens exactly when score >= A, because the
check-by-check path up to the first crossing does not depend on A: the evidence cap and the
escalation point are held at their nominal values in nats (score_config / deploy_config).

CIs: the bootstrap resamples honest SESSIONS (and AI sessions), regenerates the resampled honest
sessions from the drawn ones, and recomputes every threshold and rate. The fold maps are not
refitted.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from adr.contracts import (
    ACTIVE_MECHANISMS,
    Assessment,
    FailureMap,
    Item,
    RenderedItem,
    Router,
    parse_variant_id,
)
from adr.map import build_failure_map
from adr.replay.donors import Donor, Donors, Group
from adr.replay.simulate import ReplayResult, replay_session
from adr.router.core import RouterConfig
from adr.router.map_guard import check_router_map
from adr.router.sprt import wald_thresholds

UNREACHABLE = 1e6  # review threshold while scoring: never reached, so nothing pauses
MIN_TAIL_SESSIONS = 3  # warn when fewer real sessions reach a threshold
HONEST = Group.human("pilot")

PolicyFactory = Callable[[FailureMap, RouterConfig], Router]


# ------------------------------------------------------------------ folds and maps
def fold_map(
    ai_rows: pd.DataFrame,
    human_rows: pd.DataFrame,
    held_out_sessions: Sequence[str],
    version: str,
    n_bootstrap: int = 200,
    seed: int = 34,
) -> FailureMap:
    """map_k: the strict builder on the AI rows plus the human rows of everyone NOT held out."""
    keep = ~human_rows["session_id"].isin(set(held_out_sessions))
    frame = pd.concat([ai_rows, human_rows[keep]], ignore_index=True)
    fmap = build_failure_map(
        frame,
        version=version,
        n_bootstrap=n_bootstrap,
        seed=seed,
        note=f"cross-fit map: human rows exclude {len(set(held_out_sessions))} held-out sessions",
    )
    check_router_map(fmap)
    return fmap


def all_maps(
    ai_map_rows: pd.DataFrame,
    human_rows: pd.DataFrame,
    folds: list[list[str]],
    map_bootstrap: int = 200,
) -> list[FailureMap]:
    return [
        fold_map(ai_map_rows, human_rows, sessions, f"crossfit-fold{k}", map_bootstrap)
        for k, sessions in enumerate(folds)
    ]


# ------------------------------------------------------------------ scoring
def nominal_A(cfg: RouterConfig) -> float:
    return cfg.upper if cfg.upper is not None else wald_thresholds(cfg.alpha, cfg.beta)[0]


def _held(cfg: RouterConfig, a: float) -> RouterConfig:
    """`cfg` with review threshold `a` but the cap and escalation point of the nominal A."""
    a0 = nominal_A(cfg)
    return replace(
        cfg,
        upper=a,
        mech_cap_frac=None if cfg.mech_cap_frac is None else cfg.mech_cap_frac * a0 / a,
        escalate_frac=cfg.escalate_frac * a0 / a,
    )


def score_config(cfg: RouterConfig) -> RouterConfig:
    """Pausing off (threshold unreachable), cap and escalation at the nominal A, in nats."""
    return _held(cfg, UNREACHABLE)


def deploy_config(cfg: RouterConfig, threshold: float) -> RouterConfig:
    """The router to deploy at a calibrated threshold: pauses at `threshold`, with the same cap
    and escalation point (in nats) the threshold was calibrated with."""
    return _held(cfg, threshold)


@dataclass(frozen=True)
class Trace:
    """One replayed session: its score (the highest running score) and, per update, the running
    score and how many active checks had been served, to read off burden at any threshold."""

    who: str  # session, AI session, or "synthetic:<fold>:<i>"
    fold: int
    score: float
    running: np.ndarray
    checks: np.ndarray
    total_checks: int
    cleared: bool = False  # the replay ended "cleared" (score reached B; checks stopped there)
    # Where the outcomes of the ACTIVE checks served came from (AI sessions on the pilot bank,
    # review H3): same_type = own rows on another item of the same item_type; fallback = another
    # session of the same group; missing = no row at all (NoObservation: no evidence). Passive
    # checks are not counted: they are attempted on every item, also where they never apply.
    sources: tuple[int, int, int, int] = (0, 0, 0, 0)  # (same_type, fallback, synthetic, missing)

    def cleared_at(self, threshold: float) -> bool:
        """Cleared when deployed at `threshold`: it reached B before ever reaching `threshold`.
        Clearing is final (no more checks, score frozen), so that is: cleared and score < threshold."""
        return self.cleared and self.score < threshold

    def checks_at(self, threshold: float) -> int:
        """Active checks served before pausing at `threshold` (all of them if never paused)."""
        hit = np.nonzero(self.running >= threshold)[0]
        return int(self.checks[hit[0]]) if len(hit) else self.total_checks


def trace(result: ReplayResult, who: str, fold: int, prior: float) -> Trace:
    deltas = np.array([d for _, _, d in result.final.history], dtype=float)
    running = prior + np.cumsum(deltas) if len(deltas) else np.array([prior])
    active = np.cumsum(
        [parse_variant_id(v)[1] in ACTIVE_MECHANISMS for v, _, _ in result.final.history]
    )
    checks = active if len(active) else np.array([0])
    score = float(max(prior, running.max()))
    return Trace(
        who,
        fold,
        score,
        running,
        checks,
        result.checks_run,
        result.final.status == "cleared",
        result.active_sources,
    )


# ------------------------------------------------------------------ donors
def _index(rows: pd.DataFrame) -> dict[str, dict[tuple[str, str], list[str]]]:
    """session -> {(item_id, mechanism): outcomes} and {(item_type, mechanism): outcomes}."""
    out: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    rows = rows[rows["mechanism"].notna()].sort_values(
        ["session_id", "item_id", "mechanism", "variant_id", "outcome"]
    )
    for r in rows[["session_id", "item_id", "item_type", "mechanism", "outcome"]].itertuples(
        index=False
    ):
        out[r.session_id][(r.item_id, r.mechanism)].append(r.outcome)
        out[r.session_id][("type:" + r.item_type, r.mechanism)].append(r.outcome)
    return out


def resampled_donor(
    session_index: dict[str, dict[tuple[str, str], list[str]]],
    sessions: Sequence[str],
    items: dict[str, Item],
    rng: random.Random,
) -> Donor:
    """A synthetic honest session: every check's outcome is drawn with replacement from THESE
    sessions only (`sessions` may repeat, as in a bootstrap draw): same item if any of them answered
    it, else same item type. No map draws, no other sessions."""
    pooled: dict[tuple[str, str], list[str]] = defaultdict(list)
    pooled_type: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in sessions:
        for (key, mech), outcomes in session_index[p].items():
            if key.startswith("type:"):
                pooled_type[(key[5:], mech)].extend(outcomes)
            else:
                pooled[(key, mech)].extend(outcomes)
    return Donor(
        group=HONEST,
        session_id=None,
        _own={},
        _pooled=dict(pooled),
        _items=items,
        _fmap=None,
        _rng=rng,
        _pooled_type=dict(pooled_type),
    )


# ------------------------------------------------------------------ one policy, all folds
@dataclass
class Scored:
    """Everything the thresholds and rates are computed from, for one policy."""

    folds: list[list[str]]
    real: dict[str, Trace]  # session -> out-of-fold trace
    ai: dict[str, list[Trace]]  # AI session -> one trace per fold map
    synthetic: list[Trace]  # K x N resampled honest sessions
    maps: list[FailureMap]
    prior: float


def score_policy(
    make: PolicyFactory,
    cfg: RouterConfig,
    assessment: Assessment,
    items: dict[str, Item],
    human_rows: pd.DataFrame,
    ai_eval_rows: pd.DataFrame,
    folds: list[list[str]],
    maps: list[FailureMap],
    n_synthetic: int = 500,
    passive: tuple[str, ...] = (),
    seed: int = 0,
) -> Scored:
    if len(maps) != len(folds):
        raise ValueError("one map per fold")
    scfg = score_config(cfg)
    session_index = _index(human_rows)
    real: dict[str, Trace] = {}
    ai: dict[str, list[Trace]] = defaultdict(list)
    synthetic: list[Trace] = []
    # The AI donors don't depend on the fold: index them once (review H4). Every replay draws
    # through respondent_from(), which seeds its own rng, so results are unchanged.
    ai_donors = Donors(ai_eval_rows, items, type_match_kinds=("ai",))
    for k, (sessions, fmap) in enumerate(zip(folds, maps, strict=True)):
        check_router_map(fmap)  # #25 guard, also for maps passed in
        policy = make(fmap, scfg)
        # 1. out-of-fold: fold k's real sessions, scored with a map that never saw them
        donors = Donors(human_rows[human_rows["session_id"].isin(set(sessions))], items)
        for p in sessions:
            r = replay_session(
                policy, assessment, donors.respondent_from(HONEST, p, seed), passive, seed
            )
            real[p] = trace(r, p, k, cfg.prior_log_odds)
        # 2. resampled honest sessions from fold k's sessions only
        synthetic += _synthetic(
            policy,
            assessment,
            items,
            session_index,
            sessions,
            k,
            n_synthetic,
            passive,
            seed,
            cfg.prior_log_odds,
        )
        # AI sessions with the same map
        for g in ai_donors.groups("ai"):
            for s in ai_donors.sessions(g):
                r = replay_session(
                    policy, assessment, ai_donors.respondent_from(g, s, seed), passive, seed
                )
                who = f"{g.label}|{s}"
                ai[who].append(trace(r, who, k, cfg.prior_log_odds))
    return Scored(folds, real, dict(ai), synthetic, maps, cfg.prior_log_odds)


def _synthetic(
    policy, assessment, items, session_index, sessions, fold, n, passive, seed, prior, tag=""
) -> list[Trace]:
    out = []
    for i in range(n):
        rng = random.Random(f"synthetic:{seed}:{fold}:{i}{tag}")
        donor = resampled_donor(session_index, sessions, items, rng)
        r = replay_session(policy, assessment, donor, passive, seed + i)
        out.append(trace(r, f"synthetic:{fold}:{i}", fold, prior))
    return out


# ------------------------------------------------------------------ thresholds and rates
def threshold_at(scores: Sequence[float], alpha: float) -> float:
    """The smallest threshold A with P(score >= A) <= alpha (pause when score >= A)."""
    s = np.sort(np.asarray(scores, dtype=float))
    if not len(s):
        return math.inf
    for t in np.unique(s):
        if np.mean(s >= t) <= alpha:
            return float(t)
    return float(np.nextafter(s[-1], math.inf))  # above everyone: pauses no one


def threshold_none(scores: Sequence[float]) -> float:
    """A_max: the lowest threshold that pauses none of these scores."""
    s = np.asarray(scores, dtype=float)
    return float(np.nextafter(s.max(), math.inf)) if len(s) else math.inf


def ai_group(per_fold: list[Trace]) -> str:
    """An AI session's group, model:attack_mode:attacker_level (Trace.who = "ai:<group>|<session>")."""
    return per_fold[0].who.split("|", 1)[0].removeprefix("ai:")


SOURCE_NAMES = ("same_type", "fallback", "synthetic", "missing")

# Per attack mode: an AI group of that mode needs at least this many sessions for a TPR (document
# upload: MORE than 5); smaller groups are described session by session. Other modes: no minimum.
MIN_GROUP_SESSIONS = {"doc_upload": 6}


def group_mode(group: str) -> str:
    """"gpt-mini:doc_upload:L0" -> "doc_upload"."""
    parts = group.split(":")
    return parts[-2] if len(parts) >= 3 else ""


def describe_ai(ai: list[list[Trace]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Session by session, for an AI group too small for a TPR: its score (mean over the fold
    maps), checks, and per threshold the share of fold maps under which it is paused / cleared."""
    out = []
    for f in ai:
        row: dict[str, Any] = {
            "group": ai_group(f),
            "session": f[0].who.split("|", 1)[1],
            "score": float(np.mean([t.score for t in f])),
            "checks": float(np.mean([t.total_checks for t in f])),
        }
        for name, a in thresholds.items():
            row[f"paused@{name}"] = float(np.mean([t.score >= a for t in f]))
            row[f"cleared@{name}"] = float(np.mean([t.cleared_at(a) for t in f]))
        out.append(row)
    return out


def ai_sources(ai: list[list[Trace]]) -> dict[str, dict[str, float]]:
    """Per AI group (review H3): sessions, and per session the mean number of checks served and
    of check outcomes answered same-type / from a fallback / from the map / missing, averaged
    over the fold maps. A group with many `missing` checks stays near the prior: it would count
    as misses, so it must not be silently pooled into the TPR."""
    by: dict[str, list[list[Trace]]] = defaultdict(list)
    for f in ai:
        by[ai_group(f)].append(f)
    out = {}
    for g, sessions in sorted(by.items()):
        traces = [t for f in sessions for t in f]
        out[g] = {"sessions": len(sessions)}
        out[g]["checks"] = float(np.mean([t.total_checks for t in traces]))
        for i, name in enumerate(SOURCE_NAMES):
            out[g][name] = float(np.mean([t.sources[i] for t in traces]))
    return out


def rates(
    real: list[Trace], ai: list[list[Trace]], synthetic: list[float], thresholds: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Per threshold: TPR, real and resampled honest pause rates, checks per honest session, the
    share of honest sessions cleared (reached B first), the share of AI sessions wrongly cleared,
    and how many real sessions reach it."""
    out = {}
    for name, a in thresholds.items():
        ai_hits = [np.mean([t.score >= a for t in per_fold]) for per_fold in ai]
        out[name] = {
            "A": a,
            "tpr": float(np.mean(ai_hits)) if ai_hits else math.nan,
            "honest_pause_rate": float(np.mean([t.score >= a for t in real])),
            "resampled_pause_rate": float(np.mean(np.asarray(synthetic) >= a))
            if len(synthetic)
            else math.nan,
            "checks_per_honest_session": float(np.mean([t.checks_at(a) for t in real])),
            "honest_cleared_rate": float(np.mean([t.cleared_at(a) for t in real])),
            # AI sessions wrongly cleared: reached B before `a` (mean over the fold maps)
            "ai_cleared_rate": float(np.mean([np.mean([t.cleared_at(a) for t in f]) for f in ai]))
            if ai
            else math.nan,
            "n_real_at_or_above": int(sum(t.score >= a for t in real)),
        }
        groups = sorted({ai_group(f) for f in ai})
        if len(groups) > 1:  # TPR per model / attack mode / level too (review H3)
            for g in groups:
                hits = [np.mean([t.score >= a for t in f]) for f in ai if ai_group(f) == g]
                out[name][f"tpr[{g}]"] = float(np.mean(hits))
    return out


def thresholds_from(
    real_scores: Sequence[float], synthetic_scores: Sequence[float], alphas: Sequence[float]
) -> dict[str, float]:
    out = {f"A_{a:g}": threshold_at(synthetic_scores, a) for a in alphas}
    out["A_max"] = threshold_none(real_scores)
    return out


def summarize(
    scored: Scored,
    make: PolicyFactory,
    cfg: RouterConfig,
    assessment: Assessment,
    items: dict[str, Item],
    human_rows: pd.DataFrame,
    alphas: Sequence[float] = (0.01, 0.05),
    n_boot: int = 200,
    n_synthetic_boot: int = 100,
    passive: tuple[str, ...] = (),
    seed: int = 0,
    min_group_sessions: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Thresholds, TPR and honest rates, with 95% CIs from a bootstrap over SESSIONS: each draw
    resamples sessions (within their folds) and AI sessions (within their group), regenerates the
    resampled honest sessions from the drawn sessions only, and recomputes thresholds and rates.

    AI groups with fewer sessions than `min_group_sessions[<their attack mode>]` get no TPR (they
    are left out of the pooled TPR too) and are listed session by session under "described_ai"."""
    mins = min_group_sessions or {}
    real = list(scored.real.values())
    all_ai = list(scored.ai.values())
    size: dict[str, int] = defaultdict(int)
    for f in all_ai:
        size[ai_group(f)] += 1

    def too_small(f: list[Trace]) -> bool:
        g = ai_group(f)
        return size[g] < mins.get(group_mode(g), 0)

    ai = [f for f in all_ai if not too_small(f)]
    small = [f for f in all_ai if too_small(f)]
    syn = [t.score for t in scored.synthetic]
    th = thresholds_from([t.score for t in real], syn, alphas)
    point = rates(real, ai, syn, th)

    session_index = _index(human_rows)
    scfg = score_config(cfg)
    policies = [make(fmap, scfg) for fmap in scored.maps]
    rng = np.random.default_rng(seed)
    honest_ids = sorted(scored.real)
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(ai):
        by_group[ai_group(f)].append(i)
    draws: dict[str, dict[str, list[float]]] = {n: defaultdict(list) for n in th}
    for b in range(n_boot):
        picked = [honest_ids[i] for i in rng.integers(0, len(honest_ids), len(honest_ids))]
        ai_pick = []  # resampled within each AI group, so every group keeps its size
        for idx in by_group.values():
            ai_pick += [ai[idx[i]] for i in rng.integers(0, len(idx), len(idx))]
        by_fold: dict[int, list[str]] = defaultdict(list)
        for p in picked:
            by_fold[scored.real[p].fold].append(p)
        syn_b = []
        for k, sessions in sorted(by_fold.items()):
            syn_b += [
                t.score
                for t in _synthetic(
                    policies[k],
                    assessment,
                    items,
                    session_index,
                    sessions,
                    k,
                    n_synthetic_boot,
                    passive,
                    seed,
                    scored.prior,
                    tag=f":boot{b}",
                )
            ]
        real_b = [scored.real[p] for p in picked]
        th_b = thresholds_from([t.score for t in real_b], syn_b, alphas)
        for name, vals in rates(real_b, ai_pick, syn_b, th_b).items():
            for metric, v in vals.items():
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    draws[name][metric].append(v)
    ci = {
        name: {
            metric: (
                tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))
                if len(v) * 2 > n_boot and np.isfinite(v).all()
                else None
            )
            for metric, v in by_metric.items()
        }
        for name, by_metric in draws.items()
    }
    sizes = [len(f) for f in scored.folds]
    return {
        "thresholds": th,
        "results": point,
        "ci95": ci,
        "n_sessions": len(real),
        "n_ai_sessions": len(ai),
        "ai_sources_by_group": ai_sources(all_ai),
        "n_ai_sessions_by_group": dict(sorted(size.items())),
        "described_ai": describe_ai(small, th),
        "min_group_sessions": dict(mins),
        "n_resampled_sessions": len(syn),
        "sessions_per_fold": sizes,
        "n_boot": n_boot,
        "n_resampled_per_fold_in_boot": n_synthetic_boot,
        "boot_unit": "session (resampled honest sessions regenerated per draw) and AI session",
        "boot_seed": seed,
        "map_versions": [m.version for m in scored.maps],
    }


def tail_warnings(summary: dict[str, Any]) -> list[str]:
    """'1% tail extrapolated from N sessions' always; plus a warning for any threshold that fewer
    than MIN_TAIL_SESSIONS real sessions reach."""
    n = summary["n_sessions"]
    out = [f"1% tail extrapolated from {n} sessions"]
    for name, r in summary["results"].items():
        if r["n_real_at_or_above"] < MIN_TAIL_SESSIONS:
            out.append(
                f"{name} = {r['A']:.3g} is driven by fewer than {MIN_TAIL_SESSIONS} real sessions "
                f"({r['n_real_at_or_above']} of {n} reach it)"
            )
    return out


def cap_findings(
    summary: dict[str, Any], cfg: RouterConfig, mechanisms: Sequence[str]
) -> dict[str, str]:
    """threshold name -> explanation, for thresholds at or above the cap of the ONLY evidence
    mechanism whose measured TPR is 0. Above the cap no session can pause, so TPR is 0. Exactly
    AT the cap, sessions whose evidence saturates the cap still pause (pausing is score >= A),
    so the explanation is attached only when the TPR really is 0: a finding about the design
    (the cap, mech_cap_frac x A, deliberately stops any single check from pausing a session), not
    a failure of the check. Empty with several mechanisms or with no cap."""
    from adr.contracts import PASSIVE_MECHANISMS

    evidence = [m for m in mechanisms if m in ACTIVE_MECHANISMS or m in PASSIVE_MECHANISMS]
    if len(evidence) != 1 or cfg.mech_cap_frac is None:
        return {}
    a0 = nominal_A(cfg)
    cap = cfg.mech_cap_frac * a0
    return {
        name: (
            f"TPR = 0 here is a finding, not a failure: with {evidence[0]} as the only check, the "
            f"per-mechanism cap ({cfg.mech_cap_frac:g}A = {cap:.2f} nats, A = {a0:.2f}) correctly "
            f"stops any single check from pausing a session, and this threshold ({r['A']:.2f}) is "
            "at or above it."
        )
        for name, r in summary["results"].items()
        if r["A"] >= cap - 1e-9 and r.get("tpr") == 0
    }


def router_factory(
    name: str,
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
) -> PolicyFactory:
    """A PolicyFactory for any policy name from adr/baselines/policies.py."""
    from adr.baselines.policies import make_policy

    def make(fmap: FailureMap, cfg: RouterConfig) -> Router:
        return make_policy(name, assessment, items, variants, fmap, cfg)

    return make

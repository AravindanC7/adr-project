"""Person-level cross-fitting: calibrate the review threshold on held-out PEOPLE (step #42).

Owner: Arav. Used by scripts/run_real_router.py once pilot rows exist.

Honest people are the pilot sessions (data_origin == "pilot"); one session_id = one person.
People are split into K folds (fixed seed, sizes as equal as possible). For each fold k:
  1. map_k:  Archana's builder in STRICT mode on the AI rows (train items, held-out model and
             survey excluded, as always) plus the human rows of the OTHER folds only.
  2. A_k:    replay the policy on fold k's people, once each, with map_k, and pick the smallest
             review threshold A (on a grid) that pauses at most `alpha` of them.
  3. AI:     replay every AI evaluation session with the same map_k and threshold A_k.
Pooled over folds: TPR at the calibrated threshold, the honest pause rate, checks per honest
person, each A_k, and 95% bootstrap CIs that resample PEOPLE (and AI sessions), never rows.

The threshold is set on people whose rows never entered the map they are scored with. With few
people per fold, "at most alpha" can only be met by pausing nobody (e.g. 11 people: one pause is
9%), so the 1% and 5% thresholds can coincide; `granularity` reports the smallest nonzero rate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from adr.contracts import Assessment, FailureMap, Item, RenderedItem, Router
from adr.map import build_failure_map
from adr.replay.donors import Donors, Group
from adr.replay.simulate import ReplayResult, replay_session
from adr.router.core import RouterConfig
from adr.router.map_guard import check_router_map

# Review thresholds tried, in nats. Wald's A at alpha=0.01, beta=0.05 is 4.55.
DEFAULT_GRID = tuple(round(x, 2) for x in np.arange(0.25, 20.01, 0.25))

PolicyFactory = Callable[[FailureMap, float], Router]  # (map, review threshold A) -> policy


def person_folds(people: Sequence[str], k: int, seed: int) -> list[list[str]]:
    """Split people into k folds: shuffled with a fixed seed, dealt round-robin, so fold sizes
    differ by at most one and every person is in exactly one fold."""
    unique = sorted(set(people))
    if k < 2:
        raise ValueError("need at least 2 folds")
    if len(unique) < k:
        raise ValueError(f"{len(unique)} people cannot fill {k} folds")
    random.Random(f"crossfit:{seed}").shuffle(unique)
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, person in enumerate(unique):
        folds[i % k].append(person)
    return [sorted(f) for f in folds]


def fold_map(
    ai_rows: pd.DataFrame,
    human_rows: pd.DataFrame,
    held_out_people: Sequence[str],
    version: str,
    n_bootstrap: int = 200,
    seed: int = 34,
) -> FailureMap:
    """map_k: the strict builder on the AI rows plus the human rows of everyone NOT held out.
    Refused by the #25 guard if it is provisional (it never should be: strict mode)."""
    keep = ~human_rows["session_id"].isin(set(held_out_people))
    frame = pd.concat([ai_rows, human_rows[keep]], ignore_index=True)
    fmap = build_failure_map(
        frame,
        version=version,
        n_bootstrap=n_bootstrap,
        seed=seed,
        note=f"cross-fit map: human rows exclude {len(set(held_out_people))} held-out people",
    )
    check_router_map(fmap)
    return fmap


@dataclass
class FoldResult:
    fold: int
    people: list[str]
    threshold: float
    honest: list[ReplayResult]  # one per held-out person, at the threshold
    ai: list[ReplayResult]  # one per AI evaluation session, at the threshold
    map_version: str
    tried: list[tuple[float, float]] = field(default_factory=list)  # (A, honest pause rate)

    @property
    def honest_pause_rate(self) -> float:
        return float(np.mean([r.final.status == "review" for r in self.honest]))


def _replay_people(
    policy: Router,
    assessment: Assessment,
    donors: Donors,
    group: Group,
    sessions: Sequence[str],
    passive: tuple[str, ...],
    seed: int,
) -> list[ReplayResult]:
    return [
        replay_session(policy, assessment, donors.respondent_from(group, s, seed), passive, seed)
        for s in sessions
    ]


def calibrate(
    make: PolicyFactory,
    fmap: FailureMap,
    assessment: Assessment,
    donors: Donors,
    group: Group,
    people: Sequence[str],
    alpha: float,
    passive: tuple[str, ...] = (),
    seed: int = 0,
    grid: Sequence[float] = DEFAULT_GRID,
) -> tuple[float, list[ReplayResult], list[tuple[float, float]]]:
    """The smallest grid threshold A that pauses at most `alpha` of `people` (fold k only).

    The pause rate falls as A rises, so the grid is bisected. If even the largest A pauses too
    many, A = inf (the policy never pauses). Returns (A, the people's results at A, tried)."""
    tried: list[tuple[float, float]] = []
    cache: dict[float, list[ReplayResult]] = {}

    def run(a: float) -> list[ReplayResult]:
        if a not in cache:
            cache[a] = _replay_people(
                make(fmap, a), assessment, donors, group, people, passive, seed
            )
            rate = float(np.mean([r.final.status == "review" for r in cache[a]]))
            tried.append((a, rate))
        return cache[a]

    def ok(a: float) -> bool:
        res = run(a)
        return float(np.mean([r.final.status == "review" for r in res])) <= alpha

    lo, hi = 0, len(grid) - 1
    if not ok(grid[hi]):
        return math.inf, run(math.inf), sorted(tried)
    while lo < hi:  # smallest index with ok()
        mid = (lo + hi) // 2
        if ok(grid[mid]):
            hi = mid
        else:
            lo = mid + 1
    return grid[lo], run(grid[lo]), sorted(tried)


def crossfit(
    make: PolicyFactory,
    assessment: Assessment,
    items: dict[str, Item],
    ai_map_rows: pd.DataFrame,
    human_rows: pd.DataFrame,
    ai_eval_rows: pd.DataFrame,
    folds: list[list[str]],
    alpha: float,
    passive: tuple[str, ...] = (),
    seed: int = 0,
    grid: Sequence[float] = DEFAULT_GRID,
    map_bootstrap: int = 200,
    maps: list[FailureMap] | None = None,
) -> list[FoldResult]:
    """Run the K folds (see module doc). `maps` reuses map_k from an earlier call (the maps do
    not depend on the policy or alpha)."""
    honest_group = Group.human("pilot")
    out = []
    for k, people in enumerate(folds):
        fmap = (
            maps[k]
            if maps is not None
            else fold_map(ai_map_rows, human_rows, people, f"crossfit-fold{k}", map_bootstrap)
        )
        check_router_map(fmap)  # #25 guard, also for maps passed in: no provisional maps
        honest_donors = Donors(
            human_rows[human_rows["session_id"].isin(set(people))], items, fallback_map=fmap
        )
        threshold, honest, tried = calibrate(
            make, fmap, assessment, honest_donors, honest_group, people, alpha, passive, seed, grid
        )
        ai_donors = Donors(ai_eval_rows, items, fallback_map=fmap, type_match_kinds=("ai",))
        policy = make(fmap, threshold)
        ai = [
            r
            for g in ai_donors.groups("ai")
            for r in _replay_people(
                policy, assessment, ai_donors, g, ai_donors.sessions(g), passive, seed
            )
        ]
        out.append(FoldResult(k, list(people), threshold, honest, ai, fmap.version, tried))
    return out


def all_maps(
    ai_map_rows: pd.DataFrame,
    human_rows: pd.DataFrame,
    folds: list[list[str]],
    map_bootstrap: int = 200,
) -> list[FailureMap]:
    return [
        fold_map(ai_map_rows, human_rows, people, f"crossfit-fold{k}", map_bootstrap)
        for k, people in enumerate(folds)
    ]


# ------------------------------------------------------------------ pooled metrics
def _ai_detection_by_session(results: list[FoldResult]) -> dict[str, list[float]]:
    """AI session -> paused-for-review indicator in each fold (every fold replays every one)."""
    by: dict[str, list[float]] = {}
    for f in results:
        for r in f.ai:
            by.setdefault(f"{r.group.label}|{r.donor_session}", []).append(
                float(r.final.status == "review")
            )
    return by


def _honest_by_person(results: list[FoldResult]) -> dict[str, tuple[float, float, float]]:
    """person -> (paused, checks run, items to decision or nan), from their own fold."""
    out = {}
    for f in results:
        for r in f.honest:
            out[r.donor_session] = (
                float(r.final.status == "review"),
                float(r.checks_run),
                np.nan if r.items_to_decision is None else float(r.items_to_decision),
            )
    return out


def pooled_metrics(results: list[FoldResult], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """TPR at the cross-fitted thresholds (mean over AI sessions of the share of folds in which
    the session was paused), the honest pause rate and checks per honest person (each person
    once, in their own fold), with 95% bootstrap CIs that resample people and AI sessions."""
    ai = _ai_detection_by_session(results)
    hu = _honest_by_person(results)
    ai_vals = np.array([np.mean(v) for v in ai.values()]) if ai else np.array([])
    hu_vals = np.array(list(hu.values())) if hu else np.zeros((0, 3))

    def stats(a: np.ndarray, h: np.ndarray) -> dict[str, float | None]:
        decided = h[:, 2] if len(h) else np.array([])
        reached = decided[~np.isnan(decided)]
        return {
            "tpr": float(a.mean()) if len(a) else None,
            "honest_pause_rate": float(h[:, 0].mean()) if len(h) else None,
            "checks_per_honest_person": float(h[:, 1].mean()) if len(h) else None,
            "median_items_to_decision_honest": float(np.median(reached)) if len(reached) else None,
        }

    point = stats(ai_vals, hu_vals)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        a = ai_vals[rng.integers(0, len(ai_vals), len(ai_vals))] if len(ai_vals) else ai_vals
        h = hu_vals[rng.integers(0, len(hu_vals), len(hu_vals))] if len(hu_vals) else hu_vals
        for k, v in stats(a, h).items():
            if v is not None:
                draws[k].append(v)
    ci = {
        k: (tuple(float(x) for x in np.percentile(v, [2.5, 97.5])) if len(v) * 2 > n_boot else None)
        for k, v in draws.items()
    }
    sizes = [len(f.people) for f in results]
    return {
        **point,
        "ci95": ci,
        "thresholds": [f.threshold for f in results],
        "n_people": len(hu),
        "n_ai_sessions": len(ai),
        "people_per_fold": sizes,
        # the smallest honest pause rate above zero any fold can show: 1 / its size
        "granularity": 1 / min(sizes) if sizes else None,
        "n_boot": n_boot,
        "boot_seed": seed,
        "boot_unit": "person (honest) and AI session",
    }


def threshold_rows(policy: str, alpha: float, results: list[FoldResult]) -> list[dict]:
    return [
        {
            "policy": policy,
            "alpha": alpha,
            "fold": f.fold,
            "n_people": len(f.people),
            "A_k": f.threshold,
            "honest_paused": int(sum(r.final.status == "review" for r in f.honest)),
            "honest_pause_rate": f.honest_pause_rate,
            "ai_paused": int(sum(r.final.status == "review" for r in f.ai)),
            "n_ai_sessions": len(f.ai),
            "map_version": f.map_version,
        }
        for f in results
    ]


def router_factory(
    name: str,
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    cfg: RouterConfig,
) -> PolicyFactory:
    """A PolicyFactory for any policy name from adr/baselines/policies.py."""
    from adr.baselines.policies import make_policy

    def make(fmap: FailureMap, upper: float) -> Router:
        return make_policy(name, assessment, items, variants, fmap, replace(cfg, upper=upper))

    return make

"""Respondent-level metrics for replayed sessions (AGENTS.md section 7.5), with bootstrap CIs.

Owner: Arav. Works on anything with `.final.score`, `.final.status`, `.checks_run` and
`.items_to_decision`: ReplayResult here, and SessionResult from adr/router/synthetic.py.

  tpr_at_1pct_fpr / tpr_at_5pct_fpr   share of AI respondents flagged, at the score threshold
                                      that flags at most 1% / 5% of honest respondents
  ai_paused_for_review                share of AI sessions that ended paused for review
  honest_paused_for_review            same for honest people (the false flags we act on)
  checks_per_honest_person            mean active checks served to an honest person (burden)
  median_items_to_decision            items shown when review/cleared was first reached,
                                      over every session that reached one
  share_decided                       share of all sessions that reached review or cleared
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

METRICS = (
    "tpr_at_1pct_fpr",
    "tpr_at_5pct_fpr",
    "ai_paused_for_review",
    "honest_paused_for_review",
    "checks_per_honest_person",
    "median_items_to_decision",
    "share_decided",
)


def detection_at_fpr(
    ai_scores: list[float], human_scores: list[float], fpr: float
) -> tuple[float, float, float]:
    """Pick the lowest threshold t with P(human score > t) <= fpr; flag when score > t.

    Returns (TPR on AI, actual FPR on humans, t). Ties are never split, so the actual FPR can
    be below the target when many honest people share a score.
    """
    h = np.asarray(human_scores, dtype=float)
    a = np.asarray(ai_scores, dtype=float)
    for t in np.unique(h):  # ascending
        actual = float(np.mean(h > t))
        if actual <= fpr:
            return float(np.mean(a > t)), actual, float(t)
    raise AssertionError("unreachable: the max human score always qualifies")


def _cluster_keys(results: Sequence[Any]) -> list[Any]:
    """Which logged session each simulated respondent copied. Respondents with no donor session
    (synthetic groups, or SessionResult from adr/router/synthetic.py) are their own cluster."""
    keys: list[Any] = []
    for i, r in enumerate(results):
        session = getattr(r, "donor_session", None)
        group = getattr(r, "group", None)
        keys.append((getattr(group, "label", group), session) if session is not None else ("own", i))
    return keys


class _Arrays:
    """One respondent per array position, so a bootstrap draw is just a list of indexes.
    `clusters` holds, per donor session, the positions of the respondents that copied it."""

    def __init__(self, results: Sequence[Any]):
        by_key: dict[Any, list[int]] = {}
        for i, key in enumerate(_cluster_keys(results)):
            by_key.setdefault(key, []).append(i)
        self.clusters = [np.array(v, dtype=int) for _, v in sorted(by_key.items(), key=str)]
        self.score = np.array([r.final.score for r in results], dtype=float)
        self.review = np.array([r.final.status == "review" for r in results], dtype=float)
        self.checks = np.array([r.checks_run for r in results], dtype=float)
        self.decided = np.array(
            [np.nan if r.items_to_decision is None else r.items_to_decision for r in results],
            dtype=float,
        )


def _metrics(ai: _Arrays, hu: _Arrays, ia: np.ndarray, ih: np.ndarray) -> dict[str, float | None]:
    a_score, h_score = ai.score[ia], hu.score[ih]
    out: dict[str, float | None] = dict.fromkeys(METRICS)
    if len(ih):
        if len(ia):
            out["tpr_at_1pct_fpr"] = detection_at_fpr(a_score, h_score, 0.01)[0]
            out["tpr_at_5pct_fpr"] = detection_at_fpr(a_score, h_score, 0.05)[0]
        out["honest_paused_for_review"] = float(hu.review[ih].mean())
        out["checks_per_honest_person"] = float(hu.checks[ih].mean())
    if len(ia):
        out["ai_paused_for_review"] = float(ai.review[ia].mean())
    decided = np.concatenate([ai.decided[ia], hu.decided[ih]])
    if len(decided):
        reached = decided[~np.isnan(decided)]
        out["share_decided"] = len(reached) / len(decided)
        out["median_items_to_decision"] = float(np.median(reached)) if len(reached) else None
    return out


def evaluate(
    ai_results: Sequence[Any],
    human_results: Sequence[Any],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Point estimates plus 95% bootstrap intervals.

    The bootstrap resamples donor sessions, not simulated respondents: many respondents copy the
    same logged session, so resampling them one by one would make the intervals shrink as --n
    grows although no new real data was added. Each draw picks logged sessions with replacement
    (AI and honest separately) and takes every respondent that copied each one, then recomputes
    every metric. The interval is the 2.5th and 97.5th percentile of those draws, or None if a
    metric was undefined in more than half of them. `n_ai_sessions` / `n_human_sessions` say how
    many independent units the intervals rest on.
    """
    ai, hu = _Arrays(ai_results), _Arrays(human_results)
    n_ai, n_hu = len(ai_results), len(human_results)
    point = _metrics(ai, hu, np.arange(n_ai), np.arange(n_hu))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_boot):
        ia = _resample(ai.clusters, rng)
        ih = _resample(hu.clusters, rng)
        for name, value in _metrics(ai, hu, ia, ih).items():
            if value is not None:
                draws[name].append(value)
    ci: dict[str, tuple[float, float] | None] = {}
    for name in METRICS:
        values = draws[name]
        if n_boot and len(values) * 2 > n_boot:
            lo, hi = np.percentile(values, [2.5, 97.5])
            ci[name] = (float(lo), float(hi))
        else:
            ci[name] = None
    return {"n_ai": n_ai, "n_human": n_hu, "n_ai_sessions": len(ai.clusters),
            "n_human_sessions": len(hu.clusters), **point, "ci95": ci, "n_boot": n_boot,
            "boot_seed": seed, "boot_unit": "donor_session"}


def _resample(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if not clusters:
        return np.arange(0)
    picks = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[k] for k in picks])

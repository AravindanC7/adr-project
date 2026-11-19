"""Wald's sequential probability ratio test: thresholds and the capped score update."""

from __future__ import annotations

import math


def wald_thresholds(alpha: float, beta: float) -> tuple[float, float]:
    """Upper threshold A (flag for review) and lower threshold B (stop checking).

    alpha = target false-flag rate on honest people, beta = target miss rate on AI users.
    A = ln((1 - beta) / alpha),  B = ln(beta / (1 - alpha)).
    These assume independent observations; calibrate empirically later (step #42).
    """
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must be in (0, 1)")
    return math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))


def capped_delta(
    prior_total: float, llr: float, cap: float | None, floor: float | None = None
) -> float:
    """How much of `llr` to add, given this mechanism has already added `prior_total`.

    With a cap, one mechanism's running total stays inside [floor, +cap]. The router passes its
    lower threshold B as the floor (team decision, Nov 18): +cap < A, so one check alone can never
    pause a session, but it CAN clear one by reaching B. Repeated, correlated evidence from the
    same check (e.g. writing style on every answer) then can't dominate either way.
    floor None = symmetric [-cap, +cap] (the old bound). cap None = no bound at all.
    """
    if cap is None:
        return llr
    lo = -cap if floor is None else floor
    new_total = max(lo, min(cap, prior_total + llr))
    return new_total - prior_total


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def entropy(p: float) -> float:
    """Binary entropy in nats."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

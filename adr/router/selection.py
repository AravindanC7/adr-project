"""How useful is running a given check next? Expected evidence minus penalties.

Two objectives:

"kl" (default), Chernoff's rule for sequential testing (Chernoff 1959):
    pi * KL(P_AI || P_human) + (1 - pi) * KL(P_human || P_AI)
  = the expected amount the score moves toward the right threshold, in nats, weighted by how
  likely each hypothesis currently is. It stays large when the router is already fairly sure,
  which is exactly when it needs one more check to reach a decision.

"info_gain": expected drop in uncertainty, H(pi) - E[H(pi | outcome)]. It shrinks toward 0 as
  pi approaches 0 or 1, so on its own the router stops checking just before it could decide.
  Kept for comparison in the report.
"""

from __future__ import annotations

import math

from adr.contracts import MapCell
from adr.router.sprt import entropy


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(p[o] * math.log(p[o] / q[o]) for o in p)


def expected_evidence(pi: float, cell: MapCell) -> float:
    return pi * kl(cell.p_ai, cell.p_human) + (1 - pi) * kl(cell.p_human, cell.p_ai)


def expected_info_gain(pi: float, cell: MapCell) -> float:
    h_now = entropy(pi)
    expected_after = 0.0
    for o, p_ai in cell.p_ai.items():
        p_o = pi * p_ai + (1 - pi) * cell.p_human[o]
        if p_o > 0:
            expected_after += p_o * entropy(pi * p_ai / p_o)
    return max(0.0, h_now - expected_after)


def false_flag_risk(cell: MapCell) -> float:
    """P(an honest person produces an outcome that points toward AI) for this check."""
    return sum(cell.p_human[o] for o in cell.p_human if cell.llr(o) > 0)


def utility(
    pi: float,
    cell: MapCell,
    burden: float,
    lambda_burden: float,
    lambda_fp: float,
    objective: str = "kl",
) -> float:
    if objective == "kl":
        gain = expected_evidence(pi, cell)
    elif objective == "info_gain":
        gain = expected_info_gain(pi, cell)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    return gain - lambda_burden * burden - lambda_fp * false_flag_risk(cell)

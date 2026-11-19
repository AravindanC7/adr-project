"""The Adaptive Defense Router: SPRT scoring + choosing the most informative next check.

Implements the `adr.contracts.Router` protocol, so the server (Harshini) and the replay
simulation (Arav) call it the same way.

Each step:
  - next(state): among the items not yet shown, score every pre-built variant whose check has a
    failure-map cell, by expected information gain minus burden and false-flag penalties. Serve
    the best (near-ties broken at random). If no check is worth it, serve the next item plain.
  - update(state, variant_id, outcome): add the check's log-likelihood ratio to the score
    (capped per mechanism), then set the status from the thresholds.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from adr.contracts import (
    ACTIVE_MECHANISMS,
    PASSIVE_MECHANISMS,
    Assessment,
    Choice,
    FailureMap,
    Item,
    RenderedItem,
    SessionState,
    check_outcome,
    make_variant_id,
    parse_variant_id,
)
from adr.router.selection import utility
from adr.router.sprt import capped_delta, sigmoid, wald_thresholds

_CLEAR_TOL = 1e-9  # nats


@dataclass(frozen=True)
class RouterConfig:
    alpha: float = 0.01            # target false-flag rate on honest people
    # Target miss rate on AI users. 0.05, not the 0.2 first suggested: on the synthetic map,
    # beta=0.2 clears too many AI users early (TPR@1% FPR 0.82 vs 0.92) for ~2 fewer checks.
    beta: float = 0.05
    prior_log_odds: float = 0.0    # prior belief before any evidence (0 = 50/50)
    escalate_frac: float = 0.5     # status "escalated" once score >= escalate_frac * A
    lambda_burden: float = 0.5     # penalty per unit of respondent burden
    lambda_fp: float = 0.05        # penalty per unit of honest-person false-flag risk
    escalated_burden_scale: float = 0.25  # when escalated, burden matters less: harder checks OK
    # Max total any one mechanism may add, as a fraction of the review threshold A: cap = frac * A
    # nats, so it moves with A when A is recalibrated (#42). 0.65 = the old fixed 3.0 at A = 4.60
    # (2.96 at the default A = 4.55). Below 1, no single mechanism can trigger review on its own.
    # None = no cap.
    mech_cap_frac: float | None = 0.65
    # Optional groups of mechanisms that share ONE cap of the same size (mech_cap_frac * A),
    # e.g. {"passive": ("behaviour", "style")}: correlated signals from the same person then can't
    # add up past one cap together. None = off (each mechanism has its own cap, as before).
    # NOTE: with groups, SessionState.contrib is keyed by the GROUP name, not the mechanism: read a
    # mechanism's running total as state.contrib[router.cap_key(mechanism)] (review H6).
    mech_group_caps: dict[str, tuple[str, ...]] | None = None
    objective: str = "kl"          # "kl" (Chernoff's rule) or "info_gain"; see selection.py
    tie_eps: float = 0.01          # candidates within this of the best are a tie
    allow_reorder: bool = True     # may serve items out of display order
    attack_mode: str | None = None # which failure-map cells to use; None = pooled over modes
    upper: float | None = None     # override A (after empirical calibration, step #42)
    lower: float | None = None     # override B
    # False = never clear (router_noclear): keeps checking a low score. B still floors each
    # mechanism's total, so both policies bound evidence the same way.
    clear: bool = True


def _cap_keys(groups: dict[str, tuple[str, ...]] | None) -> dict[str, str]:
    """mechanism -> group name, checked: known mechanisms, each in at most one group, and group
    names that don't collide with a mechanism's own contrib key."""
    if not groups:
        return {}
    known = set(ACTIVE_MECHANISMS) | set(PASSIVE_MECHANISMS)
    out: dict[str, str] = {}
    for name, members in groups.items():
        if name in known:
            raise ValueError(f"mech_group_caps: group name {name!r} is a mechanism name")
        if not members:
            raise ValueError(f"mech_group_caps: group {name!r} is empty")
        for m in members:
            if m not in known:
                raise ValueError(f"mech_group_caps: unknown mechanism {m!r} in {name!r}")
            if m in out:
                raise ValueError(f"mech_group_caps: {m!r} is in both {out[m]!r} and {name!r}")
            out[m] = name
    return out


def eligible_checks(item: Item, variants: list[RenderedItem]) -> list[RenderedItem]:
    """The variants any policy may serve as a check on `item`: an active mechanism that the item
    lists in eligible_mechanisms. The router and every baseline use this one rule, so a
    comparison never gives one policy a check the others can't use."""
    allowed = set(item.eligible_mechanisms) & set(ACTIVE_MECHANISMS)
    return [v for v in variants if v.mechanism in allowed]


class SPRTRouter:
    def __init__(
        self,
        items: dict[str, Item],
        variants: dict[str, list[RenderedItem]],
        failure_map: FailureMap,
        config: RouterConfig = RouterConfig(),
    ):
        self.items = items
        self.variants = variants
        self.fmap = failure_map
        self.cfg = config
        a, b = wald_thresholds(config.alpha, config.beta)
        self.A = config.upper if config.upper is not None else a
        self.B = config.lower if config.lower is not None else b
        if config.mech_cap_frac is not None and config.mech_cap_frac <= 0:
            raise ValueError("mech_cap_frac must be > 0 (or None for no cap)")
        # Per-mechanism cap in nats, from A after any `upper` override.
        self.mech_cap = None if config.mech_cap_frac is None else config.mech_cap_frac * self.A
        # Each mechanism's (or group's) total is bounded to [B, +cap]: one check alone can reach
        # B (clear) but not A (review). None when there is no cap, or B is not finite.
        self.mech_floor = (
            self.B if self.mech_cap is not None and math.isfinite(self.B) else None
        )
        self._cap_key = _cap_keys(config.mech_group_caps)
        self._assessments: dict[str, Assessment] = {}
        self._variant_index: dict[str, RenderedItem] = {
            v.variant_id: v for vs in variants.values() for v in vs
        }

    # ------------------------------------------------------------ protocol
    def start(
        self, assessment: Assessment, session_id: str, seed: int
    ) -> tuple[SessionState, Choice]:
        missing = [i for i in assessment.item_ids if i not in self.items]
        if missing:
            raise KeyError(f"items not loaded in router: {missing[:5]}")
        self._assessments[assessment.assessment_id] = assessment
        state = SessionState(
            session_id=session_id,
            assessment_id=assessment.assessment_id,
            score=self.cfg.prior_log_odds,
            seed=seed,
        )
        choice = self.next(state)
        assert choice is not None  # an assessment has at least one item
        return state, choice

    def update(
        self, state: SessionState, variant_id: str, outcome: str | None
    ) -> SessionState:
        """Record one observation. outcome None = shown, nothing to score (NoObservation): the
        variant counts as served, but adds no evidence and leaves the status as it was."""
        item_id, mech, _ = parse_variant_id(variant_id)
        if item_id not in self.items:
            raise KeyError(f"unknown item {item_id!r}")
        new = state.model_copy(deep=True)

        if mech not in PASSIVE_MECHANISMS and variant_id not in new.served:
            new.served.append(variant_id)
        if outcome is None:
            new.history.append((variant_id, "none", 0.0))
            return new

        delta = 0.0
        if mech is None:
            if outcome != "none":
                raise ValueError('plain variants have outcome "none"')
        else:
            check_outcome(mech, outcome)
            cell = self.fmap.lookup(mech, self.items[item_id].item_type, self.cfg.attack_mode)
            if cell is not None:
                key = self.cap_key(mech)  # the mechanism, or the group it shares a cap with
                total = new.contrib.get(key, 0.0)
                # Bounded to [B, +cap]: B from this config, so it follows calibration.
                delta = capped_delta(total, cell.llr(outcome), self.mech_cap, self.mech_floor)
                new.contrib[key] = total + delta
        new.score += delta
        new.history.append((variant_id, outcome, delta))
        new.status = self._status(new)
        return new

    def next(self, state: SessionState) -> Choice | None:
        if state.status == "review":
            return None  # paused for human review
        assessment = self._assessments[state.assessment_id]
        shown = {parse_variant_id(v)[0] for v in state.served}
        remaining = [i for i in assessment.item_ids if i not in shown]
        if not remaining:
            return None
        if state.status == "cleared":
            return self._plain(remaining[0])

        pool = remaining if self.cfg.allow_reorder else remaining[:1]
        scored = self._score_candidates(state, pool)
        if not scored:
            return self._plain(remaining[0])
        best = max(u for u, _ in scored)
        if best <= 0:
            return self._plain(remaining[0])
        ties = [v for u, v in scored if u >= best - self.cfg.tie_eps]
        rng = random.Random(f"{state.seed}:{state.session_id}:{len(state.served)}")
        pick = rng.choice(ties)
        return Choice(item_id=pick.item_id, variant_id=pick.variant_id, mechanism=pick.mechanism)

    # ------------------------------------------------------------ helpers
    def _status(self, state: SessionState) -> str:
        if state.status == "review" or state.score >= self.A:
            return "review"
        # Tolerance: a mechanism at its floor makes score == B up to float rounding of the deltas.
        if self.cfg.clear and state.score <= self.B + _CLEAR_TOL:
            return "cleared"
        if state.score >= self.cfg.escalate_frac * self.A:
            return "escalated"
        return "active"

    def cap_key(self, mechanism: str) -> str:
        """The key in SessionState.contrib that `mechanism` counts toward: its group's name when
        RouterConfig.mech_group_caps puts it in a group, otherwise the mechanism itself."""
        return self._cap_key.get(mechanism, mechanism)

    def checks(self, item_id: str) -> list[RenderedItem]:
        """This item's servable checks (eligible_checks), in variant-file order."""
        return eligible_checks(self.items[item_id], self.variants.get(item_id, []))

    def _plain(self, item_id: str) -> Choice:
        plain = [v for v in self.variants.get(item_id, []) if v.mechanism is None]
        vid = plain[0].variant_id if plain else make_variant_id(item_id, None)
        return Choice(item_id=item_id, variant_id=vid, mechanism=None)

    def _score_candidates(
        self, state: SessionState, pool: list[str]
    ) -> list[tuple[float, RenderedItem]]:
        pi = sigmoid(state.score)
        burden_w = self.cfg.lambda_burden * (
            self.cfg.escalated_burden_scale if state.status == "escalated" else 1.0
        )
        out: list[tuple[float, RenderedItem]] = []
        for item_id in pool:
            item = self.items[item_id]
            for v in self.checks(item_id):
                if self.mech_cap is not None and (
                    state.contrib.get(self.cap_key(v.mechanism), 0.0) >= self.mech_cap
                ):
                    continue  # this check can't add more evidence toward AI
                cell = self.fmap.lookup(v.mechanism, item.item_type, self.cfg.attack_mode)
                if cell is None:
                    continue
                u = utility(pi, cell, v.burden, burden_w, self.cfg.lambda_fp, self.cfg.objective)
                out.append((u, v))
        return out

    def explain(self, state: SessionState, top: int = 5) -> list[dict]:
        """For the demo and debugging: the top candidate checks and their utilities."""
        assessment = self._assessments[state.assessment_id]
        shown = {parse_variant_id(v)[0] for v in state.served}
        remaining = [i for i in assessment.item_ids if i not in shown]
        scored = sorted(self._score_candidates(state, remaining), key=lambda t: -t[0])[:top]
        return [{"variant_id": v.variant_id, "mechanism": v.mechanism, "utility": round(u, 4)}
                for u, v in scored]

"""Baselines 1-4 (AGENTS.md section 7.4, step #32). They share the router's SPRT scoring, so
only the choice of check differs:

  1. Best single mechanism: one fixed check wherever the item allows it. Which check is picked
     per assessment by best_single_mechanism() on the train map (adr/baselines/plans.py).
  2. All checks: every item gets a check (plans.AllChecksRouter).
  3. Fixed plan: the check for every item is chosen before the session starts
     (plans.FixedPlanRouter).
  4. Random: a random available check for each item.

Baseline 5 (published survey traps, #33) is a fixed set of traps scored on its own, not a policy:
adr/baselines/published_traps.py and scripts/baseline5.py.

Every baseline serves only router.core.eligible_checks() (active mechanisms the item lists in
eligible_mechanisms), the same rule as the router. Every baseline goes through items in display order and, unlike the router, never clears a
respondent early: a low score does not stop the checks, so status never becomes "cleared". All
of them still pause for review at the upper threshold A, like the router. update(), including
update(..., None) for "shown, nothing to score", is the router's own.
"""

from __future__ import annotations

import random

from adr.contracts import ACTIVE_MECHANISMS, Choice, SessionState, parse_variant_id
from adr.router.core import SPRTRouter


class InOrderBaseline(SPRTRouter):
    """Items in display order, one check (or plain) per item, no early clearing."""

    def next(self, state: SessionState) -> Choice | None:
        if state.status == "review":
            return None  # paused for human review, as with the router
        assessment = self._assessments[state.assessment_id]
        shown = {parse_variant_id(v)[0] for v in state.served}
        remaining = [i for i in assessment.item_ids if i not in shown]
        if not remaining:
            return None
        return self._pick(state, remaining[0])

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        raise NotImplementedError

    def _status(self, state: SessionState) -> str:
        if state.status == "review" or state.score >= self.A:
            return "review"
        if state.score >= self.cfg.escalate_frac * self.A:
            return "escalated"
        return "active"  # never "cleared": baselines keep checking whatever the score


class SingleMechanismRouter(InOrderBaseline):
    """Baseline 1: always `mechanism` where the item allows it, else plain.

    No per-mechanism cap. The cap (RouterConfig.mech_cap_frac) limits how much one mechanism
    adds next to the others, because evidence from one mechanism is correlated and the router
    can keep choosing its strongest check. A single-mechanism policy has one mechanism by
    definition: with the default cap below A (0.65 x A), it could never reach review at all,
    so the baseline would measure the cap, not the mechanism. The other baselines keep the
    router's cap. `expected_evidence` is what best_single_mechanism() chose it on, if it did.
    """

    def __init__(self, *args, mechanism: str, expected_evidence: dict[str, float] | None = None,
                 **kw):
        super().__init__(*args, **kw)
        if mechanism not in ACTIVE_MECHANISMS:
            raise ValueError(f"{mechanism!r} is not an active mechanism")
        self.mechanism = mechanism
        self.expected_evidence = expected_evidence
        self.mech_cap = None  # one mechanism by definition: see the class docstring

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        for v in self.checks(item_id):
            if v.mechanism == self.mechanism:
                return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)
        return self._plain(item_id)


class RandomCheckRouter(InOrderBaseline):
    """Baseline 4: a random active variant of each item (seeded per session and step)."""

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        options = self.checks(item_id)
        if not options:
            return self._plain(item_id)
        rng = random.Random(f"{state.seed}:{state.session_id}:{len(state.served)}")
        v = rng.choice(options)
        return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)

"""Baselines 1-4 (AGENTS.md section 7.4, step #32). They share the router's SPRT scoring, so
only the choice of check differs:

  1. Best single mechanism: one fixed check wherever the item allows it. Which check is picked
     per assessment by best_single_mechanism() on the train map (adr/baselines/plans.py).
  2. All checks: every item gets a check (plans.AllChecksRouter).
  3. Fixed plan: the check for every item is chosen before the session starts
     (plans.FixedPlanRouter).
  4. Random: a random available check for each item.

Baseline 5 (published survey traps, #33) comes later.

Every baseline goes through items in display order and, unlike the router, never clears a
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
    """Baseline 1: always `mechanism` where the item has a variant for it, else plain."""

    def __init__(self, *args, mechanism: str, **kw):
        super().__init__(*args, **kw)
        if mechanism not in ACTIVE_MECHANISMS:
            raise ValueError(f"{mechanism!r} is not an active mechanism")
        self.mechanism = mechanism

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        for v in self.variants.get(item_id, []):
            if v.mechanism == self.mechanism:
                return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)
        return self._plain(item_id)


class RandomCheckRouter(InOrderBaseline):
    """Baseline 4: a random active variant of each item (seeded per session and step)."""

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        options = [v for v in self.variants.get(item_id, []) if v.mechanism in ACTIVE_MECHANISMS]
        if not options:
            return self._plain(item_id)
        rng = random.Random(f"{state.seed}:{state.session_id}:{len(state.served)}")
        v = rng.choice(options)
        return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)

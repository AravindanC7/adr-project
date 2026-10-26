"""Early versions of two baselines (step #32), sharing the router's SPRT scoring so that only
the choice of check differs:

  Baseline 1, best single mechanism: always run one fixed check wherever the item allows it.
  Baseline 4, random: pick a random available check for each item.

Both go through items in display order. Baselines 2 (all checks on every item), 3 (fixed plan)
and 5 (published traps) come later.
"""

from __future__ import annotations

import random

from adr.contracts import ACTIVE_MECHANISMS, Choice, SessionState, parse_variant_id
from adr.router.core import SPRTRouter


class _InOrder(SPRTRouter):
    def _remaining(self, state: SessionState) -> list[str]:
        assessment = self._assessments[state.assessment_id]
        shown = {parse_variant_id(v)[0] for v in state.served}
        return [i for i in assessment.item_ids if i not in shown]


class SingleMechanismRouter(_InOrder):
    def __init__(self, *args, mechanism: str, **kw):
        super().__init__(*args, **kw)
        if mechanism not in ACTIVE_MECHANISMS:
            raise ValueError(f"{mechanism!r} is not an active mechanism")
        self.mechanism = mechanism

    def next(self, state: SessionState) -> Choice | None:
        if state.status == "review":
            return None
        remaining = self._remaining(state)
        if not remaining:
            return None
        item_id = remaining[0]
        for v in self.variants.get(item_id, []):
            if v.mechanism == self.mechanism:
                return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)
        return self._plain(item_id)


class RandomCheckRouter(_InOrder):
    def next(self, state: SessionState) -> Choice | None:
        if state.status == "review":
            return None
        remaining = self._remaining(state)
        if not remaining:
            return None
        item_id = remaining[0]
        options = [v for v in self.variants.get(item_id, []) if v.mechanism in ACTIVE_MECHANISMS]
        if not options:
            return self._plain(item_id)
        rng = random.Random(f"{state.seed}:{state.session_id}:{len(state.served)}")
        v = rng.choice(options)
        return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)

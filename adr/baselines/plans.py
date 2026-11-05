"""Baselines 1-3 that plan their checks from the failure map before any session (step #32).

  best_single_mechanism()   picks baseline 1's one mechanism for an assessment
  AllChecksRouter           baseline 2: a check on every item
  FixedPlanRouter           baseline 3: the best check per item, chosen in advance

All of these read only the failure map they are given. Pass the map fitted on the TRAIN rows
(adr/replay/quick_map.py drops held-out rows, as will Archana's builder, #34): choosing a
baseline's checks with held-out data would tune it on the evaluation set (AGENTS.md section 8).
"""

from __future__ import annotations

from collections import defaultdict

from adr.baselines.simple import InOrderBaseline
from adr.contracts import (
    ACTIVE_MECHANISMS,
    Assessment,
    Choice,
    FailureMap,
    Item,
    RenderedItem,
    SessionState,
)
from adr.router.selection import expected_evidence, utility

PI_PLAN = 0.5  # no evidence yet: AI and human equally likely when a plan is made


def best_single_mechanism(
    fmap: FailureMap,
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    attack_mode: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Baseline 1's mechanism for this assessment: the active mechanism whose variants add the
    most expected evidence (selection.expected_evidence at pi = 0.5), summed over the items it
    can run on. Summing rewards coverage: a strong check on one item can lose to a decent check
    on every item. Returns (mechanism, {mechanism: total}); ties go to the first name A-Z."""
    totals: dict[str, float] = defaultdict(float)
    for item_id in assessment.item_ids:
        item = items[item_id]
        for mech in {v.mechanism for v in variants.get(item_id, [])} & set(ACTIVE_MECHANISMS):
            cell = fmap.lookup(mech, item.item_type, attack_mode)
            if cell is not None:
                totals[mech] += expected_evidence(PI_PLAN, cell)
    if not totals:
        raise ValueError(f"no active mechanism of {assessment.assessment_id} has a map cell")
    best = max(sorted(totals), key=lambda m: totals[m])
    return best, dict(totals)


def multi_check_items(
    assessment: Assessment, items: dict[str, Item], variants: dict[str, list[RenderedItem]]
) -> list[str]:
    """Items with variants for more than one eligible active check. AllChecksRouter runs only
    one of them, so this says how much "all checks" leaves out."""
    out = []
    for item_id in assessment.item_ids:
        eligible = set(items[item_id].eligible_mechanisms) & set(ACTIVE_MECHANISMS)
        mechs = {v.mechanism for v in variants.get(item_id, []) if v.mechanism in eligible}
        if len(mechs) > 1:
            out.append(item_id)
    return out


class AllChecksRouter(InOrderBaseline):
    """Baseline 2: every item gets its first eligible active variant (in variant-file order).

    One active check per item at most: the contracts serve one variant per item, and a variant
    carries one mechanism, so two active checks can't be stacked on the same item. Few items
    have more than one eligible check (multi_check_items() counts them; run_replay.py records
    the count in config.yaml). Passive checks are observed on every item as usual.
    """

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        eligible = self.items[item_id].eligible_mechanisms
        for v in self.variants.get(item_id, []):
            if v.mechanism in ACTIVE_MECHANISMS and v.mechanism in eligible:
                return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)
        return self._plain(item_id)


class FixedPlanRouter(InOrderBaseline):
    """Baseline 3: the check for every item is fixed before the session starts.

    Each item's variants are ranked by the router's own utility at pi = 0.5: expected evidence
    minus the same burden and false-flag penalties (RouterConfig.lambda_burden, lambda_fp). The
    best one is served if its utility is positive, otherwise the item is served plain, which is
    the router's rule too. Ties go to the earlier variant. Items come in display order, and
    nothing changes with the answers: no adaptivity and no early clearing. Comparing this with
    the router isolates what adapting to the answers adds.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.plan: dict[str, RenderedItem | None] = {}
        for item_id, vs in self.variants.items():
            item = self.items[item_id]
            best, best_u = None, 0.0
            for v in vs:
                if v.mechanism not in ACTIVE_MECHANISMS:
                    continue
                cell = self.fmap.lookup(v.mechanism, item.item_type, self.cfg.attack_mode)
                if cell is None:
                    continue
                u = utility(PI_PLAN, cell, v.burden, self.cfg.lambda_burden, self.cfg.lambda_fp,
                            self.cfg.objective)
                if u > best_u:
                    best, best_u = v, u
            self.plan[item_id] = best

    def _pick(self, state: SessionState, item_id: str) -> Choice:
        v = self.plan.get(item_id)
        if v is None:
            return self._plain(item_id)
        return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)

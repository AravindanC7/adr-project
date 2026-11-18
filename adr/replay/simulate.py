"""Replay one simulated respondent through any Router, using logged outcomes (step #27).

Owner: Arav. Uses only the Router protocol (start -> next -> update), so the SPRT router and every
baseline are evaluated by exactly the same code.

Each item the router serves:
  - plain variant: update with outcome "none" (no evidence).
  - a check: ask the donor what this respondent got, and update with it.
  - passive checks (e.g. behaviour, style) are observed on every item shown.
NoObservation (nothing to score, or no data anywhere) means no evidence. For a passive check it
is simply skipped. For an active check the variant was still shown, so the router gets
update(state, variant_id, None): "shown, nothing to score" (served, delta 0, status unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass

from adr.contracts import (
    PASSIVE_MECHANISMS,
    Assessment,
    NoObservation,
    Router,
    SessionState,
    make_variant_id,
)
from adr.replay.donors import Donor, Group


@dataclass
class ReplayResult:
    group: Group
    donor_session: str | None
    final: SessionState
    items_shown: int
    checks_run: int                # active checks served (a skipped one still counts: it was shown)
    items_to_decision: int | None  # items shown when the status first became review or cleared
    fallbacks: int                 # outcomes taken from another session of the same group
    synthetic: int                 # outcomes drawn from the failure map: no logged row
    skipped: int                   # NoObservation: no evidence given to the router
    same_type: int = 0             # outcomes from this session's own rows on another item of the
                                   # same item_type (AI sessions on the pilot bank)

    @property
    def kind(self) -> str:
        return self.group.kind


def replay_session(
    router: Router,
    assessment: Assessment,
    donor: Donor,
    passive: tuple[str, ...] = (),
    seed: int = 0,
) -> ReplayResult:
    for p in passive:
        if p not in PASSIVE_MECHANISMS:
            raise ValueError(f"{p!r} is not a passive mechanism")
    session_id = f"replay:{donor.group.label}:{seed}"
    state, choice = router.start(assessment, session_id, seed)
    shown = checks = skipped = 0
    decided_at = None
    while choice is not None:
        shown += 1
        if shown > len(assessment.item_ids):
            raise RuntimeError(f"{type(router).__name__} served more items than the assessment has")
        if choice.mechanism is None:
            state = router.update(state, choice.variant_id, "none")
        else:
            checks += 1
            try:
                state = router.update(
                    state, choice.variant_id, donor.outcome(choice.item_id, choice.mechanism)
                )
            except NoObservation:
                skipped += 1
                state = router.update(state, choice.variant_id, None)
        for p in passive:
            try:
                outcome = donor.outcome(choice.item_id, p)
            except NoObservation:
                skipped += 1
                continue
            state = router.update(state, make_variant_id(choice.item_id, p), outcome)
        if decided_at is None and state.status in ("review", "cleared"):
            decided_at = shown
        choice = router.next(state)
    return ReplayResult(
        group=donor.group,
        donor_session=donor.session_id,
        final=state,
        items_shown=shown,
        checks_run=checks,
        items_to_decision=decided_at,
        fallbacks=donor.fallbacks,
        synthetic=donor.synthetic,
        skipped=skipped,
        same_type=getattr(donor, "same_type", 0),
    )

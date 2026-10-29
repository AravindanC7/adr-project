"""Router API: what the server (Harshini) and the replay simulation (Arav) call.

Owner: Arav. DRAFT until the team freezes it.

Flow for one session:
    state, choice = router.start(assessment, session_id, seed)
    loop:
        show choice.variant_id to the respondent; score their response with each applicable
        mechanism (the active one, plus passive behaviour/style checks);
        for each (variant_id, outcome): state = router.update(state, variant_id, outcome)
        choice = router.next(state)       # None = assessment finished
Never send any of these to the respondent's browser (an AI agent reads everything it gets):
    - SessionState (score, status, contrib, history)
    - Choice.mechanism
    - any variant_id (it names the mechanism; send an opaque per-step token instead)
The browser only needs the visible item and whether the assessment is finished.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from adr.contracts.schema import Assessment, MechanismName

SessionStatus = Literal["active", "escalated", "review", "cleared"]


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    assessment_id: str
    score: float = Field(default=0.0, description="s_t, log-odds that the respondent uses AI")
    served: list[str] = Field(default_factory=list, description="variant_ids already shown")
    history: list[tuple[str, str, float]] = Field(
        default_factory=list, description="(variant_id, outcome, delta) per observation"
    )
    status: SessionStatus = "active"
    seed: int = 0
    # PROPOSED (not in the Oct 17 draft): running total per mechanism, needed for the
    # per-mechanism cap that limits overconfidence from correlated signals.
    contrib: dict[str, float] = Field(default_factory=dict)


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    variant_id: str
    mechanism: MechanismName | None


@runtime_checkable
class Router(Protocol):
    def start(
        self, assessment: Assessment, session_id: str, seed: int
    ) -> tuple[SessionState, Choice]: ...

    def update(self, state: SessionState, variant_id: str, outcome: str) -> SessionState: ...

    def next(self, state: SessionState) -> Choice | None: ...

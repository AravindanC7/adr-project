"""Web server for the respondent-facing front end (step #11 skeleton).

Owner: Harshini. Reviewer: Arav.

Wraps the router API (adr/contracts/router_api.py) over HTTP:

    GET  /assessments            the list the start page offers
    POST /session/start          {assessment_id}            -> {session_id, finished, item}
    POST /session/{id}/respond   {token, answer, telemetry} -> {session_id, finished, item}

What the browser gets, and nothing more (AGENTS.md section 6.5, "Server-side only"): the
visible question and whether the assessment is finished. An AI agent reads everything the page
receives, so these stay on the server:
  * SessionState (score, status, contrib, history). A paused-for-review session looks finished.
  * Choice.mechanism and every variant_id, which names the mechanism.
  * item_id (names the source dataset), the answer key and the variant's hidden data.
Each question carries an opaque one-time `token` instead; the server maps it back to the
variant and builds the contract Response itself.

Debugging: with ADR_DEBUG_STATE=1, GET /session/{id} returns the SessionState, for the demo
(#47) and logging. Off by default; never enable it in a session real people take.

What #11 does NOT do yet:
  * Active checks. The router runs with an empty failure map, so it serves every item plain, in
    display order. Mechanism injection is step #19; a real failure map arrives with #28.
  * Telemetry beyond time on item (#23). Responses stay in memory only: nothing is written to
    disk until the human-subjects decision (#24) allows real data.

Run from the repo root:   uvicorn server.app:app --reload      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import math
import os
import secrets
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from adr.contracts import (
    Assessment,
    Choice,
    FailureMap,
    Item,
    RenderedItem,
    Response,
    SessionState,
    Telemetry,
)
from adr.harness.copy_paste import plain_variant
from adr.router.core import SPRTRouter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANK = ROOT / "server" / "stub_items.json"
FRONTEND_DIR = ROOT / "frontend"


# ---------------------------------------------------------------- request / response bodies
class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str


class RespondRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    # Raw JSON as sent, so check_answer sees it before Response's answer field would turn
    # `true` into 1 and accept it as an option index.
    answer: Any
    telemetry: Telemetry


class ShownItem(BaseModel):
    """The part of a variant the browser may see."""

    token: str  # opaque, one per question shown; the server maps it back to the variant
    item_type: str
    stem: str
    options: list[str] | None = None
    position: int  # 1-based, for "Question 2 of 5"
    total: int


class Step(BaseModel):
    session_id: str
    finished: bool  # also true when paused for review: the browser can't tell the difference
    item: ShownItem | None


class AssessmentInfo(BaseModel):
    assessment_id: str
    title: str
    assessment_type: str
    n_items: int


# ---------------------------------------------------------------- item bank
def load_bank(path: str | Path) -> tuple[dict[str, Item], dict[str, Assessment]]:
    """Read items and assessments from JSON. Every item is validated against the contract."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = {it.item_id: it for it in (Item(**d) for d in raw["items"])}
    assessments = {a.assessment_id: a for a in (Assessment(**d) for d in raw["assessments"])}
    for a in assessments.values():
        missing = [i for i in a.item_ids if i not in items]
        if missing:
            raise ValueError(f"assessment {a.assessment_id} lists unknown items: {missing}")
        wrong = [i for i in a.item_ids if items[i].assessment_type != a.assessment_type]
        if wrong:
            raise ValueError(f"assessment {a.assessment_id} is a {a.assessment_type}: {wrong}")
    return items, assessments


def check_answer(item: Item, answer: Any) -> None:
    """Raise ValueError unless `answer` fits the item type (option index, ranking, number, text)."""
    n = len(item.options or [])
    if item.item_type in ("mcq", "likert"):
        ok = type(answer) is int and 0 <= answer < n
    elif item.item_type == "ranking":
        ok = (
            isinstance(answer, list)
            and all(type(a) is int for a in answer)
            and sorted(answer) == list(range(n))
        )
    elif item.item_type == "numeric":
        ok = type(answer) in (int, float) and math.isfinite(answer)
    else:  # open_text
        ok = isinstance(answer, str) and answer.strip() != ""
    if not ok:
        # Says what kind of answer is expected, never which item it was.
        raise ValueError(f"{answer!r} is not a valid {item.item_type} answer")


@dataclass
class _Session:
    state: SessionState
    assessment: Assessment
    current: Choice | None
    token: str | None = None  # the opaque token for `current`, as the browser knows it
    responses: list[tuple[Response, Telemetry]] = field(default_factory=list)


# ---------------------------------------------------------------- app
def create_app(bank_path: str | Path | None = None) -> FastAPI:
    items, assessments = load_bank(bank_path or os.environ.get("ADR_ITEMS", DEFAULT_BANK))
    # One plain variant per item until mechanism injection (#19) brings real variants.
    variants: dict[str, list[RenderedItem]] = {i: [plain_variant(it)] for i, it in items.items()}
    by_variant = {v.variant_id: v for vs in variants.values() for v in vs}
    router = SPRTRouter(items, variants, FailureMap(version="empty", cells=[]))
    sessions: dict[str, _Session] = {}

    app = FastAPI(title="Survey server", docs_url=None, redoc_url=None, openapi_url=None)

    def step(s: _Session) -> Step:
        if s.current is None:
            s.token = None
            return Step(session_id=s.state.session_id, finished=True, item=None)
        rendered = by_variant[s.current.variant_id]
        s.token = secrets.token_urlsafe(16)
        item = ShownItem(
            token=s.token,
            item_type=items[rendered.item_id].item_type,
            stem=rendered.visible["stem"],
            options=rendered.visible.get("options"),
            position=len(s.state.served) + 1,
            total=len(s.assessment.item_ids),
        )
        return Step(session_id=s.state.session_id, finished=False, item=item)

    def get_session(session_id: str) -> _Session:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    @app.get("/assessments")
    def list_assessments() -> list[AssessmentInfo]:
        return [
            AssessmentInfo(
                assessment_id=a.assessment_id,
                title=a.title,
                assessment_type=a.assessment_type,
                n_items=len(a.item_ids),
            )
            for a in assessments.values()
        ]

    @app.post("/session/start")
    def start(req: StartRequest) -> Step:
        assessment = assessments.get(req.assessment_id)
        if assessment is None:
            raise HTTPException(404, "unknown assessment")
        session_id = uuid.uuid4().hex
        state, choice = router.start(assessment, session_id, seed=secrets.randbelow(2**31))
        s = sessions[session_id] = _Session(state=state, assessment=assessment, current=choice)
        return step(s)

    @app.post("/session/{session_id}/respond")
    def respond(session_id: str, req: RespondRequest) -> Step:
        s = get_session(session_id)
        if s.current is None:
            raise HTTPException(409, "this session is finished")
        if s.token is None or not secrets.compare_digest(req.token, s.token):
            raise HTTPException(409, "that question is no longer open")
        choice = s.current
        try:
            check_answer(items[choice.item_id], req.answer)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        response = Response(
            session_id=session_id,
            item_id=choice.item_id,
            variant_id=choice.variant_id,
            answer=req.answer,
        )

        s.responses.append((response, req.telemetry))
        # Plain variants have outcome "none". Scoring with the active and passive mechanisms
        # (behaviour, style) comes with #19, #25 and #28.
        s.state = router.update(s.state, choice.variant_id, "none")
        s.current = router.next(s.state)
        return step(s)

    if os.environ.get("ADR_DEBUG_STATE") == "1":

        @app.get("/session/{session_id}")
        def session_state(session_id: str) -> SessionState:
            return get_session(session_id).state

    # Last, so the API routes above take precedence over static files.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()

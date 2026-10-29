"""Web server for the respondent-facing front end (step #11 skeleton).

Owner: Harshini. Reviewer: Arav.

Wraps the router API (adr/contracts/router_api.py) over HTTP, following the endpoint table in
AGENTS.md section 6.5:

    POST /session/start          {assessment_id}                    -> {session_id, status, next, item}
    POST /session/{id}/respond   {variant_id, response, telemetry}  -> {session_id, status, next, item}
    GET  /session/{id}           SessionState, for logging and the demo (the browser never calls it)

Two additions to that table, both needed to show anything in a browser:
  * `item`: what to display for `next`. A `Choice` only carries ids, so the server looks up the
    variant and sends its visible part. Never the answer key, eligible mechanisms or hidden data.
  * GET /assessments: the list the start page offers.

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
    parse_variant_id,
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

    variant_id: str
    # Kept raw so check_answer sees the JSON as sent: Response's answer field would turn `true`
    # into 1 and accept it as an option index. Validated as a Response right after that check.
    response: dict[str, Any]
    telemetry: Telemetry


class ShownItem(BaseModel):
    """The part of a variant the browser may see."""

    item_id: str
    variant_id: str
    item_type: str
    stem: str
    options: list[str] | None = None
    position: int  # 1-based, for "Question 2 of 5"
    total: int


class Step(BaseModel):
    session_id: str
    status: str
    next: Choice | None  # None = finished (or paused for review; the browser can't tell which)
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
        raise ValueError(f"{item.item_id}: {answer!r} is not a valid {item.item_type} answer")


@dataclass
class _Session:
    state: SessionState
    assessment: Assessment
    current: Choice | None
    responses: list[tuple[Response, Telemetry]] = field(default_factory=list)


# ---------------------------------------------------------------- app
def create_app(bank_path: str | Path | None = None) -> FastAPI:
    items, assessments = load_bank(bank_path or os.environ.get("ADR_ITEMS", DEFAULT_BANK))
    # One plain variant per item until mechanism injection (#19) brings real variants.
    variants: dict[str, list[RenderedItem]] = {i: [plain_variant(it)] for i, it in items.items()}
    by_variant = {v.variant_id: v for vs in variants.values() for v in vs}
    router = SPRTRouter(items, variants, FailureMap(version="empty", cells=[]))
    sessions: dict[str, _Session] = {}

    app = FastAPI(title="ADR respondent server")

    def step(s: _Session) -> Step:
        item = None
        if s.current is not None:
            rendered = by_variant[s.current.variant_id]
            item = ShownItem(
                item_id=rendered.item_id,
                variant_id=rendered.variant_id,
                item_type=items[rendered.item_id].item_type,
                stem=rendered.visible["stem"],
                options=rendered.visible.get("options"),
                position=len(s.state.served) + 1,
                total=len(s.assessment.item_ids),
            )
        return Step(session_id=s.state.session_id, status=s.state.status, next=s.current, item=item)

    def get_session(session_id: str) -> _Session:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(404, f"unknown session {session_id!r}")
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
            raise HTTPException(404, f"unknown assessment {req.assessment_id!r}")
        session_id = uuid.uuid4().hex
        state, choice = router.start(assessment, session_id, seed=secrets.randbelow(2**31))
        s = sessions[session_id] = _Session(state=state, assessment=assessment, current=choice)
        return step(s)

    @app.post("/session/{session_id}/respond")
    def respond(session_id: str, req: RespondRequest) -> Step:
        s = get_session(session_id)
        if s.current is None:
            raise HTTPException(409, "this session is finished")
        if req.variant_id != s.current.variant_id:
            raise HTTPException(409, f"expected an answer to {s.current.variant_id!r}")
        item_id = parse_variant_id(req.variant_id)[0]
        try:
            check_answer(items[item_id], req.response.get("answer"))
            r = Response(**req.response)
        except ValueError as e:  # pydantic's ValidationError is a ValueError too
            raise HTTPException(422, str(e)) from e
        if (r.session_id, r.variant_id, r.item_id) != (session_id, req.variant_id, item_id):
            raise HTTPException(422, "response ids don't match the session and variant")

        s.responses.append((r, req.telemetry))
        # Plain variants have outcome "none". Scoring with the active and passive mechanisms
        # (behaviour, style) comes with #19, #25 and #28.
        s.state = router.update(s.state, req.variant_id, "none")
        s.current = router.next(s.state)
        return step(s)

    @app.get("/session/{session_id}")
    def session_state(session_id: str) -> SessionState:
        return get_session(session_id).state

    # Last, so the API routes above take precedence over static files.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()

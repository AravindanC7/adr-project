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

Mechanism injection (#19): decoys and canaries carry `hidden_text`, which the page must show
in a way people can't see but copy-paste, document export and browser agents still get
(Archana's adr/mechanisms/CANARY_SPEC.md). The browser receives that text and where it goes
(`aux`), nothing else from `hidden`: no decoy index, no canary token on its own, no answer key.
The server scores each answer with the variant's mechanism and gives the router the outcome.

Settings (environment variables):
  ADR_PICK=all | router   all (default, for the pilot): every question that can carry a check
                          shows it. router: the SPRT router picks, which needs a failure map (#28).
  ADR_HIDE=clip | clear | clip_aria
                          How hidden text is hidden (see frontend/style.css). The screen-reader
                          test (#44) compares them. Default clip.
  ADR_STUB_MECHANISMS=1   Use the test stubs (tests/stubs.py) instead of adr/mechanisms/, for
                          trying the page before Archana's mechanisms are on main.
  ADR_DEBUG_STATE=1       GET /session/{id} returns the SessionState, for the demo (#47) and
                          logging. Never enable it in a session real people take.

Not here yet: telemetry beyond time on item (#23); a real failure map for ADR_PICK=router (#28).
Responses and outcomes stay in memory only; #39 writes them to the outcome table.

Run from the repo root:   uvicorn server.app:app --reload      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import math
import os
import random
import secrets
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from adr.contracts import (
    Assessment,
    Choice,
    FailureMap,
    Item,
    Mechanism,
    RenderedItem,
    Response,
    SessionState,
    Telemetry,
    parse_variant_id,
)
from adr.harness.copy_paste import plain_variant
from adr.harness.registry import get_mechanism
from adr.router.core import SPRTRouter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANK = ROOT / "server" / "stub_items.json"
FRONTEND_DIR = ROOT / "frontend"

# Mechanisms that change what the page shows. Opinion traps, behaviour and style only look at
# answers (passive), so the page has nothing to inject for them; glitch tokens are out for now.
INJECTED = ("decoy", "canary")
HIDDEN_POSITIONS = ("before_stem", "after_stem", "after_options")
Pick = Literal["all", "router"]
Hide = Literal["clip", "clear", "clip_aria"]


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


class Aux(BaseModel):
    """Text that goes in the page but out of sight. Neutral names: agents read this JSON too."""

    text: str
    at: str  # before_stem | after_stem | after_options
    style: str  # clip | clear | clip_aria


class ShownItem(BaseModel):
    """The part of a variant the browser may see."""

    token: str  # opaque, one per question shown; the server maps it back to the variant
    item_type: str
    stem: str
    options: list[str] | None = None
    aux: Aux | None = None
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


def load_injected_mechanisms(stubs: bool = False) -> dict[str, Mechanism]:
    """The decoy and canary objects that exist. A missing one is skipped, not an error: items
    that could carry it are served plain until its module lands in adr/mechanisms/."""
    if stubs:
        from tests.stubs import StubCanary, StubDecoy  # dev only; needs the repo root on sys.path

        return {"decoy": StubDecoy(), "canary": StubCanary()}
    found: dict[str, Mechanism] = {}
    for name in INJECTED:
        try:
            found[name] = get_mechanism(name)
        except ModuleNotFoundError:
            continue
    return found


def build_variants(
    items: dict[str, Item], mechanisms: dict[str, Mechanism], seed: int = 0
) -> dict[str, list[RenderedItem]]:
    """Plain variant first, then one per mechanism the item is eligible for and that applies."""
    variants: dict[str, list[RenderedItem]] = {}
    for item_id, item in items.items():
        vs = [plain_variant(item)]
        for name, mech in mechanisms.items():
            if name in item.eligible_mechanisms and mech.applies_to(item):
                vs.append(check_hidden(mech.generate(item, seed)))
        variants[item_id] = vs
    return variants


def check_hidden(rendered: RenderedItem) -> RenderedItem:
    """The page can only place hidden text at a known position. Until contracts v2 (#5) makes
    hidden_position required, a variant without one is treated as "after_stem", which is where
    every current mechanism puts it in copy_text."""
    if "hidden_text" not in rendered.hidden:
        return rendered
    position = rendered.hidden.setdefault("hidden_position", "after_stem")
    if position not in HIDDEN_POSITIONS:
        raise ValueError(f"{rendered.variant_id}: unknown hidden_position {position!r}")
    return rendered


@dataclass
class _Session:
    state: SessionState
    assessment: Assessment
    current: Choice | None
    token: str | None = None  # the opaque token for `current`, as the browser knows it
    responses: list[tuple[Response, Telemetry]] = field(default_factory=list)
    outcomes: list[tuple[str, str]] = field(default_factory=list)  # (variant_id, outcome)


# ---------------------------------------------------------------- app
def create_app(
    bank_path: str | Path | None = None,
    mechanisms: dict[str, Mechanism] | None = None,
    pick: Pick | None = None,
    hide: Hide | None = None,
) -> FastAPI:
    items, assessments = load_bank(bank_path or os.environ.get("ADR_ITEMS", DEFAULT_BANK))
    if mechanisms is None:
        mechanisms = load_injected_mechanisms(os.environ.get("ADR_STUB_MECHANISMS") == "1")
    pick = pick or os.environ.get("ADR_PICK", "all")  # type: ignore[assignment]
    hide = hide or os.environ.get("ADR_HIDE", "clip")  # type: ignore[assignment]
    if pick not in ("all", "router") or hide not in ("clip", "clear", "clip_aria"):
        raise ValueError(f"bad ADR_PICK={pick!r} or ADR_HIDE={hide!r}")

    variants = build_variants(items, mechanisms)
    by_variant = {v.variant_id: v for vs in variants.values() for v in vs}
    router = SPRTRouter(items, variants, FailureMap(version="empty", cells=[]))
    sessions: dict[str, _Session] = {}

    app = FastAPI(title="Survey server", docs_url=None, redoc_url=None, openapi_url=None)

    def pick_all(s: _Session) -> Choice | None:
        """Next item in display order, carrying a check whenever it has one (pilot design)."""
        shown = {parse_variant_id(v)[0] for v in s.state.served}
        remaining = [i for i in s.assessment.item_ids if i not in shown]
        if not remaining:
            return None
        item_id = remaining[0]
        active = [v for v in variants[item_id] if v.mechanism is not None]
        rng = random.Random(f"{s.state.seed}:{s.state.session_id}:{len(s.state.served)}")
        v = rng.choice(active) if active else variants[item_id][0]
        return Choice(item_id=item_id, variant_id=v.variant_id, mechanism=v.mechanism)

    def next_choice(s: _Session) -> Choice | None:
        return pick_all(s) if pick == "all" else router.next(s.state)

    def step(s: _Session) -> Step:
        if s.current is None:
            s.token = None
            return Step(session_id=s.state.session_id, finished=True, item=None)
        rendered = by_variant[s.current.variant_id]
        s.token = secrets.token_urlsafe(16)
        hidden_text = rendered.hidden.get("hidden_text")
        item = ShownItem(
            token=s.token,
            item_type=items[rendered.item_id].item_type,
            stem=rendered.visible["stem"],
            options=rendered.visible.get("options"),
            aux=(
                Aux(text=hidden_text, at=rendered.hidden["hidden_position"], style=hide)
                if hidden_text
                else None
            ),
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
        s.current = next_choice(s)
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
        # Plain variants have outcome "none"; a decoy or canary scores the answer. The passive
        # checks (behaviour, style, opinion trap) come with #25 and #28.
        rendered = by_variant[choice.variant_id]
        outcome = (
            mechanisms[rendered.mechanism].score(rendered, response, req.telemetry)
            if rendered.mechanism
            else "none"
        )
        s.outcomes.append((choice.variant_id, outcome))
        s.state = router.update(s.state, choice.variant_id, outcome)
        s.current = next_choice(s)
        return step(s)

    if os.environ.get("ADR_DEBUG_STATE") == "1":

        @app.get("/session/{session_id}")
        def session_state(session_id: str) -> SessionState:
            return get_session(session_id).state

    # Last, so the API routes above take precedence over static files.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()

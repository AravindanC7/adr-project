"""Web server for the respondent-facing front end (step #11 skeleton).

Owner: Harshini. Reviewer: Arav.

Wraps the router API (adr/contracts/router_api.py) over HTTP:

    GET  /assessments            the list the start page offers
    POST /session/start          {assessment_id}            -> {session_id, finished, item}
    POST /session/{id}/respond   {token, answer, telemetry} -> {session_id, finished, item}
    GET  /study                  {consent}: the consent text in pilot mode, else null
    POST /session/{id}/withdraw  removes the session and everything answered (Stop button)
    GET  /session/{id}/debrief   {paragraphs}: pilot only, and only once the session finished

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
                          shows it, to the last question. router: the SPRT router picks checks
                          from the failure map (ADR_MAP), and a session it pauses for review
                          ends early. Either way the router scores every answer with the map.
  ADR_MAP=<file>          The failure map the router scores with (#28, #43): FailureMap JSON as
                          adr/map/builder.py + scripts/build_failure_map.py write it
                          (results/failure_map/<version>.json). Unset: an empty map, so nothing
                          is scored and ADR_PICK=router serves every question plain. Checked at
                          startup (load_failure_map): a missing or invalid file, an empty map, a
                          provisional one (AI-only, or cells with no human data), a held-out or
                          final-evaluation map, or a synthetic map with ADR_PILOT=1 stops the
                          server. Demo map: scripts/make_demo_map.py (SYNTHETIC numbers).
  ADR_HIDE=clip | clear | clip_aria
                          How hidden text is hidden (see frontend/style.css). The screen-reader
                          test (#44) compares them. Default clip.
  ADR_STUB_MECHANISMS=1   Use the test stubs (tests/stubs.py) instead of adr/mechanisms/, for
                          trying the page before Archana's mechanisms are on main.
  ADR_PILOT=1             Human pilot (#30, #35): consent screen before question 1, a Stop
                          button, and the debrief after the last one. Text in docs/pilot/.
                          Needs ADR_DATA_DIR, so a pilot can't run without saving answers.
  ADR_DATA_DIR=<folder>   Save every session to <folder>/<session_id>.jsonl as it happens
                          (server/store.py). Withdrawing deletes the file. Unset: memory only.
  ADR_DEBUG_STATE=1       GET /session/{id} returns the SessionState, for the demo (#47) and
                          logging, with the map in an X-ADR-Map header. Never enable it in a
                          session real people take.

The map's version and file hash stay on the server: in each session's record (_Session.fmap)
and, when saving, in the saved start record. The page looks the same whatever the status.
Saved sessions are raw records; #39 turns them into outcome-table rows.

Run from the repo root:   uvicorn server.app:app --reload      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
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
from fastapi import Response as FastAPIResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from adr.contracts import (
    Assessment,
    Choice,
    FailureMap,
    Item,
    Mechanism,
    NoObservation,
    RenderedItem,
    Response,
    SessionState,
    Telemetry,
    make_variant_id,
    parse_variant_id,
)
from adr.harness.copy_paste import plain_variant
from adr.harness.registry import get_mechanism
from adr.mechanisms.behaviour import MECHANISM as BEHAVIOUR
from adr.router.core import SPRTRouter
from adr.router.map_guard import load_router_map
from server.store import SessionStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANK = ROOT / "server" / "stub_items.json"
FRONTEND_DIR = ROOT / "frontend"
PILOT_DIR = ROOT / "docs" / "pilot"

# Mechanisms that change what the page shows. Opinion traps, behaviour and style only look at
# answers (passive), so the page has nothing to inject for them; glitch tokens are out for now.
INJECTED = ("decoy", "canary")
Pick = Literal["all", "router"]
EMPTY_MAP = FailureMap(version="empty", cells=[])
# Words in a map's version that mark it as fitted on, or meant for, the final evaluation
# (AGENTS.md section 8): the live server must never score anyone with one.
HELDOUT_MARKERS = ("heldout", "held-out", "held_out", "final")
Hide = Literal["clip", "clear", "clip_aria"]


# ---------------------------------------------------------------- request / response bodies
class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    consent: bool = False  # required in pilot mode: the person pressed "I agree"


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
    # Item.passage, shown above the stem. Variants never carry it (RenderedItem.copy_text
    # leaves it out), so the server adds it from the Item, like the harness's copy_text does.
    passage: str | None = None
    stem: str
    options: list[str] | None = None
    aux: Aux | None = None
    position: int  # 1-based, for "Question 2 of 5"
    total: int


class Step(BaseModel):
    session_id: str
    finished: bool  # also true when paused for review: the browser can't tell the difference
    item: ShownItem | None


class Study(BaseModel):
    consent: list[str] | None  # paragraphs to show before starting; None = no consent step


class Debrief(BaseModel):
    paragraphs: list[str]


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


@dataclass(frozen=True)
class Flow:
    """Assessments taken one after another in one session: one consent, one debrief at the end.

    The pilot needs a quiz part and a survey part. Taken as two separate sessions, the debrief
    after the first would reveal the hidden text before the second. In the bank JSON:
        "flow": {"title": "Quiz and survey", "parts": ["pilot-quiz", "pilot-survey"]}
    """

    title: str
    parts: tuple[str, ...]


def load_flow(path: str | Path, assessments: dict[str, Assessment]) -> Flow | None:
    raw = json.loads(Path(path).read_text(encoding="utf-8")).get("flow")
    if raw is None:
        return None
    flow = Flow(title=raw["title"], parts=tuple(raw["parts"]))
    missing = [p for p in flow.parts if p not in assessments]
    if not flow.parts or missing or len(set(flow.parts)) != len(flow.parts):
        raise ValueError(f"bad flow parts {flow.parts}: unknown {missing}" if missing
                         else f"flow parts must be non-empty and distinct: {flow.parts}")
    return flow


def load_failure_map(path: str | Path, pilot: bool = False) -> tuple[FailureMap, dict[str, Any]]:
    """The ADR_MAP file as a FailureMap, plus {version, sha256, path} to record with sessions.

    Raises ValueError with the reason when the server must not use it: missing file, not a
    FailureMap, no cells, provisional (adr/router/map_guard.py, no override here), held-out or
    final-evaluation (HELDOUT_MARKERS in its version), or synthetic in a pilot."""
    path = Path(path)
    where = f"ADR_MAP={path}"
    if not path.is_file():
        raise ValueError(f"{where}: no such file. Build one with scripts/build_failure_map.py "
                         "(or scripts/make_demo_map.py for the demo), or unset ADR_MAP.")
    raw = path.read_bytes()
    try:
        fmap = load_router_map(path)  # ProvisionalMapError is a ValueError too
    except ValueError as e:  # bad JSON, a FailureMap validation error, or provisional
        raise ValueError(f"{where}: not usable as the server's failure map: {e}") from e
    if not fmap.cells:
        raise ValueError(f"{where}: map {fmap.version!r} has no cells; unset ADR_MAP for the "
                         "empty map")
    if any(m in fmap.version.lower() for m in HELDOUT_MARKERS):
        raise ValueError(f"{where}: map {fmap.version!r} is a held-out / final-evaluation map "
                         "(AGENTS.md section 8); the live server must not route with it")
    synthetic = "synthetic" in fmap.version.lower() or "SYNTHETIC" in fmap.note
    if pilot and synthetic:
        raise ValueError(f"{where}: map {fmap.version!r} is SYNTHETIC (invented numbers); "
                         "never with ADR_PILOT=1")
    info = {"version": fmap.version, "sha256": hashlib.sha256(raw).hexdigest()[:12],
            "path": str(path)}
    return fmap, info


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


def screen_text(path: str | Path) -> list[str]:
    """Paragraphs of the "## On screen" section of a docs/pilot file, so the wording lives in
    one place that reviewers can read. Plain text, one paragraph per blank-line block."""
    text = Path(path).read_text(encoding="utf-8")
    _, found, rest = text.partition("\n## On screen\n")
    if not found:
        raise ValueError(f"{path}: no '## On screen' section")
    section = rest.split("\n## ", 1)[0]
    paragraphs = [" ".join(block.split()) for block in section.split("\n\n") if block.strip()]
    if not paragraphs:
        raise ValueError(f"{path}: the '## On screen' section is empty")
    return paragraphs


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
                vs.append(mech.generate(item, seed))  # contracts check hidden_position
        variants[item_id] = vs
    return variants


@dataclass
class _Session:
    state: SessionState
    assessment: Assessment
    current: Choice | None
    token: str | None = None  # the opaque token for `current`, as the browser knows it
    consented: bool = False  # pilot: pressed "I agree" (the only thing stored about consent)
    responses: list[tuple[Response, Telemetry]] = field(default_factory=list)
    outcomes: list[tuple[str, str]] = field(default_factory=list)  # (variant_id, outcome)
    # A flow: the parts still to come, questions in the parts already done, questions in all.
    later_parts: list[Assessment] = field(default_factory=list)
    done_before: int = 0
    total: int = 0
    fmap: dict[str, Any] = field(default_factory=dict)  # {version, sha256, path} it is scored with


# ---------------------------------------------------------------- app
class _Unset:
    """create_app(data_dir=...) left out, as opposed to data_dir=None ("don't save")."""


_FROM_ENV = _Unset()


def create_app(
    bank_path: str | Path | None = None,
    mechanisms: dict[str, Mechanism] | None = None,
    pick: Pick | None = None,
    hide: Hide | None = None,
    pilot: bool | None = None,
    data_dir: str | Path | None | _Unset = _FROM_ENV,
    map_path: str | Path | None = None,
) -> FastAPI:
    """Arguments left out come from the environment (ADR_*). Saving is the exception:
    ADR_DATA_DIR is read only when both `pilot` and `data_dir` are left out, i.e. when the
    whole server is set up from the environment (uvicorn, scripts/run_pilot.ps1). Code that
    builds its own app (tests, the browser-agent harness) never saves into a folder it didn't
    name, so AI sessions can't end up next to the human pilot's answers."""
    configured_by_env = pilot is None and isinstance(data_dir, _Unset)
    if os.environ.get("ADR_DEBUG_STATE") == "1":
        configured_by_env = False  # a debug or harness server: never the pilot's folder
    if isinstance(data_dir, _Unset):
        data_dir = (os.environ.get("ADR_DATA_DIR") or None) if configured_by_env else None
    bank_path = bank_path or os.environ.get("ADR_ITEMS", DEFAULT_BANK)
    items, assessments = load_bank(bank_path)
    flow = load_flow(bank_path, assessments)
    if mechanisms is None:
        mechanisms = load_injected_mechanisms(os.environ.get("ADR_STUB_MECHANISMS") == "1")
    pick = pick or os.environ.get("ADR_PICK", "all")  # type: ignore[assignment]
    hide = hide or os.environ.get("ADR_HIDE", "clip")  # type: ignore[assignment]
    if pick not in ("all", "router") or hide not in ("clip", "clear", "clip_aria"):
        raise ValueError(f"bad ADR_PICK={pick!r} or ADR_HIDE={hide!r}")
    if pilot is None:
        pilot = os.environ.get("ADR_PILOT") == "1"
    if pilot and os.environ.get("ADR_DEBUG_STATE") == "1":
        raise ValueError("ADR_PILOT=1 with ADR_DEBUG_STATE=1 would expose real sessions' state")
    if pilot and data_dir is None:
        raise ValueError("ADR_PILOT=1 needs ADR_DATA_DIR, or the pilot's answers are lost")
    store = SessionStore(data_dir) if data_dir else None
    debug_state = os.environ.get("ADR_DEBUG_STATE") == "1"
    map_path = map_path or os.environ.get("ADR_MAP") or None
    if map_path:
        fmap, map_info = load_failure_map(map_path, pilot=pilot)
    else:
        fmap, map_info = EMPTY_MAP, {"version": EMPTY_MAP.version, "sha256": None, "path": None}
    # Read at startup, so a broken file stops the server before anyone sees the page.
    consent = screen_text(PILOT_DIR / "consent.md") if pilot else None
    debrief = screen_text(PILOT_DIR / "debrief.md") if pilot else None
    # Which consent wording each person agreed to, if it changes during the pilot.
    consent_version = (
        hashlib.sha256("\n\n".join(consent).encode()).hexdigest()[:12] if consent else None
    )

    variants = build_variants(items, mechanisms)
    by_variant = {v.variant_id: v for vs in variants.values() for v in vs}
    router = SPRTRouter(items, variants, fmap)
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
            passage=items[rendered.item_id].passage,
            stem=rendered.visible["stem"],
            options=rendered.visible.get("options"),
            aux=(
                Aux(text=hidden_text, at=rendered.hidden["hidden_position"], style=hide)
                if hidden_text
                else None
            ),
            position=s.done_before + len(s.state.served) + 1,
            total=s.total or len(s.assessment.item_ids),
        )
        return Step(session_id=s.state.session_id, finished=False, item=item)

    def get_session(session_id: str) -> _Session:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    @app.get("/assessments")
    def list_assessments() -> list[AssessmentInfo]:
        if flow:  # one entry that runs every part; starting a later part alone isn't offered
            first = assessments[flow.parts[0]]
            return [AssessmentInfo(
                assessment_id=first.assessment_id,
                title=flow.title,
                assessment_type=first.assessment_type,
                n_items=sum(len(assessments[p].item_ids) for p in flow.parts),
            )]
        return [
            AssessmentInfo(
                assessment_id=a.assessment_id,
                title=a.title,
                assessment_type=a.assessment_type,
                n_items=len(a.item_ids),
            )
            for a in assessments.values()
        ]

    @app.get("/study")
    def study() -> Study:
        return Study(consent=consent)

    @app.post("/session/start")
    def start(req: StartRequest) -> Step:
        if pilot and not req.consent:
            raise HTTPException(403, "please read and agree to the consent screen first")
        assessment = assessments.get(req.assessment_id)
        if assessment is None or (flow and req.assessment_id != flow.parts[0]):
            raise HTTPException(404, "unknown assessment")
        session_id = uuid.uuid4().hex
        state, choice = router.start(assessment, session_id, seed=secrets.randbelow(2**31))
        s = sessions[session_id] = _Session(
            state=state, assessment=assessment, current=choice, consented=req.consent,
            fmap=map_info,
        )
        if flow:
            s.later_parts = [assessments[p] for p in flow.parts[1:]]
            s.total = sum(len(assessments[p].item_ids) for p in flow.parts)
        if store:
            store.start(session_id, assessment_id=assessment.assessment_id, pilot=pilot,
                        consented=req.consent, consent_version=consent_version,
                        seed=state.seed, failure_map=map_info)
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
        n_outcomes = len(s.outcomes)
        # Plain variants have outcome "none"; a decoy or canary scores the answer. The
        # behaviour check (#25) then scores every answer from its telemetry. Style and the
        # opinion trap come with #28.
        rendered = by_variant[choice.variant_id]
        outcome: str | None = "none"
        if rendered.mechanism:
            try:
                outcome = mechanisms[rendered.mechanism].score(rendered, response, req.telemetry)
            except NoObservation:
                # Nothing to score (e.g. a too-short answer): no evidence either way. Contract
                # (router_api.py): update(..., None) marks the variant shown so the session
                # moves on, adds no evidence, and no outcome is recorded for this check.
                outcome = None
        if outcome is not None:
            s.outcomes.append((choice.variant_id, outcome))
        s.state = router.update(s.state, choice.variant_id, outcome)
        try:
            seen = BEHAVIOUR.score(
                rendered, response, req.telemetry, passage=items[choice.item_id].passage
            )
        except NoObservation:
            seen = None
        if seen is not None:
            passive_id = make_variant_id(choice.item_id, "behaviour")
            s.outcomes.append((passive_id, seen))
            s.state = router.update(s.state, passive_id, seen)
        s.current = next_choice(s)
        # Someone may have pressed Stop while this answer was being scored: then the session
        # is gone and nothing is written (the store also never re-creates a deleted file).
        saving = store is not None and session_id in sessions
        if saving:
            store.answer(
                session_id,
                assessment_id=s.assessment.assessment_id,
                item_id=choice.item_id,
                variant_id=choice.variant_id,
                mechanism=choice.mechanism,
                item_type=items[choice.item_id].item_type,
                answer=req.answer,
                telemetry=req.telemetry.model_dump(),
                outcomes=s.outcomes[n_outcomes:],
            )
        # A flow: straight on to the next part, unless the router paused the session for review
        # (ADR_PICK=router). One person takes every part, so the evidence carries over.
        while s.current is None and s.later_parts and not (
            pick == "router" and s.state.status == "review"
        ):
            s.done_before += len(s.assessment.item_ids)
            s.assessment = s.later_parts.pop(0)
            before = s.state
            s.state, _ = router.start(s.assessment, session_id, seed=before.seed)
            s.state = s.state.model_copy(update={
                "score": before.score, "status": before.status, "contrib": dict(before.contrib),
                "history": list(before.history),
            })
            s.current = next_choice(s)
            if saving:
                store.part(session_id, assessment_id=s.assessment.assessment_id)
        if saving and s.current is None:
            store.finish(session_id)
        return step(s)

    @app.post("/session/{session_id}/withdraw")
    def withdraw(session_id: str) -> dict[str, bool]:
        """Stop button or "Remove my answers": forget the session and all its answers, on
        disk too. Works after a server restart as long as the file is still there."""
        on_disk = False
        if store:
            try:
                on_disk = store.delete(session_id)
            except ValueError:
                raise HTTPException(404, "unknown session") from None
        if sessions.pop(session_id, None) is None and not on_disk:
            raise HTTPException(404, "unknown session")
        return {"withdrawn": True}

    @app.get("/session/{session_id}/debrief")
    def debrief_text(session_id: str) -> Debrief:
        """Explains the hidden checks, so only after the last question (agents read the page)."""
        s = get_session(session_id)
        if debrief is None:
            raise HTTPException(404, "no debrief outside the pilot")
        if s.current is not None:
            raise HTTPException(409, "the session isn't finished yet")
        return Debrief(paragraphs=debrief)

    if debug_state:

        @app.get("/session/{session_id}")
        def session_state(session_id: str, response: FastAPIResponse) -> SessionState:
            s = get_session(session_id)
            response.headers["X-ADR-Map"] = f"{s.fmap['version']} sha256:{s.fmap['sha256']}"
            return s.state

    # Last, so the API routes above take precedence over static files.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()

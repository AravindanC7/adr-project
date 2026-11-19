"""The live server's failure map (ADR_MAP, #43/#47): loading, refusing, and the router scoring
real sessions with it. The map is the SYNTHETIC demo map: invented numbers, tests only."""

import json
import re

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from adr.contracts import FailureMap, MapCell
from adr.router.core import RouterConfig
from adr.router.sprt import wald_thresholds
from scripts.make_demo_map import VERSION, demo_map
from server.app import create_app, load_failure_map
from server.store import read_session

# The server's router uses the default config: review at A, cleared at B.
CFG = RouterConfig()
A, B = wald_thresholds(CFG.alpha, CFG.beta)


@pytest.fixture
def map_file(tmp_path):
    return demo_map().save(tmp_path / "demo_synthetic.json")


@pytest.fixture
def debug(monkeypatch):
    monkeypatch.setenv("ADR_DEBUG_STATE", "1")
    monkeypatch.delenv("ADR_MAP", raising=False)


def _cell(**kw):
    return MapCell.from_counts("decoy", "mcq", {"picked_decoy": 10}, {"picked_correct": 10}, **kw)


# ---------------------------------------------------------------- loading
def test_demo_map_is_labelled_synthetic_and_covers_the_stub_checks(map_file):
    fmap, info = load_failure_map(map_file)
    assert fmap.version == VERSION == info["version"] and "SYNTHETIC" in fmap.note
    assert len(info["sha256"]) == 12
    have = {(c.mechanism, c.item_type) for c in fmap.cells}
    assert {("decoy", "mcq"), ("canary", "open_text"), ("behaviour", "numeric")} <= have


def test_unset_map_keeps_the_empty_map(debug):
    c = TestClient(create_app(mechanisms=None, pick="router"))
    step = c.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    assert step["item"]["aux"] is None  # no map cells: the router serves plain
    r = c.get(f"/session/{step['session_id']}")
    assert r.headers["X-ADR-Map"] == "empty sha256:None"


def test_adr_map_from_the_environment_is_used(debug, monkeypatch, map_file):
    monkeypatch.setenv("ADR_MAP", str(map_file))
    c = TestClient(create_app(mechanisms=None, pick="router"))
    step = c.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    assert step["item"]["aux"] is not None  # the router now picks a check
    assert c.get(f"/session/{step['session_id']}").headers["X-ADR-Map"].startswith(VERSION)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        create_app(mechanisms={}, map_path=tmp_path / "nope.json")


@pytest.mark.parametrize("content, match", [
    ("not json", "not usable"),
    (json.dumps({"version": "v1"}), "not usable"),  # no cells field
    (json.dumps({"version": "v1", "cells": []}), "no cells"),
])
def test_invalid_file_is_a_clear_error(tmp_path, content, match):
    path = tmp_path / "m.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        create_app(mechanisms={}, map_path=path)


@pytest.mark.parametrize("version", ["v2-heldout", "v2-held-out", "final-eval"])
def test_heldout_or_final_map_is_refused(tmp_path, version):
    path = FailureMap(version=version, cells=[_cell()]).save(tmp_path / "m.json")
    with pytest.raises(ValueError, match="held-out / final"):
        create_app(mechanisms={}, map_path=path)


def test_provisional_map_is_refused(tmp_path):
    ai_only = MapCell.from_counts("decoy", "mcq", {"picked_decoy": 10}, {})
    path = FailureMap(version="v2-ai-only", cells=[ai_only]).save(tmp_path / "m.json")
    with pytest.raises(ValueError, match="refusing failure map"):
        create_app(mechanisms={}, map_path=path)


def test_synthetic_map_is_refused_in_the_pilot(tmp_path, map_file, monkeypatch):
    monkeypatch.delenv("ADR_DEBUG_STATE", raising=False)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        create_app(mechanisms={}, map_path=map_file, pilot=True, data_dir=tmp_path)


def test_saved_session_records_the_map(debug, tmp_path, map_file):
    data = tmp_path / "data"
    c = TestClient(create_app(mechanisms=None, map_path=map_file, data_dir=data))
    sid = c.post("/session/start", json={"assessment_id": "stub-quiz"}).json()["session_id"]
    start = read_session(data / f"{sid}.jsonl")[0]
    _, info = load_failure_map(map_file)
    assert start["failure_map"] == info and info["version"] == VERSION


# ---------------------------------------------------------------- whole sessions
FAST = {"time_on_item_ms": 300}       # faster than anyone reads: behaviour "too_fast"
CAREFUL = {"time_on_item_ms": 30000}  # behaviour "normal"
# A valid answer per item type for questions without a hidden key.
VALID = {"likert": 3, "ranking": [3, 1, 0, 2], "numeric": 7, "open_text": "I like it."}


def _take(c, assessment_id, ai):
    """One respondent through the page, seeing only what the browser gets. The AI-like one
    follows the hidden answer key whenever there is one and answers instantly; the human-like
    one answers correctly (option 1 is the key for every stub MCQ) and takes its time."""
    step = c.post("/session/start", json={"assessment_id": assessment_id}).json()
    sid, shown = step["session_id"], 0
    while not step["finished"]:
        item = step["item"]
        shown += 1
        if item["item_type"] == "mcq":
            hidden = re.search(r"select option (\d+)", (item["aux"] or {}).get("text", ""))
            answer = int(hidden.group(1)) - 1 if ai and hidden else 1
        else:
            answer = VALID[item["item_type"]]
        step = c.post(f"/session/{sid}/respond", json={
            "token": item["token"], "answer": answer, "telemetry": FAST if ai else CAREFUL,
        }).json()
    return sid, shown, c.get(f"/session/{sid}").json()


@pytest.mark.parametrize("pick", ["router", "all"])
def test_ai_like_session_is_paused_and_human_like_is_cleared(debug, map_file, pick):
    c = TestClient(create_app(mechanisms=None, pick=pick, map_path=map_file))
    _, ai_shown, ai = _take(c, "stub-quiz", ai=True)
    outcomes = [o for _, o, _ in ai["history"]]
    assert "picked_decoy" in outcomes and "picked_correct" not in outcomes
    assert ai["score"] >= A and ai["status"] == "review"
    # No mechanism may push the score past its cap on its own.
    cap = CFG.mech_cap_frac * A
    assert all(v <= cap + 1e-9 for v in ai["contrib"].values())

    _, human_shown, human = _take(c, "stub-quiz", ai=False)
    assert "picked_decoy" not in [o for _, o, _ in human["history"]]
    assert human["score"] <= B and human["status"] == "cleared"

    quiz_len = next(a["n_items"] for a in c.get("/assessments").json()
                    if a["assessment_id"] == "stub-quiz")
    assert human_shown == quiz_len  # a cleared person finishes the quiz (plain from then on)
    if pick == "all":
        assert ai_shown == quiz_len  # pilot design: every question, whatever the status
    else:
        assert ai_shown <= quiz_len  # router: paused for review = finished early or at the end


def test_the_page_looks_the_same_whatever_the_status(debug, map_file):
    c = TestClient(create_app(mechanisms=None, pick="router", map_path=map_file))
    step = c.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    sid = step["session_id"]
    bodies = [step]
    while not bodies[-1]["finished"]:
        item = bodies[-1]["item"]
        hidden = re.search(r"select option (\d+)", (item["aux"] or {}).get("text", ""))
        answer = int(hidden.group(1)) - 1 if hidden else (1 if item["item_type"] == "mcq" else 7)
        bodies.append(c.post(f"/session/{sid}/respond", json={
            "token": item["token"], "answer": answer, "telemetry": FAST}).json())
    assert c.get(f"/session/{sid}").json()["status"] == "review"
    assert bodies[-1] == {"session_id": sid, "finished": True, "item": None}
    text = json.dumps(bodies).lower()
    for word in ("review", "status", "score", "flag", "synthetic", VERSION, "paused"):
        assert word not in text



def test_a_flow_stops_when_paused_and_carries_the_evidence(debug, tmp_path, map_file):
    from server.app import DEFAULT_BANK

    bank = json.loads(DEFAULT_BANK.read_text(encoding="utf-8"))
    bank["flow"] = {"title": "Quiz and survey", "parts": ["stub-quiz", "stub-survey"]}
    path = tmp_path / "flow.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    c = TestClient(create_app(bank_path=path, mechanisms=None, pick="router", map_path=map_file))
    _, _, ai = _take(c, "stub-quiz", ai=True)
    assert ai["status"] == "review" and ai["assessment_id"] == "stub-quiz"  # survey never shown
    _, human_shown, human = _take(c, "stub-quiz", ai=False)
    quiz, survey = (len(bank["assessments"][i]["item_ids"]) for i in (0, 1))
    assert human["assessment_id"] == "stub-survey" and human_shown == quiz + survey
    assert human["status"] == "cleared"  # cleared in the quiz, still cleared in the survey
    assert any("|decoy|" in v for v, _, _ in human["history"])  # quiz evidence carried over

"""A flow (#35): quiz then survey in one session, one consent, one debrief after the last part."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from server.app import DEFAULT_BANK, create_app
from server.store import read_session
from tests.stubs import StubCanary, StubDecoy
from tests.test_server import VALID, respond

MECHANISMS = {"decoy": StubDecoy(), "canary": StubCanary()}


@pytest.fixture
def bank(tmp_path):
    raw = json.loads(DEFAULT_BANK.read_text(encoding="utf-8"))
    raw["flow"] = {"title": "Quiz and survey", "parts": ["stub-quiz", "stub-survey"]}
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


@pytest.fixture
def pilot(bank, tmp_path):
    path, _ = bank
    data = tmp_path / "data"
    app = create_app(bank_path=path, mechanisms=MECHANISMS, pilot=True, data_dir=data)
    return TestClient(app), data


def sizes(raw):
    return {a["assessment_id"]: len(a["item_ids"]) for a in raw["assessments"]}


def test_start_page_offers_one_entry_for_the_whole_flow(pilot, bank):
    client, _ = pilot
    n = sizes(bank[1])
    assert client.get("/assessments").json() == [{
        "assessment_id": "stub-quiz", "title": "Quiz and survey", "assessment_type": "quiz",
        "n_items": n["stub-quiz"] + n["stub-survey"],
    }]
    # A later part can't be started on its own (its debrief would come too early).
    r = client.post("/session/start", json={"assessment_id": "stub-survey", "consent": True})
    assert r.status_code == 404


def test_quiz_then_survey_then_one_debrief(pilot, bank):
    client, data = pilot
    n = sizes(bank[1])
    total = n["stub-quiz"] + n["stub-survey"]
    step = client.post("/session/start", json={"assessment_id": "stub-quiz", "consent": True}).json()
    sid = step["session_id"]
    seen = []
    while not step["finished"]:
        item = step["item"]
        seen.append((item["position"], item["total"]))
        if len(seen) <= n["stub-quiz"]:
            # Still in the quiz part: no debrief yet, not even between the parts.
            assert client.get(f"/session/{sid}/debrief").status_code == 409
        step = respond(client, step, VALID[item["item_type"]]).json()
    assert seen == [(i, total) for i in range(1, total + 1)]  # one counter across both parts
    assert client.get(f"/session/{sid}/debrief").status_code == 200

    records = read_session(data / f"{sid}.jsonl")
    events = [r["event"] for r in records]
    assert events.count("start") == 1 and events.count("part") == 1
    assert events.count("finish") == 1 and events[-1] == "finish"
    answered = [r["assessment_id"] for r in records if r["event"] == "answer"]
    assert answered == ["stub-quiz"] * n["stub-quiz"] + ["stub-survey"] * n["stub-survey"]
    part = events.index("part")
    assert records[part]["assessment_id"] == "stub-survey"
    assert all(r["assessment_id"] == "stub-quiz" for r in records[1:part])


def test_withdraw_in_the_second_part_removes_both(pilot, bank):
    client, data = pilot
    n = sizes(bank[1])
    step = client.post("/session/start", json={"assessment_id": "stub-quiz", "consent": True}).json()
    sid = step["session_id"]
    for _ in range(n["stub-quiz"] + 1):
        step = respond(client, step, VALID[step["item"]["item_type"]]).json()
    assert client.post(f"/session/{sid}/withdraw").json() == {"withdrawn": True}
    assert not (data / f"{sid}.jsonl").exists()


@pytest.mark.parametrize("parts", [[], ["stub-quiz", "stub-quiz"], ["stub-quiz", "nope"]])
def test_bad_flows_stop_the_server(tmp_path, parts):
    raw = json.loads(DEFAULT_BANK.read_text(encoding="utf-8"))
    raw["flow"] = {"title": "x", "parts": parts}
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="flow"):
        create_app(bank_path=path, mechanisms={})


def test_without_a_flow_nothing_changes():
    client = TestClient(create_app(mechanisms={}))
    assert len(client.get("/assessments").json()) == 2

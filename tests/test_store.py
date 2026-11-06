"""Saving sessions to disk (#35): every answer is on disk as it happens; withdrawing deletes it."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from server.app import create_app
from server.store import SessionStore, read_session
from tests.stubs import StubCanary, StubDecoy
from tests.test_pilot import finish, start
from tests.test_server import VALID, respond

MECHANISMS = {"decoy": StubDecoy(), "canary": StubCanary()}


@pytest.fixture
def saved(tmp_path):
    return TestClient(create_app(mechanisms=MECHANISMS, pilot=True, data_dir=tmp_path)), tmp_path


def test_pilot_cannot_run_without_saving(monkeypatch):
    monkeypatch.delenv("ADR_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="ADR_DATA_DIR"):
        create_app(mechanisms={}, pilot=True)


def test_each_answer_is_on_disk_as_it_happens(saved):
    client, folder = saved
    step = start(client).json()
    sid = step["session_id"]
    records = read_session(folder / f"{sid}.jsonl")
    assert [r["event"] for r in records] == ["start"]
    assert records[0]["consented"] is True and records[0]["assessment_id"] == "stub-quiz"

    step = respond(client, step, VALID[step["item"]["item_type"]]).json()
    records = read_session(folder / f"{sid}.jsonl")
    assert [r["event"] for r in records] == ["start", "answer"]
    answer = records[1]
    assert answer["mechanism"] == "decoy" and answer["item_type"] == "mcq"
    assert answer["telemetry"]["time_on_item_ms"] == 1200
    assert [o for _, o in answer["outcomes"]][1] == "too_fast"  # the behaviour check, passive
    assert answer["outcomes"][1][0].endswith("|behaviour|0")

    finish(client, step)
    records = read_session(folder / f"{sid}.jsonl")
    assert records[-1]["event"] == "finish"
    assert sum(r["event"] == "answer" for r in records) == 3


def test_nothing_identifying_is_saved(saved):
    client, folder = saved
    sid = finish(client, start(client).json())["session_id"]
    keys = set()
    for r in read_session(folder / f"{sid}.jsonl"):
        keys |= set(r) | set(r.get("telemetry", {}))
    for word in ("name", "email", "ip", "user_agent", "host", "asu"):
        assert not [k for k in keys if word in k.lower()], word


@pytest.mark.parametrize("when", ["midway", "finished"])
def test_withdraw_deletes_the_file(saved, when):
    client, folder = saved
    step = start(client).json()
    sid = step["session_id"]
    if when == "finished":
        finish(client, step)
    assert (folder / f"{sid}.jsonl").exists()
    assert client.post(f"/session/{sid}/withdraw").json() == {"withdrawn": True}
    assert not (folder / f"{sid}.jsonl").exists()
    assert client.post(f"/session/{sid}/withdraw").status_code == 404


def test_withdraw_still_works_after_a_restart(tmp_path):
    first = TestClient(create_app(mechanisms=MECHANISMS, pilot=True, data_dir=tmp_path))
    sid = finish(first, start(first).json())["session_id"]
    restarted = TestClient(create_app(mechanisms=MECHANISMS, pilot=True, data_dir=tmp_path))
    assert restarted.post(f"/session/{sid}/withdraw").json() == {"withdrawn": True}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", ["../secret", "..", "abc", "A" * 32, "0" * 31 + "/"])
def test_session_ids_cannot_name_other_files(tmp_path, bad):
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.path(bad)
    if "/" not in bad and bad != "..":  # those never reach the route at all
        client = TestClient(create_app(mechanisms={}, pilot=True, data_dir=tmp_path))
        assert client.post(f"/session/{bad}/withdraw").status_code == 404


def test_outside_the_pilot_saving_is_optional(tmp_path, monkeypatch):
    monkeypatch.delenv("ADR_DATA_DIR", raising=False)
    client = TestClient(create_app(mechanisms={}))
    assert finish(client, client.post("/session/start", json={"assessment_id": "stub-quiz"})
                  .json())["finished"]
    client = TestClient(create_app(mechanisms={}, data_dir=tmp_path))
    sid = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()["session_id"]
    first = json.loads((tmp_path / f"{sid}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["pilot"] is False and first["consented"] is False

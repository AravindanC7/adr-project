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

    # 500 ms is under any reading floor (at least ANSWER_MS = 750 ms): too_fast
    step = respond(client, step, VALID[step["item"]["item_type"]], ms=500).json()
    records = read_session(folder / f"{sid}.jsonl")
    assert [r["event"] for r in records] == ["start", "answer"]
    answer = records[1]
    assert answer["mechanism"] == "decoy" and answer["item_type"] == "mcq"
    assert answer["telemetry"]["time_on_item_ms"] == 500
    assert [o for _, o in answer["outcomes"]][1] == "too_fast"  # the behaviour check, passive
    assert answer["outcomes"][1][0].endswith("|behaviour|0")

    finish(client, step)
    records = read_session(folder / f"{sid}.jsonl")
    assert records[-1]["event"] == "finish"
    assert sum(r["event"] == "answer" for r in records) == 4


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


def test_no_clock_times_are_saved(saved):
    client, folder = saved
    sid = finish(client, start(client).json())["session_id"]
    records = read_session(folder / f"{sid}.jsonl")
    assert len(records[0]["started_on"]) == 10  # a date, YYYY-MM-DD, no time of day
    assert all("at" not in r for r in records)
    assert all(isinstance(r["t"], int) and r["t"] >= 0 for r in records[1:])
    assert all(":" not in json.dumps(r.get("started_on", "")) for r in records)


def test_start_records_which_consent_text_was_agreed(saved):
    client, folder = saved
    sid = start(client).json()["session_id"]
    version = read_session(folder / f"{sid}.jsonl")[0]["consent_version"]
    assert len(version) == 12 and int(version, 16) >= 0


def test_an_answer_after_withdrawing_never_recreates_the_file(tmp_path):
    store = SessionStore(tmp_path)
    sid = "a" * 32
    store.start(sid, assessment_id="x")
    assert store.delete(sid)
    assert store.answer(sid, item_id="x:1") is False  # in flight when Stop was pressed
    assert store.finish(sid) is False
    assert list(tmp_path.iterdir()) == []


def test_remove_by_code_from_the_command_line(tmp_path, capsys):
    from server.store import main

    store = SessionStore(tmp_path)
    sid = "b" * 32
    store.start(sid, assessment_id="x")
    assert main(["remove", sid.upper(), "--dir", str(tmp_path)]) == 0  # forgiving about case
    assert "Removed" in capsys.readouterr().out
    assert main(["remove", sid, "--dir", str(tmp_path)]) == 1
    assert main(["remove", "not-a-code", "--dir", str(tmp_path)]) == 2


def test_an_app_built_in_code_never_saves_into_the_environments_folder(tmp_path, monkeypatch):
    # e.g. the browser-agent harness run in a shell where the pilot's ADR_DATA_DIR is set
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "pilot_data"))
    for app in (create_app(mechanisms={}, pilot=False), create_app(mechanisms={}, data_dir=None)):
        client = TestClient(app)
        client.post("/session/start", json={"assessment_id": "stub-quiz"})
    assert not (tmp_path / "pilot_data").exists()
    # Set up entirely from the environment (uvicorn), it does save there.
    monkeypatch.delenv("ADR_PILOT", raising=False)
    client = TestClient(create_app(mechanisms={}))
    client.post("/session/start", json={"assessment_id": "stub-quiz"})
    assert len(list((tmp_path / "pilot_data").iterdir())) == 1


def test_a_debug_server_never_saves_into_the_environments_folder(tmp_path, monkeypatch):
    # The browser-agent harness sets ADR_DEBUG_STATE=1 and may pass no data_dir at all.
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "pilot_data"))
    monkeypatch.setenv("ADR_DEBUG_STATE", "1")
    monkeypatch.delenv("ADR_PILOT", raising=False)
    client = TestClient(create_app(mechanisms={}))
    client.post("/session/start", json={"assessment_id": "stub-quiz"})
    assert not (tmp_path / "pilot_data").exists()


def test_status_counts_without_reading_out_answers(tmp_path, capsys):
    import os
    import time

    from server.store import main, status

    store = SessionStore(tmp_path)
    for code, n_answers, done in (("a" * 32, 3, True), ("b" * 32, 1, False), ("c" * 32, 2, False)):
        store.start(code, assessment_id="x")
        for i in range(n_answers):
            store.answer(code, item_id=f"x:{i}", answer="secret answer text")
        if done:
            store.finish(code)
    old = time.time() - 2 * 3600  # "c" was last touched two hours ago
    os.utime(tmp_path / f"{'c' * 32}.jsonl", (old, old))

    s = status(tmp_path)
    assert (s["finished"], s["in_progress"], s["abandoned"]) == (1, 1, 1)
    assert len(s["minutes"]) == 1 and sum(s["by_day"].values()) == 3
    assert main(["status", "--dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "finished     1" in printed and "abandoned    1" in printed
    assert "secret" not in printed and "a" * 32 not in printed


def test_status_of_a_folder_that_doesnt_exist_yet(tmp_path):
    from server.store import status

    s = status(tmp_path / "pilot_data")
    assert (s["finished"], s["in_progress"], s["abandoned"]) == (0, 0, 0)

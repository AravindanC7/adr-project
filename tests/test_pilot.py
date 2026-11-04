"""Pilot mode (#30): consent before starting, Stop / remove my answers, debrief only at the end."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from server.app import PILOT_DIR, create_app, screen_text
from tests.stubs import StubCanary, StubDecoy
from tests.test_server import VALID, respond, walk

CONSENT = screen_text(PILOT_DIR / "consent.md")
DEBRIEF = screen_text(PILOT_DIR / "debrief.md")
# The consent screen uses partial disclosure: none of these may appear before the debrief.
GIVES_IT_AWAY = ["hidden", "decoy", "canary", "trap", "detect", "instruction", "really about"]


@pytest.fixture
def pilot():
    return TestClient(
        create_app(mechanisms={"decoy": StubDecoy(), "canary": StubCanary()}, pilot=True)
    )


def start(client, consent=True):
    return client.post("/session/start", json={"assessment_id": "stub-quiz", "consent": consent})


def finish(client, step):
    while not step["finished"]:
        step = respond(client, step, VALID[step["item"]["item_type"]]).json()
    return step


def test_screen_text_reads_the_on_screen_section():
    assert CONSENT[0].startswith("Thanks for helping") and len(CONSENT) == 6
    assert DEBRIEF[0] == "Thank you! Here is what the study is really about."
    assert all("\n" not in p and "  " not in p for p in CONSENT + DEBRIEF)


def test_consent_and_page_give_nothing_away():
    text = " ".join(CONSENT).lower()
    assert [w for w in GIVES_IT_AWAY if w in text] == []
    # The page itself must not contain the debrief: an agent reads the whole page.
    c = TestClient(create_app(mechanisms={}, pilot=True))
    page = "".join(c.get(p).text for p in ("/", "/app.js", "/style.css", "/telemetry.js"))
    assert all(p not in page for p in DEBRIEF)
    assert "really about" not in page


def test_outside_the_pilot_there_is_no_consent_step(client=None):
    c = TestClient(create_app(mechanisms={}))
    assert c.get("/study").json() == {"consent": None}
    assert len(walk(c, "stub-quiz")) == 4  # starts without consent, as before


def test_pilot_needs_consent_to_start(pilot):
    assert pilot.get("/study").json() == {"consent": CONSENT}
    r = start(pilot, consent=False)
    assert r.status_code == 403 and "consent" in r.json()["detail"]
    assert start(pilot).status_code == 200


def test_debrief_only_after_the_last_question(pilot):
    step = start(pilot).json()
    sid = step["session_id"]
    assert pilot.get(f"/session/{sid}/debrief").status_code == 409
    finish(pilot, step)
    assert pilot.get(f"/session/{sid}/debrief").json() == {"paragraphs": DEBRIEF}


def test_no_debrief_outside_the_pilot():
    c = TestClient(create_app(mechanisms={}))
    sid = walk(c, "stub-quiz")[0]["session_id"]
    assert c.get(f"/session/{sid}/debrief").status_code == 404


@pytest.mark.parametrize("when", ["midway", "after_debrief"])
def test_withdraw_removes_the_session(pilot, when):
    step = start(pilot).json()
    sid = step["session_id"]
    if when == "after_debrief":
        finish(pilot, step)
    assert pilot.post(f"/session/{sid}/withdraw").json() == {"withdrawn": True}
    assert respond(pilot, step, 1).status_code == 404
    assert pilot.get(f"/session/{sid}/debrief").status_code == 404
    assert pilot.post(f"/session/{sid}/withdraw").status_code == 404


def test_pilot_refuses_debug_state(monkeypatch):
    monkeypatch.setenv("ADR_DEBUG_STATE", "1")
    with pytest.raises(ValueError, match="ADR_PILOT"):
        create_app(mechanisms={}, pilot=True)

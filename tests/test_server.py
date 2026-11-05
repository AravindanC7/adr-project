import json

import pytest

pytest.importorskip("fastapi")  # pip install -e ".[server]"
from fastapi.testclient import TestClient

from adr.contracts import Item
from server.app import DEFAULT_BANK, check_answer, check_hidden, create_app, load_bank
from tests.stubs import StubCanary, StubDecoy

# One valid answer per item type, for the stub bank.
VALID = {"mcq": 1, "likert": 3, "ranking": [3, 1, 0, 2], "numeric": 7, "open_text": "I like it."}
ITEMS, ASSESSMENTS = load_bank(DEFAULT_BANK)

# Words that must never reach the browser (AGENTS.md 6.5): ids, check names, router state.
SECRETS = ["handwritten:", "|plain|", "variant", "mechanism", "decoy", "canary", "opinion_trap",
           "status", "score", "contrib", "history", "correct_option", "eligible", "escalated",
           "review"]


@pytest.fixture
def client():
    return TestClient(create_app(mechanisms={}))


@pytest.fixture
def traps(monkeypatch):
    """Stub decoy and canary on, plus the debug state endpoint to see the scored outcomes."""
    monkeypatch.setenv("ADR_DEBUG_STATE", "1")
    return TestClient(create_app(mechanisms={"decoy": StubDecoy(), "canary": StubCanary()}))


def respond(client, step, answer, ms=1200, token=None):
    return client.post(f"/session/{step['session_id']}/respond", json={
        "token": token or step["item"]["token"],
        "answer": answer,
        "telemetry": {"time_on_item_ms": ms},
    })


def walk(client, assessment_id):
    """Answer every question validly; return every body the browser received."""
    bodies = [client.post("/session/start", json={"assessment_id": assessment_id}).json()]
    while not bodies[-1]["finished"]:
        item = bodies[-1]["item"]
        bodies.append(respond(client, bodies[-1], VALID[item["item_type"]]).json())
    return bodies


def test_stub_bank_covers_every_item_type():
    assert {it.item_type for it in ITEMS.values()} == set(VALID)
    assert set(ASSESSMENTS) == {"stub-quiz", "stub-survey"}


def test_lists_assessments(client):
    got = {a["assessment_id"]: a["n_items"] for a in client.get("/assessments").json()}
    assert got == {"stub-quiz": 3, "stub-survey": 5}


def test_full_survey_walk_in_display_order(client):
    bodies = walk(client, "stub-survey")
    shown = [b["item"] for b in bodies if b["item"]]
    expected = [ITEMS[i].stem for i in ASSESSMENTS["stub-survey"].item_ids]
    assert [it["stem"] for it in shown] == expected
    assert [it["position"] for it in shown] == [1, 2, 3, 4, 5]
    assert bodies[-1] == {"session_id": bodies[0]["session_id"], "finished": True, "item": None}


@pytest.mark.parametrize("assessment_id", ["stub-quiz", "stub-survey"])
@pytest.mark.parametrize("app", ["client", "traps"])
def test_browser_never_sees_ids_checks_or_router_state(request, app, assessment_id):
    client = request.getfixturevalue(app)
    bodies = walk(client, assessment_id)
    for b in bodies:
        assert set(b) == {"session_id", "finished", "item"}
        if b["item"]:
            assert set(b["item"]) == {
                "token", "item_type", "stem", "options", "aux", "position", "total",
            }
            if b["item"]["aux"]:
                assert set(b["item"]["aux"]) == {"text", "at", "style"}
                b["item"]["aux"]["text"] = ""  # the trap text itself has to reach the page
    sent = json.dumps(bodies) + json.dumps(client.get("/assessments").json())
    leaked = [word for word in SECRETS if word in sent]
    assert leaked == []


def test_without_mechanisms_every_question_is_plain(client):
    assert all(b["item"] is None or b["item"]["aux"] is None for b in walk(client, "stub-quiz"))


def test_every_eligible_question_carries_its_check(traps):
    quiz = [b["item"] for b in walk(traps, "stub-quiz") if b["item"]]
    survey = [b["item"] for b in walk(traps, "stub-survey") if b["item"]]
    # q1, q2 are decoy-eligible; s1 is canary-eligible; everything else is plain.
    assert [bool(it["aux"]) for it in quiz] == [True, True, False]
    assert [bool(it["aux"]) for it in survey] == [True, False, False, False, False]
    assert quiz[0]["aux"]["at"] == "after_stem" and quiz[0]["aux"]["style"] == "clip"
    assert "marigold" in survey[0]["aux"]["text"]
    # Only the hidden text itself, never the decoy index or the answer key behind it.
    assert "decoy_option_index" not in json.dumps(quiz) and "canary_token" not in json.dumps(survey)


# Mechanism names, as the registry and variant ids spell them. An agent reading the page must
# not learn which check a question carries, and the leak test above blanks aux.text first.
MECHANISM_NAMES = ["decoy", "canary", "opinion_trap", "opinion trap", "glitch", "behaviour"]


@pytest.mark.parametrize("assessment_id", ["stub-quiz", "stub-survey"])
def test_hidden_text_never_names_its_own_check(traps, assessment_id):
    for b in walk(traps, assessment_id):
        if b["item"] and b["item"]["aux"]:
            text = b["item"]["aux"]["text"].lower()
            assert [name for name in MECHANISM_NAMES if name in text] == []


def test_answers_are_scored_by_the_check_they_carried(traps):
    step = traps.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    decoy = int(step["item"]["aux"]["text"].split("option ")[1].rstrip(".]")) - 1
    step = respond(traps, step, decoy).json()       # q1: AI-style, follows the hidden key
    step = respond(traps, step, 1).json()           # q2: picks the right answer
    step = respond(traps, step, 7).json()           # q3: plain
    history = traps.get(f"/session/{step['session_id']}").json()["history"]
    assert [(v.split("|")[1], outcome) for v, outcome, _ in history] == [
        ("decoy", "picked_decoy"), ("decoy", "picked_correct"), ("plain", "none"),
    ]

    step = traps.post("/session/start", json={"assessment_id": "stub-survey"}).json()
    step = respond(traps, step, "I like the people. marigold0").json()
    history = traps.get(f"/session/{step['session_id']}").json()["history"]
    assert history[0][1] == "followed_instruction"


@pytest.mark.parametrize("hide", ["clip", "clear", "clip_aria"])
def test_hiding_style_is_a_server_setting(hide):
    c = TestClient(create_app(mechanisms={"decoy": StubDecoy()}, hide=hide))
    step = c.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    assert step["item"]["aux"]["style"] == hide


def test_router_mode_serves_plain_until_there_is_a_failure_map():
    c = TestClient(create_app(mechanisms={"decoy": StubDecoy()}, pick="router"))
    assert all(b["item"] is None or b["item"]["aux"] is None for b in walk(c, "stub-quiz"))


def test_bad_settings_fail_at_startup():
    with pytest.raises(ValueError):
        create_app(mechanisms={}, hide="display_none")


def test_hidden_position_defaults_until_contracts_v2_and_must_be_known():
    rendered = StubDecoy().generate(ITEMS["handwritten:q1"], 0)
    rendered.hidden.pop("hidden_position", None)
    assert check_hidden(rendered).hidden["hidden_position"] == "after_stem"
    rendered.hidden["hidden_position"] = "in_the_footer"
    with pytest.raises(ValueError, match="hidden_position"):
        check_hidden(rendered)


def test_tokens_are_one_time_and_unguessable(client):
    step = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    assert respond(client, step, 1, token="handwritten:q1|plain|0").status_code == 409
    nxt = respond(client, step, 1).json()
    assert nxt["item"]["token"] != step["item"]["token"]
    assert respond(client, step, 1).status_code == 409  # old token, already answered


def test_state_endpoint_only_in_debug_mode(client, monkeypatch):
    sid = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()["session_id"]
    assert "score" not in client.get(f"/session/{sid}").text  # falls through to static files

    monkeypatch.setenv("ADR_DEBUG_STATE", "1")
    debug = TestClient(create_app(mechanisms={}))
    bodies = walk(debug, "stub-quiz")
    state = debug.get(f"/session/{bodies[0]['session_id']}").json()
    assert [v.split("|")[1] for v in state["served"]] == ["plain"] * 3


@pytest.mark.parametrize("answer", [9, -1, True, "1", [0, 1], None])
def test_rejects_bad_mcq_answers(client, answer):
    step = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    r = respond(client, step, answer)
    assert r.status_code == 422 and "handwritten" not in r.text


def test_finished_session_and_unknown_ids(client):
    bodies = walk(client, "stub-quiz")
    assert respond(client, bodies[-2], 7).status_code == 409
    assert client.post("/session/nope/respond", json={
        "token": "x", "answer": 1, "telemetry": {"time_on_item_ms": 1}}).status_code == 404
    assert client.post("/session/start", json={"assessment_id": "nope"}).status_code == 404


def test_serves_the_front_end_with_no_third_party_requests(client):
    page = client.get("/")
    assert page.status_code == 200 and 'id="stem"' in page.text
    for path in ("/", "/style.css", "/app.js"):
        assert "https://" not in client.get(path).text.replace("https://github.com", "")
    assert client.get("/fonts/Nunito-wght.ttf").status_code == 200


def _item(item_type, options=None):
    return Item(item_id=f"handwritten:{item_type}", source="handwritten",
                assessment_type="survey", item_type=item_type, stem="Q?", options=options)


@pytest.mark.parametrize("item_type,options,good,bad", [
    ("likert", ["a", "b", "c"], [0, 2], [3, -1, 1.0, True, "a"]),
    ("ranking", ["a", "b", "c"], [[0, 1, 2], [2, 0, 1]], [[0, 1], [0, 0, 1], [0, 1, 3], 1]),
    ("numeric", None, [0, 3.5, -2], [float("nan"), float("inf"), "3", None]),
    ("open_text", None, ["hi", "  hi  "], ["", "   ", 3, None]),
])
def test_check_answer(item_type, options, good, bad):
    item = _item(item_type, options)
    for a in good:
        check_answer(item, a)
    for a in bad:
        with pytest.raises(ValueError):
            check_answer(item, a)

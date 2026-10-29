import pytest

pytest.importorskip("fastapi")  # pip install -e ".[server]"
from fastapi.testclient import TestClient

from adr.contracts import Item
from server.app import DEFAULT_BANK, check_answer, create_app, load_bank

# One valid answer per item type, for the stub bank.
VALID = {"mcq": 1, "likert": 3, "ranking": [3, 1, 0, 2], "numeric": 7, "open_text": "I like it."}


@pytest.fixture
def client():
    return TestClient(create_app())


def respond(client, step, answer, ms=1200):
    sid, item = step["session_id"], step["item"]
    return client.post(f"/session/{sid}/respond", json={
        "variant_id": item["variant_id"],
        "response": {"session_id": sid, "item_id": item["item_id"],
                     "variant_id": item["variant_id"], "answer": answer},
        "telemetry": {"time_on_item_ms": ms},
    })


def test_stub_bank_covers_every_item_type():
    items, assessments = load_bank(DEFAULT_BANK)
    assert {it.item_type for it in items.values()} == set(VALID)
    assert set(assessments) == {"stub-quiz", "stub-survey"}


def test_lists_assessments(client):
    got = {a["assessment_id"]: a["n_items"] for a in client.get("/assessments").json()}
    assert got == {"stub-quiz": 3, "stub-survey": 5}


def test_full_survey_walk_in_display_order(client):
    step = client.post("/session/start", json={"assessment_id": "stub-survey"}).json()
    seen = []
    while step["item"] is not None:
        seen.append(step["item"]["item_id"])
        assert step["item"]["position"] == len(seen)
        step = respond(client, step, VALID[step["item"]["item_type"]]).json()
        assert step.get("session_id"), step
    assert step["next"] is None
    assert seen == [f"handwritten:s{i}" for i in range(1, 6)]
    state = client.get(f"/session/{step['session_id']}").json()
    assert [v.split("|")[1] for v in state["served"]] == ["plain"] * 5


def test_browser_never_sees_answer_key_or_hidden_data(client):
    step = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    assert set(step) == {"session_id", "status", "next", "item"}
    assert set(step["item"]) == {
        "item_id", "variant_id", "item_type", "stem", "options", "position", "total",
    }
    assert "score" not in step


@pytest.mark.parametrize("answer", [9, -1, True, "1", [0, 1], None])
def test_rejects_bad_mcq_answers(client, answer):
    step = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    assert respond(client, step, answer).status_code == 422


def test_rejects_answer_to_the_wrong_item(client):
    step = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    step["item"]["variant_id"] = "handwritten:q2|plain|0"
    step["item"]["item_id"] = "handwritten:q2"
    assert respond(client, step, 1).status_code == 409


def test_finished_session_and_unknown_ids(client):
    step = client.post("/session/start", json={"assessment_id": "stub-quiz"}).json()
    for answer in (1, 1, 7):
        last, step = step, respond(client, step, answer).json()
    assert step["item"] is None
    assert respond(client, last, 7).status_code == 409
    assert client.get("/session/nope").status_code == 404
    assert client.post("/session/start", json={"assessment_id": "nope"}).status_code == 404


def test_serves_the_front_end(client):
    page = client.get("/")
    assert page.status_code == 200 and 'id="stem"' in page.text


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

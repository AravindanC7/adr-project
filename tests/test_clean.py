"""Pilot cleaning (#39): saved sessions -> outcome-table rows."""

import json

import pytest

from adr.contracts.outcome import read_outcomes
from server.clean import clean, main

AGREE = ["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"]
BANK = {
    "items": [
        {"item_id": "arc:q1", "source": "arc", "assessment_type": "quiz", "item_type": "mcq",
         "stem": "Q?", "options": ["a", "b", "c", "d"], "correct_option": 1,
         "eligible_mechanisms": ["decoy"]},
        {"item_id": "opinionqa:w41:X", "source": "opinionqa", "assessment_type": "survey",
         "item_type": "mcq", "stem": "Future?", "options": ["Up", "Same", "Gone"],
         "eligible_mechanisms": ["opinion_trap"], "human_dist_key": "opinionqa:w41:X:human"},
        {"item_id": "ipip:e1", "source": "ipip", "assessment_type": "survey",
         "item_type": "likert", "stem": "I am the life of the party.", "options": AGREE,
         "reverse_keyed_with": "ipip:e2", "eligible_mechanisms": ["opinion_trap"],
         "human_dist_key": "ipip:e1:human"},
        {"item_id": "ipip:e2", "source": "ipip", "assessment_type": "survey",
         "item_type": "likert", "stem": "I don't talk a lot.", "options": AGREE,
         "reverse_keyed_with": "ipip:e1", "eligible_mechanisms": ["opinion_trap"],
         "human_dist_key": "ipip:e2:human"},
        {"item_id": "handwritten:o1", "source": "handwritten", "assessment_type": "survey",
         "item_type": "open_text", "stem": "Tell us.", "eligible_mechanisms": ["canary"]},
    ],
    "assessments": [
        {"assessment_id": "q", "title": "Q", "assessment_type": "quiz", "item_ids": ["arc:q1"]},
        {"assessment_id": "s", "title": "S", "assessment_type": "survey",
         "item_ids": ["opinionqa:w41:X", "ipip:e1", "ipip:e2", "handwritten:o1"]},
    ],
    "human_distributions": [
        {"key": "opinionqa:w41:X:human", "source": "opinionqa",
         "option_probs": [0.6, 0.38, 0.02], "n_respondents": 2524},
        {"key": "ipip:e1:human", "source": "ipip", "option_probs": [0.2] * 5,
         "n_respondents": 1000},
        {"key": "ipip:e2:human", "source": "ipip", "option_probs": [0.2] * 5,
         "n_respondents": 1000},
    ],
}
TELEMETRY = {"time_on_item_ms": 9000, "paste_events": 0}


def session(folder, sid, answers, *, finished=True, consented=True):
    """answers: {item_id: (answer, [[variant_id, outcome], ...])}"""
    lines = [{"event": "start", "session_id": sid, "assessment_id": "q", "pilot": True,
              "consented": consented, "consent_version": "v1", "seed": 1,
              "started_on": "2026-11-10"}]
    for t, (item_id, (answer, outcomes)) in enumerate(answers.items(), start=1):
        lines.append({"event": "answer", "item_id": item_id, "variant_id": outcomes[0][0],
                      "answer": answer, "telemetry": TELEMETRY, "outcomes": outcomes, "t": t})
    if finished:
        lines.append({"event": "finish", "t": 99})
    path = folder / f"{sid}.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def plain(item_id):
    return [[f"{item_id}|plain|0", "none"], [f"{item_id}|behaviour|0", "normal"]]


@pytest.fixture
def setup(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(BANK), encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    return data, bank


def full(e1=0, e2=4, x=0, text="I learned to cook from my grandmother."):
    return {
        "arc:q1": (2, [["arc:q1|decoy|0", "picked_decoy"], ["arc:q1|behaviour|0", "too_fast"]]),
        "opinionqa:w41:X": (x, plain("opinionqa:w41:X")),
        "ipip:e1": (e1, plain("ipip:e1")),
        "ipip:e2": (e2, plain("ipip:e2")),
        "handwritten:o1": (text, [["handwritten:o1|canary|0", "clean"],
                                  ["handwritten:o1|behaviour|0", "normal"]]),
    }


def by_check(rows):
    return {(r.item_id, r.mechanism): r.outcome for r in rows}


def test_only_finished_consented_sessions_are_used(setup):
    data, bank = setup
    session(data, "a" * 32, full())
    session(data, "b" * 32, full(), finished=False)
    session(data, "c" * 32, full(), consented=False)
    session(data, "d" * 32, full())
    rows, rep = clean(data, bank, origin="pilot", run_id="r", exclude=frozenset({"d" * 32}))
    assert rep.used == ["a" * 32 + ".jsonl"]
    assert rep.dropped == {"b" * 32 + ".jsonl": "not finished",
                           "c" * 32 + ".jsonl": "no consent",
                           "d" * 32 + ".jsonl": "excluded on request"}
    assert {r.session_id for r in rows} == {"a" * 32}


def test_rows_carry_served_behaviour_and_opinion_trap_outcomes(setup):
    data, bank = setup
    session(data, "a" * 32, full(x=2))
    rows, _ = clean(data, bank, origin="pilot", run_id="r")
    got = by_check(rows)
    assert got[("arc:q1", "decoy")] == "picked_decoy"
    assert got[("arc:q1", "behaviour")] == "too_fast"
    assert got[("handwritten:o1", "canary")] == "clean"
    assert got[("opinionqa:w41:X", None)] == "none"
    assert got[("opinionqa:w41:X", "opinion_trap")] == "implausible"  # a 2% answer
    assert all(r.respondent_kind == "human" and r.data_origin == "pilot" for r in rows)
    assert all(r.raw_ref.startswith("a" * 32 + ".jsonl#L") for r in rows)
    assert not any(r.model or r.attack_mode for r in rows)


@pytest.mark.parametrize("e1, e2, expected", [
    (0, 4, "plausible"),       # disagree "life of the party", agree "don't talk a lot"
    (4, 4, "contradiction"),   # agree with both
    (0, 1, "contradiction"),   # disagree with both
    (2, 4, "plausible"),       # neutral on one: not a contradiction
])
def test_reverse_keyed_pairs(setup, e1, e2, expected):
    data, bank = setup
    session(data, "a" * 32, full(e1=e1, e2=e2))
    rows, _ = clean(data, bank, origin="pilot", run_id="r")
    got = by_check(rows)
    assert got[("ipip:e2", "opinion_trap")] == expected
    assert got[("ipip:e1", "opinion_trap")] == "plausible"  # judged on the second of the pair
    no_pairs, _ = clean(data, bank, origin="pilot", run_id="r", pairs=False)
    assert by_check(no_pairs)[("ipip:e2", "opinion_trap")] == "plausible"


@pytest.mark.parametrize("text, kind", [
    ("Email me at sam.lee@asu.edu please", "email"),
    ("My name is Sam and I learned guitar", "name"),
    ("Call 480-555-0199 any time", "phone"),
    ("See www.example.com", "url"),
])
def test_open_text_that_may_identify_someone_is_listed(setup, text, kind):
    data, bank = setup
    session(data, "a" * 32, full(text=text))
    _, rep = clean(data, bank, origin="pilot", run_id="r")
    assert [f["looks_like"] for f in rep.flagged] == [kind]
    session(data, "a" * 32, full())  # the same session with an ordinary answer
    assert clean(data, bank, origin="pilot", run_id="r")[1].flagged == []


def test_command_writes_a_valid_outcome_table_and_a_run_record(setup, tmp_path, capsys):
    data, bank = setup
    session(data, "a" * 32, full())
    session(data, "b" * 32, full(), finished=False)
    out = tmp_path / "results"
    assert main(["--data", str(data), "--bank", str(bank), "--origin", "dry_run",
                 "--run-id", "dry-1", "--out", str(out)]) == 0
    df = read_outcomes(out / "outcomes" / "dry-1.parquet")  # validates every row
    assert set(df["data_origin"]) == {"dry_run"} and len(df) == 5 * 2 + 3  # + 3 opinion trap
    config = (out / "runs" / "dry-1" / "config.yaml").read_text(encoding="utf-8")
    assert "sessions_used: 1" in config and "not finished" in config
    assert "1 of 2 sessions used" in capsys.readouterr().out
    assert "I learned to cook" not in df.to_string()  # no answer text in the table


def test_an_item_missing_from_the_bank_stops_the_run(setup):
    data, bank = setup
    session(data, "a" * 32, {"arc:nope": (1, plain("arc:nope"))})
    with pytest.raises(ValueError, match="isn't in"):
        clean(data, bank, origin="pilot", run_id="r")

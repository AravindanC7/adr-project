"""Pilot cleaning (#39): saved sessions -> outcome-table rows."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict

import pytest
import yaml

from adr.contracts import OUTCOME_LABELS
from adr.contracts.outcome import COLUMNS, read_outcomes
from adr.mechanisms.style import StyleProfile
from server.clean import clean, main, style_record

AGREE = ["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"]
BANK = {
    "items": [
        {
            "item_id": "arc:q1",
            "source": "arc",
            "assessment_type": "quiz",
            "item_type": "mcq",
            "stem": "Q?",
            "options": ["a", "b", "c", "d"],
            "correct_option": 1,
            "eligible_mechanisms": ["decoy"],
        },
        {
            "item_id": "opinionqa:w41:X",
            "source": "opinionqa",
            "assessment_type": "survey",
            "item_type": "mcq",
            "stem": "Future?",
            "options": ["Up", "Same", "Gone"],
            "eligible_mechanisms": ["opinion_trap"],
            "human_dist_key": "opinionqa:w41:X:human",
        },
        {
            "item_id": "ipip:e1",
            "source": "ipip",
            "assessment_type": "survey",
            "item_type": "likert",
            "stem": "I am the life of the party.",
            "options": AGREE,
            "reverse_keyed_with": "ipip:e2",
            "eligible_mechanisms": ["opinion_trap"],
            "human_dist_key": "ipip:e1:human",
        },
        {
            "item_id": "ipip:e2",
            "source": "ipip",
            "assessment_type": "survey",
            "item_type": "likert",
            "stem": "I don't talk a lot.",
            "options": AGREE,
            "reverse_keyed_with": "ipip:e1",
            "eligible_mechanisms": ["opinion_trap"],
            "human_dist_key": "ipip:e2:human",
        },
        {
            "item_id": "handwritten:o1",
            "source": "handwritten",
            "assessment_type": "survey",
            "item_type": "open_text",
            "stem": "Tell us.",
            "eligible_mechanisms": ["canary"],
        },
    ],
    "assessments": [
        {"assessment_id": "q", "title": "Q", "assessment_type": "quiz", "item_ids": ["arc:q1"]},
        {
            "assessment_id": "s",
            "title": "S",
            "assessment_type": "survey",
            "item_ids": ["opinionqa:w41:X", "ipip:e1", "ipip:e2", "handwritten:o1"],
        },
    ],
    "human_distributions": [
        {
            "key": "opinionqa:w41:X:human",
            "source": "opinionqa",
            "option_probs": [0.6, 0.38, 0.02],
            "n_respondents": 2524,
        },
        {
            "key": "ipip:e1:human",
            "source": "ipip",
            "option_probs": [0.2] * 5,
            "n_respondents": 1000,
        },
        {
            "key": "ipip:e2:human",
            "source": "ipip",
            "option_probs": [0.2] * 5,
            "n_respondents": 1000,
        },
    ],
}
TELEMETRY = {"time_on_item_ms": 9000, "paste_events": 0}


def session(folder, sid, answers, *, finished=True, consented=True, telemetry=None):
    """answers: {item_id: (answer, [[variant_id, outcome], ...])}; telemetry: {item_id: dict},
    TELEMETRY for the rest, a value of None leaves the record without telemetry."""
    telemetry = telemetry or {}
    lines = [
        {
            "event": "start",
            "session_id": sid,
            "assessment_id": "q",
            "pilot": True,
            "consented": consented,
            "consent_version": "v1",
            "seed": 1,
            "started_on": "2026-11-10",
        }
    ]
    for t, (item_id, (answer, outcomes)) in enumerate(answers.items(), start=1):
        lines.append(
            {
                "event": "answer",
                "item_id": item_id,
                "variant_id": outcomes[0][0],
                "answer": answer,
                "telemetry": telemetry.get(item_id, TELEMETRY),
                "outcomes": outcomes,
                "t": t,
            }
        )
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
        "handwritten:o1": (
            text,
            [["handwritten:o1|canary|0", "clean"], ["handwritten:o1|behaviour|0", "normal"]],
        ),
    }


def by_check(rows):
    return {(r.item_id, r.mechanism): r.outcome for r in rows}


def profile(mid=1e9, high=1e9):
    """Synthetic style profile. The default cutoffs are out of reach, so every answer is "low";
    profile(0, 0) puts every answer at "high"."""
    return StyleProfile(
        feature_means=(4.0, 0.8, 4.5, 15.0, 0.1),
        feature_stds=(1.0, 0.1, 0.5, 5.0, 0.05),
        mid_threshold=mid,
        high_threshold=high,
        n_training=20,
        sources=("synthetic",),
    )


def test_only_finished_consented_sessions_are_used(setup):
    data, bank = setup
    session(data, "a" * 32, full())
    session(data, "b" * 32, full(), finished=False)
    session(data, "c" * 32, full(), consented=False)
    session(data, "d" * 32, full())
    rows, rep = clean(data, bank, origin="pilot", run_id="r", exclude=frozenset({"d" * 32}))
    assert rep.used == ["a" * 32 + ".jsonl"]
    assert rep.dropped == {
        "b" * 32 + ".jsonl": "not finished",
        "c" * 32 + ".jsonl": "no consent",
        "d" * 32 + ".jsonl": "excluded on request",
    }
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


# Option indices 0-4 are the 1-5 points of the agree scale (index 2 = 3 = Neutral).
@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (4, 0, "plausible"),  # 5/1: agree "life of the party", disagree "don't talk a lot"
        (0, 4, "plausible"),  # 1/5: the other consistent way round
        (4, 4, "contradiction"),  # 5/5: agree with both
        (0, 1, "contradiction"),  # 1/2: disagree with both
        (2, 4, None),  # 3/5: neutral on the first -> no observation, no row
        (4, 2, None),  # 5/3: neutral on the second -> no row
        (2, 2, None),  # 3/3: neutral on both -> no row
        (None, 4, None),  # missing first answer -> no row
        ("4", 0, None),  # non-integer answer -> no row
        (4, 7, None),  # out of range -> no row
        (-1, 4, None),  # out of range -> no row
    ],
)
def test_reverse_keyed_pairs(setup, e1, e2, expected):
    data, bank = setup
    session(data, "a" * 32, full(e1=e1, e2=e2))
    rows, rep = clean(data, bank, origin="pilot", run_id="r")
    got = by_check(rows)
    assert got.get(("ipip:e2", "opinion_trap")) == expected
    assert got[("ipip:e1", "opinion_trap")] == "plausible"  # judged on the second of the pair
    assert sum(1 for r in rows if r.item_id == "ipip:e2" and r.mechanism == "opinion_trap") == (
        0 if expected is None else 1
    )
    assert sum(rep.rows[k] for k in rep.rows if k[0] == "opinion_trap") == (
        2 if expected is None else 3
    )
    no_pairs, _ = clean(data, bank, origin="pilot", run_id="r", pairs=False)
    assert by_check(no_pairs)[("ipip:e2", "opinion_trap")] == "plausible"


def test_pairs_are_scored_by_the_shared_pair_outcome(setup, monkeypatch):
    # The pilot uses the same pair rule as the AI runs (adr/mechanisms/opinion_trap.py).
    import server.clean as clean_module
    from adr.mechanisms.opinion_trap import pair_outcome

    calls = []

    def spy(*args):
        calls.append(args)
        return pair_outcome(*args)

    monkeypatch.setattr(clean_module, "pair_outcome", spy)
    data, bank = setup
    session(data, "a" * 32, full(e1=4, e2=4))
    got = by_check(clean(data, bank, origin="pilot", run_id="r")[0])
    assert calls == [(4, 4, 5, "plausible")]  # only the second item of the pair is judged
    assert got[("ipip:e2", "opinion_trap")] == "contradiction"


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Email me at sam.lee@asu.edu please", "email"),
        ("My name is Sam and I learned guitar", "name"),
        ("Call 480-555-0199 any time", "phone"),
        ("See www.example.com", "url"),
    ],
)
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
    assert (
        main(
            [
                "--data",
                str(data),
                "--bank",
                str(bank),
                "--origin",
                "dry_run",
                "--run-id",
                "dry-1",
                "--out",
                str(out),
                "--no-style",  # style has its own test; keep this one off the local profile
            ]
        )
        == 0
    )
    df = read_outcomes(out / "outcomes" / "dry-1.parquet")  # validates every row
    assert set(df["data_origin"]) == {"dry_run"} and len(df) == 5 * 2 + 3  # + 3 opinion trap
    config = (out / "runs" / "dry-1" / "config.yaml").read_text(encoding="utf-8")
    assert "sessions_used: 1" in config and "not finished" in config
    assert "1 of 2 sessions used" in capsys.readouterr().out
    assert "I learned to cook" not in df.to_string()  # no answer text in the table


def test_style_is_on_by_default_and_needs_a_profile(setup, tmp_path, capsys):
    data, bank = setup
    session(data, "a" * 32, full())
    session(data, "b" * 32, full(text=""))  # empty answer: no style observation
    out = tmp_path / "results"
    args = ["--data", str(data), "--bank", str(bank), "--run-id", "p", "--out", str(out)]
    with pytest.raises(SystemExit):  # default on: a missing profile stops the run
        main([*args, "--style-profile", str(tmp_path / "missing.json")])
    assert "no style profile" in capsys.readouterr().err

    prof = tmp_path / "style_profile.json"
    prof.write_text(json.dumps({"profile": asdict(profile(0, 0))}), encoding="utf-8")
    assert main([*args, "--style-profile", str(prof)]) == 0
    df = read_outcomes(out / "outcomes" / "p.parquet")
    assert list(df.columns) == COLUMNS  # no new column, no text column
    style = df[df["mechanism"] == "style"]
    assert list(style["session_id"]) == ["a" * 32] and set(style["outcome"]) == {"high"}
    assert len(df) == 2 * (5 * 2 + 3) + 1
    config = (out / "runs" / "p" / "config.yaml").read_text(encoding="utf-8")
    assert "no_observation: 1" in config and "style:high: 1" in config
    assert "no observation (empty answer): 1" in capsys.readouterr().out
    assert "I learned to cook" not in df.to_string()

    assert main([*args, "--style-profile", str(tmp_path / "missing.json"), "--no-style"]) == 0
    df = read_outcomes(out / "outcomes" / "p.parquet")
    assert "style" not in set(df["mechanism"].dropna()) and len(df) == 2 * (5 * 2 + 3)


def test_style_rows_only_for_open_text_answers(setup):
    data, bank = setup
    session(data, "a" * 32, full())
    without, _ = clean(data, bank, origin="pilot", run_id="r")
    rows, rep = clean(data, bank, origin="pilot", run_id="r", style_profile=profile())
    style = [r for r in rows if r.mechanism == "style"]
    assert len(style) == 1 and len(rows) == len(without) + 1  # nothing else changes
    (row,) = style
    assert row.item_id == "handwritten:o1" and row.item_type == "open_text"
    assert row.variant_id == "handwritten:o1|style|0" and row.outcome == "low"
    assert row.outcome in OUTCOME_LABELS["style"]
    assert row.respondent_kind == "human" and row.data_origin == "pilot"
    assert row.raw_ref.startswith("a" * 32 + ".jsonl#L")
    assert rep.rows[("style", "low")] == 1 and rep.style_no_observation == 0
    assert by_check(rows)[("handwritten:o1", "canary")] == "clean"  # served check kept
    high, _ = clean(data, bank, origin="pilot", run_id="r", style_profile=profile(0, 0))
    assert by_check(high)[("handwritten:o1", "style")] == "high"


@pytest.mark.parametrize("answer", ["", "   ", None, 3])
def test_unscorable_open_text_is_no_observation(setup, answer):
    data, bank = setup
    session(data, "a" * 32, full(text=answer))
    rows, rep = clean(data, bank, origin="pilot", run_id="r", style_profile=profile())
    assert not any(r.mechanism == "style" for r in rows)
    assert rep.style_no_observation == 1 and not any(m == "style" for m, _ in rep.rows)


def test_style_rows_hold_no_answer_text(setup):
    data, bank = setup
    secret = "Zanzibar quokka marmalade expedition"
    session(data, "a" * 32, full(text=secret))
    rows, _ = clean(data, bank, origin="pilot", run_id="r", style_profile=profile())
    dumped = json.dumps([r.model_dump() for r in rows])
    assert "Zanzibar" not in dumped and "quokka" not in dumped


def test_an_item_missing_from_the_bank_stops_the_run(setup):
    data, bank = setup
    session(data, "a" * 32, {"arc:nope": (1, plain("arc:nope"))})
    with pytest.raises(ValueError, match="isn't in"):
        clean(data, bank, origin="pilot", run_id="r")


def profile_file(path, doc_profile, where):
    """A style_profile.json like prepare_style_profile.py writes: the profile plus metadata that
    records OS paths and the decisions file's sha."""
    doc = {
        "profile": doc_profile,
        "inputs": [{"raw": f"{where}/data/raw/asurre.csv", "raw_sha256": "ab" * 32}],
        "decisions_sha256": hashlib.sha256(where.encode()).hexdigest(),
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def canonical(doc_profile):
    blob = json.dumps(doc_profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def decisions(path, sha):
    path.write_text(json.dumps({"style_profile": {"profile_sha256": sha}}), encoding="utf-8")
    return path


def test_profile_hash_ignores_paths_and_metadata(tmp_path):
    prof = asdict(profile(1.0, 2.0))
    mac = profile_file(tmp_path / "mac.json", prof, "/Users/a/adr-project")
    win = profile_file(tmp_path / "win.json", prof, r"C:\Users\h\adr-project")
    dec = decisions(tmp_path / "dec.json", canonical(prof))
    a, b = style_record(mac, dec), style_record(win, dec)
    assert a["profile_file_sha256_12"] != b["profile_file_sha256_12"]  # the files differ
    assert a["profile_sha256"] == b["profile_sha256"] == canonical(prof)
    assert a["profile_sha256_12"] == canonical(prof)[:12]
    assert a["profile_matches_decision"] is True and b["profile_matches_decision"] is True

    changed = dict(prof, mid_threshold=1.5)
    other = profile_file(tmp_path / "other.json", changed, "/Users/a/adr-project")
    assert style_record(other, dec)["profile_sha256"] != a["profile_sha256"]


def test_config_records_canonical_hash_and_warns_on_a_different_profile(setup, tmp_path, capsys):
    data, bank = setup
    session(data, "a" * 32, full())
    prof = asdict(profile(0, 0))
    path = profile_file(tmp_path / "style_profile.json", prof, "/Users/a/adr-project")
    out = tmp_path / "results"
    args = [
        "--data",
        str(data),
        "--bank",
        str(bank),
        "--out",
        str(out),
        "--style-profile",
        str(path),
    ]

    assert (
        main(
            [
                *args,
                "--run-id",
                "ok",
                "--decisions",
                str(decisions(tmp_path / "d1.json", canonical(prof))),
            ]
        )
        == 0
    )
    assert "WARNING" not in capsys.readouterr().out
    style = yaml.safe_load((out / "runs" / "ok" / "config.yaml").read_text(encoding="utf-8"))[
        "style"
    ]
    assert style["profile_sha256"] == canonical(prof)
    assert style["profile_sha256_12"] == canonical(prof)[:12]
    assert style["profile_matches_decision"] is True
    assert len(style["profile_file_sha256_12"]) == 12

    locked = "6b43d6ed5372" + "0" * 52
    assert (
        main(
            [*args, "--run-id", "exp", "--decisions", str(decisions(tmp_path / "d2.json", locked))]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "WARNING" in printed and "6b43d6ed5372" in printed and canonical(prof)[:12] in printed
    style = yaml.safe_load((out / "runs" / "exp" / "config.yaml").read_text(encoding="utf-8"))[
        "style"
    ]
    assert style["profile_matches_decision"] is False
    assert style["profile_sha256"] == canonical(prof)  # still the profile actually used


# --rescore-behaviour. "Q?" is one word: the old floor (1000 ms + 100 ms a word) is 1100 ms, the
# current one (750 + 50) is 800 ms, so a 900 ms answer is too_fast live and normal re-scored.
SECRET = "Zanzibar quokka marmalade expedition"
BORDERLINE = {
    "arc:q1": {"time_on_item_ms": 900, "paste_events": 0},
    # typed, so the untyped-text rule doesn't call the open-text answer a paste
    "handwritten:o1": {"time_on_item_ms": 9000, "paste_events": 0, "keystroke_count": 60},
}


def live_session(data, sid="a" * 32, telemetry=None):
    """Live outcomes as the server scored them with the old floor."""
    answers = full(text=SECRET)
    session(data, sid, answers, telemetry={**BORDERLINE, **(telemetry or {})})


def old_floor(monkeypatch):
    import adr.mechanisms.behaviour as b

    monkeypatch.setattr(b, "ANSWER_MS", 1000)
    monkeypatch.setattr(b, "READ_MS_PER_WORD", 100)


def behaviour_rows(rows):
    return {r.item_id: r.outcome for r in rows if r.mechanism == "behaviour"}


def test_rescore_is_off_by_default_and_copies_live_outcomes(setup):
    data, bank = setup
    live_session(data)
    rows, rep = clean(data, bank, origin="pilot", run_id="r")
    assert behaviour_rows(rows)["arc:q1"] == "too_fast"  # live, though today's floor says normal
    assert rep.behaviour_transitions == Counter() and rep.behaviour_not_rescorable == Counter()


def test_rescore_with_the_old_floor_reproduces_live_outcomes(setup, monkeypatch):
    pytest.importorskip("fastapi")  # re-scoring rebuilds served variants with server.app
    data, bank = setup
    live_session(data)
    old_floor(monkeypatch)
    key = lambda r: r.model_dump(exclude={"created_at"})
    live, _ = clean(data, bank, origin="pilot", run_id="r", style_profile=profile())
    again, rep = clean(
        data, bank, origin="pilot", run_id="r", style_profile=profile(), rescore=True
    )
    assert [key(r) for r in again] == [key(r) for r in live]
    assert sum(rep.behaviour_transitions.values()) == 5
    assert all(a == b for a, b in rep.behaviour_transitions)
    assert rep.behaviour_not_rescorable == Counter()


def test_rescore_with_the_current_floor_flips_a_borderline_answer(setup):
    pytest.importorskip("fastapi")  # re-scoring rebuilds served variants with server.app
    data, bank = setup
    live_session(data)
    live, _ = clean(data, bank, origin="pilot", run_id="r")
    rows, rep = clean(data, bank, origin="pilot", run_id="r", rescore=True)
    assert behaviour_rows(live)["arc:q1"] == "too_fast"
    assert behaviour_rows(rows)["arc:q1"] == "normal"
    assert rep.behaviour_transitions == Counter(
        {("too_fast", "normal"): 1, ("normal", "normal"): 4}
    )
    others = lambda rs: [
        r.model_dump(exclude={"created_at"}) for r in rs if r.mechanism != "behaviour"
    ]
    assert others(rows) == others(live)  # nothing but behaviour changes


def test_answers_without_telemetry_keep_the_live_outcome(setup):
    pytest.importorskip("fastapi")  # re-scoring rebuilds served variants with server.app
    data, bank = setup
    live_session(data, telemetry={"arc:q1": None, "ipip:e1": {"time_on_item_ms": -5}})
    rows, rep = clean(data, bank, origin="pilot", run_id="r", rescore=True)
    assert behaviour_rows(rows)["arc:q1"] == "too_fast"  # kept
    assert rep.behaviour_not_rescorable == Counter({"no telemetry": 1, "invalid telemetry": 1})
    assert sum(rep.behaviour_transitions.values()) == 3


def test_command_records_how_behaviour_was_scored(setup, tmp_path, capsys, monkeypatch):
    pytest.importorskip("fastapi")  # re-scoring rebuilds served variants with server.app
    import adr.mechanisms.behaviour as b

    data, bank = setup
    live_session(data)
    out = tmp_path / "results"
    args = ["--data", str(data), "--bank", str(bank), "--out", str(out), "--no-style"]

    def config(run_id):
        return yaml.safe_load((out / "runs" / run_id / "config.yaml").read_text(encoding="utf-8"))

    assert main([*args, "--run-id", "live"]) == 0
    assert config("live")["behaviour"]["rescored"] is False
    assert "re-scored" not in capsys.readouterr().out

    assert main([*args, "--run-id", "new", "--rescore-behaviour"]) == 0
    beh = config("new")["behaviour"]
    assert beh["rescored"] is True
    assert (beh["answer_ms"], beh["read_ms_per_word"]) == (b.ANSWER_MS, b.READ_MS_PER_WORD)
    assert beh["transitions"] == {"normal->normal": 4, "too_fast->normal": 1}
    assert beh["changed"] == 1 and beh["not_rescorable"] == 0
    printed = capsys.readouterr().out
    assert "too_fast->normal 1" in printed and "not_rescorable 0" in printed
    df = read_outcomes(out / "outcomes" / "new.parquet")
    assert dict(df[df["mechanism"] == "behaviour"]["outcome"].value_counts()) == {"normal": 5}

    monkeypatch.setattr(b, "ANSWER_MS", 1234)  # read from behaviour.py, not hard-coded
    assert main([*args, "--run-id", "patched", "--rescore-behaviour"]) == 0
    assert config("patched")["behaviour"]["answer_ms"] == 1234

    texts = [printed, capsys.readouterr().out, df.to_string()]
    texts += [(out / "runs" / r / "config.yaml").read_text("utf-8") for r in ("new", "patched")]
    assert not any(word in t for t in texts for word in ("Zanzibar", "quokka", "marmalade"))


def test_strict_stops_when_an_answer_cannot_be_rescored(setup, tmp_path, capsys):
    pytest.importorskip("fastapi")  # re-scoring rebuilds served variants with server.app
    data, bank = setup
    live_session(data, telemetry={"arc:q1": None})
    out = tmp_path / "results"
    args = ["--data", str(data), "--bank", str(bank), "--out", str(out), "--no-style"]
    args += ["--run-id", "s", "--rescore-behaviour"]
    assert main([*args, "--strict"]) == 1
    assert not (out / "outcomes" / "s.parquet").exists()
    assert "nothing written" in capsys.readouterr().out
    assert main(args) == 0  # without --strict: reported, live outcome kept
    beh = yaml.safe_load((out / "runs" / "s" / "config.yaml").read_text(encoding="utf-8"))
    assert beh["behaviour"]["not_rescorable"] == 1
    assert beh["behaviour"]["not_rescorable_reasons"] == {"no telemetry": 1}

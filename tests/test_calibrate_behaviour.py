"""Behaviour calibration sweep (#42) on synthetic sessions only."""

import json

import pandas as pd
import pytest

import scripts.calibrate_behaviour as cal
from adr.replay.folds import DEFAULT_K, DEFAULT_SEED
from scripts.calibrate_behaviour import (
    OUTCOMES,
    Thresholds,
    fold_table,
    load_features,
    main,
    match_prefixes,
    sanity_check,
    score,
    select,
    settings,
)

SECRET = "SECRET-ANSWER-TEXT that must never leave memory"
BANK = {
    "items": [
        {  # stem 5 words + passage 5 words: default floor 1000 + 100 * 10 = 2000 ms
            "item_id": "arc:q1",
            "source": "arc",
            "assessment_type": "quiz",
            "item_type": "mcq",
            "stem": "Which one is the answer?",
            "passage": "Read this short passage first.",
            "options": ["a", "b", "c", "d"],
            "correct_option": 1,
            "eligible_mechanisms": ["decoy"],
        },
        {  # 2 words: default floor 1200 ms
            "item_id": "handwritten:o1",
            "source": "handwritten",
            "assessment_type": "survey",
            "item_type": "open_text",
            "stem": "Tell us.",
            "eligible_mechanisms": [],
        },
    ],
    "assessments": [
        {
            "assessment_id": "q",
            "title": "Q",
            "assessment_type": "quiz",
            "item_ids": ["arc:q1", "handwritten:o1"],
        },
    ],
}
NORMAL = {"time_on_item_ms": 9000}
FAST = {"time_on_item_ms": 1500}  # under 2000 (default) on arc:q1, over 1200 on o1
AWAY = {"time_on_item_ms": 9000, "focus_loss_count": 1, "away_ms": 5000}
PASTE = {"time_on_item_ms": 9000, "paste_events": 1, "pasted_chars": 25}
TYPED = {"time_on_item_ms": 9000, "keystroke_count": 60, "mean_interkey_ms": 120.0}


def write_session(folder, sid, answers, *, finished=True, consented=True):
    """answers: list of (item_id, answer, telemetry)."""
    lines = [
        {
            "event": "start",
            "session_id": sid,
            "assessment_id": "q",
            "pilot": True,
            "consented": consented,
            "consent_version": "v1",
            "seed": 1,
        }
    ]
    for item_id, answer, tel in answers:
        lines.append(
            {
                "event": "answer",
                "item_id": item_id,
                "variant_id": f"{item_id}|plain|0",
                "answer": answer,
                "telemetry": tel,
                "outcomes": [],
            }
        )
    if finished:
        lines.append({"event": "finish"})
    (folder / f"{sid}.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )


@pytest.fixture
def pilot(tmp_path):
    data = tmp_path / "pilot_data"
    data.mkdir()
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(BANK), encoding="utf-8")
    # 10 sessions x 10 answers. Session i: i too_fast answers on arc:q1, 1 focus_loss in the
    # even sessions, 1 paste in session 0, one typed open-text answer, the rest normal.
    for i in range(10):
        answers = [("arc:q1", 1, FAST)] * i
        answers += [("arc:q1", 1, AWAY)] if i % 2 == 0 else []
        answers += [("arc:q1", 1, PASTE)] if i == 0 else []
        answers += [("handwritten:o1", SECRET, TYPED)]
        answers += [("arc:q1", 1, NORMAL)] * (10 - len(answers))
        write_session(data, f"{i:02d}aaaaaa", answers)
    write_session(data, "unfinished", [("arc:q1", 1, FAST)], finished=False)
    write_session(data, "noconsent", [("arc:q1", 1, FAST)], consented=False)
    return data, bank


def test_loads_finished_consented_and_matches_behaviour_py(pilot):
    df = load_features(*pilot)
    assert df["session_id"].nunique() == 10 and len(df) == 100
    counts = sanity_check(df, None)  # re-implementation == behaviour.py's score() per answer
    assert counts == {"paste": 1, "focus_loss": 5, "too_fast": 45, "normal": 49}


def test_sanity_check_stops_on_mismatch(pilot):
    df = load_features(*pilot)
    with pytest.raises(SystemExit, match="cleaned table"):
        sanity_check(df, {"paste": 0, "focus_loss": 0, "too_fast": 0, "normal": 100})
    df.loc[0, "live_outcome"] = "paste" if df.loc[0, "ref_outcome"] != "paste" else "normal"
    with pytest.raises(SystemExit, match="live server"):
        sanity_check(df, None)


def test_thresholds_move_the_rates(pilot):
    df = load_features(*pilot)

    def n(th, o):
        return int((score(df, th) == OUTCOMES.index(o)).sum())

    assert n(Thresholds(), "too_fast") == 45
    assert n(Thresholds(per_word_ms=50), "too_fast") == 0  # floor 1500: 1500 is not < 1500
    assert n(Thresholds(away_ms=6000), "focus_loss") == 0
    assert n(Thresholds(paste_min_chars=30), "paste") == 0
    assert n(Thresholds(key_gap_ms=130), "too_fast") == 45 + 10  # the typed answers now too fast


def test_fold_rates_are_held_out_fold_rates(pilot):
    df = load_features(*pilot)
    folds = {f"{i:02d}aaaaaa": i % 2 for i in range(10)}  # fold 0: even sessions
    ft = fold_table(df, folds, 2, [("too_fast", Thresholds())], {"all": set()})
    fast = ft.set_index("fold")["rate_too_fast"]
    assert fast[0] == pytest.approx((0 + 2 + 4 + 6 + 8) / 50)
    assert fast[1] == pytest.approx((1 + 3 + 5 + 7 + 9) / 50)
    assert ft.set_index("fold")["rate_focus_loss"][0] == pytest.approx(5 / 50)
    excl = fold_table(df, folds, 2, [("too_fast", Thresholds())], {"x": {"08aaaaaa"}})
    assert excl.set_index("fold")["n_answers"][0] == 40


def test_selection_uses_other_folds_only(pilot):
    df = load_features(*pilot)
    folds = {f"{i:02d}aaaaaa": i % 2 for i in range(10)}
    grid = {
        "base_floor_ms": [500, 1000],
        "per_word_ms": [50, 100],
        "key_gap_ms": [30],
        "away_ms": [2000],
        "paste_min_chars": [20],
    }
    ft = fold_table(df, folds, 2, settings(grid), {"all": set()})
    sel = select(ft, [0.45], 2).set_index(["outcome", "fold"])
    # fold 1 held out: train on fold 0 (40% too fast at the defaults, <= 45%) -> defaults
    row = sel.loc[("too_fast", 1)]
    assert (row["base_floor_ms"], row["per_word_ms"]) == (1000, 100)
    assert row["train_rate"] == pytest.approx(0.40)
    assert row["heldout_rate"] == pytest.approx(0.50)
    # fold 0 held out: fold 1 has 50% at the defaults, too many -> a stricter-for-nobody
    # setting (0% on the other fold); fold 0's own 40% never enters the choice
    row = sel.loc[("too_fast", 0)]
    assert row["train_rate"] == 0 and row["heldout_rate"] == 0


def test_match_prefixes():
    assert match_prefixes(["d3ba4f10abc", "ffff"], ["d3ba4f10"]) == {"d3ba4f10abc"}
    with pytest.raises(SystemExit):  # a prefix matching no session stops the run
        match_prefixes(["d3ba4f10abc", "ffff"], ["d3ba4f10", "0a87374a"])
    with pytest.raises(SystemExit):  # an ambiguous prefix stops the run
        match_prefixes(["d3ba4f10abc", "d3ba4f10xyz"], ["d3ba4f10"])


def test_main_uses_shared_folds_with_low_effort_passed(pilot, tmp_path, monkeypatch):
    data, bank = pilot
    calls = []
    real = cal.assign_session_folds

    def spy(session_ids, **kw):
        calls.append((sorted(session_ids), kw))
        return real(session_ids, **kw)

    monkeypatch.setattr(cal, "assign_session_folds", spy)
    argv = ["--data-dir", str(data), "--bank", str(bank), "--out", str(tmp_path / "out")]
    argv += ["--exclude-sessions", "03aa", "07aa", "--no-expected", "--floor-ms", "1000"]
    argv += ["--per-word-ms", "100", "--key-gap-ms", "30", "--away-ms", "2000"]
    argv += ["--paste-chars", "20"]
    assert main(argv) == 0
    assert len(calls) == 1
    sids, kw = calls[0]
    assert sids == [f"{i:02d}aaaaaa" for i in range(10)]
    # full ids, passed explicitly (never read from pilot_data/), at the shared defaults
    assert kw == {"k": DEFAULT_K, "seed": DEFAULT_SEED, "low_effort": ["03aaaaaa", "07aaaaaa"]}


def test_outputs_hold_no_text(pilot, tmp_path):
    data, bank = pilot
    out = tmp_path / "out"
    assert (
        main(
            [
                "--data-dir",
                str(data),
                "--bank",
                str(bank),
                "--out",
                str(out),
                "--k",
                "2",
                "--exclude-sessions",
                "03aaaaaa",
                "--no-expected",
                "--floor-ms",
                "1000",
                "--per-word-ms",
                "50,100",
                "--key-gap-ms",
                "30",
                "--away-ms",
                "2000",
                "--paste-chars",
                "20",
            ]
        )
        == 0
    )
    for name in ("folds.csv", "summary.csv", "selection.csv", "summary.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert "SECRET" not in text and "passage" not in text.lower()
    for name in ("folds.csv", "summary.csv", "selection.csv"):
        df = pd.read_csv(out / name)
        text_cols = set(df.select_dtypes(include=["object", "string"]).columns)
        assert text_cols <= {"sessions", "group", "outcome"}  # labels only
        assert "session_id" not in df.columns
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "HUMAN side only" in md and "excluding_low_effort" in md

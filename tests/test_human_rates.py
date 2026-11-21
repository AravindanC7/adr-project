"""scripts/human_rates.py on synthetic outcome rows only (no pilot data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adr.contracts import OUTCOME_LABELS, OutcomeRow, rows_to_frame, write_outcomes
from adr.map.builder import _clustered_interval
from scripts.human_rates import (
    human_rates,
    low_effort_ids,
    main,
    rates_for,
    session_draws,
    wilson,
)

ITEM_TYPE = {"decoy": "mcq", "canary": "open_text", "style": "open_text"}


def _row(session: str, mechanism: str, outcome: str, n: int, item_type: str | None = None):
    return OutcomeRow(
        run_id="pilot-test",
        session_id=session,
        respondent_kind="human",
        data_origin="pilot",
        model=None,
        attack_mode=None,
        attacker_level=None,
        assessment_type="survey",
        item_type=item_type or ITEM_TYPE.get(mechanism, "mcq"),
        item_id=f"item:{mechanism}:{n}",
        split="train",
        heldout_model=False,
        mechanism=mechanism,
        variant_id=f"item:{mechanism}:{n}|{mechanism}|0",
        outcome=outcome,
        raw_ref="raw/test.json",
    )


def _rows() -> list[OutcomeRow]:
    """s1 picks the decoy 3 of 4 times, s2 once, s3 and s4 never; canary always clean;
    behaviour: s1 too_fast twice, the rest normal. s4 is the 'low-effort' session."""
    rows = []
    decoy = {"s1": [1, 1, 1, 0], "s2": [1, 0, 0, 0], "s3": [0, 0, 0, 0], "s4": [0, 0, 0, 0]}
    for s, flags in decoy.items():
        for i, f in enumerate(flags):
            rows.append(_row(s, "decoy", "picked_decoy" if f else "picked_correct", i))
            rows.append(_row(s, "behaviour", "too_fast" if s == "s1" and i < 2 else "normal", i))
        rows.append(_row(s, "canary", "clean", 0))
        rows.append(_row(s, "opinion_trap", "contradiction" if s == "s2" else "plausible", 0,
                         "likert"))
        rows.append(_row(s, "opinion_trap", "implausible" if s == "s3" else "plausible", 1))
    return rows


def _frame() -> pd.DataFrame:
    return rows_to_frame(_rows())


def test_wilson_known_values():
    lo, hi = wilson(0, 88)
    assert lo == 0.0 and hi == pytest.approx(0.0418, abs=1e-3)
    lo, hi = wilson(10, 100)
    assert (lo, hi) == pytest.approx((0.0552, 0.1744), abs=1e-3)


def test_pooled_rates_counts_and_sessions():
    table = human_rates(_frame(), n_boot=200, seed=1)
    pooled = table[table["item_type"].eq("all")].set_index(["check", "flag"])
    decoy = pooled.loc[("decoy", "picked_decoy")]
    assert (decoy["flagged"], decoy["rows"]) == (4, 16)
    assert decoy["rate"] == pytest.approx(0.25)
    assert (decoy["sessions_flagged"], decoy["sessions"]) == (2, 4)
    assert decoy["boot_lo"] <= 0.25 <= decoy["boot_hi"]
    assert decoy["session_rate_max"] == pytest.approx(0.75)
    # add-one smoothing over the 3 decoy labels: (4 + 1) / (16 + 3)
    assert decoy["smoothed"] == pytest.approx(5 / 19)
    canary = pooled.loc[("canary", "any flag")]
    assert canary["flagged"] == 0 and canary["rule_of_three"] == pytest.approx(3 / 4)
    assert canary["boot_hi"] == 0.0
    trap = pooled.loc[("opinion_trap", "any flag")]
    assert (trap["flagged"], trap["rows"], trap["sessions_flagged"]) == (2, 8, 2)
    beh = pooled.loc[("behaviour", "any non-normal")]
    assert (beh["flagged"], beh["rows"]) == (2, 16)
    # per item type rows exist for the opinion trap (likert and mcq)
    trap_types = set(table[table["check"].eq("opinion_trap")]["item_type"])
    assert trap_types == {"all", "likert", "mcq"}


def test_bootstrap_matches_failure_map_builder_resampling():
    """Same session draws as builder._clustered_interval; with a vanishing prior its smoothed
    interval is our raw interval."""
    frame = _frame()
    rows = frame[frame["mechanism"].eq("decoy")]
    draws = session_draws(4, 500, seed=7)
    ours = rates_for(rows, "decoy", "picked_decoy", ("picked_decoy",), draws=draws)
    theirs = _clustered_interval(
        rows, OUTCOME_LABELS["decoy"], n_bootstrap=500, seed=7, prior=1e-12
    )["picked_decoy"]
    assert (ours["boot_lo"], ours["boot_hi"]) == pytest.approx(theirs, abs=1e-9)


def test_session_draws_are_reproducible():
    assert np.array_equal(session_draws(5, 10, 3), session_draws(5, 10, 3))


def test_low_effort_prefixes_and_file(tmp_path):
    frame = _frame()
    assert low_effort_ids(frame, None, ["s4"]) == {"s4"}
    with pytest.raises(SystemExit):
        low_effort_ids(frame, None, ["s"])  # ambiguous
    with pytest.raises(SystemExit):
        low_effort_ids(frame, tmp_path / "missing.txt", [])
    ids = tmp_path / "ids.txt"
    ids.write_text("s4\n", encoding="utf-8")
    assert low_effort_ids(frame, ids, []) == {"s4"}


def test_main_writes_all_outputs(tmp_path):
    src = tmp_path / "outcomes.parquet"
    write_outcomes(_rows(), src)
    out = tmp_path / "out"
    main(["--outcomes", str(src), "--low-effort-prefixes", "s4", "--n-boot", "100",
          "--out", str(out)])
    table = pd.read_csv(out / "human_rates.csv")
    assert set(table["variant"]) == {"all", "no_low_effort"}
    no_low = table[table["variant"].eq("no_low_effort") & table["item_type"].eq("all")]
    assert no_low.set_index(["check", "flag"]).loc[("decoy", "picked_decoy"), "rows"] == 12
    tex = (out / "human_rates_table.tex").read_text(encoding="utf-8")
    assert r"\label{tab:human-rates}" in tex and r"\toprule" in tex
    assert "Without 1 low-effort (3)" in (out / "human_rates.md").read_text(encoding="utf-8")
    assert (out / "notes.md").exists()

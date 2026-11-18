"""adr/analysis/bootstrap.py (paired version) and scripts/rq4_analysis.py on synthetic files."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from adr.analysis.bootstrap import paired_difference_bootstrap, two_level_bootstrap
from adr.contracts import OutcomeRow, write_outcomes

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rq4_analysis.py"
_spec = importlib.util.spec_from_file_location("rq4_analysis", _PATH)
rq4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq4)


# ---------- paired bootstrap ----------
def test_identical_conditions_give_exactly_zero_when_sessions_are_fixed():
    # one session per item: the inner level has nothing to resample, so the same item draws
    # make every replicate's difference exactly 0
    a = [np.array([1.0]), np.array([0.0]), np.array([1.0])]
    out = paired_difference_bootstrap(a, [x.copy() for x in a], 500, np.random.default_rng(0))
    assert np.all(out == 0)


def test_paired_is_tighter_than_unpaired_when_items_vary_a_lot():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 150)  # item difficulty varies widely
    a = [(rng.random(5) < q).astype(float) for q in p]
    b = [(rng.random(5) < max(q - 0.1, 0)).astype(float) for q in p]  # 10 pp lower, same items
    paired = paired_difference_bootstrap(a, b, 1000, np.random.default_rng(2))
    r = np.random.default_rng(3)
    unpaired = two_level_bootstrap(b, 1000, r) - two_level_bootstrap(a, 1000, r)
    assert paired.std() < 0.8 * unpaired.std()  # session noise within items remains
    lo, hi = np.quantile(paired, [0.025, 0.975])
    assert lo < -0.05 and hi < 0  # the 10 pp drop is detected


def test_item_draws_are_used_as_given():
    per_item = [np.array([1.0]), np.array([0.0])]
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    out = two_level_bootstrap(per_item, 3, np.random.default_rng(0), item_draws=draws)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.5])


def test_paired_checks_shapes_and_empty_items():
    with pytest.raises(ValueError, match="same items"):
        paired_difference_bootstrap(
            [np.ones(2)], [np.ones(2), np.ones(1)], 5, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="at least one observation"):
        paired_difference_bootstrap(
            [np.ones(2), np.array([])], [np.ones(2), np.ones(1)], 5, np.random.default_rng(0)
        )
    assert np.isnan(paired_difference_bootstrap([], [], 4, np.random.default_rng(0))).all()


def test_default_draws_unchanged_for_the_m2_figure():
    # Moving the function must not change the RNG sequence the M2 numbers were made with.
    data = [np.array([1.0, 0, 1]), np.array([0.0]), np.array([1.0, 1])]
    a = two_level_bootstrap(data, 50, np.random.default_rng(576))
    rng = np.random.default_rng(576)
    items = rng.integers(0, 3, size=(50, 3))  # what the function draws first...
    b = two_level_bootstrap(data, 50, rng, item_draws=items)  # ...then sessions, same stream
    np.testing.assert_array_equal(a, b)


# ---------- the RQ4 script on synthetic runs ----------
def _write(tmp: Path, rid: str, level: str, mode: str, rates: dict[str, float], sessions: int):
    rows = []
    for item, p in rates.items():
        for s in range(sessions):
            hit = (s + 0.5) / sessions < p
            rows.append(
                OutcomeRow(
                    run_id=rid,
                    session_id=f"{rid}:gpt-mini:{s}",
                    respondent_kind="ai",
                    data_origin="harness",
                    model="gpt-mini",
                    attack_mode=mode,
                    attacker_level=level,
                    assessment_type="quiz",
                    item_type="mcq",
                    item_id=item,
                    split="train",
                    heldout_model=False,
                    mechanism="decoy",
                    variant_id=f"{item}|decoy|0",
                    outcome="picked_decoy" if hit else "picked_correct",
                    raw_ref="x",
                    prompt_version="t",
                )
            )
    write_outcomes(rows, tmp / "outcomes" / f"{rid}.parquet")


def _l3_responses(tmp: Path, rid: str, items: list[str], sessions: int):
    run = tmp / "runs" / rid
    run.mkdir(parents=True)
    with (run / "responses.jsonl").open("w", encoding="utf-8") as f:
        for k, item in enumerate(items):
            for s in range(sessions):
                flagged = s < 2  # 2 of 5 first answers flagged
                final = flagged and k % 2 == 0  # half the items stay caught on the retry
                f.write(
                    json.dumps(
                        {
                            "model": "gpt-mini",
                            "item_id": item,
                            "variant_id": f"{item}|decoy|0",
                            "flagged": flagged,
                            "final_flagged": final,
                        }
                    )
                    + "\n"
                )
        f.write(
            json.dumps(
                {"model": "gpt-mini", "item_id": items[0], "variant_id": f"{items[0]}|plain|0"}
            )
            + "\n"
        )  # ignored


@pytest.fixture
def synthetic(tmp_path):
    items = [f"q{i}" for i in range(20)]
    for bank in ("mmlu", "arc"):
        for level, p in {"L0": 0.8, "L1": 0.6, "L2": 0.4, "L3": 0.2, "L4": 0.8}.items():
            _write(
                tmp_path,
                rq4.run_id("cp", level, bank, "s5"),
                level,
                "copy_paste",
                dict.fromkeys(items, p),
                5,
            )
        for level, p in {"L0": 1.0, "L1": 1.0, "L2": 0.0, "L4": 1.0}.items():
            _write(
                tmp_path,
                rq4.run_id("du", level, bank, "q20"),
                level,
                "doc_upload",
                dict.fromkeys(items, p),
                1,
            )
        _l3_responses(tmp_path, rq4.run_id("cp", "L3", bank, "s5"), items, 5)
    return tmp_path


def test_rq4_rates_drops_and_l3_stats(synthetic):
    t = rq4.compute(synthetic / "outcomes", synthetic / "runs", 300, 0)
    assert len(t) == 2 * 5 + 2 * 4  # cp L0-L4 and du L0, L1, L2, L4 per bank
    cp = t[(t["mode"] == "Copy-paste") & (t.bank == "MMLU")].set_index("level")
    assert cp.loc["L0", "rate"] == pytest.approx(0.8) and cp.loc["L0", "drop"] == 0
    assert cp.loc["L2", "drop"] == pytest.approx(-0.4)
    assert cp.loc["L4", "drop"] == pytest.approx(0.0)
    assert (cp.loc[["L1", "L2", "L3", "L4"], "n_items_paired"] == 20).all()
    assert cp.loc["L1", "drop_ci_low"] <= -0.2 + 1e-9 <= cp.loc["L1", "drop_ci_high"]
    du = t[(t["mode"] == "Doc upload") & (t.bank == "ARC")].set_index("level")
    assert "L3" not in du.index
    assert du.loc["L2", "drop"] == pytest.approx(-1.0)
    assert du.loc["L2", "drop_ci_low"] == pytest.approx(-1.0)  # every item flips: no spread
    l3 = cp.loc["L3"]
    assert l3.l3_n_answers == 100 and l3.l3_n_first_flagged == 40
    assert l3.l3_first_flag_rate == pytest.approx(0.4)
    assert l3.l3_got_away_given_flag == pytest.approx(0.5)
    assert l3.l3_n_got_away == 20


def test_rq4_outputs(synthetic):
    out = synthetic / "rq4"
    out.mkdir()
    t = rq4.compute(synthetic / "outcomes", synthetic / "runs", 100, 0)
    tex = rq4.latex(t)
    assert tex.count("\\\\") > 18 and "booktabs" in tex and "nan" not in tex.lower()
    assert "L3 one retry after a flag" in tex and "40.0\\% of first answers flagged" in tex
    # one row per mode x level (cp L0-L4, du L0/L1/L2/L4), MMLU and ARC side by side
    assert (
        sum(line.startswith(("L0 &", "L1 &", "L2 &", "L3 &", "L4 &")) for line in tex.splitlines())
        == 9
    )
    assert "\\multicolumn{2}{c}{MMLU}" in tex and "\\textit{Doc upload}" in tex
    rq4.plot(t, out)
    assert (out / "rq4_drop.png").stat().st_size > 0 and (out / "rq4_drop.pdf").exists()

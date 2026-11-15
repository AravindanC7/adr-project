"""scripts/m2_decoy_figure.py: the two-level bootstrap behind the M2 decoy-rate CIs."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "m2_decoy_figure.py"
_spec = importlib.util.spec_from_file_location("m2_decoy_figure", _PATH)
m2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m2)


def boot(per_item, n_boot=2000, seed=0):
    return m2.two_level_bootstrap(per_item, n_boot, np.random.default_rng(seed))


def test_uneven_session_counts_weight_items_by_their_observations():
    # Item A: 5 sessions, all hits. Item B: 1 session, a miss. Two items are drawn per replicate:
    # AA -> 1, BB -> 0, AB -> 5/6 (a pooled rate over observations, not a mean of item means).
    out = boot([np.ones(5), np.zeros(1)])
    assert set(np.round(out, 9)) <= {1.0, 0.0, round(5 / 6, 9)}
    assert {1.0, 0.0, round(5 / 6, 9)} <= set(np.round(out, 9))  # all three cases occur


def test_inner_level_resamples_sessions_within_an_item():
    # One item with sessions [1, 0]: the item is always drawn, its 2 sessions are resampled.
    out = boot([np.array([1.0, 0.0])])
    assert set(out) == {0.0, 0.5, 1.0}
    assert abs(out.mean() - 0.5) < 0.05


def test_single_observation():
    assert np.all(boot([np.array([1.0])]) == 1.0)
    assert np.all(boot([np.array([0.0])]) == 0.0)
    assert set(boot([np.array([0.0]), np.array([1.0])])) <= {0.0, 0.5, 1.0}


def test_empty_group_gives_nan_not_a_crash():
    out = boot([], n_boot=50)
    assert out.shape == (50,) and np.isnan(out).all()
    assert np.isnan(boot([np.array([])], n_boot=10)).all()  # an item with no observations


def test_fixed_seed_is_reproducible():
    data = [np.array([1.0, 0, 1]), np.array([0.0]), np.array([1.0, 1])]
    np.testing.assert_array_equal(boot(data, seed=576), boot(data, seed=576))


def test_latex_shows_na_for_an_empty_group():
    rows = []
    for mode, bank, run_id in m2.CELLS:
        for key, label, _ in m2.GROUPS:
            empty = key == "unknown" and bank == "ARC"
            rows.append(
                {
                    "mode": mode,
                    "bank": bank,
                    "run_id": run_id,
                    "group": key,
                    "group_label": label,
                    "n_items": 0 if empty else 10,
                    "rate": np.nan if empty else 0.5,
                    "ci_low": np.nan if empty else 0.4,
                    "ci_high": np.nan if empty else 0.6,
                }
            )
    tex = m2.latex(pd.DataFrame(rows))
    assert tex.count("n/a {\\tiny(0)}") == 2
    assert "nan" not in tex


@pytest.mark.parametrize("n_boot", [1, 7])
def test_output_length_matches_n_boot(n_boot):
    assert boot([np.ones(3), np.zeros(2)], n_boot=n_boot).shape == (n_boot,)

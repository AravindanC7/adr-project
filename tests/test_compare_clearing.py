"""scripts/compare_clearing.py: the session-level router vs router_noclear counts."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_clearing

from adr.replay.crossfit import Trace


def _t(who, running, cleared_step=None):
    run = np.array(running, dtype=float)
    return Trace(who, 0, float(max(0.0, run.max())), run, np.arange(1, len(run) + 1), len(run),
                 cleared=cleared_step is not None, cleared_step=cleared_step)


def test_counts_lost_and_shared_misses_per_group():
    th = {"router": {"A_0.05": 1.0, "A_0.01": 1.0}, "router_noclear": {"A_0.05": 1.0, "A_0.01": 1.0}}
    traces = {
        "router": {
            "ai:m:copy_paste:L2|a": [_t("ai:m:copy_paste:L2|a", [-1.0, -3.0], cleared_step=1)],
            "ai:m:copy_paste:L2g|b": [_t("ai:m:copy_paste:L2g|b", [-1.5, -3.2], cleared_step=1)],
            "ai:m:copy_paste:L0|c": [_t("ai:m:copy_paste:L0|c", [0.5, 1.5])],
        },
        "router_noclear": {
            "ai:m:copy_paste:L2|a": [_t("ai:m:copy_paste:L2|a", [-1.0, -3.0, -2.5])],  # misses too
            "ai:m:copy_paste:L2g|b": [_t("ai:m:copy_paste:L2g|b", [-1.5, -3.2, 0.7, 1.2])],  # pauses
            "ai:m:copy_paste:L0|c": [_t("ai:m:copy_paste:L0|c", [0.5, 1.5])],
        },
    }
    c = compare_clearing.compare(traces, th)["A_0.01"]
    assert c["pairs"] == 3 and c["router_clears"] == 2
    assert c["cleared_and_noclear_misses"] == 1 and c["cleared_but_noclear_pauses"] == 1
    assert c["noclear_pauses_router_misses"] == 1 and c["router_pauses_noclear_misses"] == 0
    assert c["cost_in:m:copy_paste:L2g"] == 1


def test_a_session_paused_before_it_would_clear_is_not_counted_as_cleared():
    th = {"router": {"A_0.05": 0.5, "A_0.01": 0.5}, "router_noclear": {"A_0.05": 0.5, "A_0.01": 0.5}}
    tr = _t("ai:m:copy_paste:L2|a", [0.8, -3.0], cleared_step=1)  # reaches 0.5 first
    traces = {"router": {"x": [tr]}, "router_noclear": {"x": [tr]}}
    assert compare_clearing.compare(traces, th)["A_0.01"]["router_clears"] == 0

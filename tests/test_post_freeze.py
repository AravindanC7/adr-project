"""scripts/post_freeze.py: the pieces that need no replay (exploratory, after freeze)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import post_freeze as pf


def _ai(who, prior=0.0):
    return [SimpleNamespace(who=who, running=np.array([prior + 1.0]))]


def test_evidence_counts_each_check_until_the_session_pauses():
    g = "claude-haiku:doc_upload:L2"
    who = f"ai:{g}|composite:{g}:3"
    ev = [(who, 0, ["decoy", "style", "canary", "style"], np.array([1.0, 0.5, 2.0, 3.0]))]
    th = {"A_0.05": 1.4, "A_0.01": 10.0}  # pauses after the style step; never
    out = pf.evidence_by_check(ev, th, {who: _ai(who)})[g]
    assert out["A_0.05"] == {"decoy": {"mean_llr": 1.0, "share_positive": 1.0},
                             "style": {"mean_llr": 0.5, "share_positive": 1.0}}
    assert out["A_0.01"]["style"]["mean_llr"] == 3.5 and out["A_0.01"]["canary"]["mean_llr"] == 2.0


def _draw_file(tmp_path, run, policy, point, draws, n):
    d = tmp_path / "draws"
    d.mkdir(exist_ok=True)
    det = lambda v: {t: {"g": v} for t in pf.THRESHOLDS}
    (d / f"{run}--{policy}.json").write_text(json.dumps({
        "point": {"thresholds": {}, "detected": det(point), "n": {"g": n}},
        "draws": [{"detected": det(v)} for v in draws]}))


def test_paired_differences_are_taken_draw_by_draw_and_pooled_by_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "OUT", tmp_path)
    for r in pf.RUNS:
        n = 20 if "model" in r else 10
        _draw_file(tmp_path, r, "router", 0.9 * n, [0.9 * n, 0.8 * n, 1.0 * n], n)
        _draw_file(tmp_path, r, "best_single", 0.7 * n, [0.8 * n, 0.8 * n, 0.7 * n], n)
        _draw_file(tmp_path, r, "all_checks", 0.9 * n, [0.9 * n, 0.8 * n, 1.0 * n], n)
    rows = pf.diff_rows()
    one = next(r for r in rows if r["set"] == pf.RUNS[0] and r["comparison"] == "router - best_single"
               and r["threshold"] == "A_0.05")
    assert one["diff"] == pytest.approx(0.2) and one["n_boot"] == 3
    assert one["share_draws_router_better"] == pytest.approx(2 / 3)  # 0.1, 0.0, 0.3
    pooled = next(r for r in rows if r["set"].startswith("pooled-all")
                  and r["comparison"] == "router - all_checks")
    assert pooled["diff"] == pytest.approx(0.0) and pooled["router_tpr"] == pytest.approx(0.9)
    assert all(r["share_draws_router_worse"] == 0 for r in rows)

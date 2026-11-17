"""Provisional failure maps (AI-only, or cells without human rows) are refused for routing."""

import subprocess
import sys
from pathlib import Path

import pytest

from adr.contracts import FailureMap, MapCell
from adr.router.map_guard import (
    ProvisionalMapError,
    check_router_map,
    load_router_map,
    provisional_reasons,
)
from tests.test_replay import _run_replay, _stub_table

ROOT = Path(__file__).resolve().parents[1]
AI = {"picked_decoy": 8, "picked_correct": 2, "picked_other": 0}
HUMAN = {"picked_decoy": 1, "picked_correct": 9, "picked_other": 0}
NO_HUMAN = {"picked_decoy": 0, "picked_correct": 0, "picked_other": 0}


def _map(version="v1", human=HUMAN, pooled_human=HUMAN) -> FailureMap:
    return FailureMap(
        version=version,
        cells=[
            MapCell.from_counts("decoy", "mcq", AI, pooled_human),
            MapCell.from_counts("decoy", "mcq", AI, human, attack_mode="copy_paste"),
        ],
    )


# ---------- unit: what counts as provisional ----------
def test_normal_map_is_unaffected(tmp_path):
    fmap = _map()
    assert provisional_reasons(fmap) == []
    assert check_router_map(fmap) == []
    loaded = load_router_map(fmap.save(tmp_path / "m.json"))
    assert loaded == fmap


def test_ai_only_version_is_refused_by_default(tmp_path):
    path = _map(version="v1-ai-only").save(tmp_path / "m.json")
    with pytest.raises(ProvisionalMapError, match="ends in '-ai-only'") as e:
        load_router_map(path)
    msg = str(e.value)
    assert "--allow-provisional-map" in msg and "honest people would be flagged" in msg


def test_any_cell_with_zero_human_counts_is_refused_by_default():
    # A normal-looking version, but one cell has no human rows (p_human = uniform prior).
    fmap = _map(human=NO_HUMAN)
    with pytest.raises(ProvisionalMapError, match="1 of 2 cells have zero human counts") as e:
        check_router_map(fmap)
    assert "decoy/mcq/copy_paste/*" in str(e.value)


def test_provisional_map_is_allowed_with_the_flag(tmp_path):
    fmap = _map(version="v1-ai-only", human=NO_HUMAN, pooled_human=NO_HUMAN)
    reasons = check_router_map(fmap, allow_provisional=True)
    assert len(reasons) == 2  # both reasons are still reported, for the warning
    assert load_router_map(fmap.save(tmp_path / "m.json"), allow_provisional=True) == fmap


def test_plain_failure_map_load_stays_unguarded(tmp_path):
    # Plotting or inspecting a provisional map is fine; only routing is guarded.
    fmap = _map(version="v1-ai-only", human=NO_HUMAN, pooled_human=NO_HUMAN)
    assert FailureMap.load(fmap.save(tmp_path / "m.json")) == fmap


# ---------- run_replay.py --map ----------
def _replay_cmd(tmp_path, outcomes, map_path, *extra):
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_replay.py"),
        "--stub",
        "--outcomes",
        str(outcomes),
        "--map",
        str(map_path),
        "--n",
        "20",
        "--boot",
        "20",
        "--policies",
        "router",
        "--results-dir",
        str(tmp_path),
        "--run-id",
        "r",
        *extra,
    ]


def test_run_replay_refuses_a_provisional_map_by_default(tmp_path):
    outcomes = _stub_table(tmp_path, [])
    map_path = _map(version="v1-ai-only").save(tmp_path / "m.json")
    proc = subprocess.run(
        _replay_cmd(tmp_path, outcomes, map_path),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode != 0
    assert "refusing failure map 'v1-ai-only' for routing" in proc.stderr
    assert not (tmp_path / "runs" / "r" / "metrics.json").exists()


def test_run_replay_allows_a_provisional_map_with_the_flag(tmp_path):
    outcomes = _stub_table(tmp_path, [])
    map_path = _map(version="v1-ai-only").save(tmp_path / "m.json")
    out, m = _run_replay(tmp_path, outcomes, "--map", str(map_path), "--allow-provisional-map")
    assert "PROVISIONAL MAP (--allow-provisional-map)" in out
    assert m["provisional_map_allowed"] is True
    assert any("PROVISIONAL MAP" in w for w in m["warnings"])


def test_run_replay_with_a_normal_map_is_unaffected(tmp_path):
    outcomes = _stub_table(tmp_path, [])
    map_path = _map().save(tmp_path / "m.json")
    out, m = _run_replay(tmp_path, outcomes, "--map", str(map_path))
    assert "PROVISIONAL" not in out
    assert m["provisional_map_allowed"] is False
    assert not any("PROVISIONAL" in w for w in m["warnings"])

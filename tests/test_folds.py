"""adr/replay/folds.py: the one session-to-fold assignment for every pilot analysis."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from adr.replay import folds
from adr.replay.folds import (
    DEFAULT_K,
    DEFAULT_SEED,
    assign_session_folds,
    fold_lists,
    read_low_effort,
    recorded_defaults,
)

ROOT = Path(__file__).resolve().parents[1]
# Scripts that split pilot sessions into folds. Each must import assign_session_folds; a script
# that is not on this branch yet is checked once it lands.
FOLD_USERS = ("run_real_router.py", "calibrate_behaviour.py")

SESSIONS = [f"{i:032x}" for i in range(44)]  # 44 pilot-like full session ids
LOW = SESSIONS[3:7]  # 4 low-effort sessions


def test_defaults_are_the_recorded_ones():
    assert recorded_defaults() == (DEFAULT_K, DEFAULT_SEED) == (4, "adr-pilot-folds-v1")


def test_identical_across_calls_and_input_order():
    a = assign_session_folds(SESSIONS, low_effort=LOW)
    shuffled = SESSIONS[::-1] + SESSIONS[:5]  # other order, duplicates
    random.Random(1).shuffle(shuffled)
    assert assign_session_folds(shuffled, low_effort=list(reversed(LOW))) == a
    assert assign_session_folds(SESSIONS, 4, "adr-pilot-folds-v1", LOW) == a
    # pinned: a change to the rule shows up here, not silently in the results
    assert [a[s] for s in LOW] == [1, 0, 3, 2]


def test_partition_sizes_and_low_effort_spread():
    a = assign_session_folds(SESSIONS, low_effort=LOW)
    assert set(a) == set(SESSIONS) and set(a.values()) == {0, 1, 2, 3}
    lists = fold_lists(a, 4)
    assert sorted(len(f) for f in lists) == [11, 11, 11, 11]
    assert [len(set(f) & set(LOW)) for f in lists] == [1, 1, 1, 1]
    assert sorted(s for f in lists for s in f) == sorted(SESSIONS)


def test_seed_and_k_change_the_assignment():
    a = assign_session_folds(SESSIONS, low_effort=LOW)
    assert assign_session_folds(SESSIONS, seed="other", low_effort=LOW) != a
    assert set(assign_session_folds(SESSIONS, k=5, low_effort=LOW).values()) == set(range(5))


def test_excluding_sessions_keeps_everyone_elses_fold():
    a = assign_session_folds(SESSIONS, low_effort=LOW)
    kept = [s for s in SESSIONS if s not in LOW]
    without = fold_lists(a, 4, keep=kept)
    assert without == [[s for s in f if s not in LOW] for f in fold_lists(a, 4)]


def test_low_effort_id_not_found_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="not among the 44 sessions"):
        assign_session_folds(SESSIONS, low_effort=[*LOW, "d3ba4f10"])  # a prefix is not an id
    f = tmp_path / "low.txt"
    f.write_text("# low-effort\n" + "\n".join(LOW) + "\n\nnot-a-session\n")
    with pytest.raises(ValueError, match="not-a-session"):
        assign_session_folds(SESSIONS, low_effort=f)


def test_low_effort_file(tmp_path, monkeypatch):
    f = tmp_path / "low.txt"
    f.write_text("# comment\n" + "\n".join(LOW) + "\n" + LOW[0] + "\n")
    assert read_low_effort(f) == sorted(LOW)
    assert assign_session_folds(SESSIONS, low_effort=f) == assign_session_folds(
        SESSIONS, low_effort=LOW
    )
    assert assign_session_folds(SESSIONS, low_effort=str(f)) == assign_session_folds(
        SESSIONS, low_effort=LOW
    )
    # None reads the default file; a missing one is an error, never "no low-effort sessions"
    monkeypatch.setattr(folds, "LOW_EFFORT_FILE", tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError):
        assign_session_folds(SESSIONS)
    monkeypatch.setattr(folds, "LOW_EFFORT_FILE", f)
    assert assign_session_folds(SESSIONS) == assign_session_folds(SESSIONS, low_effort=LOW)


def test_bad_k():
    with pytest.raises(ValueError):
        assign_session_folds(SESSIONS, k=1, low_effort=[])
    with pytest.raises(ValueError):
        assign_session_folds(SESSIONS[:3], k=4, low_effort=[])


@pytest.mark.parametrize("name", FOLD_USERS)
def test_fold_scripts_use_the_shared_function(name):
    """Across scripts: every script that assigns pilot folds imports assign_session_folds from
    adr.replay.folds and has no fold function of its own."""
    path = ROOT / "scripts" / name
    if not path.is_file():
        pytest.skip(f"scripts/{name} is not on this branch yet")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "adr.replay.folds"
        for alias in node.names
    }
    assert "assign_session_folds" in imported, f"{name} must import assign_session_folds"
    own = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and "fold" in node.name.lower()
        and any(isinstance(n, ast.Attribute) and n.attr == "shuffle" for n in ast.walk(node))
    ]
    assert not own, f"{name} defines its own fold assignment: {own}"


def test_no_other_module_shuffles_sessions_into_folds():
    """No second fold assignment anywhere in adr/ or scripts/."""
    offenders = []
    for path in [*(ROOT / "adr").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        if path == ROOT / "adr" / "replay" / "folds.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and "fold" in node.name.lower()
                and any(
                    isinstance(n, ast.Attribute) and n.attr == "shuffle" for n in ast.walk(node)
                )
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.name}")
    assert not offenders, offenders

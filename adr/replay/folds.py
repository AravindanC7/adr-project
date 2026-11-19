"""The one session-to-fold assignment for the pilot (step #42). Owner: Arav.

Every script that splits pilot sessions into folds imports `assign_session_folds` from here:
scripts/run_real_router.py (cross-fitting the router's failure maps and review threshold) and
scripts/calibrate_behaviour.py (choosing the behaviour-check thresholds). So a pilot session held
out in one analysis is held out in the other, and neither can tune on a session the other treats
as unseen. Do not copy this function into a script; import it.

The defaults (K and the seed string) are recorded in configs/data_decisions.json under
"pilot_folds"; a test checks they match. Change them there and here together, never per run.

Rule: sessions (unique pilot session_ids, sorted) are dealt round-robin into K folds after a
seeded shuffle. The low-effort sessions are shuffled and dealt first, the rest continue the same
deal, so the low-effort sessions are spread across the folds (at most one apart) and fold sizes
differ by at most one. The assignment depends only on the set of session ids, the low-effort
ids, K and the seed string.

Assign folds on ALL pilot sessions, then drop any excluded ones from their folds: the "without
low-effort" run then uses exactly the folds of the "all sessions" run.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISIONS = ROOT / "configs" / "data_decisions.json"
LOW_EFFORT_FILE = ROOT / "pilot_data" / "low_effort_ids.txt"

DEFAULT_K = 4
DEFAULT_SEED = "adr-pilot-folds-v1"


def read_low_effort(path: str | Path = LOW_EFFORT_FILE) -> list[str]:
    """Full pilot session_ids, one per line; blank lines and # comments are ignored.
    A missing file is an error, not an empty list."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"low-effort id file not found: {path}. It lists the low-effort pilot sessions "
            "(full session_ids, one per line)."
        )
    lines = (s.strip() for s in path.read_text(encoding="utf-8").splitlines())
    return sorted({s for s in lines if s and not s.startswith("#")})


def assign_session_folds(
    session_ids: Iterable[str],
    k: int = DEFAULT_K,
    seed: str = DEFAULT_SEED,
    low_effort: Iterable[str] | str | Path | None = None,
) -> dict[str, int]:
    """{session_id: fold in 0..k-1} for every unique session id.

    low_effort: None reads pilot_data/low_effort_ids.txt; a str or Path is a file to read; any
    other iterable is the ids themselves (pass [] for none). Every low-effort id must be one of
    `session_ids` (full ids, exact match), or this raises: a typo or a stale file would
    otherwise silently drop the stratification."""
    unique = sorted(set(session_ids))
    if k < 2:
        raise ValueError("need at least 2 folds")
    if len(unique) < k:
        raise ValueError(f"{len(unique)} sessions cannot fill {k} folds")
    if low_effort is None:
        low = read_low_effort(LOW_EFFORT_FILE)
    elif isinstance(low_effort, (str, Path)):
        low = read_low_effort(low_effort)
    else:
        low = sorted(set(low_effort))
    missing = sorted(set(low) - set(unique))
    if missing:
        raise ValueError(
            f"{len(missing)} low-effort id(s) are not among the {len(unique)} sessions given: "
            f"{missing}. Low-effort ids must be full session_ids, and folds are assigned on "
            "all pilot sessions before any are excluded."
        )
    marked = set(low)
    first = [s for s in unique if s in marked]
    rest = [s for s in unique if s not in marked]
    random.Random(f"{seed}:low_effort").shuffle(first)
    random.Random(f"{seed}:rest").shuffle(rest)
    return {s: i % k for i, s in enumerate(first + rest)}


def fold_lists(
    assignment: dict[str, int], k: int, keep: Iterable[str] | None = None
) -> list[list[str]]:
    """The assignment as k sorted lists of session ids, optionally only the `keep` sessions."""
    kept = set(assignment) if keep is None else set(keep) & set(assignment)
    folds: list[list[str]] = [[] for _ in range(k)]
    for s in sorted(kept):
        folds[assignment[s]].append(s)
    return folds


def recorded_defaults(path: str | Path = DECISIONS) -> tuple[int, str]:
    """(k, seed) as recorded in configs/data_decisions.json["pilot_folds"]."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))["pilot_folds"]
    return int(rec["k"]), str(rec["seed"])


__all__: Sequence[str] = (
    "DEFAULT_K",
    "DEFAULT_SEED",
    "LOW_EFFORT_FILE",
    "assign_session_folds",
    "fold_lists",
    "read_low_effort",
    "recorded_defaults",
)

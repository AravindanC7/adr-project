"""Saving sessions to disk, for the pilot (#35) and the cleaning step (#39).

Owner: Harshini.

One file per session, `<data_dir>/<session_id>.jsonl`, one JSON object per line, appended as it
happens so a crash or restart loses nothing already answered:

    {"event": "start",  "session_id", "assessment_id", "pilot", "consented", "consent_version",
                        "seed", "started_on"}
    {"event": "answer", "item_id", "variant_id", "mechanism", "item_type", "answer",
                        "telemetry": {...}, "outcomes": [[variant_id, outcome], ...], "t"}
    {"event": "finish", "t"}

What is NOT stored (AGENTS.md rule 7, docs/pilot/consent.md): no name, email, ASU ID, IP
address or browser details; the session id is random. Telemetry is counts and timings only.
No clock times either: with about ten classmates, "answered at 19:42 on Nov 10" could point to
one person. The start record has the date only (`started_on`, UTC) and every other record has
`t`, whole seconds since the session started.

Withdrawing ("Stop" or "Remove my answers") deletes the file, and nothing can re-create it: only
`start` makes a new file, so an answer still in flight when someone presses Stop is dropped.
#39 reads every file that has a "finish" line; a file without one is a partial session and is
dropped there.

Removing a session later, from the code shown on the done screen:

    python -m server.store remove <code> [--dir pilot_data]
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SESSION_ID = re.compile(r"[0-9a-f]{32}")  # uuid4().hex: nothing that could name another path


class SessionStore:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # an append and a delete never interleave
        self._t0: dict[str, float] = {}

    def path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError(f"not a session id: {session_id!r}")
        return self.dir / f"{session_id}.jsonl"

    def _write(self, session_id: str, record: dict[str, Any], create: bool = False) -> bool:
        p = self.path(session_id)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            if not create and not p.exists():
                return False  # withdrawn (or never started): don't bring it back
            with p.open("a", encoding="utf-8") as f:
                f.write(line)
        return True

    def _since_start(self, session_id: str) -> int | None:
        t0 = self._t0.get(session_id)
        return None if t0 is None else round(time.monotonic() - t0)

    def start(self, session_id: str, **fields: Any) -> None:
        self._t0[session_id] = time.monotonic()
        today = datetime.now(timezone.utc).date().isoformat()
        record = {"event": "start", "session_id": session_id, **fields, "started_on": today}
        self._write(session_id, record, create=True)

    def answer(self, session_id: str, **fields: Any) -> bool:
        record = {"event": "answer", **fields, "t": self._since_start(session_id)}
        return self._write(session_id, record)

    def finish(self, session_id: str) -> bool:
        return self._write(session_id, {"event": "finish", "t": self._since_start(session_id)})

    def delete(self, session_id: str) -> bool:
        """Remove everything stored for the session. True if there was a file."""
        p = self.path(session_id)
        with self._lock:
            existed = p.exists()
            p.unlink(missing_ok=True)
        self._t0.pop(session_id, None)
        return existed


def read_session(path: str | Path) -> list[dict[str, Any]]:
    """The records of one session file, in order (for #39)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m server.store")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rm = sub.add_parser("remove", help="delete one session, e.g. when someone emails their code")
    rm.add_argument("code", help="the code from the done screen")
    rm.add_argument("--dir", default="pilot_data")
    args = parser.parse_args(argv)
    try:
        removed = SessionStore(args.dir).delete(args.code.strip().lower())
    except ValueError:
        print(f"That isn't a session code: {args.code!r}")
        return 2
    print("Removed." if removed else "No session with that code (already removed?).")
    return 0 if removed else 1


if __name__ == "__main__":
    raise SystemExit(main())

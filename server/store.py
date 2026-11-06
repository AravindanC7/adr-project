"""Saving sessions to disk, for the pilot (#35) and the cleaning step (#39).

Owner: Harshini.

One file per session, `<data_dir>/<session_id>.jsonl`, one JSON object per line, appended as it
happens so a crash or restart loses nothing already answered:

    {"event": "start",  "session_id", "assessment_id", "pilot", "consented", "seed", "at"}
    {"event": "answer", "item_id", "variant_id", "mechanism", "item_type", "answer",
                        "telemetry": {...}, "outcomes": [[variant_id, outcome], ...], "at"}
    {"event": "finish", "at"}

What is NOT stored (AGENTS.md rule 7, docs/pilot/consent.md): no name, email, ASU ID, IP
address or browser details; the session id is random. Telemetry is counts and timings only.
`at` is UTC, to the second.

Withdrawing ("Stop" or "Remove my answers") deletes the file. #39 reads every file that has a
"finish" line; a file without one is a partial session and is dropped there.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SESSION_ID = re.compile(r"[0-9a-f]{32}")  # uuid4().hex: nothing that could name another path


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SessionStore:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError(f"not a session id: {session_id!r}")
        return self.dir / f"{session_id}.jsonl"

    def _append(self, session_id: str, record: dict[str, Any]) -> None:
        line = json.dumps({**record, "at": _now()}, ensure_ascii=False)
        with self.path(session_id).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def start(self, session_id: str, **fields: Any) -> None:
        self._append(session_id, {"event": "start", "session_id": session_id, **fields})

    def answer(self, session_id: str, **fields: Any) -> None:
        self._append(session_id, {"event": "answer", **fields})

    def finish(self, session_id: str) -> None:
        self._append(session_id, {"event": "finish"})

    def delete(self, session_id: str) -> bool:
        """Remove everything stored for the session. True if there was a file."""
        p = self.path(session_id)
        existed = p.exists()
        p.unlink(missing_ok=True)
        return existed


def read_session(path: str | Path) -> list[dict[str, Any]]:
    """The records of one session file, in order (for #39)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

"""Which code a run used: the git commit and whether the working tree had uncommitted changes.

Call `git_state()` when a run STARTS and write the result into the run's config.yaml. Reading it
at the end is wrong: a commit made while the run was going (or right after the code was written
and before it was committed) would be recorded instead of the code that actually ran.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                          check=True, cwd=cwd).stdout


def git_state(cwd: Path | None = None) -> dict[str, str | bool | None]:
    """{"git_commit": short HEAD, "git_dirty": True if tracked files are modified or untracked,
    non-ignored files exist}. git_dirty=True means the code that ran may differ from git_commit
    (e.g. a new, not yet committed script). Both are None outside a git checkout."""
    try:
        commit = _git("rev-parse", "--short", "HEAD", cwd=cwd).strip()
        dirty = bool(_git("status", "--porcelain", cwd=cwd).strip())
    except Exception:  # noqa: BLE001 - no git, not a repo: record that we don't know
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}

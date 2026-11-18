"""Run configs record the code that ran: HEAD at the start, plus a dirty flag."""

import subprocess

from adr.harness.provenance import git_state


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_state_reports_head_and_dirty(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "start")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, check=True,
                          capture_output=True, text=True).stdout.strip()
    assert git_state(tmp_path) == {"git_commit": head, "git_dirty": False}
    (tmp_path / "new_script.py").write_text("x = 1\n")   # untracked, not yet committed code
    assert git_state(tmp_path) == {"git_commit": head, "git_dirty": True}


def test_outside_a_repo_both_are_unknown(tmp_path):
    assert git_state(tmp_path) == {"git_commit": None, "git_dirty": None}

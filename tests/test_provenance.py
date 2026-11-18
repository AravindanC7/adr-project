"""Run configs record the code that ran: HEAD at the start, plus a dirty flag (tracked files)."""

import subprocess

from adr.harness.provenance import git_state


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo,
                   check=True, capture_output=True)


def _head(repo):
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_git_state_reports_head_and_tracked_changes(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-q", "-m", "start")
    assert git_state(tmp_path) == {"git_commit": _head(tmp_path), "git_dirty": False}
    (tmp_path / "code.py").write_text("x = 2\n")          # tracked file changed
    assert git_state(tmp_path) == {"git_commit": _head(tmp_path), "git_dirty": True}


def test_an_untracked_file_does_not_set_the_flag(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "start")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "out.csv").write_text("a,b\n")   # e.g. an untracked results folder
    (tmp_path / "new_script.py").write_text("x = 1\n")
    assert git_state(tmp_path) == {"git_commit": _head(tmp_path), "git_dirty": False}


def test_outside_a_repo_both_are_unknown(tmp_path):
    assert git_state(tmp_path) == {"git_commit": None, "git_dirty": None}

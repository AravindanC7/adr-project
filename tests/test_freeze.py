"""scripts/freeze_manifest.py, scripts/freeze_report.py and scripts/freeze_run.sh (dry run only:
nothing is run)."""

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_manifest as fm
import freeze_report as fr


def _bank(root: Path, newline: str = "\n") -> Path:
    for rel in fm.BANK_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f'{{"file": "{rel}"}}{newline}{{"n": 1}}{newline}'.encode())
    (root / "manifest.json").write_text(json.dumps({"assessment": str(root / "x.json")}))
    return root


def _zip(bank: Path, out: Path) -> Path:
    with zipfile.ZipFile(out, "w") as z:
        for rel in [*fm.BANK_FILES, "manifest.json"]:
            z.writestr(rel, (bank / rel).read_bytes())
    return out


def test_content_sha_ignores_windows_line_endings():
    assert fm.content_sha(b"a\r\nb\r\n") == fm.content_sha(b"a\nb\n")
    assert fm.sha256(b"a\r\nb\r\n") != fm.sha256(b"a\nb\n")


def test_check_heldout_matches_a_windows_build_and_catches_a_real_change(tmp_path, capsys):
    built = _bank(tmp_path / "mac")
    win = _zip(_bank(tmp_path / "win", "\r\n"), tmp_path / "win.zip")
    args = type("A", (), {"bank": built, "against": win})
    assert fm.cmd_check_heldout(args) == 0
    assert "MATCH" in capsys.readouterr().out
    (built / "items.jsonl").write_text('{"changed": true}\n')
    assert fm.cmd_check_heldout(args) == 1


def test_prep_verify_and_add_final(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # configs/ are read from the checkout
    bank = _bank(tmp_path / "bank")
    pilot = tmp_path / "pilot.parquet"
    pilot.write_bytes(b"pilot")
    low = tmp_path / "low.txt"
    low.write_text("id\n")
    train = tmp_path / "t.parquet"
    train.write_bytes(b"train")
    listing = tmp_path / "train.txt"
    listing.write_text(f"{train}\n")
    out = tmp_path / "m.json"
    prep = type("A", (), {"out": out, "heldout_bank": bank, "pilot": pilot, "low_effort": low,
                          "training_files": listing, "expected_bank_zip": None})
    assert fm.cmd_prep(prep) == 0
    m = json.loads(out.read_text())
    assert {e["path"] for e in m["training_outcomes"]} == {str(train.resolve())}
    assert set(m["configs"]) == set(fm.CONFIGS) and m["git"]["git_commit"]
    verify = type("A", (), {"manifest": out, "heldout_bank": None})
    assert fm.cmd_verify(verify) == 0
    final = tmp_path / "final.parquet"
    final.write_bytes(b"final")
    assert fm.cmd_add_final(type("A", (), {"manifest": out, "files": [str(final)]})) == 0
    assert fm.cmd_verify(verify) == 0
    final.write_bytes(b"changed")  # an input changed after it was recorded
    assert fm.cmd_verify(verify) == 1


def test_prep_refuses_a_bank_that_differs_from_the_handoff(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    bank = _bank(tmp_path / "bank")
    other = _bank(tmp_path / "other")
    (other / "items.jsonl").write_text('{"other": 1}\n')
    for name, data in (("pilot", b"p"), ("low", b"l"), ("t", b"t")):
        (tmp_path / name).write_bytes(data)
    (tmp_path / "list.txt").write_text(f"{tmp_path / 't'}\n")
    prep = type("A", (), {"out": tmp_path / "m.json", "heldout_bank": bank,
                          "pilot": tmp_path / "pilot", "low_effort": tmp_path / "low",
                          "training_files": tmp_path / "list.txt",
                          "expected_bank_zip": _zip(other, tmp_path / "h.zip")})
    assert fm.cmd_prep(prep) == 1
    assert not (tmp_path / "m.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="bash driver")
def test_freeze_driver_dry_run_plans_42_attacker_runs_and_4_router_runs(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "PY": "python"}
    proc = subprocess.run(["bash", str(ROOT / "scripts" / "freeze_run.sh")], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    out = proc.stdout
    ids = set(re.findall(r"--run-id (v1-final-\S+)", out))
    assert len(ids) == 42 and all("final" in i for i in ids)
    assert sum("claude-haiku" in i for i in ids) == 36
    gpt = {i for i in ids if "gpt-mini" in i}
    assert len(gpt) == 6 and all("osmi-heldout" in i for i in gpt)  # gpt-mini: held-out bank only
    assert out.count("run_real_router.py --final") == 4
    routers = [ln for ln in out.splitlines() if "run_real_router.py --final" in ln]
    for ln in routers:
        finals = set(re.findall(r"results/outcomes/(v1-final-\S+)\.parquet", ln))
        if "unseen-model" in ln:  # claude-haiku on the training banks only
            assert len(finals) == 30
            assert all("claude-haiku" in f and "osmi" not in f for f in finals)
        else:  # both models on the held-out OSMI bank only
            assert "unseen-survey" in ln and len(finals) == 12
            assert all("osmi-heldout" in f for f in finals)
    assert "freeze_report.py --out results/freeze-2026-11-22/README.md" in out
    assert "--reference-no-low-effort results/real-router/rq2-l012-floor750-reopen-no-low-effort" in out
    assert "--jobs 8 --resume" in out and "--exclude-sessions" in out
    assert out.count("rescore --final") == 30  # every survey / written-answer run is re-scored
    assert "freeze-2026-11-22-claude-haiku (cap $10)" in out
    assert "freeze-2026-11-22-gpt-mini (cap $10)" in out
    for line in out.splitlines():  # every attacker call is a --final call at temperature 0.7
        if "--run-id v1-final" in line:
            assert "--final" in line and "--temperature 0.7" in line and "--samples 20" in line


# ------------------------------------------------------------------ freeze_report.py
SRC = {"checks": 4.0, "same_type": 4.0, "fallback": 0.0, "synthetic": 0.0, "missing": 0.0}


def _router(kind="unseen-survey", thr=(0.008, 0.951), haiku_src=None, sessions=None, n=44):
    """A metrics.json router block with the complete expected group matrix for `kind`;
    `sessions` overrides per-group session counts (None = drop the group)."""
    counts = {g: 20 for g in fr.EXPECTED_GROUPS[kind]}
    for g, c in (sessions or {}).items():
        if c is None:
            counts.pop(g, None)
        else:
            counts[g] = c
    groups = {g: {**SRC, "sessions": c, **(haiku_src if g.startswith("claude-haiku:") and haiku_src
                                           else {})}
              for g, c in counts.items()}
    if "gpt-mini:copy_paste:L0" in groups:  # gpt-mini's fallback/synthetic are not checked
        groups["gpt-mini:copy_paste:L0"].update(fallback=2.0, synthetic=1.0)
    res = {t: {"honest_pause_rate": 1 / n, **{f"tpr[{g}]": 0.5 for g in groups}}
           for t in fr.THRESHOLDS}
    ci = {t: {f"tpr[{g}]": [0.3, 0.7] for g in groups} for t in fr.THRESHOLDS}
    return {"thresholds": dict(zip(fr.THRESHOLDS, thr, strict=True)), "results": res, "ci95": ci,
            "n_sessions": n, "ai_sources_by_group": groups, "n_ai_sessions_by_group": counts}


def _write(run_dir: Path, router: dict) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"results": {"router": router}}))
    return run_dir


def test_expected_group_matrix():
    assert len(fr.EXPECTED_GROUPS["unseen-model"]) == 6  # claude-haiku only, by design
    assert all(g.startswith("claude-haiku:") for g in fr.EXPECTED_GROUPS["unseen-model"])
    assert len(fr.EXPECTED_GROUPS["unseen-survey"]) == 12
    assert {g.split(":")[0] for g in fr.EXPECTED_GROUPS["unseen-survey"]} == {"gpt-mini",
                                                                               "claude-haiku"}


@pytest.mark.parametrize("kind", ["unseen-model", "unseen-survey"])
def test_complete_group_matrix_passes(kind):
    assert fr.group_problems(_router(kind), fr.EXPECTED_GROUPS[kind]) == []


@pytest.mark.parametrize(
    "kind, sessions, message",
    [
        ("unseen-survey", {"gpt-mini:doc_upload:L2g": None},
         "missing group gpt-mini:doc_upload:L2g (0 sessions, expected 20)"),
        ("unseen-model", {"claude-haiku:copy_paste:L2": 19},
         "group claude-haiku:copy_paste:L2 has 19 sessions, expected exactly 20"),
        ("unseen-survey", {"claude-haiku:doc_upload:L0": 21},
         "group claude-haiku:doc_upload:L0 has 21 sessions, expected exactly 20"),
        ("unseen-model", {"gpt-mini:copy_paste:L0": 20},  # gpt-mini is not evaluated here
         "unexpected group gpt-mini:copy_paste:L0 (20 sessions)"),
        ("unseen-survey", {"claude-haiku:copy_paste:L1": 20},
         "unexpected group claude-haiku:copy_paste:L1 (20 sessions)"),
    ],
)
def test_missing_short_overfull_or_unexpected_group_fails(kind, sessions, message):
    assert fr.group_problems(_router(kind, sessions=sessions), fr.EXPECTED_GROUPS[kind]) == [message]


@pytest.mark.parametrize("bad", [{"synthetic": 0.5}, {"fallback": 1.0}])
def test_source_check_flags_claude_haiku_borrowed_outcomes_only(bad):
    assert fr.source_problems(_router()) == []  # gpt-mini's fallback/synthetic are not checked
    problems = fr.source_problems(_router(haiku_src=bad))
    assert len(problems) == 6 and all(p.startswith("claude-haiku:") for p in problems)


def test_threshold_check_against_51():
    assert fr.threshold_problems(_router(), _router()) == []
    assert len(fr.threshold_problems(_router(thr=(0.008, 0.9)), _router())) == 1


def test_reference_is_picked_by_honest_sessions(tmp_path):
    ref_all, ref_no = tmp_path / "all", tmp_path / "no"
    assert fr.reference_for(Path("x-unseen-model-all"), ref_all, ref_no) == ref_all
    assert fr.reference_for(Path("x-unseen-model-no-low-effort"), ref_all, ref_no) == ref_no


def _report(tmp_path, haiku_src=None, thr=(0.008, 0.951), survey_sessions=None):
    runs = {}
    for kind in ("unseen-model", "unseen-survey"):
        for who in ("all", "no-low-effort"):
            sessions = survey_sessions if kind == "unseen-survey" else None
            runs[f"{kind}-{who}"] = _write(tmp_path / f"{kind}-{who}",
                                           _router(kind, thr, haiku_src, sessions))
    ref_all = _write(tmp_path / "ref-all", _router())
    ref_no = _write(tmp_path / "ref-no", _router())
    out = tmp_path / "README.md"
    cmd = [sys.executable, str(ROOT / "scripts" / "freeze_report.py"), "--out", str(out),
           "--unseen-model", str(runs["unseen-model-all"]), str(runs["unseen-model-no-low-effort"]),
           "--unseen-survey", str(runs["unseen-survey-all"]),
           str(runs["unseen-survey-no-low-effort"]),
           "--reference-all", str(ref_all), "--reference-no-low-effort", str(ref_no)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    return proc, out.read_text(encoding="utf-8")


def test_report_writes_the_fpr_rule_and_both_reports(tmp_path):
    proc, readme = _report(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert fr.FPR_RULE in readme
    assert "## Unseen model" in readme and "## Unseen survey" in readme
    model_part, survey_part = readme.split("## Unseen survey")
    assert "gpt-mini" not in model_part.split("## Unseen model")[1]  # claude-haiku only
    assert "| gpt-mini | L0 | copy-paste | 20 | 0.50 [0.30, 0.70] | 0.50 [0.30, 0.70]" in survey_part
    assert "FAILED" not in readme


def test_report_fails_loudly_on_borrowed_claude_haiku_outcomes(tmp_path):
    proc, readme = _report(tmp_path, haiku_src={"fallback": 0.25})
    assert proc.returncode == 1
    assert "POST-RUN CHECK FAILED" in proc.stderr and "claude-haiku:copy_paste:L0" in proc.stderr
    assert "- FAILED: unseen-model-all: claude-haiku:copy_paste:L0" in readme


def test_report_fails_loudly_on_a_short_group_naming_run_group_and_count(tmp_path):
    proc, readme = _report(tmp_path, survey_sessions={"gpt-mini:doc_upload:L2": 19})
    assert proc.returncode == 1
    msg = ("unseen-survey-no-low-effort: group gpt-mini:doc_upload:L2 has 19 sessions, "
           "expected exactly 20")
    assert f"POST-RUN CHECK FAILED: {msg}" in proc.stderr
    assert f"- FAILED: {msg}" in readme


def test_report_fails_when_a_threshold_differs_from_51(tmp_path):
    proc, _ = _report(tmp_path, thr=(0.01, 0.951))
    assert proc.returncode == 1 and "threshold differs from #51" in proc.stderr

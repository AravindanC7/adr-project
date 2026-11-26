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
import yaml

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


def _run_root(root: Path) -> Path:
    """A checkout-like root with the files prep hashes besides the outcome files."""
    for rel in (fm.PILOT_BANK, *fm.TRAINING_BANK_FILES):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'{{"file": "{rel}"}}\n')
    prof = root / fm.STYLE_PROFILE
    prof.parent.mkdir(parents=True, exist_ok=True)
    prof.write_text(json.dumps({"profile": {"means": [1, 2]}, "note": "x"}))
    return root


@pytest.fixture
def locked_profile(tmp_path, monkeypatch):
    """configs/data_decisions.json stand-in locking the synthetic profile's sha."""
    monkeypatch.chdir(ROOT)  # configs/ are read from the checkout
    root = _run_root(tmp_path / "root")
    sha = fm.profile_sha(root / fm.STYLE_PROFILE)
    dec = tmp_path / "decisions.json"
    dec.write_text(json.dumps({"style_profile": {"profile_sha256": sha}}))
    monkeypatch.setattr(fm, "DECISIONS", dec)
    monkeypatch.setattr(fm, "STYLE_PROFILE_SHA12", sha[:12])
    return root


def _prep_args(tmp_path, root, **kw):
    for name, data in (("pilot.parquet", b"pilot"), ("low.txt", b"id\n"), ("t.parquet", b"t")):
        (tmp_path / name).write_bytes(data)
    (tmp_path / "train.txt").write_text(f"{tmp_path / 't.parquet'}\n")
    base = {"out": tmp_path / "m.json", "heldout_bank": _bank(tmp_path / "bank"),
            "pilot": tmp_path / "pilot.parquet", "low_effort": tmp_path / "low.txt",
            "training_files": tmp_path / "train.txt", "expected_bank_zip": None, "root": root}
    return type("A", (), {**base, **kw})


def test_prep_verify_and_add_final(tmp_path, locked_profile):
    prep = _prep_args(tmp_path, locked_profile)
    assert fm.cmd_prep(prep) == 0
    m = json.loads(prep.out.read_text())
    assert {e["path"] for e in m["training_outcomes"]} == {str((tmp_path / "t.parquet").resolve())}
    assert set(m["configs"]) == set(fm.CONFIGS) and m["git"]["git_commit"]
    assert m["spend_caps_usd"] == {"claude-haiku": 8.0, "gpt-mini": 2.0}
    assert m["max_tokens"]["claude-haiku"]["copy_paste"] == 4096 and m["max_tokens"]["reason"]
    du = m["max_tokens"]["claude-haiku"]["doc_upload"]
    assert du["quiz_and_survey"] == 8192 and du["written_answers"] == 8192
    assert m["style_profile"]["profile_sha256"] == fm.STYLE_PROFILE_SHA12 + m["style_profile"][
        "profile_sha256"][12:]
    assert Path(m["pilot_bank"]["path"]).parts[-2:] == ("server", "pilot_items.json")
    assert len(m["training_banks"]) == len(fm.TRAINING_BANK_FILES) == 12  # 2 items files + 5 banks x (assessment, variants)
    verify = type("A", (), {"manifest": prep.out, "heldout_bank": None})
    assert fm.cmd_verify(verify) == 0
    final = tmp_path / "final.parquet"
    final.write_bytes(b"final")
    assert fm.cmd_add_final(type("A", (), {"manifest": prep.out, "files": [str(final)]})) == 0
    assert fm.cmd_verify(verify) == 0
    final.write_bytes(b"changed")  # an input changed after it was recorded
    assert fm.cmd_verify(verify) == 1


@pytest.mark.parametrize("rel", ["server/pilot_items.json",
                                 "data/processed/runs_v1/variants/quiz-arc-200.jsonl",
                                 "data/processed/archana-survey-written-v1/assessments/survey-gss.json"])
def test_verify_catches_a_changed_pilot_bank_or_training_bank_file(tmp_path, locked_profile, rel):
    prep = _prep_args(tmp_path, locked_profile)
    assert fm.cmd_prep(prep) == 0
    (locked_profile / rel).write_text("changed\n")
    assert fm.cmd_verify(type("A", (), {"manifest": prep.out, "heldout_bank": None})) == 1


def test_verify_requires_the_manifest_commit_to_be_head(tmp_path, locked_profile, capsys):
    prep = _prep_args(tmp_path, locked_profile)
    assert fm.cmd_prep(prep) == 0
    m = json.loads(prep.out.read_text())
    m["git"]["git_commit"] = "0000000"
    prep.out.write_text(json.dumps(m))
    assert fm.cmd_verify(type("A", (), {"manifest": prep.out, "heldout_bank": None})) == 1
    assert "git commit: manifest 0000000" in capsys.readouterr().err


def test_prep_refuses_a_style_profile_that_is_not_the_locked_one(tmp_path, locked_profile):
    (locked_profile / fm.STYLE_PROFILE).write_text(json.dumps({"profile": {"means": [9]}}))
    with pytest.raises(SystemExit, match="is not the locked"):
        fm.cmd_prep(_prep_args(tmp_path, locked_profile))


def test_the_real_lock_is_6b43d6ed5372():
    dec = json.loads((ROOT / "configs" / "data_decisions.json").read_text(encoding="utf-8"))
    assert dec["style_profile"]["profile_sha256"].startswith("6b43d6ed5372")
    assert fm.STYLE_PROFILE_SHA12 == "6b43d6ed5372"


def test_prep_refuses_a_bank_that_differs_from_the_handoff(tmp_path, locked_profile):
    other = _bank(tmp_path / "other")
    (other / "items.jsonl").write_text('{"other": 1}\n')
    prep = _prep_args(tmp_path, locked_profile, expected_bank_zip=_zip(other, tmp_path / "h.zip"))
    assert fm.cmd_prep(prep) == 1
    assert not prep.out.exists()


# ------------------------------------------------------------------ weekly budget preflight
def _client(tmp_path, spent_by_ts):
    from adr.harness.llm_client import LLMClient

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "_spend_ledger.jsonl").write_text("".join(
        json.dumps({"ts": ts, "model": "gpt-mini", "in": 1, "out": 1, "usd": usd}) + "\n"
        for ts, usd in spent_by_ts))
    return LLMClient(models={}, cache_dir=cache, weekly_spend_cap_usd=25.0)


def test_budget_refuses_when_the_week_has_less_than_the_two_caps_left(tmp_path, capsys):
    client = _client(tmp_path, [("2026-11-19T20:00:00+00:00", 8.57),
                                ("2026-11-21T18:00:00+00:00", 6.32)])
    assert sum(fm.SPEND_CAPS_USD.values()) == 10.0  # $8 (from $12, Nov 26) + $2 (from $10, Nov 25)
    need = 22.0  # the caps when the freeze was first planned (Nov 21)
    # Sun Nov 22 16:59 Arizona = 23:59 UTC: still ISO week 47, $14.89 spent, $10.11 left
    assert fm.budget_check(client, need, "2026-11-22T23:59:00+00:00") == 1
    out = capsys.readouterr()
    assert "$14.89 of the $25.00 weekly cap, $10.11 left" in out.out
    assert "NOT STARTING" in out.err and "17:00 Arizona time" in out.err
    # Sun Nov 22 17:00 Arizona = 00:00 UTC Monday: a new ISO week with the full $25
    assert fm.budget_check(client, need, "2026-11-23T00:00:00+00:00") == 0
    assert "2026-W48 (UTC): ledger spend $0.00" in capsys.readouterr().out


def test_budget_passes_after_the_fourth_halt_with_the_lowered_caps(tmp_path, capsys):
    # week 48 after the fourth halt: $11.25 spent, $13.75 left; $12 + $2 needed $14
    client = _client(tmp_path, [("2026-11-26T03:42:00+00:00", 11.25)])
    assert fm.budget_check(client, sum(fm.SPEND_CAPS_USD.values()), "2026-11-26T05:00:00+00:00") == 0
    assert "$13.75 left" in capsys.readouterr().out
    assert fm.budget_check(client, 14.0, "2026-11-26T05:00:00+00:00") == 1



# ------------------------------------------------------------------ post-run check of one run
def _run_dir(tmp_path, n_rows=8000, refused=0, errors=0, truncated=0):
    d = tmp_path / "runs" / "v1-final-du-L2-quiz-arc-200-claude-haiku-s20"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({
        "finished_at": "2026-11-26T03:42:19+00:00", "n_rows": n_rows,
        "n_truncated_replies": truncated, "stopped_for_budget": False,
        "n_refused_replies": 1 if refused else 0, "n_refused_questions": refused}))
    (d / "errors.jsonl").write_text("".join('{"error": "x"}\n' for _ in range(errors)))
    return d


def test_check_run_passes_refusals_with_no_limit(tmp_path):
    ok, msg = fm.check_run(_run_dir(tmp_path, refused=20))  # run 10: 0.25%
    assert ok and "8000 rows, 20 refused" in msg
    ok, msg = fm.check_run(_run_dir(tmp_path / "b", n_rows=380, refused=1620))  # run 16: 81%
    assert ok and "380 rows, 1620 refused" in msg


def test_check_run_still_fails_a_cut_off_or_a_call_error(tmp_path):
    ok, msg = fm.check_run(_run_dir(tmp_path, truncated=1, errors=1))
    assert not ok and "'truncated': 1" in msg
    ok, msg = fm.check_run(_run_dir(tmp_path / "b", errors=1, refused=20))
    assert not ok and "'errors': 1" in msg


@pytest.mark.skipif(sys.platform == "win32", reason="bash driver")
def test_freeze_driver_dry_run_plans_42_attacker_runs_and_8_router_replays(tmp_path):
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
    assert out.count("run_real_router.py --final") == 8  # 4 main + 4 pre-registered ablations
    routers = [ln for ln in out.splitlines() if "run_real_router.py --final" in ln]
    rids = [re.search(r"--run-id (\S+)", ln).group(1) for ln in routers]
    abl = [ln for ln in routers if "--clear-requires-style-human" in ln]
    assert len(abl) == 4 and all(re.search(r"--run-id \S+-style-clear\b", ln) for ln in abl)
    assert {r + "-style-clear" for r in rids if not r.endswith("-style-clear")} == {
        r for r in rids if r.endswith("-style-clear")}  # every main run has its partner
    for ln in routers:
        finals = set(re.findall(r"results/outcomes/(v1-final-\S+)\.parquet", ln))
        if "unseen-model" in ln:  # claude-haiku on the training banks only
            assert len(finals) == 30
            assert all("claude-haiku" in f and "osmi" not in f for f in finals)
        else:  # both models on the held-out OSMI bank only
            assert "unseen-survey" in ln and len(finals) == 12
            assert all("osmi-heldout" in f for f in finals)
    assert "freeze_report.py --out results/freeze-2026-11-26b/README.md" in out
    assert "--unseen-model-ablation results/real-router/freeze-2026-11-26b-unseen-model-all-style-clear" in out
    assert "--reference-no-low-effort results/real-router/rq2-l012-floor750-reopen-no-low-effort" in out
    assert "--jobs 8 --resume" in out and "--exclude-sessions" in out
    assert out.count("rescore --final") == 30  # every survey / written-answer run is re-scored
    assert "freeze-2026-11-26b-claude-haiku (cap $8)" in out
    assert "freeze-2026-11-26b-gpt-mini (cap $2)" in out
    assert list(tmp_path.iterdir()) == []  # a dry run creates no folders
    cp = [ln for ln in out.splitlines() if "--run-id v1-final-cp-" in ln and "DRY:" in ln]
    du = [ln for ln in out.splitlines() if "--run-id v1-final-du-" in ln and "DRY:" in ln]
    assert len(cp) == len(du) == 21
    for ln in cp:  # copy-paste: 4096 for claude-haiku only; gpt-mini keeps the training limits
        assert ("--max-tokens 4096" in ln) == ("claude-haiku" in ln)
    for ln in du:  # document upload keeps the training limits for both models
        assert "--max-tokens 4096" not in ln
        # claude-haiku: 8192 on every bank, incl. ASURRE and OSMI written answers;
        # gpt-mini keeps the training's (no override on written-answer banks)
        assert ("--max-tokens 8192" in ln) == ("claude-haiku" in ln)
        if "gpt-mini" in ln and "--bank assignment-" in ln:
            assert "--max-tokens" not in ln
    for line in out.splitlines():  # every attacker call is a --final call at temperature 0.7
        if "--run-id v1-final" in line:
            assert "--final" in line and "--temperature 0.7" in line and "--samples 20" in line


# ------------------------------------------------------------------ freeze_report.py
SRC = {"checks": 4.0, "same_type": 4.0, "fallback": 0.0, "synthetic": 0.0, "missing": 0.0}


def _router(kind="unseen-survey", thr=(0.008, 0.951), n_by=None, src_by=None, n=44):
    """A complete metrics.json router block for `kind`: every expected group in both session
    tables with 20 sessions and clean sources. `n_by` / `src_by` override one table per group
    (None drops the group from that table; a dict for src_by updates its source fields)."""
    groups = sorted(fr.EXPECTED_GROUPS[kind])
    n_tab = {g: 20 for g in groups}
    src_tab = {g: {**SRC, "sessions": 20} for g in groups}
    if kind == "unseen-survey":  # OSMI: only canary has rows, the other checks are missing
        for g in groups:
            src_tab[g].update(missing=3.0)
    for g, c in (n_by or {}).items():
        if c is None:
            n_tab.pop(g, None)
        else:
            n_tab[g] = c
    for g, v in (src_by or {}).items():
        if v is None:
            src_tab.pop(g, None)
        else:
            src_tab[g] = {**src_tab.get(g, {**SRC, "sessions": 20}), **v}
    shown = set(n_tab) | set(src_tab)
    res = {t: {"honest_pause_rate": 1 / n, "honest_cleared_rate": 36 / n,
               "checks_per_honest_session": 6.4,
               **{f"tpr[{g}]": 0.5 for g in shown},
               **{f"ai_cleared_rate[{g}]": 0.6 for g in shown}}
           for t in fr.THRESHOLDS}
    ci = {t: {f"tpr[{g}]": [0.3, 0.7] for g in shown} for t in fr.THRESHOLDS}
    return {"thresholds": dict(zip(fr.THRESHOLDS, thr, strict=True)), "results": res, "ci95": ci,
            "n_sessions": n, "ai_sources_by_group": src_tab, "n_ai_sessions_by_group": n_tab}


def _write(run_dir: Path, router: dict, ablation: bool | None = None) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"results": {"router": router}}))
    flag = run_dir.name.endswith(fr.ABLATION_SUFFIX) if ablation is None else ablation
    (run_dir / "config.yaml").write_text(
        f"router_config:\n  clear_requires_style_human: {str(flag).lower()}\n")
    return run_dir


def _groups(kind, router):
    return fr.group_problems(router, fr.EXPECTED_GROUPS[kind])


def test_expected_group_matrix():
    assert len(fr.EXPECTED_GROUPS["unseen-model"]) == 6  # claude-haiku only, by design
    assert all(g.startswith("claude-haiku:") for g in fr.EXPECTED_GROUPS["unseen-model"])
    assert len(fr.EXPECTED_GROUPS["unseen-survey"]) == 12
    assert {g.split(":")[0] for g in fr.EXPECTED_GROUPS["unseen-survey"]} == {"gpt-mini",
                                                                               "claude-haiku"}


@pytest.mark.parametrize("kind", ["unseen-model", "unseen-survey"])
def test_complete_matrix_passes_every_check(kind):
    router = _router(kind)
    assert _groups(kind, router) == []
    assert fr.source_problems(router, kind) == []


def test_no_groups_at_all_fails():
    router = {**_router("unseen-model"), "n_ai_sessions_by_group": {}, "ai_sources_by_group": {}}
    assert _groups("unseen-model", router) == [
        "no AI groups at all (expected 6 groups of 20 sessions)"]


G = "claude-haiku:doc_upload:L0"


@pytest.mark.parametrize(
    "kind, n_by, src_by, message",
    [
        # missing from both tables / from one table only
        ("unseen-survey", {"gpt-mini:doc_upload:L2g": None}, {"gpt-mini:doc_upload:L2g": None},
         [("missing group gpt-mini:doc_upload:L2g: not in n_ai_sessions_by_group (expected 20 "
          "sessions)"), ("missing group gpt-mini:doc_upload:L2g: not in ai_sources_by_group "
          "(expected 20 sessions)")]),
        ("unseen-model", None, {G: None},
         [f"missing group {G}: not in ai_sources_by_group (expected 20 sessions)"]),
        # short and over-full, in both tables
        ("unseen-model", {"claude-haiku:copy_paste:L2": 19}, {"claude-haiku:copy_paste:L2":
                                                             {"sessions": 19}},
         [("group claude-haiku:copy_paste:L2 has 19 sessions in n_ai_sessions_by_group, expected "
          "exactly 20"), ("group claude-haiku:copy_paste:L2 has 19 sessions in ai_sources_by_group, "
          "expected exactly 20")]),
        ("unseen-model", {"claude-haiku:copy_paste:L0": 7}, None,  # Harshini's n=7 case
         [("group claude-haiku:copy_paste:L0 has 7 sessions in n_ai_sessions_by_group, expected "
          "exactly 20")]),
        ("unseen-survey", {G: 21}, {G: {"sessions": 21}},
         [f"group {G} has 21 sessions in n_ai_sessions_by_group, expected exactly 20",
          f"group {G} has 21 sessions in ai_sources_by_group, expected exactly 20"]),
        # Archana's case: the two tables disagree (20 vs 19)
        ("unseen-model", None, {G: {"sessions": 19}},
         [f"group {G} has 19 sessions in ai_sources_by_group, expected exactly 20"]),
        ("unseen-model", {G: 19}, None,
         [f"group {G} has 19 sessions in n_ai_sessions_by_group, expected exactly 20"]),
        # unexpected model / unexpected group
        ("unseen-model", {"gpt-mini:copy_paste:L0": 20}, {"gpt-mini:copy_paste:L0": {}},
         [("unexpected model gpt-mini: group gpt-mini:copy_paste:L0 (n_ai_sessions_by_group 20, "
          "ai_sources_by_group 20 sessions) is not in this report's matrix")]),
        ("unseen-survey", {"claude-haiku:copy_paste:L1": 20}, None,
         [("unexpected group claude-haiku:copy_paste:L1 (n_ai_sessions_by_group 20 sessions) is "
          "not in this report's matrix")]),
    ],
)
def test_missing_short_overfull_disagreeing_or_unexpected_groups_fail(kind, n_by, src_by, message):
    assert _groups(kind, _router(kind, n_by=n_by, src_by=src_by)) == message


@pytest.mark.parametrize("field", ["synthetic", "fallback"])
@pytest.mark.parametrize("kind, model", [("unseen-model", "claude-haiku"),
                                         ("unseen-survey", "claude-haiku"),
                                         ("unseen-survey", "gpt-mini")])  # gpt-mini OSMI too
def test_synthetic_or_fallback_fails_for_every_group(kind, field, model):
    g = f"{model}:copy_paste:L2g"
    problems = fr.source_problems(_router(kind, src_by={g: {field: 0.5}}), kind)
    assert len(problems) == 1 and problems[0].startswith(f"{g}: synthetic")


def test_missing_checks_fail_in_unseen_model_only():
    g = "claude-haiku:copy_paste:L0"
    problems = fr.source_problems(_router("unseen-model", src_by={g: {"missing": 1.0}}),
                                  "unseen-model")
    assert problems == [(f"{g}: missing 1.0 checks per session with no per-session record "
                        "(must be 0, or explained by refusals)")]
    assert fr.source_problems(_router("unseen-survey"), "unseen-survey") == []  # missing 3.0 ok


def test_missing_checks_pass_when_every_session_has_a_recorded_refusal():
    g = "claude-haiku:doc_upload:L2"
    router = _router("unseen-model", src_by={g: {"missing": 0.1}})
    router["ai_missing_by_session"] = {g: {f"composite:{g}:3": 1.0, f"composite:{g}:16": 1.0}}
    assert fr.source_problems(router, "unseen-model", {g: {3, 16, 7}}) == []
    lines = "\n".join(fr.missing_lines(router, {g: {3, 16, 7}}))
    assert f"| {g} | 2 | 2 | 1.00 |" in lines


def test_unexplained_missing_checks_still_fail():
    g = "claude-haiku:doc_upload:L2"
    router = _router("unseen-model", src_by={g: {"missing": 0.1}})
    router["ai_missing_by_session"] = {g: {f"composite:{g}:3": 1.0, f"composite:{g}:5": 1.0}}
    problems = fr.source_problems(router, "unseen-model", {g: {3}})  # sample 5 refused nothing
    assert len(problems) == 1 and "1 of 2 sessions with missing checks" in problems[0]
    # a refusal in another group does not explain this one
    assert fr.source_problems(router, "unseen-model",
                              {"claude-haiku:doc_upload:L2g": {3, 5}}) != []
    # synthetic and fallback stay strict even with refusals
    router["ai_sources_by_group"][g]["fallback"] = 0.5
    assert any("fallback 0.5" in p for p in fr.source_problems(router, "unseen-model", {g: {3, 5}}))


def test_threshold_check_against_51():
    assert fr.threshold_problems(_router(), _router()) == []
    assert len(fr.threshold_problems(_router(thr=(0.008, 0.9)), _router())) == 1


def test_reference_is_picked_by_honest_sessions(tmp_path):
    ref_all, ref_no = tmp_path / "all", tmp_path / "no"
    assert fr.reference_for(Path("x-unseen-model-all"), ref_all, ref_no) == ref_all
    assert fr.reference_for(Path("x-unseen-model-no-low-effort"), ref_all, ref_no) == ref_no


def _report(tmp_path, thr=(0.008, 0.951), model_src=None, survey_n=None, abl_n=None,
            abl_flag=None, abl_names=None):
    """The 4 main runs and their 4 -style-clear partners; abl_n overrides the unseen-model
    ablation partners' n_ai_sessions_by_group; abl_flag their recorded flag."""
    runs, abls = {}, {}
    for kind in ("unseen-model", "unseen-survey"):
        for who in ("all", "no-low-effort"):
            name = f"{kind}-{who}"
            r = (_router(kind, thr, src_by=model_src) if kind == "unseen-model"
                 else _router(kind, thr, n_by=survey_n))
            runs[name] = _write(tmp_path / name, r)
            ra = _router(kind, thr, n_by=abl_n if kind == "unseen-model" else None)
            ra["results"]["A_0.05"]["ai_cleared_rate[claude-haiku:copy_paste:L2]"] = 0.35
            abl_name = (abl_names or {}).get(name, name + fr.ABLATION_SUFFIX)
            abls[name] = _write(tmp_path / abl_name, ra, abl_flag)
    ref_all = _write(tmp_path / "ref-all", _router())
    ref_no = _write(tmp_path / "ref-no", _router())
    out = tmp_path / "README.md"
    m, s_ = ("unseen-model-all", "unseen-model-no-low-effort"), ("unseen-survey-all",
                                                                 "unseen-survey-no-low-effort")
    cmd = [sys.executable, str(ROOT / "scripts" / "freeze_report.py"), "--out", str(out),
           "--unseen-model", *(str(runs[k]) for k in m),
           "--unseen-survey", *(str(runs[k]) for k in s_),
           "--unseen-model-ablation", *(str(abls[k]) for k in m),
           "--unseen-survey-ablation", *(str(abls[k]) for k in s_),
           "--reference-all", str(ref_all), "--reference-no-low-effort", str(ref_no)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    return proc, out.read_text(encoding="utf-8")


def test_report_writes_the_fpr_rule_both_reports_and_the_wording(tmp_path):
    proc, readme = _report(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert fr.FPR_RULE in readme
    model_part, survey_part = readme.split("## Unseen survey")
    model_part = model_part.split("## Unseen model")[1]
    assert "gpt-mini |" not in model_part  # claude-haiku only
    assert '"Unseen model" means an unseen model, not unseen items' in model_part
    assert "canary and style only and is not comparable to the headline" in survey_part
    assert "Style is passive" in readme and "not in the source table" in readme
    assert "| gpt-mini | L0 | copy-paste | 20 | 0.50 [0.30, 0.70] | 0.50 [0.30, 0.70]" in survey_part
    assert "FAILED" not in readme


def test_report_fails_loudly_on_borrowed_outcomes(tmp_path):
    proc, readme = _report(tmp_path, model_src={"claude-haiku:copy_paste:L0": {"fallback": 0.25}})
    assert proc.returncode == 1
    assert "POST-RUN CHECK FAILED" in proc.stderr and "claude-haiku:copy_paste:L0" in proc.stderr
    assert "- FAILED: unseen-model-all: claude-haiku:copy_paste:L0: synthetic 0.0, fallback 0.25" in readme


def test_report_fails_loudly_on_a_short_group_naming_run_group_and_count(tmp_path):
    proc, readme = _report(tmp_path, survey_n={"gpt-mini:doc_upload:L2": 19})
    assert proc.returncode == 1
    msg = ("unseen-survey-no-low-effort: group gpt-mini:doc_upload:L2 has 19 sessions in "
           "n_ai_sessions_by_group, expected exactly 20")
    assert f"POST-RUN CHECK FAILED: {msg}" in proc.stderr
    assert f"- FAILED: {msg}" in readme


def test_report_fails_when_a_threshold_differs_from_51(tmp_path):
    proc, _ = _report(tmp_path, thr=(0.01, 0.951))
    assert proc.returncode == 1 and "threshold differs from #51" in proc.stderr


# ------------------------------------------------------------------ pre-registered ablation
def test_report_shows_the_ablation_beside_the_main_policy(tmp_path):
    proc, readme = _report(tmp_path)
    assert proc.returncode == 0, proc.stderr
    model_part, survey_part = readme.split("## Unseen survey")
    for part in (model_part, survey_part):
        assert part.count("#### Pre-registered ablation:") == 2  # all and no-low-effort
        assert "stays the headline" in part
        assert "| **primary:** L2 wrongly cleared, claude-haiku copy-paste | A_0.05" in part
        assert "| cost: honest sessions cleared |" in part
        assert "| cost: checks per honest person |" in part
        assert "| guardrail: honest false pauses [exact 95% CI] |" in part
    assert ("| **primary:** L2 wrongly cleared, claude-haiku copy-paste | A_0.05 (0.008 / 0.008) | "
            "12 of 20 (0.60) | 7 of 20 (0.35) |") in model_part
    assert "L2 wrongly cleared, gpt-mini" in survey_part  # both models on OSMI
    assert "L2 wrongly cleared, gpt-mini" not in model_part  # claude-haiku only


def test_ablation_partners_get_the_same_group_checks(tmp_path):
    proc, readme = _report(tmp_path, abl_n={"claude-haiku:doc_upload:L2g": 19})
    assert proc.returncode == 1
    msg = ("unseen-model-all-style-clear: group claude-haiku:doc_upload:L2g has 19 sessions in "
           "n_ai_sessions_by_group, expected exactly 20")
    assert f"POST-RUN CHECK FAILED: {msg}" in proc.stderr and f"- FAILED: {msg}" in readme


def test_ablation_partner_must_record_the_flag(tmp_path):
    proc, _ = _report(tmp_path, abl_flag=False)
    assert proc.returncode == 1
    assert "-style-clear: clear_requires_style_human is False, expected True" in proc.stderr


def test_ablation_partners_must_pair_with_the_main_runs(tmp_path):
    proc, _ = _report(tmp_path, abl_names={"unseen-survey-all": "something-else"})
    assert proc.returncode == 1 and "do not pair with" in proc.stderr


@pytest.mark.parametrize("k, lo, hi", [(3, 1.4, 18.7), (2, 0.6, 15.5), (5, 3.8, 24.6),
                                       (0, 0.0, 8.0), (44, 92.0, 100.0)])
def test_clopper_pearson_matches_the_published_51_intervals(k, lo, hi):
    """The exact intervals docs/notes/rq2_calibration.md reports for 44 honest sessions."""
    got = fr.clopper_pearson(k, 44)
    assert (round(100 * got[0], 1), round(100 * got[1], 1)) == (lo, hi)


def test_ablation_guardrail_has_exact_cis_and_secondary_tpr_is_side_by_side(tmp_path):
    proc, readme = _report(tmp_path)
    assert proc.returncode == 0, proc.stderr
    model_part, survey_part = readme.split("## Unseen survey")
    # fixture: honest_pause_rate 1/44 in every run
    assert ("| guardrail: honest false pauses [exact 95% CI] | A_0.05 (0.008 / 0.008) | "
            "1 of 44 = 2.3% [0.1%, 12.0%] | 1 of 44 = 2.3% [0.1%, 12.0%] |") in model_part
    assert model_part.count("Secondary outcome: per-group TPR") == 2
    assert ("| claude-haiku | L2 | copy-paste | 0.50 [0.30, 0.70] / 0.50 [0.30, 0.70] | "
            "0.50 [0.30, 0.70] / 0.50 [0.30, 0.70] |") in model_part
    assert survey_part.count("canary + style evidence only") == 2  # beside each ablation
    assert "canary + style evidence only" not in model_part


def test_report_lists_refused_questions_per_run_and_per_group(tmp_path):
    root = tmp_path / "results"
    runs = {"v1-final-du-L2-quiz-arc-200-claude-haiku-s20": 20,
            "v1-final-du-L2-quiz-mmlu-200-claude-haiku-s20": 0,
            "v1-final-cp-L2-generic-quiz-arc-200-claude-haiku-s20": None}  # no count recorded
    for rid, n in runs.items():
        (root / "runs" / rid).mkdir(parents=True)
        (root / "runs" / rid / "config.yaml").write_text(
            yaml.safe_dump({} if n is None else {"n_refused_questions": n}))
    router = tmp_path / "router"
    router.mkdir()
    (router / "inputs.txt").write_text("ai_outcomes:\n" + "".join(
        f"  {root / 'outcomes' / (rid + '.parquet')}\n" for rid in runs)
        + f"  {root / 'outcomes' / 'v1-cp-L0-quiz-mmlu-200-s5.parquet'}\n")  # training: ignored
    assert fr.refusals(router) == {"v1-final-du-L2-quiz-arc-200-claude-haiku-s20": 20,
                                   "v1-final-du-L2-quiz-mmlu-200-claude-haiku-s20": 0,
                                   "v1-final-cp-L2-generic-quiz-arc-200-claude-haiku-s20": 0}
    text = "\n".join(fr.refusal_lines(router))
    assert "20 in all" in text
    assert "| claude-haiku:doc_upload:L2 | 20 |" in text
    assert "| claude-haiku:copy_paste:L2g | 0 |" in text
    assert "| v1-final-du-L2-quiz-arc-200-claude-haiku-s20 | 20 |" in text


def test_replay_records_missing_checks_per_session():
    from types import SimpleNamespace

    from adr.replay.crossfit import ai_missing_by_session

    def trace(who, missing):
        return SimpleNamespace(who=who, sources=(0, 0, 0, missing))

    g = "claude-haiku:doc_upload:L2"
    ai = [[trace(f"ai:{g}|composite:{g}:16", 2), trace(f"ai:{g}|composite:{g}:16", 0)],
          [trace(f"ai:{g}|composite:{g}:3", 0)]]
    assert ai_missing_by_session(ai) == {g: {f"composite:{g}:16": 1.0}}


@pytest.mark.skipif(sys.platform == "win32", reason="bash driver")
def test_router_only_dry_run_runs_no_attacker_and_all_router_replays(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "ROUTER_ONLY": "1", "PY": "python"}
    proc = subprocess.run(["bash", str(ROOT / "scripts" / "freeze_run.sh")], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    out = proc.stdout
    assert out.count("router-only: needs finished v1-final-") == 42
    assert "run_copy_paste.py" not in out and "run_doc_upload.py" not in out
    assert "run_survey_banks.py" not in out                  # no attacker call, no re-score
    assert out.count("scripts/run_real_router.py") == 8      # every router replay
    assert "freeze_report.py --out" in out
    assert list(tmp_path.iterdir()) == []


def test_router_only_skips_the_budget_check_but_keeps_tag_and_manifest_checks():
    text = (ROOT / "scripts" / "freeze_run.sh").read_text(encoding="utf-8")
    pre = text[text.index("# ------------------------------------------------------------------ 0. preflight"):
               text.index("# ------------------------------------------------------------------ (a)")]
    assert 'router-only restart, no API calls: weekly budget check skipped' in pre
    assert pre.index('HEAD is not the $FREEZE tag') < pre.index('ROUTER_ONLY')
    assert "freeze_manifest.py verify" in pre and 'no $MANIFEST' in pre
    assert 'ADR_JOB_SPEND_CAP_USD=0' in text

"""scripts/freeze_manifest.py and scripts/freeze_run.sh (dry run only: nothing is run)."""

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
def test_freeze_driver_dry_run_plans_42_attacker_runs_and_2_router_runs(tmp_path):
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
    assert out.count("run_real_router.py --final") == 2
    assert "--jobs 8 --resume" in out and "--exclude-sessions" in out
    assert out.count("rescore --final") == 30  # every survey / written-answer run is re-scored
    assert "freeze-2026-11-22-claude-haiku (cap $10)" in out
    assert "freeze-2026-11-22-gpt-mini (cap $10)" in out
    for line in out.splitlines():  # every attacker call is a --final call at temperature 0.7
        if "--run-id v1-final" in line:
            assert "--final" in line and "--temperature 0.7" in line and "--samples 20" in line

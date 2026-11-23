"""The held-out guard (adr/harness/final_guard.py) and the drivers' --final mode. No API calls: the
subprocess tests run with every provider key removed from the environment."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from adr.contracts import Assessment, Item, RenderedItem
from adr.harness.final_guard import HELDOUT_SOURCE, check_final

ROOT = Path(__file__).resolve().parents[1]


class _Client:
    def __init__(self, heldout: set[str]):
        self.heldout = heldout

    def spec(self, model):
        return SimpleNamespace(heldout=model in self.heldout)


def _item(iid, source="asurre_nature_phd_2019", split="train"):
    return Item(item_id=iid, source=source, assessment_type="survey", item_type="open_text",
                stem="Question?", eligible_mechanisms=["canary"], split=split)


TRAIN = {"a:1": _item("a:1")}
OSMI = {"o:1": _item("o:1", source=HELDOUT_SOURCE, split="heldout_survey")}
HELD_SPLIT = {"h:1": _item("h:1", split="heldout_survey")}


def test_training_runs_pass_without_final():
    check_final(_Client(set()), ["gpt-mini"], TRAIN, ["a:1"], final=False, run_id="v1-cp-L0-x")


@pytest.mark.parametrize(
    "models, items, what",
    [
        (["claude-haiku"], TRAIN, "held-out model"),
        (["gpt-mini"], OSMI, "held-out survey"),
        (["gpt-mini"], HELD_SPLIT, "held-out survey"),
    ],
)
def test_held_out_model_or_survey_needs_final(models, items, what):
    with pytest.raises(SystemExit, match=what):
        check_final(_Client({"claude-haiku"}), models, items, list(items), False, "v1-final-x")


def test_final_runs_need_final_in_the_run_id():
    client = _Client({"claude-haiku"})
    with pytest.raises(SystemExit, match="final"):
        check_final(client, ["claude-haiku"], TRAIN, ["a:1"], True, "v1-cp-L0-quiz")
    check_final(client, ["claude-haiku"], OSMI, ["o:1"], True, "v1-final-cp-L0-osmi")


# ------------------------------------------------------------------- run_survey_banks --final
def _heldout_bank(root: Path, items: dict[str, Item], source=HELDOUT_SOURCE):
    """A synthetic bank laid out like scripts/build_runs_v1.py --heldout-asurre-only output."""
    from scripts import run_survey_banks as rsb

    bank = rsb.HELDOUT_BANK
    (root / "assessments").mkdir(parents=True)
    (root / "variants").mkdir()
    a = Assessment(assessment_id=bank, title="t", assessment_type="survey", item_ids=list(items))
    (root / "assessments" / f"{bank}.json").write_text(a.model_dump_json())
    (root / "items.jsonl").write_text("\n".join(i.model_dump_json() for i in items.values()))
    vs = [RenderedItem(item_id=i, mechanism=None, variant_id=f"{i}|plain|0", visible={"stem": "s"})
          for i in items]
    (root / "variants" / f"{bank}.jsonl").write_text("\n".join(v.model_dump_json() for v in vs))
    (root / "manifest.json").write_text(json.dumps({
        "heldout_asurre_source": source,
        "assessments": [{"assessment_id": bank, "n_items": len(items), "n_variants": len(vs),
                         "assessment": r"data\processed\osmi-heldout-final\x.json",
                         "variants": r"data\processed\osmi-heldout-final\x.jsonl"}],
    }))
    return bank


def test_survey_driver_reads_the_held_out_bank_only_with_final(tmp_path, monkeypatch):
    from scripts import run_survey_banks as rsb

    bank = _heldout_bank(tmp_path, OSMI)
    monkeypatch.setattr(rsb, "HELDOUT_DATA", tmp_path)
    with pytest.raises(SystemExit, match="held-out survey"):
        rsb.load(bank)
    assessment, _items, variants = rsb.load(bank, final=True)  # Windows manifest paths ignored
    assert assessment.item_ids == ["o:1"] and set(variants) == {"o:1"}
    assert bank not in rsb.TRAINING_BANKS  # check / estimate / default rescore never touch it


def test_held_out_bank_must_hold_only_osmi_items(tmp_path, monkeypatch):
    from scripts import run_survey_banks as rsb

    bank = _heldout_bank(tmp_path, {**OSMI, **TRAIN})
    monkeypatch.setattr(rsb, "HELDOUT_DATA", tmp_path)
    with pytest.raises(SystemExit, match="item in the held-out bank"):
        rsb.load(bank, final=True)


def test_survey_driver_fake_client_marks_claude_haiku_held_out():
    from scripts import run_survey_banks as rsb

    assert rsb.client_for(True, "claude-haiku").spec("claude-haiku").heldout
    assert not rsb.client_for(True, "gpt-mini").spec("gpt-mini").heldout


# ------------------------------------------------------------------- quiz drivers
def _no_keys_env():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONPATH"] = str(ROOT)
    return env


@pytest.mark.parametrize("script", ["run_copy_paste.py", "run_doc_upload.py"])
def test_quiz_drivers_refuse_the_held_out_model_without_final(script, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--stub", "--models", "claude-haiku",
         "--run-id", "guard-test"],
        cwd=ROOT, env=_no_keys_env(), capture_output=True, text=True, encoding="utf-8",
        check=False,
    )
    assert proc.returncode != 0
    assert "without --final" in proc.stderr
    assert not (ROOT / "results" / "runs" / "guard-test").exists()

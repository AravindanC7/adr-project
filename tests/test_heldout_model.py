"""The held-out model is the same everywhere: configs/models.yaml and configs/data_decisions.json.

Team decision, Nov 17 2026: claude-haiku replaces gemini-flash (Gemini needs a $50 minimum
prepay). The held-out model never contributes to the failure map, thresholds or tuning, and runs
only in the final --final evaluation (AGENTS.md section 8).
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _heldout(path: str) -> list[dict]:
    cfg = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return [m for m in cfg["models"] if m.get("heldout")]


def test_exactly_one_heldout_model_and_it_matches_the_recorded_decision():
    decision = json.loads((ROOT / "configs/data_decisions.json").read_text(encoding="utf-8"))
    decision = decision["heldout_model"]
    heldout = _heldout("configs/models.yaml")
    assert [m["name"] for m in heldout] == [decision["name"]] == ["claude-haiku"]
    assert heldout[0]["model_id"] == decision["model_id"]
    assert decision["replaces"] == "gemini-flash" and decision["decided"] == "2026-11-17"


def test_gemini_flash_is_no_longer_held_out():
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))
    gemini = next(m for m in cfg["models"] if m["name"] == "gemini-flash")
    assert gemini["heldout"] is False


def test_example_config_marks_the_same_model_family_as_heldout():
    heldout = _heldout("configs/models.example.yaml")
    assert len(heldout) == 1 and heldout[0]["model_id"] == "claude-haiku-5-5"

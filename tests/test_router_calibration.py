"""The calibrated router (step #42) recorded in configs/data_decisions.json["router_calibration"]
is what scripts/run_real_router.py and RouterConfig run by default."""

import json
import sys
from pathlib import Path

import pytest

from adr.router.core import RouterConfig, SPRTRouter
from adr.router.sprt import wald_thresholds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_real_router

CAL = json.loads((ROOT / "configs" / "data_decisions.json").read_text(encoding="utf-8"))[
    "router_calibration"
]


def _defaults():
    return run_real_router.build_parser().parse_args(["--ai-outcomes", "x", "--pilot-outcomes", "y"])


def test_script_defaults_match_the_calibration_decision():
    args = _defaults()
    assert args.cap_variant[0] == CAL["cap_design"] == "per_mech"
    assert args.clear_policy == CAL["clear_policy"] == "reopen"
    assert args.neg_floor_frac == CAL["neg_floor_frac"] == 1.0
    assert run_real_router.CAP_VARIANTS[CAL["cap_design"]] == {}  # per_mech = RouterConfig as is


def test_router_defaults_match_the_calibration_decision():
    cfg = RouterConfig()
    assert cfg.mech_cap_frac == CAL["mech_cap_frac"]
    assert cfg.mech_group_caps is None  # per_mech: every mechanism has its own cap
    assert cfg.clear_policy == CAL["clear_policy"]
    assert cfg.neg_floor_frac == CAL["neg_floor_frac"]
    assert (cfg.alpha, cfg.beta) == (CAL["wald"]["alpha"], CAL["wald"]["beta"])
    a, b = wald_thresholds(cfg.alpha, cfg.beta)
    assert a == pytest.approx(CAL["wald"]["A"], abs=1e-4)
    assert b == pytest.approx(CAL["wald"]["B"], abs=1e-4)


def test_threshold_rule_matches_the_script():
    assert run_real_router.HEADLINE == CAL["headline_threshold"] == "A_0.05"
    assert CAL["secondary_threshold"] == "A_0.01"
    assert run_real_router.ORDER[:2] == ("A_0.05", "A_0.01")  # headline first, then secondary


def test_evidence_bounds_follow_from_the_recorded_fractions():
    router = SPRTRouter({}, {}, None, RouterConfig())  # type: ignore[arg-type]
    assert router.mech_cap == pytest.approx(CAL["mech_cap_frac"] * CAL["wald"]["A"], abs=1e-3)
    assert router.mech_floor == pytest.approx(CAL["neg_floor_frac"] * CAL["wald"]["B"], abs=1e-3)
    assert router.mech_cap < router.A  # one check alone cannot reach the NOMINAL A (it can reach
    # a calibrated threshold below the cap, e.g. A_0.01 ~ 0.95: see rq2_calibration.md)
    assert router.mech_floor <= router.B  # one check alone can clear


def test_ablations_are_recorded_and_runnable():
    assert set(CAL["ablations"]) == {"no_cap", "twocheck"}
    assert run_real_router.CAP_VARIANTS["no_cap"] == {"mech_cap_frac": None}
    assert "no_cap" in run_real_router.ABLATIONS
    # twocheck = --neg-floor-frac 0.65; the option itself is tested in tests/test_mech_floor.py
    assert "--neg-floor-frac" in run_real_router.build_parser().format_help()


def test_preregistered_style_clearing_ablation_is_recorded_and_off_by_default():
    """The pre-registered ablation is written down before the held-out run, and the frozen main
    policy (#51) does not use it: off in RouterConfig and in the script's defaults."""
    pre = json.loads((ROOT / "configs" / "data_decisions.json").read_text(encoding="utf-8"))[
        "preregistered_ablations"
    ]["clear_requires_style_human"]
    for key in ("rule", "status", "primary_outcome", "costs", "guardrail", "chosen_by",
                "registered", "offline_check"):
        assert pre[key], key
    assert "headline" in pre["status"] and "L2 wrongly cleared" in pre["primary_outcome"]
    assert "Harshini" in pre["chosen_by"] and "first clean test" in pre["chosen_by"]
    assert RouterConfig().clear_requires_style_human is False
    assert _defaults().clear_requires_style_human is False

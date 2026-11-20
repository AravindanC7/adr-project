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
    assert router.mech_cap < router.A  # one check alone cannot pause
    assert router.mech_floor <= router.B  # one check alone can clear


def test_ablations_are_recorded_and_runnable():
    assert set(CAL["ablations"]) == {"no_cap", "twocheck"}
    assert run_real_router.CAP_VARIANTS["no_cap"] == {"mech_cap_frac": None}
    assert "no_cap" in run_real_router.ABLATIONS
    # twocheck = --neg-floor-frac 0.65; the option itself is tested in tests/test_mech_floor.py
    assert "--neg-floor-frac" in run_real_router.build_parser().format_help()

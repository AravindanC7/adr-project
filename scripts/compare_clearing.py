"""What clearing costs, session by session: the router vs router_noclear on one real-router run.

    python scripts/compare_clearing.py --run results/real-router/<run_id>

Replays every AI evaluation session under each fold map with the run's own inputs, folds, split,
seed and router config (all read from the run's config.yaml), for "router" and "router_noclear"
(the same router with clearing off). At each threshold, each policy uses its OWN calibrated
threshold from the run's metrics.json. For every (AI session, fold map) pair it counts:

  router_clears                 the router clears it before reaching its threshold
                                (cleared = no evidence of AI found)
  cleared_and_noclear_misses    ... and router_noclear does not pause it either
  cleared_but_noclear_pauses    ... but router_noclear pauses it: detection lost to clearing
  noclear_pauses_router_misses  router_noclear pauses, the router does not
  router_pauses_noclear_misses  the router pauses, router_noclear does not

and, for the cleared-but-paused pairs, the AI group they are in. Replay only: no model calls, no
bootstrap. Prints counts only (no session ids, no item ids, no answer text).

The 72 / 480 result in docs/notes/rq2_calibration.md is
    python scripts/compare_clearing.py --run results/real-router/rq2-l012-floor750-reopen-all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_real_router as rrr

from adr.contracts import PASSIVE_MECHANISMS
from adr.replay.composite import add_attacker_variant, join_ai_sessions
from adr.replay.crossfit import ai_group, all_maps, router_factory, score_policy
from adr.replay.folds import assign_session_folds, fold_lists
from adr.replay.quick_map import split_rows
from adr.router.core import RouterConfig

POLICIES = ("router", "router_noclear")
THRESHOLDS = ("A_0.05", "A_0.01")


def compare(traces: dict[str, dict], thresholds: dict[str, dict[str, float]]) -> dict[str, Counter]:
    """traces[policy][ai session] = one Trace per fold map; thresholds[policy][name] = A."""
    out: dict[str, Counter] = {}
    for t in THRESHOLDS:
        a_r, a_n = thresholds["router"][t], thresholds["router_noclear"][t]
        c: Counter = Counter()
        for who, per_fold in traces["router"].items():
            for tr, tn in zip(per_fold, traces["router_noclear"][who], strict=True):
                det_r, det_n = tr.score >= a_r, tn.score >= a_n
                cleared = tr.ever_cleared_at(a_r)
                c["pairs"] += 1
                c["router_clears"] += cleared
                c["cleared_and_noclear_misses"] += cleared and not det_n
                c["cleared_but_noclear_pauses"] += cleared and det_n
                c["noclear_pauses_router_misses"] += det_n and not det_r
                c["router_pauses_noclear_misses"] += det_r and not det_n
                if cleared and det_n:
                    c[f"cost_in:{ai_group(per_fold)}"] += 1
        out[t] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a results/real-router/<run_id> folder")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["results"]

    ai = rrr.read_rows(cfg["ai_outcomes"])
    ai = ai[ai["respondent_kind"].eq("ai")].reset_index(drop=True)
    ai = join_ai_sessions(add_attacker_variant(ai)).rows
    rows = rrr.read_rows(cfg["pilot_outcomes"])
    human_all = rows[rows["respondent_kind"].eq("human") & rows["data_origin"].eq("pilot")]
    assignment = assign_session_folds(
        human_all["session_id"].unique(), cfg["fold_k"], cfg["fold_seed"], cfg["low_effort_sessions"]
    )
    human = human_all[~human_all["session_id"].isin(set(cfg["excluded_sessions"]))]
    mechs = rrr.usable_mechanisms(ai, human)
    a = ai[ai["mechanism"].isna() | ai["mechanism"].isin(mechs)]
    h = human[human["mechanism"].isna() | human["mechanism"].isin(mechs)]
    passive = tuple(m for m in mechs if m in PASSIVE_MECHANISMS)
    assessment, items, variants = rrr.load_bank(cfg["bank"], None, mechs)
    splits = split_rows(a, cfg["calibration_frac"], cfg["split_seed"])
    folds = fold_lists(assignment, cfg["fold_k"], keep=sorted(h["session_id"].unique()))
    maps = all_maps(splits.train, h, folds)
    rc = RouterConfig(**{k: v for k, v in cfg["router_config"].items() if k != "mech_group_caps"})

    traces = {}
    for pol in POLICIES:
        make = router_factory(pol, assessment, items, variants)
        scored = score_policy(
            make, rc, assessment, items, h, splits.calibration, folds, maps, 0, passive,
            cfg["replay_seed"],
        )
        traces[pol] = scored.ai
    thresholds = {p: metrics[p]["thresholds"] for p in POLICIES}
    print(f"run {cfg['run_id']}: {len(traces['router'])} AI sessions x {len(maps)} fold maps")
    for t, c in compare(traces, thresholds).items():
        print(f"  {t}: router A {thresholds['router'][t]:.3f}, router_noclear A "
              f"{thresholds['router_noclear'][t]:.3f} -> {dict(c)}")


if __name__ == "__main__":
    main()

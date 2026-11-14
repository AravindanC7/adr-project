"""Router prototype on SYNTHETIC data (step #13). Invented numbers: do not report as results.

Compares the router with two early baselines on a synthetic quiz and survey, then prints one
example session so you can see the router pick checks and the score move.

    python scripts/router_synthetic_demo.py            # 1000 AI + 1000 honest sessions each
    python scripts/router_synthetic_demo.py --n 200    # faster
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.baselines.simple import RandomCheckRouter, SingleMechanismRouter
from adr.router.core import RouterConfig, SPRTRouter
from adr.router.synthetic import (
    detection_at_fpr,
    simulate_session,
    synthetic_bank,
    synthetic_failure_map,
)


def evaluate(router, assessment, fmap, n, passive):
    ai = [simulate_session(router, assessment, fmap, "ai", s, passive) for s in range(n)]
    hu = [simulate_session(router, assessment, fmap, "human", 100_000 + s, passive)
          for s in range(n)]
    a_sc, h_sc = [x.final.score for x in ai], [x.final.score for x in hu]
    decided = [x.items_to_decision for x in ai + hu if x.items_to_decision is not None]
    return {
        "tpr_at_1pct_fpr": detection_at_fpr(a_sc, h_sc, 0.01)[0],
        "tpr_at_5pct_fpr": detection_at_fpr(a_sc, h_sc, 0.05)[0],
        "ai_paused_for_review": sum(x.final.status == "review" for x in ai) / n,
        "honest_paused_for_review": sum(x.final.status == "review" for x in hu) / n,
        "checks_per_honest_person": statistics.mean(x.checks_run for x in hu),
        "median_items_to_decision": statistics.median(decided) if decided else None,
        "share_decided": len(decided) / (2 * n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--passive", action="store_true", help="also observe behaviour on every item")
    args = ap.parse_args()
    passive = ("behaviour",) if args.passive else ()

    fmap = synthetic_failure_map()
    cfg = RouterConfig()
    print(f"SYNTHETIC failure map ({fmap.version}). alpha={cfg.alpha}, beta={cfg.beta}, "
          f"mech_cap={cfg.mech_cap}, passive={list(passive) or 'none'}\n")
    out = {}
    for kind in ("quiz", "survey"):
        assessment, items, variants = synthetic_bank(assessment_type=kind)
        strongest = "decoy" if kind == "quiz" else "canary"
        policies = {
            "router": SPRTRouter(items, variants, fmap, cfg),
            "random check per item": RandomCheckRouter(items, variants, fmap, cfg),
            f"only {strongest}": SingleMechanismRouter(items, variants, fmap, cfg,
                                                       mechanism=strongest),
            "only opinion_trap": SingleMechanismRouter(items, variants, fmap, cfg,
                                                       mechanism="opinion_trap"),
        }
        print(f"== {kind}: {len(assessment.item_ids)} items ==")
        print(f"{'policy':24s} {'TPR@1%':>7s} {'TPR@5%':>7s} {'AI paused':>9s} "
              f"{'honest paused':>13s} {'checks/honest':>13s} {'items to decide':>15s}")
        out[kind] = {}
        for name, router in policies.items():
            m = evaluate(router, assessment, fmap, args.n, passive)
            out[kind][name] = m
            med = m["median_items_to_decision"]
            print(f"{name:24s} {m['tpr_at_1pct_fpr']:7.2f} {m['tpr_at_5pct_fpr']:7.2f} "
                  f"{m['ai_paused_for_review']:9.2f} {m['honest_paused_for_review']:13.3f} "
                  f"{m['checks_per_honest_person']:13.1f} {med if med is not None else '-':>15}")
        print()

    # One example AI session, step by step
    assessment, items, variants = synthetic_bank(assessment_type="quiz")
    router = SPRTRouter(items, variants, fmap, cfg)
    res = simulate_session(router, assessment, fmap, "ai", seed=7, passive=passive)
    print("Example AI session (quiz):")
    for vid, outcome, delta in res.final.history:
        print(f"  {vid:34s} {outcome:22s} {delta:+.2f}")
    print(f"  final score {res.final.score:+.2f} (review at {router.A:.2f}, clear at "
          f"{router.B:.2f}) -> {res.final.status}")

    run_dir = Path("results/runs/synthetic-router-demo")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(
        {"WARNING": "SYNTHETIC data, invented numbers", "config": cfg.__dict__,
         "n_per_class": args.n, "passive": list(passive), "results": out}, indent=2), encoding="utf-8")
    print(f"\nSaved {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

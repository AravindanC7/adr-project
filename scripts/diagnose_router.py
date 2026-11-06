"""Diagnose what limits the router on one assessment (feeds threshold calibration, #42).

    python scripts/diagnose_router.py --synthetic survey                  # SYNTHETIC: method only
    python scripts/diagnose_router.py --outcomes results/outcomes/ \
        --assessment data/processed/assessments.jsonl --assessment-id survey-slice \
        --items data/processed/items.jsonl --variants data/processed/variants/survey-slice.jsonl

Data loading is exactly run_replay.py's (same flags; train map, held-out evaluation rows).
Prints:
  0. the log-likelihood ratio of every outcome of every check in the bank
  1. the router's budget curve (TPR vs checks per honest person over --betas) at each
     --cap-fracs value, next to all_checks
  2. for AI sessions at --beta: router vs all_checks on (a) which checks run, (b) final scores and
     statuses, (c) sessions that end with a mechanism at its cap but below the review threshold
     A, and (d) items the router serves plain before clearing: either no check is left (none
     with a map cell, or all capped) or the best check's utility is <= 0
  3. the honest score tail that sets the 1%-FPR threshold, router vs all_checks, per cap
Diagnosis only: never tune final values on this. Tune on held-out replay (#42).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.baselines.policies import make_policy
from adr.contracts import ACTIVE_MECHANISMS, NoObservation, parse_variant_id
from adr.replay.compare import BETAS, budget_curve, compare, run_policy
from adr.replay.metrics import detection_at_fpr
from adr.router.core import RouterConfig, SPRTRouter
from scripts.run_replay import load_logged, load_synthetic, loud


def parse_frac(text: str) -> float | None:
    return None if text.lower() == "none" else float(text)


def llr_table(fmap, assessment, items, variants) -> list[dict]:
    seen, rows = set(), []
    for item_id in assessment.item_ids:
        item_type = items[item_id].item_type
        for v in variants.get(item_id, []):
            key = (v.mechanism, item_type)
            if v.mechanism in ACTIVE_MECHANISMS and key not in seen:
                seen.add(key)
                cell = fmap.lookup(v.mechanism, item_type)
                if cell is not None:
                    rows.append({"mechanism": v.mechanism, "item_type": item_type,
                                 "llr": {o: round(cell.llr(o), 2) for o in cell.p_ai}})
    return rows


def ai_sessions(router, assessment, items, donors, groups, n) -> list[dict]:
    """Replay n sessions per AI group, recording why each plain item was served plain."""
    out = []
    instrument = type(router) is SPRTRouter  # only the router can serve plain "for no value"
    for g in groups:
        for s in range(n):
            donor = donors.respondent(g, s)
            state, choice = router.start(assessment, f"diag:{g.label}:{s}", s)
            plain_no_value: Counter = Counter()
            while choice is not None:
                if choice.mechanism is None:
                    if instrument and state.status != "cleared":
                        shown = {parse_variant_id(v)[0] for v in state.served}
                        remaining = [i for i in assessment.item_ids if i not in shown]
                        scored = router._score_candidates(state, remaining)
                        item_type = items[choice.item_id].item_type
                        if not scored:  # no remaining check with a map cell (or all capped)
                            plain_no_value[f"{item_type}: no check left"] += 1
                        elif max(u for u, _ in scored) <= 0:
                            plain_no_value[f"{item_type}: best utility <= 0"] += 1
                    state = router.update(state, choice.variant_id, "none")
                else:
                    try:
                        outcome = donor.outcome(choice.item_id, choice.mechanism)
                    except NoObservation:
                        outcome = None
                    state = router.update(state, choice.variant_id, outcome)
                choice = router.next(state)
            out.append({"state": state, "plain_no_value": plain_no_value})
    return out


def summarize_ai(name, sessions, A, cap) -> dict:
    mechs = Counter(parse_variant_id(v)[1] for x in sessions for v, o, _ in x["state"].history
                    if parse_variant_id(v)[1] in ACTIVE_MECHANISMS and o != "none")
    scores = np.array([x["state"].score for x in sessions])
    capped = {}
    if cap is not None:
        for m in sorted(mechs):
            at_cap = [x["state"] for x in sessions if x["state"].contrib.get(m, 0) >= cap - 1e-9]
            capped[m] = {"at_cap": len(at_cap), "below_A": sum(st.score < A for st in at_cap)}
    plain = sum((x["plain_no_value"] for x in sessions), Counter())
    return {
        "policy": name,
        "sessions": len(sessions),
        "checks_run": dict(mechs),
        "checks_per_session": round(sum(mechs.values()) / max(len(sessions), 1), 2),
        "score_quantiles_10_25_50_75_90": np.round(
            np.quantile(scores, [.1, .25, .5, .75, .9]), 2).tolist(),
        "share_score_at_least_A": round(float(np.mean(scores >= A)), 3),
        "statuses": dict(Counter(x["state"].status for x in sessions)),
        "mean_contribution": {m: round(statistics.mean(x["state"].contrib.get(m, 0.0)
                                                       for x in sessions), 2) for m in mechs},
        "capped_but_below_A": capped,
        "plain_before_clearing_by_reason": dict(sorted(plain.items())),
        "plain_before_clearing_per_session": round(sum(plain.values()) / max(len(sessions), 1), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--synthetic", choices=["quiz", "survey"])
    ap.add_argument("--outcomes", nargs="+")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--assessment"), ap.add_argument("--assessment-id")
    ap.add_argument("--items"), ap.add_argument("--variants"), ap.add_argument("--map")
    ap.add_argument("--cap-fracs", default="0.65,1.1,none",
                    help="mech_cap_frac values to compare (none = no cap)")
    ap.add_argument("--betas", default=",".join(str(b) for b in BETAS))
    ap.add_argument("--beta", type=float, default=0.01, help="beta for parts 2 and 3")
    ap.add_argument("--n", type=int, default=500, help="respondents per group for parts 1 and 3")
    ap.add_argument("--n-ai", type=int, default=200, help="AI sessions per group for part 2")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--json", type=Path, help="also save everything as JSON here")
    args = ap.parse_args()

    warnings: list[str] = []
    if args.synthetic:
        assessment, items, variants, fmap, donors, _ = load_synthetic(args.synthetic, warnings)
    elif args.outcomes:
        assessment, items, variants, fmap, donors, _ = load_logged(args, warnings)
    else:
        raise SystemExit("give --outcomes (logged rows) or --synthetic quiz|survey")
    for w in warnings:
        loud(w)
    fracs = [parse_frac(f) for f in args.cap_fracs.split(",") if f]
    betas = tuple(float(b) for b in args.betas.split(",") if b)
    ai_groups, human_groups = donors.groups("ai"), donors.groups("human")
    report: dict = {"assessment_id": assessment.assessment_id, "warnings": warnings}

    print("== 0. log-likelihood ratio per outcome (nats), from the map in use")
    report["llr"] = llr_table(fmap, assessment, items, variants)
    for row in report["llr"]:
        print(f"   {row['mechanism']:13s} {row['item_type']:9s} {row['llr']}")

    print(f"\n== 1. router budget curve per mech_cap_frac ({args.n} respondents per group)")
    report["budget_curves"] = {}
    for frac in fracs:
        cfg = replace(RouterConfig(), mech_cap_frac=frac)
        allc = compare({"all_checks": make_policy("all_checks", assessment, items, variants,
                                                  fmap, cfg)},
                       assessment, donors, args.n, 0, n_boot=args.boot)
        curve = budget_curve(assessment, items, variants, fmap, donors, args.n, 0, allc, betas,
                             cfg, n_boot=args.boot)
        report["budget_curves"][str(frac)] = curve.to_dict(orient="records")
        print(f"-- mech_cap_frac={frac}")
        print(curve[["policy", "beta", "tpr_at_1pct_fpr", "tpr_at_1pct_fpr_lo",
                     "tpr_at_1pct_fpr_hi", "tpr_at_5pct_fpr",
                     "checks_per_honest_person"]].round(3).to_string(index=False))

    cfg = replace(RouterConfig(), beta=args.beta)
    router = SPRTRouter(items, variants, fmap, cfg)
    allc = make_policy("all_checks", assessment, items, variants, fmap, cfg)
    print(f"\n== 2. {args.n_ai} AI sessions per AI group at beta={args.beta}, "
          f"mech_cap_frac={cfg.mech_cap_frac}: A={router.A:.2f}, B={router.B:.2f}, "
          f"cap={router.mech_cap if router.mech_cap is None else round(router.mech_cap, 2)}")
    report["ai_sessions"] = []
    for name, pol in (("router", router), ("all_checks", allc)):
        summary = summarize_ai(name, ai_sessions(pol, assessment, items, donors, ai_groups,
                                                 args.n_ai), router.A, router.mech_cap)
        report["ai_sessions"].append(summary)
        print(f"-- {name}")
        for key, value in summary.items():
            if key != "policy":
                print(f"   {key}: {value}")

    print(f"\n== 3. honest score tail vs the 1%-FPR threshold ({args.n} respondents per group)")
    report["honest_tail"] = []
    for frac in fracs:
        c = replace(cfg, mech_cap_frac=frac)
        for name in ("router", "all_checks"):
            runs = run_policy(make_policy(name, assessment, items, variants, fmap, c), assessment,
                              donors, ai_groups + human_groups, args.n, 0)
            ai = np.array([r.final.score for g in ai_groups for r in runs[g]])
            hu = np.array([r.final.score for g in human_groups for r in runs[g]])
            tpr, _, t = detection_at_fpr(ai, hu, 0.01)
            row = {"mech_cap_frac": frac, "policy": name,
                   "honest_50_90_99_max": np.round(np.quantile(hu, [.5, .9, .99, 1]), 2).tolist(),
                   "threshold_1pct": round(t, 2), "ai_median": round(float(np.median(ai)), 2),
                   "tpr_at_1pct_fpr": round(tpr, 3),
                   "honest_checks": round(float(np.mean([r.checks_run for g in human_groups
                                                         for r in runs[g]])), 1)}
            report["honest_tail"].append(row)
            print(f"   frac {frac!s:5s} {name:10s} honest 50/90/99/max "
                  f"{row['honest_50_90_99_max']}  1% threshold {row['threshold_1pct']}  "
                  f"AI median {row['ai_median']}  TPR@1% {row['tpr_at_1pct_fpr']}  "
                  f"honest checks {row['honest_checks']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()

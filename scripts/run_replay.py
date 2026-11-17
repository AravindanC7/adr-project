"""Replay logged outcomes through the router and baselines (step #27): offline evaluation.

Every simulated respondent copies one real logged session (see adr/replay/donors.py), so the
numbers come from real model and human answers, not from adr/router/synthetic.py.

Smoke test on the 4 handwritten items and the two first real runs:
    python scripts/run_replay.py --stub \
        --outcomes results/outcomes/first-real-L0.parquet results/outcomes/first-real-du-L0.parquet

Real use:
    python scripts/run_replay.py --outcomes results/outcomes/ \
        --assessment data/processed/assessments.jsonl --assessment-id quiz-slice \
        --items data/processed/items.jsonl --variants data/processed/variants/quiz-slice.jsonl \
        --compare --n 500 --seed 0

Policies (--policies, comma list): router, router_noclear, best_single, all_checks,
fixed_plan, random, single:<mechanism> (see adr/baselines/policies.py). --compare runs all of
them except single:* on the same respondents and also writes comparison.csv, comparison.md and
budget_curve.csv (the router at each --betas value, plus one row per other policy). Plot it with
scripts/plot_budget_curve.py.

SYNTHETIC check of the whole pipeline (invented numbers, never results):
    python scripts/run_replay.py --synthetic quiz --compare --n 500

Three splits (adr/replay/quick_map.py):
  train        fits the failure map (--map to use a saved one instead)
  calibration  --calibration-frac of the non-held-out items (default 30%, by item type, fixed
               --split-seed). The DEFAULT evaluation: development runs, and where thresholds
               and the per-mechanism cap are calibrated (#42). The replay serves only these
               items, so sessions are shorter than the real thing.
  held-out     the held-out model and the held-out survey. Read ONLY with --final, which prints
               a banner: report those numbers, never tune on them (AGENTS.md section 8).
If the evaluated rows have no AI respondents, only the honest-side metrics (false flags,
burden) are computed. Writes results/runs/<run_id>/metrics.json and config.yaml.

--map refuses a provisional failure map (version ending in -ai-only, or any cell with zero
human counts: adr/router/map_guard.py) unless --allow-provisional-map is passed, for
development runs only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.baselines.plans import multi_check_items
from adr.baselines.policies import COMPARE, make_policy
from adr.contracts import (
    COLUMNS,
    Assessment,
    Item,
    RenderedItem,
    read_jsonl,
    read_outcomes,
)
from adr.contracts.outcome import utc_now
from adr.harness.copy_paste import _git_commit, plain_variant
from adr.replay.compare import BETAS, budget_curve, compare, comparison_table, write_comparison
from adr.replay.donors import Donors, Group
from adr.replay.quick_map import (
    CALIBRATION_FRAC,
    SPLIT_SEED,
    fit_rows,
    quick_failure_map,
    split_rows,
)
from adr.router.core import RouterConfig
from adr.router.map_guard import (
    ALLOW_FLAG,
    ProvisionalMapError,
    load_router_map,
    provisional_reasons,
)


def load_bank(args) -> tuple[Assessment, dict[str, Item], dict[str, list[RenderedItem]]]:
    if args.stub:  # exactly what run_copy_paste.py / run_doc_upload.py --stub served
        from tests.stubs import ITEMS, StubCanary, StubDecoy

        decoy, canary = StubDecoy(), StubCanary()
        variants: dict[str, list[RenderedItem]] = {}
        for it in ITEMS:
            vs = [plain_variant(it)]
            vs += [m.generate(it, 0) for m in (decoy, canary) if m.applies_to(it)]
            variants[it.item_id] = vs
        assessment = Assessment(assessment_id="stub", title="Stub", assessment_type="quiz",
                                item_ids=[it.item_id for it in ITEMS])
        return assessment, {it.item_id: it for it in ITEMS}, variants
    if not (args.assessment and args.items and args.variants):
        raise SystemExit("give --assessment, --items and --variants (or use --stub)")
    assessments = read_jsonl(args.assessment, Assessment)
    if args.assessment_id:
        assessments = [a for a in assessments if a.assessment_id == args.assessment_id]
    if len(assessments) != 1:
        raise SystemExit(f"{args.assessment}: need exactly one assessment (use --assessment-id)")
    items = {it.item_id: it for it in read_jsonl(args.items, Item)}
    variants = defaultdict(list)
    for v in read_jsonl(args.variants, RenderedItem):
        variants[v.item_id].append(v)
    return assessments[0], items, dict(variants)


def load_logged(args, warnings: list[str]):
    """Logged outcome rows: map from train rows; evaluation on the calibration split, or on the
    held-out rows with --final (see module doc). Returns the assessment as evaluated: only its
    calibration items unless --final."""
    final = getattr(args, "final", False)
    frac = getattr(args, "calibration_frac", CALIBRATION_FRAC)
    split_seed = getattr(args, "split_seed", SPLIT_SEED)
    paths = []
    for p in args.outcomes:
        path = Path(p)
        paths += sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    table = read_outcomes(paths)
    assessment, items, variants = load_bank(args)
    table = table[table["item_id"].isin(assessment.item_ids)]
    if table.empty:
        raise SystemExit("no outcome rows for this assessment's items")

    splits = split_rows(table, frac, split_seed)
    allow = getattr(args, "allow_provisional_map", False)
    if args.map:
        try:
            fmap = load_router_map(args.map, allow_provisional=allow)
        except ProvisionalMapError as e:
            raise SystemExit(str(e)) from None
        reasons = provisional_reasons(fmap)
        if reasons:
            loud(f"PROVISIONAL MAP ({ALLOW_FLAG}): " + "; ".join(reasons) + ". Development "
                 "only: false-flag and TPR numbers from this run are not results.")
            warnings.append("PROVISIONAL MAP: " + "; ".join(reasons))
    else:
        fmap = quick_failure_map(splits.train)
        warnings.append(f"map: {fmap.note}")
    if final:
        loud("FINAL EVALUATION ON HELD-OUT DATA (--final). Report these numbers; never tune "
             "prompts, thresholds, caps or features on them (AGENTS.md section 8).")
        if splits.heldout.empty:
            raise SystemExit("--final: there are no held-out rows (held-out model or survey)")
        evaluated_on, eval_rows, eval_assessment = "heldout", splits.heldout, assessment
    else:
        ids = [i for i in assessment.item_ids if i in splits.calibration_item_ids]
        if not ids:
            raise SystemExit("no calibration items in this assessment: raise --calibration-frac")
        evaluated_on, eval_rows = "calibration", splits.calibration
        eval_assessment = assessment.model_copy(update={"item_ids": ids})
        print(f"Development evaluation on the calibration split: {len(ids)} of "
              f"{len(assessment.item_ids)} items. Held-out rows are not read (use --final).")
    eval_ai = eval_rows.query("respondent_kind == 'ai'")
    humans = eval_rows.query("respondent_kind == 'human'")
    if humans.empty and final:
        humans = fit_rows(table).query("respondent_kind == 'human'")
        if not humans.empty:
            warnings.append("no held-out human rows: honest respondents reuse the non-held-out "
                            "human rows (including the map's own), so false-flag rates are "
                            "optimistic.")
    extra: tuple[Group, ...] = ()
    if humans.empty:
        extra = (Group.human("synthetic"),)
        warnings.append("NO HUMAN ROWS: every honest respondent is drawn from the map's "
                        "p_human, so FPR-based numbers say nothing about real people.")
    if eval_ai.empty:
        warnings.append(f"NO AI ROWS IN THE {evaluated_on.upper()} ROWS: only the honest-side "
                        "metrics (false flags, burden) are computed; TPR is blank.")
    donors = Donors(pd.concat([eval_ai, humans]), items, fallback_map=fmap, extra_groups=extra)
    info = {
        "outcome_files": [str(p) for p in paths],
        "source_run_ids": sorted(table["run_id"].unique()),
        "map": args.map or "quick (TEMPORARY, adr/replay/quick_map.py)",
        "provisional_map_allowed": bool(args.map and allow),
        "evaluated_on": evaluated_on,
        "final": final,
        "n_items_assessment": len(assessment.item_ids),
        "eval_item_ids": list(eval_assessment.item_ids),
        "calibration_frac": frac,
        "split_seed": split_seed,
        "calibration_item_ids": sorted(splits.calibration_item_ids),
        "n_rows_train": len(splits.train),
        "n_rows_calibration": len(splits.calibration),
        "n_rows_heldout_read": len(splits.heldout) if final else 0,
        "n_rows_eval_ai": len(eval_ai),
        "n_rows_human": len(humans),
    }
    return eval_assessment, items, variants, fmap, donors, info


def load_synthetic(kind: str, warnings: list[str]):
    """SYNTHETIC: the synthetic bank, and every outcome drawn from synthetic_failure_map."""
    from adr.router.synthetic import synthetic_bank, synthetic_failure_map

    assessment, items, variants = synthetic_bank(assessment_type=kind)
    fmap = synthetic_failure_map()
    donors = Donors(pd.DataFrame(columns=COLUMNS), items, fallback_map=fmap,
                    extra_groups=(Group.ai("synthetic", "copy_paste", "L0"),
                                  Group.human("synthetic")))
    warnings.append(f"SYNTHETIC {kind}: invented numbers from adr/router/synthetic.py. "
                    "Never report these as results.")
    info = {"synthetic": kind, "map": "synthetic_failure_map (INVENTED)",
            "evaluated_on": "synthetic", "final": False}
    return assessment, items, variants, fmap, donors, info


def loud(msg: str) -> None:
    bar = "!" * 78
    print(f"\n{bar}\n!! {msg}\n{bar}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--outcomes", nargs="+", help="outcome-table files or folders")
    ap.add_argument("--synthetic", choices=["quiz", "survey"],
                    help="SYNTHETIC bank and map instead of logged rows (invented numbers)")
    ap.add_argument("--stub", action="store_true", help="the 4 handwritten items and stub traps")
    ap.add_argument("--assessment"), ap.add_argument("--assessment-id")
    ap.add_argument("--items"), ap.add_argument("--variants")
    ap.add_argument("--map", help="a saved FailureMap JSON; default: TEMPORARY quick map")
    ap.add_argument(ALLOW_FLAG, action="store_true",
                    help="use a provisional --map anyway (version ending in -ai-only, or cells "
                         "with zero human counts); development only, never for results")
    ap.add_argument("--policies", default=",".join(COMPARE))
    ap.add_argument("--compare", action="store_true",
                    help=f"run {', '.join(COMPARE)}; write comparison.csv / .md and "
                         "budget_curve.csv (router at each --betas)")
    ap.add_argument("--betas", default=",".join(str(b) for b in BETAS),
                    help="router betas for the budget curve (with --compare)")
    ap.add_argument("--passive", default="", help="comma list, e.g. behaviour,style")
    ap.add_argument("--n", type=int, default=200, help="simulated respondents per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws for the 95%% CIs")
    ap.add_argument("--final", action="store_true",
                    help="FINAL evaluation on the held-out rows (never tune on its numbers)")
    ap.add_argument("--calibration-frac", type=float, default=CALIBRATION_FRAC,
                    help="share of non-held-out items set aside for calibration (default 0.3)")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--run-id")
    args = ap.parse_args()
    run_id = args.run_id or f"replay-{datetime.now():%Y%m%d-%H%M%S}"
    passive = tuple(p for p in args.passive.split(",") if p)

    warnings: list[str] = []
    if args.synthetic:
        assessment, items, variants, fmap, donors, info = load_synthetic(args.synthetic, warnings)
    elif args.outcomes:
        assessment, items, variants, fmap, donors, info = load_logged(args, warnings)
    else:
        raise SystemExit("give --outcomes (logged rows) or --synthetic quiz|survey")
    ai_groups, human_groups = donors.groups("ai"), donors.groups("human")
    for w in warnings:
        loud(w)

    names = list(COMPARE) if args.compare else [p for p in args.policies.split(",") if p]
    cfg = RouterConfig()
    try:  # fmap is fitted on train rows only, so the baselines' choices never see held-out data
        policies = {n: make_policy(n, assessment, items, variants, fmap, cfg) for n in names}
    except ValueError as e:
        raise SystemExit(str(e)) from e
    best_single = None
    if "best_single" in names:  # the policy's own choice (attack-mode aware), not a recompute
        bs = policies["best_single"]
        best_single = {"mechanism": bs.mechanism, "expected_evidence": bs.expected_evidence}
        print(f"best_single: {best_single['mechanism']} (expected evidence per mechanism, "
              f"train map: {best_single['expected_evidence']})")
    multi = multi_check_items(assessment, items, variants)
    print(f"run {run_id}: {len(assessment.item_ids)} items ({info['evaluated_on']}), "
          f"map {fmap.version}, "
          f"{args.n} respondents per group, seed {args.seed}")
    print("groups: " + ", ".join(f"{g.label} ({len(donors.sessions(g))} sessions)"
                                 for g in ai_groups + human_groups))
    summaries = compare(policies, assessment, donors, args.n, args.seed, passive, args.boot)
    results = {n: {"overall": s["overall"], "by_group": s["by_group"]}
               for n, s in summaries.items()}
    draws = {n: s["sources"] for n, s in summaries.items()}

    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    print(f"\n{'policy':16s} {'TPR@1%':>7s} {'TPR@5%':>7s} {'AI paused':>9s} "
          f"{'honest paused':>13s} {'checks/honest':>13s} {'items to decide':>15s}")
    for name, res in results.items():
        m = res["overall"]
        print(f"{name:16s} {fmt(m['tpr_at_1pct_fpr'], '7.2f')} {fmt(m['tpr_at_5pct_fpr'], '7.2f')} "
              f"{fmt(m['ai_paused_for_review'], '9.2f')} "
              f"{fmt(m['honest_paused_for_review'], '13.3f')} "
              f"{fmt(m['checks_per_honest_person'], '13.1f')} "
              f"{fmt(m['median_items_to_decision'], '>15')}")
    print("\nWhere outcomes came from (per group, summed over respondents):")
    for name, per_group in draws.items():
        for label, d in per_group.items():
            print(f"  {name:16s} {label:34s} checks {d['checks_run']:5d}  "
                  f"fallbacks {d['fallbacks']:5d}  synthetic {d['synthetic_draws']:5d}  "
                  f"skipped {d['skipped_no_observation']:4d}  "
                  f"donor sessions {d['donor_sessions_used']}")

    run_dir = Path(args.results_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({
        "warnings": warnings,
        "evaluated_on": info["evaluated_on"],
        "final": info["final"],
        "provisional_map_allowed": info.get("provisional_map_allowed", False),
        "best_single": best_single,
        "results": results,
        "outcome_sources": draws,
    }, indent=2), encoding="utf-8")
    betas = tuple(float(b) for b in args.betas.split(",") if b)
    if args.compare:
        _, md_path = write_comparison(comparison_table(results), run_dir)
        print(f"\n{md_path.read_text(encoding='utf-8')}")
        curve = budget_curve(assessment, items, variants, fmap, donors, args.n, args.seed,
                             {n: s for n, s in summaries.items() if n != "router"}, betas, cfg,
                             passive, args.boot)
        curve.to_csv(run_dir / "budget_curve.csv", index=False)
        print("Budget curve (router at each beta, then the baselines):")
        print(curve[["policy", "beta", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr",
                     "checks_per_honest_person"]].round(3).to_string(index=False))
        print(f"Plot it: python scripts/plot_budget_curve.py {run_dir / 'budget_curve.csv'}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "mode": "replay",
        **info,
        "assessment_id": assessment.assessment_id,
        "n_items": len(assessment.item_ids),
        "map_version": fmap.version,
        "map_source_run_ids": fmap.source_run_ids,
        "groups": {g.label: len(donors.sessions(g)) for g in ai_groups + human_groups},
        "policies": list(policies),
        "router_config": cfg.__dict__,
        "best_single": best_single["mechanism"] if best_single else None,
        # all_checks runs one active check per item; these items had more than one eligible
        "n_items_multi_check": len(multi),
        "items_multi_check": multi,
        "passive": list(passive),
        "n_per_group": args.n,
        "seed": args.seed,
        "n_boot": args.boot,
        "budget_curve_betas": list(betas) if args.compare else None,
        "git_commit": _git_commit(),
        "created_at": utc_now(),
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    extra_files = ", comparison.csv, comparison.md, budget_curve.csv" if args.compare else ""
    print(f"\nSaved {run_dir / 'metrics.json'}, config.yaml{extra_files}")


if __name__ == "__main__":
    main()

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

Policies (--policies, comma list): router, best_single, all_checks, fixed_plan, random,
single:<mechanism> (see adr/baselines/policies.py). --compare runs router and baselines 1-4 on
the same respondents and also writes comparison.csv and comparison.md.

The failure map is fitted on train rows only (--map to use a saved one instead). Evaluation
uses held-out rows (held-out model or held-out survey) when there are any; otherwise it
evaluates on the map's own rows and says so loudly. Writes results/runs/<run_id>/metrics.json
and config.yaml.
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

from adr.baselines.plans import best_single_mechanism, multi_check_items
from adr.baselines.policies import COMPARE, make_policy
from adr.contracts import Assessment, FailureMap, Item, RenderedItem, read_jsonl, read_outcomes
from adr.contracts.outcome import utc_now
from adr.harness.copy_paste import _git_commit, plain_variant
from adr.replay.compare import compare, comparison_table, write_comparison
from adr.replay.donors import Donors, Group
from adr.replay.quick_map import quick_failure_map, train_eval_split
from adr.router.core import RouterConfig


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


def loud(msg: str) -> None:
    bar = "!" * 78
    print(f"\n{bar}\n!! {msg}\n{bar}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--outcomes", nargs="+", required=True, help="outcome-table files or folders")
    ap.add_argument("--stub", action="store_true", help="the 4 handwritten items and stub traps")
    ap.add_argument("--assessment"), ap.add_argument("--assessment-id")
    ap.add_argument("--items"), ap.add_argument("--variants")
    ap.add_argument("--map", help="a saved FailureMap JSON; default: TEMPORARY quick map")
    ap.add_argument("--policies", default=",".join(COMPARE))
    ap.add_argument("--compare", action="store_true",
                    help=f"run {', '.join(COMPARE)} and write comparison.csv / .md")
    ap.add_argument("--passive", default="", help="comma list, e.g. behaviour,style")
    ap.add_argument("--n", type=int, default=200, help="simulated respondents per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws for the 95%% CIs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--run-id")
    args = ap.parse_args()
    run_id = args.run_id or f"replay-{datetime.now():%Y%m%d-%H%M%S}"
    passive = tuple(p for p in args.passive.split(",") if p)

    paths = []
    for p in args.outcomes:
        path = Path(p)
        paths += sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    table = read_outcomes(paths)
    assessment, items, variants = load_bank(args)
    table = table[table["item_id"].isin(assessment.item_ids)]
    if table.empty:
        raise SystemExit("no outcome rows for this assessment's items")

    warnings: list[str] = []
    fit, held, same_data = train_eval_split(table)
    fmap = FailureMap.load(args.map) if args.map else quick_failure_map(table)
    if not args.map:
        warnings.append(f"map: {fmap.note}")
    eval_ai = (fit if same_data else held).query("respondent_kind == 'ai'")
    if same_data:
        warnings.append("EVALUATING ON THE SAME ROWS THE MAP WAS FIT ON: numbers are optimistic. "
                        "Add held-out-model or held-out-survey rows for a fair evaluation.")
    humans = held.query("respondent_kind == 'human'") if not same_data else pd.DataFrame()
    if humans.empty:
        humans = fit.query("respondent_kind == 'human'")
        if not humans.empty and not same_data:
            warnings.append("no held-out human rows: honest respondents reuse the map's own "
                            "human rows, so false-flag rates are optimistic.")
    extra: tuple[Group, ...] = ()
    if humans.empty:
        extra = (Group.human("synthetic"),)
        warnings.append("NO HUMAN ROWS AT ALL: every honest respondent is drawn from the map's "
                        "p_human, so FPR-based numbers say nothing about real people.")
    donors = Donors(pd.concat([eval_ai, humans]), items, fallback_map=fmap, extra_groups=extra)
    ai_groups, human_groups = donors.groups("ai"), donors.groups("human")
    if not ai_groups:
        raise SystemExit("no AI rows to evaluate on")
    for w in warnings:
        loud(w)

    names = list(COMPARE) if args.compare else [p for p in args.policies.split(",") if p]
    cfg = RouterConfig()
    try:  # fmap is fitted on train rows only, so the baselines' choices never see held-out data
        policies = {n: make_policy(n, assessment, items, variants, fmap, cfg) for n in names}
    except ValueError as e:
        raise SystemExit(str(e)) from e
    best_single = None
    if "best_single" in names:
        best_single = dict(zip(("mechanism", "expected_evidence"),
                               best_single_mechanism(fmap, assessment, items, variants)))
        print(f"best_single: {best_single['mechanism']} (expected evidence per mechanism, "
              f"train map: {best_single['expected_evidence']})")
    multi = multi_check_items(assessment, items, variants)
    print(f"run {run_id}: {len(assessment.item_ids)} items, map {fmap.version}, "
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
        "evaluated_on_fit_rows": same_data,
        "best_single": best_single,
        "results": results,
        "outcome_sources": draws,
    }, indent=2))
    if args.compare:
        _, md_path = write_comparison(comparison_table(results), run_dir)
        print(f"\n{md_path.read_text()}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "mode": "replay",
        "outcome_files": [str(p) for p in paths],
        "source_run_ids": sorted(table["run_id"].unique()),
        "assessment_id": assessment.assessment_id,
        "n_items": len(assessment.item_ids),
        "map": args.map or "quick (TEMPORARY, adr/replay/quick_map.py)",
        "map_version": fmap.version,
        "map_source_run_ids": fmap.source_run_ids,
        "n_rows_fit": len(fit),
        "n_rows_eval_ai": len(eval_ai),
        "n_rows_human": len(humans),
        "evaluated_on_fit_rows": same_data,
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
        "git_commit": _git_commit(),
        "created_at": utc_now(),
    }, sort_keys=False))
    extra_files = ", comparison.csv, comparison.md" if args.compare else ""
    print(f"\nSaved {run_dir / 'metrics.json'}, config.yaml{extra_files}")


if __name__ == "__main__":
    main()

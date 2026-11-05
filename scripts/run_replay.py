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
        --policies router,random,single:decoy --n 500 --seed 0

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

from adr.baselines.simple import RandomCheckRouter, SingleMechanismRouter
from adr.contracts import Assessment, FailureMap, Item, RenderedItem, read_jsonl, read_outcomes
from adr.contracts.outcome import utc_now
from adr.harness.copy_paste import _git_commit, plain_variant
from adr.replay.donors import Donors, Group
from adr.replay.metrics import evaluate
from adr.replay.quick_map import quick_failure_map, train_eval_split
from adr.replay.simulate import replay_session
from adr.router.core import RouterConfig, SPRTRouter


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


def make_policies(names: list[str], items, variants, fmap: FailureMap) -> dict:
    cfg = RouterConfig()
    out = {}
    for name in names:
        if name == "router":
            out[name] = SPRTRouter(items, variants, fmap, cfg)
        elif name == "random":
            out[name] = RandomCheckRouter(items, variants, fmap, cfg)
        elif name.startswith("single:"):
            out[name] = SingleMechanismRouter(items, variants, fmap, cfg,
                                              mechanism=name.split(":", 1)[1])
        else:
            raise SystemExit(f"unknown policy {name!r}: use router, random or single:<mechanism>")
    return out


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
    ap.add_argument("--policies", default="router,random,single:decoy")
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

    policies = make_policies(args.policies.split(","), items, variants, fmap)
    print(f"run {run_id}: {len(assessment.item_ids)} items, map {fmap.version}, "
          f"{args.n} respondents per group, seed {args.seed}")
    print("groups: " + ", ".join(f"{g.label} ({len(donors.sessions(g))} sessions)"
                                 for g in ai_groups + human_groups))
    results: dict = {}
    draws: dict = {}
    for name, router in policies.items():
        runs = {g: [replay_session(router, assessment, donors.respondent(g, args.seed + i),
                                   passive, seed=args.seed + i) for i in range(args.n)]
                for g in ai_groups + human_groups}
        honest = [r for g in human_groups for r in runs[g]]
        ai_all = [r for g in ai_groups for r in runs[g]]
        results[name] = {
            "overall": evaluate(ai_all, honest, n_boot=args.boot, seed=args.seed),
            "by_group": {g.label: evaluate(runs[g], honest, n_boot=args.boot, seed=args.seed)
                         for g in ai_groups},
        }
        draws[name] = {g.label: {
            "respondents": len(runs[g]),
            "donor_sessions_used": len({r.donor_session for r in runs[g]} - {None}),
            "fallbacks": sum(r.fallbacks for r in runs[g]),
            "synthetic_draws": sum(r.synthetic for r in runs[g]),
            "skipped_no_observation": sum(r.skipped for r in runs[g]),
            "checks_run": sum(r.checks_run for r in runs[g]),
        } for g in ai_groups + human_groups}

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
        "results": results,
        "outcome_sources": draws,
    }, indent=2))
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
        "router_config": RouterConfig().__dict__,
        "passive": list(passive),
        "n_per_group": args.n,
        "seed": args.seed,
        "n_boot": args.boot,
        "git_commit": _git_commit(),
        "created_at": utc_now(),
    }, sort_keys=False))
    print(f"\nSaved {run_dir / 'metrics.json'} and config.yaml")


if __name__ == "__main__":
    main()

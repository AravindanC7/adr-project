"""The first real router + baselines run (step #42): one command, thresholds calibrated on
held-out PEOPLE by person-level cross-fitting (adr/replay/crossfit.py).

    python scripts/run_real_router.py \\
        --ai-outcomes results/outcomes/v1-cp-L0-quiz-mmlu-200-s5.parquet ... \\
        --pilot-outcomes results/outcomes/pilot-2026-11-13.parquet \\
        [--exclude-sessions low_effort.txt] [--folds 4] [--alpha 0.01] [--group-passive] [--final]

What it does:
  1. Reads the AI rows (any files or folders) and the pilot rows (data_origin "pilot"; one
     session_id = one person). --exclude-sessions drops the listed people (one session_id per
     line), for the with/without low-effort sensitivity run.
  2. Keeps only the mechanisms that have BOTH AI and human rows, and says which ones.
  3. Replays on the pilot bank (--bank, default server/pilot_items.json: what the router would
     serve). Honest people replay their own sessions. The AI runs answered other items, so an AI
     session answers each check from its own rows on items of the same type (adr/replay/
     donors.py, "same-type draws"): real behaviour, the map's own exchangeability assumption.
  4. AI rows are split by item as in run_replay.py: the map's AI side uses the train items, and
     the AI sessions are scored on the calibration items (on the held-out rows with --final).
  5. For each of K folds of people: map_k = Archana's builder in STRICT mode (never --ai-only)
     on the AI train rows + the other folds' people; A_k = the review threshold that pauses at
     most alpha of fold k's people; the AI sessions are replayed with map_k and A_k. Done for
     the router, baselines 1-4 and the router at each beta (the budget curve), at --alpha and
     at 5% FPR (few people cannot pin down a 1% tail).

Writes results/real-router/<run_id>/: comparison.csv/.md, thresholds.csv, budget_curve.csv/.png,
metrics.json and config.yaml. --final reads held-out rows, as run_replay does; never by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.baselines.policies import COMPARE
from adr.contracts import (
    ACTIVE_MECHANISMS,
    PASSIVE_MECHANISMS,
    Assessment,
    Item,
    RenderedItem,
    read_outcomes,
)
from adr.contracts.outcome import utc_now
from adr.harness.copy_paste import _git_commit, plain_variant
from adr.harness.registry import get_mechanism
from adr.replay.compare import BETAS
from adr.replay.crossfit import (
    all_maps,
    crossfit,
    person_folds,
    pooled_metrics,
    router_factory,
    threshold_rows,
)
from adr.replay.quick_map import CALIBRATION_FRAC, SPLIT_SEED, split_rows
from adr.router.core import RouterConfig

GROUP_PASSIVE = {"passive": tuple(sorted(PASSIVE_MECHANISMS))}
EXTRA_ALPHA = 0.05  # always reported too


def read_rows(paths: list[str]) -> pd.DataFrame:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        files += sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no outcome files in {paths}")
    return read_outcomes(files)


def read_exclusions(path: str | None) -> set[str]:
    if not path:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {s.strip() for s in lines if s.strip() and not s.strip().startswith("#")}


def exclude_people(human: pd.DataFrame, sessions: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop exactly the listed session_ids. Returns the rows kept and the listed ids that were
    not found (reported, so a typo in the file is visible)."""
    present = set(human["session_id"])
    return human[~human["session_id"].isin(sessions)], sorted(sessions - present)


def usable_mechanisms(ai: pd.DataFrame, human: pd.DataFrame) -> list[str]:
    """Mechanisms with both AI and human rows (held-out model and survey rows don't count)."""
    ok_ai = ai[
        ai["mechanism"].notna()
        & ~ai["heldout_model"].fillna(False).astype(bool)
        & ai["split"].ne("heldout_survey")
    ]
    return sorted(set(ok_ai["mechanism"]) & set(human["mechanism"].dropna()))


def load_bank(path: str, assessment_id: str | None, mechanisms: list[str]):
    """The pilot bank: items, the replayed assessment (the flow's parts in order, or one
    --assessment-id) and the variants: plain + one per usable active mechanism."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = {it.item_id: it for it in (Item(**d) for d in raw["items"])}
    assessments = {a["assessment_id"]: Assessment(**a) for a in raw["assessments"]}
    if assessment_id:
        assessment = assessments[assessment_id]
    else:
        parts = (raw.get("flow") or {}).get("parts") or list(assessments)
        ids = [i for p in parts for i in assessments[p].item_ids]
        assessment = Assessment(
            assessment_id="+".join(parts),
            title="pilot flow",
            assessment_type=assessments[parts[0]].assessment_type,
            item_ids=ids,
        )
    active = [m for m in mechanisms if m in ACTIVE_MECHANISMS]
    mechs = {m: get_mechanism(m) for m in active}
    variants: dict[str, list[RenderedItem]] = {}
    for item_id in assessment.item_ids:
        item = items[item_id]
        vs = [plain_variant(item)]
        for name, mech in mechs.items():
            if name in item.eligible_mechanisms and mech.applies_to(item):
                vs.append(mech.generate(item, 0))
        variants[item_id] = vs
    return assessment, items, variants


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ai-outcomes", nargs="+", required=True, help="AI outcome files/folders")
    ap.add_argument("--pilot-outcomes", nargs="+", required=True, help="pilot outcome file(s)")
    ap.add_argument("--exclude-sessions", help="file: human session_ids to drop, one per line")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.01, help="max honest pause rate per fold")
    ap.add_argument(
        "--group-passive",
        action="store_true",
        help="behaviour and style share ONE evidence cap (mech_group_caps)",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="FINAL: score the held-out AI rows (never tune on these numbers)",
    )
    ap.add_argument("--bank", default="server/pilot_items.json", help="the pilot bank JSON")
    ap.add_argument("--assessment-id", help="replay one part of the bank (default: the flow)")
    ap.add_argument("--betas", default=",".join(str(b) for b in BETAS))
    ap.add_argument("--seed", type=int, default=0, help="fold split and replay seed")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--calibration-frac", type=float, default=CALIBRATION_FRAC)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--run-id")
    args = ap.parse_args()
    run_id = args.run_id or f"real-router-{datetime.now():%Y%m%d-%H%M%S}"

    ai = read_rows(args.ai_outcomes)
    ai = ai[ai["respondent_kind"].eq("ai")]
    rows = read_rows(args.pilot_outcomes)
    human = rows[rows["respondent_kind"].eq("human") & rows["data_origin"].eq("pilot")]
    if human.empty:
        raise SystemExit(
            "no pilot rows (respondent_kind human, data_origin pilot) in "
            f"{args.pilot_outcomes}. Without them use scripts/run_replay.py."
        )
    excluded = read_exclusions(args.exclude_sessions)
    human, not_found = exclude_people(human, excluded)
    if not_found:
        print(
            f"WARNING: {len(not_found)} --exclude-sessions ids are not pilot sessions: "
            f"{not_found[:5]}"
        )
    people = sorted(human["session_id"].unique())

    mechanisms = usable_mechanisms(ai, human)
    if not mechanisms:
        raise SystemExit("no mechanism has both AI and human rows")
    print(f"Mechanisms with BOTH AI and human rows (the only ones used): {', '.join(mechanisms)}")
    if mechanisms == ["decoy"]:
        print(
            "  -> decoy only: the router has ONE check type, so this run measures how well "
            "decoys separate AI from people at a person-calibrated threshold; it cannot show "
            "adaptive selection across mechanisms yet."
        )
    ai = ai[ai["mechanism"].isna() | ai["mechanism"].isin(mechanisms)]
    human = human[human["mechanism"].isna() | human["mechanism"].isin(mechanisms)]
    passive = tuple(m for m in mechanisms if m in PASSIVE_MECHANISMS)

    assessment, items, variants = load_bank(args.bank, args.assessment_id, mechanisms)
    splits = split_rows(ai, args.calibration_frac, args.split_seed)
    if args.final:
        bar = "!" * 78
        print(
            f"\n{bar}\n!! FINAL EVALUATION: the AI sessions are the HELD-OUT rows. Report "
            f"these numbers; never tune on them (AGENTS.md section 8).\n{bar}\n"
        )
        ai_eval = splits.heldout[splits.heldout["respondent_kind"].eq("ai")]
        if ai_eval.empty:
            raise SystemExit("--final: there are no held-out AI rows")
    else:
        ai_eval = splits.calibration
    if ai_eval.empty:
        raise SystemExit("no AI evaluation rows (raise --calibration-frac)")

    cfg = RouterConfig()
    if args.group_passive:
        cfg = replace(cfg, mech_group_caps=GROUP_PASSIVE)
    folds = person_folds(people, args.folds, args.seed)
    print(
        f"{len(people)} people in {args.folds} folds of {[len(f) for f in folds]}; "
        f"{len(excluded)} excluded. AI evaluation sessions: "
        f"{ai_eval['session_id'].nunique()} ({'held-out' if args.final else 'calibration'} "
        "items). Fitting the per-fold maps..."
    )
    maps = all_maps(splits.train, human, folds)
    alphas = sorted({args.alpha, EXTRA_ALPHA})

    def run(name: str, alpha: float, c: RouterConfig):
        make = router_factory(name, assessment, items, variants, c)
        res = crossfit(
            make,
            assessment,
            items,
            splits.train,
            human,
            ai_eval,
            folds,
            alpha,
            passive,
            args.seed,
            maps=maps,
        )
        return res, pooled_metrics(res, args.boot, args.seed)

    results: dict[str, dict[str, dict]] = {}
    thresholds: list[dict] = []
    for name in COMPARE:
        results[name] = {}
        for alpha in alphas:
            res, m = run(name, alpha, cfg)
            results[name][f"{alpha:g}"] = m
            thresholds += threshold_rows(name, alpha, res)
        print(
            f"  {name:15s} "
            + "  ".join(
                f"alpha {a}: TPR {results[name][a]['tpr']:.2f}, honest paused "
                f"{results[name][a]['honest_pause_rate']:.2f}, A_k {results[name][a]['thresholds']}"
                for a in results[name]
            )
        )

    curve = []
    for beta in (float(b) for b in args.betas.split(",") if b):
        row = {"policy": "router", "beta": beta}
        for alpha, col in ((0.01, "tpr_at_1pct_fpr"), (0.05, "tpr_at_5pct_fpr")):
            _, m = run("router", alpha, replace(cfg, beta=beta))
            row[col], (row[f"{col}_lo"], row[f"{col}_hi"]) = (
                m["tpr"],
                m["ci95"]["tpr"] or (None, None),
            )
            if alpha == 0.01:
                row["checks_per_honest_person"] = m["checks_per_honest_person"]
                lohi = m["ci95"]["checks_per_honest_person"] or (None, None)
                row["checks_per_honest_person_lo"], row["checks_per_honest_person_hi"] = lohi
        curve.append(row)
    for name in COMPARE:
        if name == "router":
            continue
        row = {"policy": name, "beta": None}
        for alpha, col in ((0.01, "tpr_at_1pct_fpr"), (0.05, "tpr_at_5pct_fpr")):
            m = results[name].get(f"{alpha:g}") or run(name, alpha, cfg)[1]
            row[col], (row[f"{col}_lo"], row[f"{col}_hi"]) = (
                m["tpr"],
                m["ci95"]["tpr"] or (None, None),
            )
        m = results[name].get("0.01") or run(name, 0.01, cfg)[1]
        row["checks_per_honest_person"] = m["checks_per_honest_person"]
        lohi = m["ci95"]["checks_per_honest_person"] or (None, None)
        row["checks_per_honest_person_lo"], row["checks_per_honest_person_hi"] = lohi
        curve.append(row)

    out = Path(args.results_dir) / "real-router" / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_comparison(results, out)
    pd.DataFrame(thresholds).to_csv(out / "thresholds.csv", index=False)
    pd.DataFrame(curve).to_csv(out / "budget_curve.csv", index=False)
    try:  # scripts/ is not a package: load the plotting script by path
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plot_budget_curve", Path(__file__).resolve().parent / "plot_budget_curve.py"
        )
        plot_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_mod)
        plot_mod.plot(out / "budget_curve.csv", title=f"{run_id}: cross-fitted thresholds")
    except ImportError as e:  # matplotlib missing: the CSV is still written
        print(f"no budget_curve.png ({e}); pip install -e '.[plot]'")
    granularity = results["router"][f"{args.alpha:g}"]["granularity"]
    warnings = []
    if granularity and granularity > args.alpha:
        warnings.append(
            f"With {min(len(f) for f in folds)} people in the smallest fold, one pause is "
            f"{granularity:.0%} > alpha {args.alpha:g}: each A_k is the 'pause nobody in the fold' "
            "threshold, so the 1% and 5% thresholds can coincide."
        )
    for w in warnings:
        print("WARNING: " + w)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "warnings": warnings,
                "final": args.final,
                "mechanisms": mechanisms,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "mode": "real_router_crossfit",
                "final": args.final,
                "ai_outcomes": args.ai_outcomes,
                "pilot_outcomes": args.pilot_outcomes,
                "bank": args.bank,
                "assessment_id": assessment.assessment_id,
                "n_items": len(assessment.item_ids),
                "mechanisms": mechanisms,
                "passive": list(passive),
                "n_people": len(people),
                "folds": folds,
                "fold_seed": args.seed,
                "excluded_sessions": sorted(excluded),
                "excluded_not_found": not_found,
                "alphas": alphas,
                "ai_eval": "heldout" if args.final else "calibration items",
                "n_ai_eval_sessions": int(ai_eval["session_id"].nunique()),
                "calibration_frac": args.calibration_frac,
                "split_seed": args.split_seed,
                "map_versions": [m.version for m in maps],
                "router_config": {
                    k: (v if not isinstance(v, tuple) else list(v)) for k, v in cfg.__dict__.items()
                },
                "group_passive": args.group_passive,
                "policies": list(COMPARE),
                "budget_curve_betas": [r["beta"] for r in curve if r["policy"] == "router"],
                "n_boot": args.boot,
                "git_commit": _git_commit(),
                "created_at": utc_now(),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"\n{(out / 'comparison.md').read_text(encoding='utf-8')}")
    print(
        f"Saved {out}/: comparison.csv/.md, thresholds.csv, budget_curve.csv/.png, "
        "metrics.json, config.yaml"
    )


COLUMNS = (
    "tpr",
    "honest_pause_rate",
    "checks_per_honest_person",
    "median_items_to_decision_honest",
)


def write_comparison(results: dict[str, dict[str, dict]], out: Path) -> None:
    """One row per policy and alpha: each metric with its 95% CI, plus the A_k per fold."""
    rows = []
    for name, by_alpha in results.items():
        for alpha, m in by_alpha.items():
            row = {
                "policy": name,
                "alpha": float(alpha),
                "n_people": m["n_people"],
                "n_ai_sessions": m["n_ai_sessions"],
                "A_k": " ".join(f"{a:g}" for a in m["thresholds"]),
            }
            for c in COLUMNS:
                ci = m["ci95"][c] or (None, None)
                row[c], row[f"{c}_lo"], row[f"{c}_hi"] = m[c], ci[0], ci[1]
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)

    def cell(r, c):
        if r[c] is None or pd.isna(r[c]):
            return "-"
        lo, hi = r[f"{c}_lo"], r[f"{c}_hi"]
        return f"{r[c]:.3f}" if lo is None or pd.isna(lo) else f"{r[c]:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = [
        "| policy | alpha | A_k per fold | " + " | ".join(COLUMNS) + " |",
        "|---|---|---|" + "---|" * len(COLUMNS),
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['policy']} | {r['alpha']:g} | {r['A_k']} | "
            + " | ".join(cell(r, c) for c in COLUMNS)
            + " |"
        )
    (out / "comparison.md").write_text(
        "Cross-fitted thresholds: each A_k is set on fold k's held-out people. TPR = AI sessions "
        "paused for review; 95% bootstrap CIs resample people and AI sessions.\n\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

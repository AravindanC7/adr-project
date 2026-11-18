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
  5. Threshold from replayed honest sessions (adr/replay/crossfit.py). For each of K folds of
     people, map_k = Archana's builder in STRICT mode (never --ai-only) on the AI train rows +
     the OTHER folds' people. Each real person is scored out of fold (with the map that never
     saw them), and --n-synthetic honest sessions per fold are resampled from that fold's people
     only. Three thresholds: A at 1% and at 5% of the pooled resampled sessions, and A_max =
     the lowest threshold that pauses none of the real people. AI sessions are scored with
     every map_k. Done for the router, baselines 1-4 and the router at each beta (the budget
     curve). CIs resample PEOPLE and regenerate the resampled sessions per draw.

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
    cap_warnings,
    deploy_config,
    person_folds,
    router_factory,
    score_policy,
    summarize,
    tail_warnings,
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
    ap.add_argument("--boot", type=int, default=200, help="bootstrap draws over people")
    ap.add_argument(
        "--n-synthetic", type=int, default=500, help="resampled honest sessions per fold"
    )
    ap.add_argument(
        "--n-synthetic-boot",
        type=int,
        default=100,
        help="resampled honest sessions per fold inside each bootstrap draw",
    )
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
    alphas = sorted({args.alpha, 0.01, EXTRA_ALPHA})

    def run(name: str, c: RouterConfig) -> dict:
        make = router_factory(name, assessment, items, variants)
        scored = score_policy(
            make,
            c,
            assessment,
            items,
            human,
            ai_eval,
            folds,
            maps,
            args.n_synthetic,
            passive,
            args.seed,
        )
        return summarize(
            scored,
            make,
            c,
            assessment,
            items,
            human,
            alphas,
            args.boot,
            args.n_synthetic_boot,
            passive,
            args.seed,
        )

    results: dict[str, dict] = {}
    for name in COMPARE:
        results[name] = run(name, cfg)
        r = results[name]["results"]
        print(
            f"  {name:15s} "
            + "  ".join(
                f"{t}={v['A']:.2f}: TPR {v['tpr']:.2f}, honest paused {v['honest_pause_rate']:.2f}"
                for t, v in r.items()
            )
        )

    def key(a: float) -> str:
        return f"A_{a:g}"

    def curve_row(policy: str, beta, summ: dict) -> dict:
        row = {"policy": policy, "beta": beta}
        for a, col in ((0.01, "tpr_at_1pct_fpr"), (0.05, "tpr_at_5pct_fpr")):
            v, ci = summ["results"][key(a)], summ["ci95"][key(a)].get("tpr")
            row[col], (row[f"{col}_lo"], row[f"{col}_hi"]) = v["tpr"], ci or (None, None)
        v, ci = summ["results"][key(0.01)], summ["ci95"][key(0.01)].get("checks_per_honest_person")
        row["checks_per_honest_person"] = v["checks_per_honest_person"]
        row["checks_per_honest_person_lo"], row["checks_per_honest_person_hi"] = ci or (None, None)
        return row

    curve = [
        curve_row("router", float(b), run("router", replace(cfg, beta=float(b))))
        for b in args.betas.split(",")
        if b
    ]
    curve += [curve_row(n, None, results[n]) for n in COMPARE if n != "router"]

    out = Path(args.results_dir) / "real-router" / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_comparison(results, out)
    write_thresholds(results, out)
    pd.DataFrame(curve).to_csv(out / "budget_curve.csv", index=False)
    try:  # scripts/ is not a package: load the plotting script by path
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plot_budget_curve", Path(__file__).resolve().parent / "plot_budget_curve.py"
        )
        plot_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_mod)
        plot_mod.plot(out / "budget_curve.csv", title=f"{run_id}: thresholds from honest replay")
    except ImportError as e:  # matplotlib missing: the CSV is still written
        print(f"no budget_curve.png ({e}); pip install -e '.[plot]'")

    warnings = tail_warnings(results["router"]) + cap_warnings(results["router"], cfg, mechanisms)
    print()
    for w in warnings:
        print(("" if w.startswith("1% tail") else "WARNING: ") + w)
    router_th = results["router"]["thresholds"]
    deploy = {
        name: {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in deploy_config(cfg, a).__dict__.items()
        }
        for name, a in router_th.items()
        if a != float("inf")
    }
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "warnings": warnings,
                "final": args.final,
                "mechanisms": mechanisms,
                "results": results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "mode": "real_router_crossfit",
                "threshold_method": "out-of-fold honest scores + resampled honest sessions "
                "(adr/replay/crossfit.py)",
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
                "n_synthetic_per_fold": args.n_synthetic,
                "n_synthetic_per_fold_in_boot": args.n_synthetic_boot,
                "ai_eval": "heldout" if args.final else "calibration items",
                "n_ai_eval_sessions": int(ai_eval["session_id"].nunique()),
                "calibration_frac": args.calibration_frac,
                "split_seed": args.split_seed,
                "map_versions": [m.version for m in maps],
                "router_config": {
                    k: (v if not isinstance(v, tuple) else list(v)) for k, v in cfg.__dict__.items()
                },
                "router_thresholds": {k: float(v) for k, v in router_th.items()},
                "router_deploy_config": deploy,
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


COLUMNS = ("tpr", "honest_pause_rate", "checks_per_honest_person")


def _fmt(v, ci) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.3f}" if not ci else f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def write_comparison(results: dict[str, dict], out: Path) -> None:
    """One row per policy and threshold (A_0.01, A_0.05, A_max): the threshold, TPR, the real
    honest pause rate and checks per honest person, with 95% CIs (bootstrap over people)."""
    rows = []
    for name, summ in results.items():
        for t, v in summ["results"].items():
            ci = summ["ci95"][t]
            row = {
                "policy": name,
                "threshold": t,
                "A": v["A"],
                "A_lo": (ci.get("A") or (None, None))[0],
                "A_hi": (ci.get("A") or (None, None))[1],
                "n_real_at_or_above": v["n_real_at_or_above"],
                "n_people": summ["n_people"],
                "n_ai_sessions": summ["n_ai_sessions"],
            }
            for c in COLUMNS:
                row[c] = v[c]
                row[f"{c}_lo"], row[f"{c}_hi"] = ci.get(c) or (None, None)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    lines = [
        "| policy | threshold | A | " + " | ".join(COLUMNS) + " |",
        "|---|---|---|" + "---|" * len(COLUMNS),
    ]
    for name, summ in results.items():
        for t, v in summ["results"].items():
            ci = summ["ci95"][t]
            lines.append(
                f"| {name} | {t} | {_fmt(v['A'], ci.get('A'))} | "
                + " | ".join(_fmt(v[c], ci.get(c)) for c in COLUMNS)
                + " |"
            )
    n = next(iter(results.values()))["n_people"]
    (out / "comparison.md").write_text(
        "Thresholds from replayed honest sessions (cross-fitted: every honest score uses a map "
        "that never saw that person). A_0.01 / A_0.05: smallest threshold pausing at most 1% / "
        "5% of the resampled honest sessions; A_max: lowest threshold pausing none of the "
        f"{n} real people. 95% CIs: bootstrap over people (resampled sessions regenerated per "
        f"draw) and AI sessions. 1% tail extrapolated from {n} people.\n\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def write_thresholds(results: dict[str, dict], out: Path) -> None:
    rows = []
    for name, summ in results.items():
        for t, v in summ["results"].items():
            lohi = summ["ci95"][t].get("A") or (None, None)
            rows.append(
                {
                    "policy": name,
                    "threshold": t,
                    "A": v["A"],
                    "A_lo": lohi[0],
                    "A_hi": lohi[1],
                    "n_real_at_or_above": v["n_real_at_or_above"],
                    "honest_pause_rate_real": v["honest_pause_rate"],
                    "pause_rate_resampled": v["resampled_pause_rate"],
                    "tpr": v["tpr"],
                    "n_people": summ["n_people"],
                    "people_per_fold": " ".join(map(str, summ["people_per_fold"])),
                    "n_resampled_sessions": summ["n_resampled_sessions"],
                }
            )
    pd.DataFrame(rows).to_csv(out / "thresholds.csv", index=False)


if __name__ == "__main__":
    main()

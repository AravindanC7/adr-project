"""The first real router + baselines run (step #42): one command, thresholds calibrated on
held-out honest sessions by session-level cross-fitting (adr/replay/crossfit.py).

    python scripts/run_real_router.py \\
        --ai-outcomes results/outcomes/v1-cp-L0-quiz-mmlu-200-s5.parquet ... \\
        --pilot-outcomes results/outcomes/pilot-2026-11-13.parquet \\
        [--low-effort FILE | --no-low-effort] [--exclude-sessions FILE]
        [--alpha 0.01] [--cap-variant per_mech|passive_group|open_text_group ...] [--final]

What it does:
  1. Reads the AI rows (any files or folders) and the pilot rows (data_origin "pilot"). The unit
     is the session (one pilot session_id); one person may span sessions. Folds come from
     adr/replay/folds.assign_session_folds, shared with scripts/calibrate_behaviour.py: K and
     the seed string are the ones recorded in configs/data_decisions.json ("pilot_folds"), and
     the low-effort sessions (--low-effort, default pilot_data/low_effort_ids.txt, full ids;
     any id that is not a pilot session stops the run) are spread across the folds.
     Folds are assigned on ALL pilot sessions; --exclude-sessions FILE then drops the listed
     sessions from their folds, so the with/without low-effort runs use the same folds.
  2. Keeps only the mechanisms that have BOTH AI and human rows, and says which ones.
  3. Replays on the pilot bank (--bank, default server/pilot_items.json: what the router would
     serve). Honest sessions are replayed from their own rows. The AI runs answered other items, so an AI
     session answers each check from its own rows on items of the same type (adr/replay/
     donors.py, "same-type draws"): real behaviour, the map's own exchangeability assumption.
  4. AI rows are split by item as in run_replay.py: the map's AI side uses the train items, and
     the AI sessions are scored on the calibration items (on the held-out rows with --final).
  5. Threshold from replayed honest sessions (adr/replay/crossfit.py). For each of K folds of
     sessions, map_k = Archana's builder in STRICT mode (never --ai-only) on the AI train rows +
     the OTHER folds' sessions. Each real session is scored out of fold (with the map that never
     saw it), and --n-synthetic honest sessions per fold are resampled from that fold's sessions
     only. Three thresholds: A at 1% and at 5% of the pooled resampled sessions, and A_max =
     the lowest threshold that pauses none of the real sessions. AI sessions are scored with
     every map_k. Done for the router, baselines 1-4 and the router at each beta (the budget
     curve). CIs resample SESSIONS and regenerate the resampled sessions per draw.

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
    cap_findings,
    deploy_config,
    router_factory,
    score_policy,
    summarize,
    tail_warnings,
)
from adr.replay.folds import (
    DEFAULT_K,
    DEFAULT_SEED,
    LOW_EFFORT_FILE,
    assign_session_folds,
    fold_lists,
    read_low_effort,
)
from adr.replay.quick_map import (
    CALIBRATION_FRAC,
    SPLIT_SEED,
    decided_heldout_model,
    final_eval_rows,
    split_rows,
)
from adr.router.core import RouterConfig

# Cap designs to compare, as RouterConfig changes from the default.
CAP_VARIANTS: dict[str, dict] = {
    "per_mech": {},  # every mechanism has its own cap (the default)
    "passive_group": {"mech_group_caps": {"passive": tuple(sorted(PASSIVE_MECHANISMS))}},
    "open_text_group": {"mech_group_caps": {"open_text": ("canary", "style")}},
    "no_cap": {"mech_cap_frac": None},  # ABLATION only: one check may decide on its own
}
ABLATIONS = {"no_cap": "router[no_cap] (ablation: single check may decide)"}
EXTRA_ALPHA = 0.05  # always reported too
HEADLINE = "A_0.05"  # the pilot headline: TPR at 5% FPR (1% is secondary: a far tail)


def read_rows(paths: list[str]) -> pd.DataFrame:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        files += sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no outcome files in {paths}")
    return read_outcomes(files)


def read_exclusions(path: str | None) -> set[str]:
    """Full pilot session_ids, one per line; blank lines and # comments are ignored. Any other
    line (a prefix, prose) is an error in main(), never a silent non-match (review A1)."""
    if not path:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {s.strip() for s in lines if s.strip() and not s.strip().startswith("#")}


def exclude_sessions(human: pd.DataFrame, sessions: set[str]) -> tuple[pd.DataFrame, list[str]]:
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
    low = ap.add_mutually_exclusive_group()
    low.add_argument(
        "--low-effort",
        default=str(LOW_EFFORT_FILE),
        help="file: low-effort session_ids, full ids, one per line (default %(default)s); they "
        "are spread across the folds. Any id that is not a pilot session stops the run",
    )
    low.add_argument(
        "--no-low-effort",
        action="store_true",
        help="no low-effort stratification (synthetic data and tests only)",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_K,
        help="K; keep the recorded default (configs/data_decisions.json) for reported runs",
    )
    ap.add_argument(
        "--fold-seed",
        default=DEFAULT_SEED,
        help="fold seed string; keep the recorded default for reported runs",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="extra honest pause rate to report a threshold at (A_<alpha>), pooled over the "
        "resampled honest sessions of all folds; A_0.01, A_0.05 and A_max are always reported",
    )
    ap.add_argument(
        "--cap-variant",
        nargs="+",
        choices=list(CAP_VARIANTS),
        default=["per_mech"],
        help="cap design(s): per_mech (default), passive_group (behaviour + style share one "
        "cap), open_text_group (canary + style share one cap), no_cap (ablation: cap off). The "
        "first is used for every policy and the headline; each extra one adds a "
        "router[<variant>] row on the same folds and split, with thresholds from the same "
        "procedure. no_cap can only be an extra (ablation) row",
    )
    ap.add_argument(
        "--group-passive", action="store_true", help="same as --cap-variant passive_group"
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="FINAL: score the held-out AI rows (never tune on these numbers)",
    )
    ap.add_argument("--bank", default="server/pilot_items.json", help="the pilot bank JSON")
    ap.add_argument("--assessment-id", help="replay one part of the bank (default: the flow)")
    ap.add_argument("--betas", default=",".join(str(b) for b in BETAS))
    ap.add_argument(
        "--seed", type=int, default=0, help="replay and bootstrap seed (folds: --fold-seed)"
    )
    ap.add_argument("--boot", type=int, default=200, help="bootstrap draws over sessions")
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
    ai = ai[ai["respondent_kind"].eq("ai")].reset_index(drop=True)  # unique index (H1)
    rows = read_rows(args.pilot_outcomes)
    human = rows[rows["respondent_kind"].eq("human") & rows["data_origin"].eq("pilot")]
    if human.empty:
        raise SystemExit(
            "no pilot rows (respondent_kind human, data_origin pilot) in "
            f"{args.pilot_outcomes}. Without them use scripts/run_replay.py."
        )
    # Folds on ALL pilot sessions, before any exclusion: the same folds in every variant of the
    # run, and the same as scripts/calibrate_behaviour.py's (adr/replay/folds.py).
    low_effort = [] if args.no_low_effort else args.low_effort
    assignment = assign_session_folds(
        human["session_id"].unique(), args.folds, args.fold_seed, low_effort
    )
    low_found = [] if args.no_low_effort else read_low_effort(args.low_effort)
    excluded = read_exclusions(args.exclude_sessions)
    human, not_found = exclude_sessions(human, excluded)
    if not_found:  # review A1: never match zero sessions silently
        short = [x for x in not_found if len(x) < 32]
        raise SystemExit(
            f"--exclude-sessions: {len(not_found)} line(s) are not full pilot session_ids"
            + (
                f" ({len(short)} are shorter than a full 32-character id, e.g. a prefix or a "
                "line of prose)"
                if short
                else ""
            )
            + ". Use a machine-only file: one full session_id per line, # for comments "
            "(like pilot_data/low_effort_ids.txt)."
        )
    sessions = sorted(human["session_id"].unique())

    mechanisms = usable_mechanisms(ai, human)
    if not mechanisms:
        raise SystemExit("no mechanism has both AI and human rows")
    print(f"Mechanisms with BOTH AI and human rows (the only ones used): {', '.join(mechanisms)}")
    if mechanisms == ["decoy"]:
        print(
            "  -> decoy only: the router has ONE check type, so this run measures how well "
            "decoys separate AI sessions from honest sessions at a threshold calibrated on "
            "honest sessions; it cannot show adaptive selection across mechanisms yet."
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
        # Review H1: only the DECIDED held-out model (configs/data_decisions.json) and training
        # models on the held-out survey; never an earlier held-out model's rows (gemini-flash).
        heldout_model = decided_heldout_model()
        kept = final_eval_rows(splits.heldout, heldout_model)
        dropped = splits.heldout.loc[splits.heldout.index.difference(kept.index)]
        if not dropped.empty:
            print(
                f"WARNING: --final: dropped {len(dropped)} held-out AI rows from other held-out "
                f"models {sorted(dropped['model'].dropna().unique())}: only {heldout_model} is "
                "the held-out model (configs/data_decisions.json)"
            )
        ai_eval = kept[kept["respondent_kind"].eq("ai")]
        if ai_eval.empty:
            raise SystemExit("--final: there are no held-out AI rows")
    else:
        ai_eval = splits.calibration
    if ai_eval.empty:
        raise SystemExit("no AI evaluation rows (raise --calibration-frac)")

    variants_cap = list(dict.fromkeys(args.cap_variant))  # in order, no repeats
    if args.group_passive:  # the older flag: passive_group becomes the main design
        rest = (
            []
            if variants_cap == ["per_mech"]
            else [v for v in variants_cap if v != "passive_group"]
        )
        variants_cap = ["passive_group", *rest]
    if variants_cap[0] in ABLATIONS:
        raise SystemExit(
            f"{variants_cap[0]} is an ablation only, never the main design or the "
            "headline: list a main design first, e.g. --cap-variant per_mech no_cap"
        )
    for v in variants_cap:
        groups = CAP_VARIANTS[v].get("mech_group_caps") or {}
        for g, members in groups.items():
            present = [m for m in members if m in mechanisms]
            if len(present) < 2:
                print(
                    f"NOTE: cap variant {v} has no effect here: of {g} = {list(members)}, only "
                    f"{present or 'none'} have both AI and human rows."
                )
    cfg = replace(RouterConfig(), **CAP_VARIANTS[variants_cap[0]])

    folds = fold_lists(assignment, args.folds, keep=sessions)
    if any(not f for f in folds):
        raise SystemExit(f"a fold is empty after the exclusions: sizes {[len(f) for f in folds]}")
    print(
        f"{len(sessions)} honest sessions in {args.folds} folds of {[len(f) for f in folds]} "
        f"(one person may span sessions; the session is the unit); low-effort sessions per "
        f"fold: {[len(set(f) & set(low_found)) for f in folds]}; "
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

    for v in variants_cap[1:]:  # extra cap designs: the router only, same split and folds;
        # thresholds from the same procedure on this design's own honest scores
        name = ABLATIONS.get(v, f"router[{v}]")
        results[name] = run("router", replace(RouterConfig(), **CAP_VARIANTS[v]))
        r = results[name]["results"]
        print(
            f"  {name:15s} "
            + "  ".join(f"{t}={x['A']:.2f}: TPR {x['tpr']:.2f}" for t, x in r.items())
        )

    def key(a: float) -> str:
        return f"A_{a:g}"

    def curve_row(policy: str, beta, summ: dict) -> dict:
        row = {"policy": policy, "beta": beta}
        for a, col in ((0.01, "tpr_at_1pct_fpr"), (0.05, "tpr_at_5pct_fpr")):
            v, ci = summ["results"][key(a)], summ["ci95"][key(a)].get("tpr")
            row[col], (row[f"{col}_lo"], row[f"{col}_hi"]) = v["tpr"], ci or (None, None)
        # plot_budget_curve.py reads the column "checks_per_honest_person": here it holds checks
        # per honest SESSION at the 1% threshold
        v, ci = summ["results"][key(0.01)], summ["ci95"][key(0.01)].get("checks_per_honest_session")
        row["checks_per_honest_person"] = v["checks_per_honest_session"]
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
    findings = cap_findings(results["router"], cfg, mechanisms)
    headline = headline_text(results["router"], findings.get("A_0.01"))
    notes = [f"{name}: {text}" for name, text in findings.items() if name != "A_0.01"]
    notes += [
        f"{label} is an ablation (the evidence cap switched off, so a single check may decide on "
        "its own): never the headline. Its thresholds come from the same procedure on its own "
        "honest scores, on the same folds."
        for v, label in ABLATIONS.items()
        if label in results
    ]
    write_comparison(results, out, headline, notes)
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

    warnings = tail_warnings(results["router"])
    for g, v in (results["router"].get("ai_sources_by_group") or {}).items():
        print(
            f"AI group {g}: {v['sessions']} sessions; per session {v['checks']:.1f} checks, "
            f"{v['same_type']:.1f} same-type, {v['fallback']:.1f} fallback, "
            f"{v['missing']:.1f} missing"
        )
        if v["missing"] > 0:
            print(f"WARNING: AI group {g} has checks with no rows (missing): they count as misses")
    print(f"\n{headline}")
    for name, text in findings.items():
        if name != "A_0.01":
            print(f"NOTE ({name}): {text}")
    for w in warnings:
        print(("" if w.startswith("1% tail") else "WARNING: ") + w)
    router_th = results["router"]["thresholds"]
    deploy = {
        name: plain(deploy_config(cfg, a).__dict__)
        for name, a in router_th.items()
        if a != float("inf")
    }
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "headline": headline,
                "cap_findings": findings,
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
                "n_sessions": len(sessions),
                "folds": folds,
                "fold_k": args.folds,
                "fold_seed": args.fold_seed,
                "fold_assignment": "adr/replay/folds.assign_session_folds, on all pilot "
                "sessions before exclusions",
                "replay_seed": args.seed,
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
                "router_config": plain(cfg.__dict__),
                "cap_variants": variants_cap,
                "low_effort_sessions": low_found,
                "low_effort_per_fold": [len(set(f) & set(low_found)) for f in folds],
                "unit": "session (one pilot session_id; one person may span sessions)",
                "router_thresholds": {k: float(v) for k, v in router_th.items()},
                "router_deploy_config": deploy,
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
    "ai_cleared_rate",
    "honest_pause_rate",
    "honest_cleared_rate",
    "checks_per_honest_session",
)
ORDER = ("A_0.05", "A_0.01", "A_max")  # headline first


def _columns(results: dict[str, dict]) -> list[str]:
    """COLUMNS, with TPR per AI group (tpr[model:attack_mode:level]) after the pooled TPR when
    there is more than one group (review H3)."""
    keys = {k for summ in results.values() for r in summ["results"].values() for k in r}
    groups = sorted(k for k in keys if k.startswith("tpr["))
    return [COLUMNS[0], *groups, *COLUMNS[1:]]


def sources_text(results: dict[str, dict]) -> str:
    """Per AI group: sessions and, per session (mean over the fold maps, router policy), the
    checks served and where their outcomes came from (review H3)."""
    src = results["router"].get("ai_sources_by_group") or {}
    if not src:
        return ""
    lines = [
        "",
        (
            "AI groups (router): active checks per session and where their outcomes came from "
            "(passive checks such as style are not counted), mean per "
            "session over the fold maps. same_type = the session's own rows on another item of the "
            "same item_type (e.g. quiz MCQ decoy rows answer survey MCQ items: both are mcq); "
            "fallback = another session of the group; missing = no row at all, so the check gives no "
            "evidence and the session stays near the prior (it counts as a miss)."
        ),
        "",
        "| AI group | sessions | checks | same_type | fallback | synthetic | missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, v in src.items():
        lines.append(
            f"| {g} | {v['sessions']} | {v['checks']:.1f} | {v['same_type']:.1f} | "
            f"{v['fallback']:.1f} | {v['synthetic']:.1f} | {v['missing']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def plain(obj):
    """Tuples (e.g. inside mech_group_caps) to lists, so YAML and JSON can write them."""
    return json.loads(json.dumps(obj, default=str))


def _ci_text(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "[no CI]"


def headline_text(summ: dict, finding_1pct: str | None = None) -> str:
    """TPR at 5% FPR is the pilot headline; TPR at 1% FPR (a tail the pilot cannot pin down) is
    reported with its CI as secondary, followed by the cap explanation when it applies."""
    h, s1 = summ["results"]["A_0.05"], summ["results"]["A_0.01"]
    text = (
        f"HEADLINE (pilot): router TPR at 5% FPR = {h['tpr']:.2f} "
        f"{_ci_text(summ['ci95']['A_0.05'].get('tpr'))} (A = {h['A']:.2f}). "
        f"Secondary: TPR at 1% FPR = {s1['tpr']:.2f} "
        f"{_ci_text(summ['ci95']['A_0.01'].get('tpr'))} (A = {s1['A']:.2f})."
    )
    return f"{text} {finding_1pct}" if finding_1pct else text


def _ordered(summ: dict) -> list[str]:
    names = list(summ["results"])
    return [n for n in ORDER if n in names] + [n for n in names if n not in ORDER]


def _fmt(v, ci) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.3f}" if not ci else f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def write_comparison(
    results: dict[str, dict], out: Path, headline: str, notes: list[str] = ()
) -> None:
    """One row per policy and threshold (A_0.01, A_0.05, A_max): the threshold, TPR, the real
    honest pause rate and checks per honest session, with 95% CIs (bootstrap over sessions)."""
    rows = []
    for name, summ in results.items():
        for t in _ordered(summ):
            v = summ["results"][t]
            ci = summ["ci95"][t]
            row = {
                "policy": name,
                "threshold": t,
                "A": v["A"],
                "A_lo": (ci.get("A") or (None, None))[0],
                "A_hi": (ci.get("A") or (None, None))[1],
                "n_real_at_or_above": v["n_real_at_or_above"],
                "n_sessions": summ["n_sessions"],
                "n_ai_sessions": summ["n_ai_sessions"],
            }
            for c in _columns(results):
                row[c] = v.get(c)
                row[f"{c}_lo"], row[f"{c}_hi"] = ci.get(c) or (None, None)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    lines = [
        "| policy | threshold | A | " + " | ".join(_columns(results)) + " |",
        "|---|---|---|" + "---|" * len(_columns(results)),
    ]
    for name, summ in results.items():
        for t in _ordered(summ):
            v = summ["results"][t]
            ci = summ["ci95"][t]
            lines.append(
                f"| {name} | {t} | {_fmt(v['A'], ci.get('A'))} | "
                + " | ".join(_fmt(v.get(c), ci.get(c)) for c in _columns(results))
                + " |"
            )
    n = next(iter(results.values()))["n_sessions"]
    (out / "comparison.md").write_text(
        f"{headline}\n\n"
        "Thresholds from replayed honest sessions (cross-fitted: every honest score uses a map "
        "that never saw that session). A_0.05 (headline) / A_0.01: smallest threshold pausing at "
        "most 5% / 1% of the resampled honest sessions; A_max: lowest threshold pausing none of "
        f"the {n} real sessions. 95% CIs: bootstrap over honest sessions (resampled sessions "
        f"regenerated per draw) and AI sessions. 1% tail extrapolated from {n} sessions. The unit "
        "is the session: one person may span sessions.\n\n"
        + "\n".join(lines)
        + "\n"
        + "".join(f"\n- {note}" for note in notes)
        + ("\n" if notes else "")
        + sources_text(results),
        encoding="utf-8",
    )


def write_thresholds(results: dict[str, dict], out: Path) -> None:
    rows = []
    for name, summ in results.items():
        for t in _ordered(summ):
            v = summ["results"][t]
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
                    "n_sessions": summ["n_sessions"],
                    "sessions_per_fold": " ".join(map(str, summ["sessions_per_fold"])),
                    "n_resampled_sessions": summ["n_resampled_sessions"],
                }
            )
    pd.DataFrame(rows).to_csv(out / "thresholds.csv", index=False)


if __name__ == "__main__":
    main()

"""Freeze reports and post-run checks for the final router runs (scripts/freeze_run.sh, step c).

    python scripts/freeze_report.py --out results/freeze-2026-11-22/README.md \\
        --unseen-model RUN_DIR [RUN_DIR] --unseen-survey RUN_DIR [RUN_DIR] \\
        [--reference-all results/real-router/rq2-l012-floor750-reopen-all] \\
        [--reference-no-low-effort results/real-router/rq2-l012-floor750-reopen-no-low-effort]

Writes the README of the final run: the FPR rule, then two separate reports, each with TPR at the
A_0.05 and A_0.01 thresholds per attacker level and mode (95% bootstrap CIs over sessions):

  unseen model   claude-haiku composite sessions on the training banks
  unseen survey  gpt-mini and claude-haiku on the held-out OSMI bank

Checks, each failing loudly (exit 1) after writing what it found:
  * no claude-haiku group has synthetic or fallback outcomes > 0 in the per-group source table:
    that would mean its checks were answered from other sessions or from the failure map, which
    is fitted on gpt-mini rows, rather than from claude-haiku's own rows;
  * the honest side is the pilot's: each run's thresholds equal #51's run with the same honest
    sessions (all 44, or without the low-effort ones; picked by the run folder's name), because
    the folds, pilot and config are the same.
Reads aggregates only (metrics.json); no session ids, no answer text.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

FPR_RULE = (
    "False-flag rates come only from the 44 pilot honest sessions, with the same folds and "
    "calibrated thresholds as #51; the held-out OSMI bank has no human side, so no FPR is "
    "reported on it."
)
THRESHOLDS = ("A_0.05", "A_0.01")
LEVELS = ("L0", "L2", "L2g")
MODES = ("copy_paste", "doc_upload")


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["results"]["router"]


def source_problems(router: dict) -> list[str]:
    """claude-haiku groups whose checks did not all come from claude-haiku's own rows."""
    bad = []
    for group, src in (router.get("ai_sources_by_group") or {}).items():
        if group.startswith("claude-haiku:") and (src["synthetic"] > 0 or src["fallback"] > 0):
            bad.append(f"{group}: synthetic {src['synthetic']}, fallback {src['fallback']} per session")
    return bad


def threshold_problems(router: dict, reference: dict, tol: float = 1e-9) -> list[str]:
    return [
        f"{t}: {router['thresholds'][t]} vs reference {reference['thresholds'][t]}"
        for t in THRESHOLDS
        if not math.isclose(router["thresholds"][t], reference["thresholds"][t], abs_tol=tol)
    ]


def reference_for(run: Path, ref_all: Path | None, ref_no_low: Path | None) -> Path | None:
    """#51's run with the same honest sessions as this one."""
    return ref_no_low if run.name.endswith("no-low-effort") else ref_all


def _cell(router: dict, t: str, group: str) -> str:
    v = router["results"][t].get(f"tpr[{group}]")
    if v is None:
        return "–"
    ci = router["ci95"][t].get(f"tpr[{group}]")
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def table(router: dict, models: tuple[str, ...]) -> list[str]:
    n_by = router.get("n_ai_sessions_by_group") or {}
    src = router.get("ai_sources_by_group") or {}
    lines = [
        ("| model | level | mode | AI sessions | TPR at A_0.05 [95% CI] | TPR at A_0.01 [95% CI] "
         "| missing checks per session |"),
        "|---|---|---|---|---|---|---|",
    ]
    for model in models:
        for level in LEVELS:
            for mode in MODES:
                g = f"{model}:{mode}:{level}"
                if g not in n_by:
                    continue
                miss = src.get(g, {}).get("missing")
                lines.append(
                    f"| {model} | {level} | {mode.replace('_', '-')} | {n_by[g]} | "
                    f"{_cell(router, 'A_0.05', g)} | {_cell(router, 'A_0.01', g)} | "
                    f"{'–' if miss is None else f'{miss:.1f}'} |"
                )
    return lines


def honest(router: dict) -> list[str]:
    n = router["n_sessions"]
    out = []
    for t in THRESHOLDS:
        v = router["results"][t]
        k = round(v["honest_pause_rate"] * n)
        out.append(f"{t} = {router['thresholds'][t]:.3f}: {k} of {n} real honest sessions paused "
                   f"(nominal {v.get('nominal_fpr', '–')})")
    return out


def report(name: str, desc: str, runs: list[Path], models: tuple[str, ...]) -> list[str]:
    lines = [f"## {name}", "", desc, ""]
    for run in runs:
        router = load(run)
        lines += [f"### {run.name} ({router['n_sessions']} honest sessions)", "",
                  *table(router, models), "",
                  "Honest side (the pilot's; see the FPR rule): " + "; ".join(honest(router)), ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unseen-model", type=Path, nargs="+", required=True)
    ap.add_argument("--unseen-survey", type=Path, nargs="+", required=True)
    ap.add_argument("--reference-all", type=Path, help="#51's rq2-l012-floor750-reopen-all")
    ap.add_argument("--reference-no-low-effort", type=Path,
                    help="#51's rq2-l012-floor750-reopen-no-low-effort")
    a = ap.parse_args()

    lines = ["# Final run (experiment freeze, 2026-11-22)", "", f"**FPR rule.** {FPR_RULE}", ""]
    lines += report(
        "Unseen model",
        "claude-haiku (the held-out model) composite sessions on the training banks (quiz MMLU/ARC,"
        " survey IPIP/GSS, assignment-asurre-train).",
        a.unseen_model, ("claude-haiku",),
    )
    lines += report(
        "Unseen survey",
        "gpt-mini and claude-haiku on the held-out OSMI bank (written answers only). Only the "
        "pilot's open-text checks (canary, style) have rows here; its other checks have none, so "
        "they show as missing and give no evidence.",
        a.unseen_survey, ("gpt-mini", "claude-haiku"),
    )

    problems = []
    for run in [*a.unseen_model, *a.unseen_survey]:
        router = load(run)
        problems += [f"{run.name}: {p}" for p in source_problems(router)]
        ref = reference_for(run, a.reference_all, a.reference_no_low_effort)
        if ref is None:
            problems.append(f"{run.name}: no #51 reference run to check its thresholds against")
        else:
            problems += [f"{run.name}: threshold differs from #51 ({ref.name}), {p}"
                         for p in threshold_problems(router, load(ref))]
    lines += ["## Post-run checks", ""]
    lines += [f"- FAILED: {p}" for p in problems] or [
        ("- claude-haiku groups: every check answered from claude-haiku's own rows "
         "(synthetic 0, fallback 0)."),
        "- thresholds: equal to #51's runs with the same honest sessions.",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    if problems:
        for p in problems:
            print(f"POST-RUN CHECK FAILED: {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

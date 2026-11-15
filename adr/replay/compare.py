"""Run several policies on the same simulated respondents and tabulate the results.

Owner: Arav. Every policy sees the same respondents: respondent i of group g copies the same
donor session, with the same seed, under every policy. So differences between rows of the
table come from the policies, not from who happened to be sampled.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from adr.contracts import Assessment, FailureMap, Item, RenderedItem, Router
from adr.replay.donors import Donors, Group
from adr.replay.metrics import METRICS, evaluate
from adr.replay.simulate import ReplayResult, replay_session
from adr.router.core import RouterConfig, SPRTRouter

BETAS = (0.01, 0.02, 0.05, 0.1, 0.2)
CURVE_METRICS = ("tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "checks_per_honest_person")


def run_policy(
    router: Router,
    assessment: Assessment,
    donors: Donors,
    groups: list[Group],
    n: int,
    seed: int,
    passive: tuple[str, ...] = (),
) -> dict[Group, list[ReplayResult]]:
    return {g: [replay_session(router, assessment, donors.respondent(g, seed + i), passive,
                               seed=seed + i) for i in range(n)]
            for g in groups}


def summarize(
    runs: dict[Group, list[ReplayResult]], n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Metrics overall (every AI group vs every honest group) and per AI group, plus where the
    outcomes came from."""
    ai_groups = [g for g in runs if g.kind == "ai"]
    honest = [r for g in runs if g.kind == "human" for r in runs[g]]
    return {
        "overall": evaluate([r for g in ai_groups for r in runs[g]], honest, n_boot, seed),
        "by_group": {g.label: evaluate(runs[g], honest, n_boot, seed) for g in ai_groups},
        "sources": {g.label: {
            "respondents": len(rs),
            "donor_sessions_used": len({r.donor_session for r in rs} - {None}),
            "fallbacks": sum(r.fallbacks for r in rs),
            "synthetic_draws": sum(r.synthetic for r in rs),
            "skipped_no_observation": sum(r.skipped for r in rs),
            "checks_run": sum(r.checks_run for r in rs),
        } for g, rs in runs.items()},
    }


def compare(
    policies: dict[str, Router],
    assessment: Assessment,
    donors: Donors,
    n: int,
    seed: int,
    passive: tuple[str, ...] = (),
    n_boot: int = 1000,
) -> dict[str, dict[str, Any]]:
    groups = donors.groups("ai") + donors.groups("human")
    return {name: summarize(run_policy(router, assessment, donors, groups, n, seed, passive),
                            n_boot, seed)
            for name, router in policies.items()}


def comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per policy: each metric with its 95% CI (columns <metric>, _lo, _hi)."""
    rows = []
    for name, res in results.items():
        m = res["overall"]
        row: dict[str, Any] = {"policy": name, "n_ai": m["n_ai"], "n_human": m["n_human"]}
        for metric in METRICS:
            ci = m["ci95"][metric]
            row[metric] = m[metric]
            row[f"{metric}_lo"], row[f"{metric}_hi"] = ci if ci else (None, None)
        rows.append(row)
    return pd.DataFrame(rows)


def write_comparison(table: pd.DataFrame, run_dir: str | Path) -> tuple[Path, Path]:
    """comparison.csv (every number) and comparison.md (value [lo, hi] per cell)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = run_dir / "comparison.csv", run_dir / "comparison.md"
    table.to_csv(csv_path, index=False)

    def cell(row: pd.Series, metric: str) -> str:
        value, lo, hi = row[metric], row[f"{metric}_lo"], row[f"{metric}_hi"]
        if value is None or pd.isna(value):
            return "-"
        if lo is None or pd.isna(lo):
            return f"{value:.3f}"
        return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = ["| policy | " + " | ".join(METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for _, row in table.iterrows():
        lines.append(f"| {row['policy']} | " + " | ".join(cell(row, m) for m in METRICS) + " |")
    md_path.write_text("Metrics with 95% bootstrap intervals [lo, hi].\n\n" + "\n".join(lines)
                       + "\n", encoding="utf-8")
    return csv_path, md_path


def _curve_row(policy: str, beta: float | None, m: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"policy": policy, "beta": beta}
    for metric in CURVE_METRICS:
        ci = m["ci95"][metric]
        row[metric] = m[metric]
        row[f"{metric}_lo"], row[f"{metric}_hi"] = ci if ci else (None, None)
    return row


def budget_curve(
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    fmap: FailureMap,
    donors: Donors,
    n: int,
    seed: int,
    baselines: dict[str, dict[str, Any]],
    betas: tuple[float, ...] = BETAS,
    cfg: RouterConfig | None = None,
    passive: tuple[str, ...] = (),
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Detection vs checks per honest person: the router at each beta (its target miss rate),
    on the same respondents, plus one row per baseline from `baselines` (compare() output).

    A larger beta raises the clearing threshold B, so honest people are cleared sooner and get
    fewer checks; it also lowers A a little. The curve shows what each extra check buys.
    """
    cfg = cfg or RouterConfig()
    groups = donors.groups("ai") + donors.groups("human")
    rows = []
    for beta in betas:
        router = SPRTRouter(items, variants, fmap, replace(cfg, beta=beta))
        runs = run_policy(router, assessment, donors, groups, n, seed, passive)
        rows.append(_curve_row("router", beta, summarize(runs, n_boot, seed)["overall"]))
    for name, res in baselines.items():
        rows.append(_curve_row(name, None, res["overall"]))
    return pd.DataFrame(rows)

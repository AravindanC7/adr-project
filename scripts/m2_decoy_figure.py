"""M2 results figure: how often gpt-mini picks the planted decoy, per attack mode x bank.

Uses the repeated runs only (several sessions each), not the single-session runs:
    v1-cp-L0-<bank>-s5    copy-paste, 5 samples at temperature 0.7 (5 sessions)
    v1-du-L0-<bank>-q20   doc-upload, 20 questions per PDF (10 decoy sessions per bank)

Rate = share of decoy-variant answers that picked the decoy. 95% CI from a two-level
bootstrap: resample items with replacement, then, within each sampled item, resample that
item's sessions with replacement (2000 reps, fixed seed). In doc-upload each item is in one
decoy PDF, so the inner level has one observation per item and only items vary.

Each cell is also split by `knows` from results/runs/v1-item-difficulty.csv (made by
scripts/item_difficulty.py): 1 if the model answered the item's PLAIN version correctly in the
temperature-0 copy-paste run. The same per-item split is used for the doc-upload cells.
An empty group (e.g. the model knows every item) is NaN in the CSV and "n/a" in the table/plot.

Run from the repo root (needs results/outcomes/ and the difficulty CSV):
    python scripts/m2_decoy_figure.py
Writes results/m2/decoy_rates.csv, decoy_rates.png, decoy_rates.pdf and decoy_table.tex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.analysis import style as figstyle
from adr.analysis.bootstrap import two_level_bootstrap
from adr.contracts import read_outcomes

MODEL = "gpt-mini"
CELLS = [  # (mode label, bank label, run id)
    ("Copy-paste", "MMLU", "v1-cp-L0-quiz-mmlu-200-s5"),
    ("Copy-paste", "ARC", "v1-cp-L0-quiz-arc-200-s5"),
    ("Doc upload", "MMLU", "v1-du-L0-quiz-mmlu-200-q20"),
    ("Doc upload", "ARC", "v1-du-L0-quiz-arc-200-q20"),
]
GROUPS = [  # (column suffix / key, legend label, filter on knows)
    ("all", "All items", None),
    ("knows", "Model knows answer", 1),
    ("unknown", "Model doesn't", 0),
]
N_BOOT = 2000
SEED = 576


def compute(outcomes_dir: Path, difficulty: Path, n_boot: int, seed: int) -> pd.DataFrame:
    knows = pd.read_csv(difficulty)
    knows = knows[knows.model == MODEL].set_index("item_id").knows
    rng = np.random.default_rng(seed)
    rows = []
    for mode, bank, run_id in CELLS:
        df = read_outcomes(outcomes_dir / f"{run_id}.parquet")
        df = df[(df.mechanism == "decoy") & (df.model == MODEL)].copy()
        df["hit"] = (df.outcome == "picked_decoy").astype(float)
        df["knows"] = df.item_id.map(knows)
        if df.knows.isna().any():
            raise ValueError(f"{run_id}: items missing from {difficulty}")
        for key, label, flag in GROUPS:
            sub = df if flag is None else df[df.knows == flag]
            per_item = [g.hit.to_numpy() for _, g in sub.groupby("item_id", sort=True)]
            boot = two_level_bootstrap(per_item, n_boot, rng)
            lo, hi = (np.nan, np.nan) if sub.empty else np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "mode": mode,
                    "bank": bank,
                    "run_id": run_id,
                    "group": key,
                    "group_label": label,
                    "n_items": len(per_item),
                    "n_obs": len(sub),
                    "n_sessions": sub.session_id.nunique(),
                    "picked_decoy": int(sub.hit.sum()),
                    "rate": sub.hit.mean() if len(sub) else np.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figstyle.apply(plt)
    ink, muted, grid = figstyle.INK, figstyle.MUTED, figstyle.GRID
    # Okabe-Ito (colour-blind safe), plus hatching so the bars survive greyscale printing.
    style = {
        "all": {"color": "#6E6E6E", "hatch": None},
        "knows": {"color": "#0072B2", "hatch": "///"},
        "unknown": {"color": "#E69F00", "hatch": "..."},
    }
    fig, ax = plt.subplots(figsize=(3.3, 2.15))  # single ACL column is ~3.3 in
    cells = [(m, b) for m, b, _ in CELLS]
    x = np.arange(len(cells))
    width = 0.26
    for j, (key, label, _) in enumerate(GROUPS):
        sub = table[table.group == key].set_index(["mode", "bank"]).loc[cells]
        pos = x + (j - 1) * width
        for p in pos[sub.rate.isna().to_numpy()]:  # empty group: no bar, say so
            ax.text(p, 1.5, "n/a", ha="center", va="bottom", rotation=90, fontsize=5.5, color=muted)
        sub = sub.fillna(0.0)
        ax.bar(
            pos,
            sub.rate * 100,
            width * 0.92,
            label=label,
            color=style[key]["color"],
            hatch=style[key]["hatch"],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax.errorbar(
            pos,
            sub.rate * 100,
            yerr=[(sub.rate - sub.ci_low) * 100, (sub.ci_high - sub.rate) * 100],
            fmt="none",
            ecolor=ink,
            elinewidth=0.7,
            capsize=1.6,
            capthick=0.7,
            zorder=3,
        )
    ax.set_xticks(x, [f"{m}\n{b}" for m, b in cells], color=ink)
    ax.set_ylabel("Picked the decoy (%)", color=ink)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color=grid, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    figstyle.tidy_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.9,
        handletextpad=0.4,
        labelcolor=ink,
    )
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"decoy_rates.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex(table: pd.DataFrame) -> str:
    def cell(r) -> str:
        # Rate (items) on top, CI underneath: keeps the table inside one ACL column.
        if pd.isna(r.rate):
            return f"n/a {{\\tiny({r.n_items})}}"
        return (
            "\\begin{tabular}[c]{@{}c@{}}"
            f"{r.rate * 100:.1f} {{\\tiny({r.n_items})}}\\\\"
            f"{{\\scriptsize[{r.ci_low * 100:.1f}, {r.ci_high * 100:.1f}]}}"
            "\\end{tabular}"
        )

    lines = [
        "% Generated by scripts/m2_decoy_figure.py; do not edit by hand.",
        "% Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        (
            "Mode & Bank & All items & \\begin{tabular}[c]{@{}c@{}}Knows\\\\answer\\end{tabular} & "
            "\\begin{tabular}[c]{@{}c@{}}Doesn't\\\\know\\end{tabular} \\\\"
        ),
        "\\midrule",
    ]
    for mode, bank, _ in CELLS:
        sub = table[(table["mode"] == mode) & (table.bank == bank)].set_index("group")
        lines.append(
            f"{mode} & {bank} & {cell(sub.loc['all'])} & {cell(sub.loc['knows'])} & "
            f"{cell(sub.loc['unknown'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        (
            "\\caption{Share of decoy-variant answers in which gpt-mini (L0) picked the planted "
            "decoy, \\% with 95\\% CI (two-level bootstrap over items, then sessions within items; "
            f"{N_BOOT} reps). Numbers in parentheses are items per cell; the ``doesn't know'' "
            "cells are small, and a cell where every answer picked the decoy has a zero-width "
            "interval. Copy-paste: 5 sessions at "
            "temperature 0.7; doc upload: 20 questions per PDF. ``Knows answer'': the model "
            "answered the plain version correctly in the temperature-0 copy-paste run; the same "
            "per-item split is used for the doc-upload cells. n/a: no items in that group.}"
        ),
        "\\label{tab:decoy-rates}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outcomes", type=Path, default=Path("results/outcomes"))
    ap.add_argument("--difficulty", type=Path, default=Path("results/runs/v1-item-difficulty.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/m2"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    table = compute(args.outcomes, args.difficulty, args.boot, args.seed)
    table.to_csv(args.out / "decoy_rates.csv", index=False, float_format="%.4f")
    plot(table, args.out)
    (args.out / "decoy_table.tex").write_text(latex(table), encoding="utf-8")
    show = table.assign(
        rate=(table.rate * 100).round(1),
        ci_low=(table.ci_low * 100).round(1),
        ci_high=(table.ci_high * 100).round(1),
    )
    print(
        show[
            [
                "mode",
                "bank",
                "group",
                "n_items",
                "n_sessions",
                "picked_decoy",
                "n_obs",
                "rate",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

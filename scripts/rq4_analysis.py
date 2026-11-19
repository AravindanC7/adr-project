"""RQ4 (step #41): how much does decoy detection drop when the attacker is more careful?

gpt-mini only, on the two decoy quiz banks (MMLU, ARC), each attack mode at each attacker level:
    copy-paste   v1-cp-<L>-quiz-<bank>-200-s5    L0-L4 + L2-generic; 5 samples at temperature 0.7
    doc upload   v1-du-<L>-quiz-<bank>-200-q20   L0, L1, L2, L2-generic, L4; 20 questions per PDF

L2 ("informed") uses the harness's L2 prompt, which names our checks, the decoy first ("hidden
'answer key' text pointing to a wrong option"). L2-generic (adr/harness/prompts.py
SYSTEM_VARIANTS, prompt versions cp-v1+l2-generic-v1 / du-v1+l2-generic-v1) has the same informed
role but only warns about hidden instructions in general and names no check; its rows keep
attacker_level L2. Same items, samples and settings as L2.

For every mode x bank x level:
  rate   share of decoy-variant answers that picked the decoy, with a 95% two-level bootstrap
         CI (items, then sessions within items; adr/analysis/bootstrap.py, as in the M2 figure)
  drop   rate(L_n) - rate(L0), the RQ4 number, with a 95% PAIRED bootstrap: the same item draws
         are used for both levels (sessions resampled within each). Negative = fewer decoys.
For L3 (copy-paste only), from results/runs/<run>/responses.jsonl:
  first flagged   how often the first answer picked the decoy (and so triggered the retry)
  got away        flagged first, not flagged on the retry (share of all decoy answers, and of
                  the flagged ones)

Run from the repo root (needs results/outcomes/ and results/runs/ for these runs):
    python scripts/rq4_analysis.py
Writes results/rq4/rq4_rates.csv, rq4_table.tex, rq4_l2_generic.md, rq4_drop.png and
rq4_drop.pdf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.analysis import style as figstyle
from adr.analysis.bootstrap import paired_difference_bootstrap, two_level_bootstrap
from adr.contracts import read_outcomes

MODEL = "gpt-mini"
BANKS = [("MMLU", "mmlu"), ("ARC", "arc")]
MODES = [  # (label, run-id code, run-id suffix, levels)
    ("Copy-paste", "cp", "s5", ["L0", "L1", "L2", "L2-generic", "L3", "L4"]),
    ("Doc upload", "du", "q20", ["L0", "L1", "L2", "L2-generic", "L4"]),
]
LEVEL_NAMES = {"L0": "naive", "L1": "warned", "L2": "informed", "L2-generic": "informed, generic",
               "L3": "retry", "L4": "persona"}
LABEL = {"L2-generic": "L2g"}  # short row / tick label (keeps the table in one ACL column)
N_BOOT = 2000
SEED = 576


def run_id(code: str, level: str, bank: str, suffix: str) -> str:
    return f"v1-{code}-{level}-quiz-{bank}-200-{suffix}"


def decoy_hits(outcomes_dir: Path, rid: str) -> dict[str, np.ndarray]:
    """item_id -> 0/1 array (picked the decoy), one entry per session, for gpt-mini."""
    df = read_outcomes(outcomes_dir / f"{rid}.parquet")
    df = df[(df.mechanism == "decoy") & (df.model == MODEL)]
    if df.empty:
        raise ValueError(f"{rid}: no {MODEL} decoy rows")
    hits = (df.outcome == "picked_decoy").astype(float)
    return {i: g.to_numpy() for i, g in hits.groupby(df.item_id, sort=True)}


def ci(boot: np.ndarray) -> tuple[float, float]:
    if np.isnan(boot).all():
        return (np.nan, np.nan)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def l3_records(runs_dir: Path, rid: str) -> pd.DataFrame:
    """One row per checked (decoy) L3 answer: item, flagged first, flagged after the retry."""
    rows = []
    with (runs_dir / rid / "responses.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["model"] != MODEL or "flagged" not in r or "|decoy|" not in r["variant_id"]:
                continue
            rows.append(
                {
                    "item_id": r["item_id"],
                    "flagged": bool(r["flagged"]),
                    "final_flagged": bool(r.get("final_flagged")),
                }
            )
    if not rows:
        raise ValueError(f"{rid}: no L3 decoy records in responses.jsonl")
    return pd.DataFrame(rows)


def per_item(df: pd.DataFrame, col: str) -> list[np.ndarray]:
    return [g[col].astype(float).to_numpy() for _, g in df.groupby("item_id", sort=True)]


def compute(outcomes_dir: Path, runs_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for mode, code, suffix, levels in MODES:
        for bank_label, bank in BANKS:
            base_id = run_id(code, "L0", bank, suffix)
            base = decoy_hits(outcomes_dir, base_id)
            for level in levels:
                rid = run_id(code, level, bank, suffix)
                hits = base if level == "L0" else decoy_hits(outcomes_dir, rid)
                items = sorted(hits)
                arrays = [hits[i] for i in items]
                n_obs = sum(len(a) for a in arrays)
                rate = float(np.concatenate(arrays).mean())
                lo, hi = ci(two_level_bootstrap(arrays, n_boot, rng))
                row = {
                    "mode": mode,
                    "bank": bank_label,
                    "level": level,
                    "level_name": LEVEL_NAMES[level],
                    "run_id": rid,
                    "n_items": len(items),
                    "n_obs": n_obs,
                    "picked_decoy": int(sum(a.sum() for a in arrays)),
                    "rate": rate,
                    "ci_low": lo,
                    "ci_high": hi,
                }
                if level == "L0":
                    row.update(
                        drop=0.0, drop_ci_low=np.nan, drop_ci_high=np.nan, n_items_paired=len(items)
                    )
                else:
                    # point estimate and CI over the same items (all of them unless a level is
                    # missing an item, e.g. a failed call)
                    paired = sorted(set(items) & set(base))
                    a = [base[i] for i in paired]
                    b = [hits[i] for i in paired]
                    dlo, dhi = ci(paired_difference_bootstrap(a, b, n_boot, rng))
                    drop = float(np.concatenate(b).mean() - np.concatenate(a).mean())
                    row.update(
                        drop=drop, drop_ci_low=dlo, drop_ci_high=dhi, n_items_paired=len(paired)
                    )
                if level == "L3":
                    rec = l3_records(runs_dir, rid)
                    rec["got_away"] = rec.flagged & ~rec.final_flagged
                    f_lo, f_hi = ci(two_level_bootstrap(per_item(rec, "flagged"), n_boot, rng))
                    g_lo, g_hi = ci(two_level_bootstrap(per_item(rec, "got_away"), n_boot, rng))
                    flagged = rec[rec.flagged]
                    c_lo, c_hi = ci(two_level_bootstrap(per_item(flagged, "got_away"), n_boot, rng))
                    row.update(
                        l3_n_answers=len(rec),
                        l3_n_first_flagged=int(rec.flagged.sum()),
                        l3_first_flag_rate=float(rec.flagged.mean()),
                        l3_first_flag_ci_low=f_lo,
                        l3_first_flag_ci_high=f_hi,
                        l3_n_got_away=int(rec.got_away.sum()),
                        l3_got_away_rate=float(rec.got_away.mean()),
                        l3_got_away_ci_low=g_lo,
                        l3_got_away_ci_high=g_hi,
                        l3_got_away_given_flag=float(flagged.got_away.mean())
                        if len(flagged)
                        else np.nan,
                        l3_got_away_given_flag_ci_low=c_lo,
                        l3_got_away_given_flag_ci_high=c_hi,
                    )
                out.append(row)
    return pd.DataFrame(out)


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}"


def _pp(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}".replace("-", "$-$")


def latex(table: pd.DataFrame) -> str:
    """One row per mode x level, MMLU and ARC side by side: picked-decoy rate and the drop vs L0,
    each with its 95% CI underneath (keeps the table inside one ACL column)."""

    def stack(top: str, bottom: str) -> str:
        return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{top}\\\\{{\\scriptsize {bottom}}}\\end{{tabular}}"

    def rate_cell(r) -> str:
        return stack(_pct(r.rate), f"[{_pct(r.ci_low)}, {_pct(r.ci_high)}]")

    def drop_cell(r) -> str:
        if r.level == "L0":
            return "---"
        return stack(_pp(r["drop"]), f"[{_pp(r.drop_ci_low)}, {_pp(r.drop_ci_high)}]")

    banks = [b for b, _ in BANKS]
    lines = [
        "% Generated by scripts/rq4_analysis.py; do not edit by hand.",
        "% Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.6pt}",
        "\\begin{tabular}{l" + "cc" * len(banks) + "}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{b}}}" for b in banks) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(banks))),
        "Level" + " & Decoy (\\%) & Drop (pp)" * len(banks) + " \\\\",
    ]
    for mode, _, _, levels in MODES:
        lines += [
            "\\midrule",
            f"\\multicolumn{{{1 + 2 * len(banks)}}}{{@{{}}l}}{{\\textit{{{mode}}}}} \\\\",
        ]
        for level in levels:
            cells = []
            for b in banks:
                r = table[(table["mode"] == mode) & (table.bank == b) & (table.level == level)]
                r = r.iloc[0]
                cells += [rate_cell(r), drop_cell(r)]
            lines.append(f"{LABEL.get(level, level)} & " + " & ".join(cells) + " \\\\")
    l3 = table[table.level == "L3"].set_index("bank")
    l3_text = "; ".join(
        f"{b}: {_pct(l3.loc[b].l3_first_flag_rate)}\\% of first answers flagged, "
        f"{_pct(l3.loc[b].l3_got_away_given_flag)}\\% of those not flagged on the retry"
        for b in banks
    )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        (
            "\\caption{RQ4: share of decoy answers in which gpt-mini picked the planted decoy "
            "(Decoy) and the change versus L0 in percentage points (Drop; negative = fewer decoys "
            "picked), with 95\\% CIs underneath (rate: two-level bootstrap over items, then "
            "sessions; drop: paired bootstrap over the same items; "
            f"{N_BOOT} reps). Levels: L0 naive, L1 warned, L2 informed of our checks (the prompt "
            "names the decoy), L2g the same informed role warned only about hidden "
            "instructions in general (names no check), L3 one "
            "retry after a flag (copy-paste only), L4 human persona. Copy-paste: 5 sessions at "
            "temperature 0.7; doc upload: 20 questions per PDF. L3 "
            f"{l3_text}. Decoy items on quiz banks only, one model.}}"
        ),
        "\\label{tab:rq4}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def l2_markdown(table: pd.DataFrame) -> str:
    """Decoy rate at L2 (prompt names the decoy) and L2 generic (names no check), side by side."""
    lines = ["| Mode | Bank | L0 | L2: names our checks | L2 generic: names no check |",
             "|---|---|---|---|---|"]

    def cell(r) -> str:
        return (f"{int(r.picked_decoy)}/{int(r.n_obs)} = {_pct(r.rate)}% "
                f"[{_pct(r.ci_low)}, {_pct(r.ci_high)}]")

    for mode, _, _, _ in MODES:
        for b, _ in BANKS:
            sub = table[(table["mode"] == mode) & (table.bank == b)].set_index("level")
            lines.append(f"| {mode} | {b} | {cell(sub.loc['L0'])} | {cell(sub.loc['L2'])} | "
                         f"{cell(sub.loc['L2-generic'])} |")
    return "\n".join(lines) + "\n"


def plot(table: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figstyle.apply(plt)
    series = {  # Okabe-Ito blue / orange, plus distinct markers for greyscale
        "Copy-paste": {"color": "#0072B2", "marker": "o", "dx": -0.09},
        "Doc upload": {"color": "#E69F00", "marker": "s", "dx": 0.09},
    }
    levels = ["L1", "L2", "L2-generic", "L3", "L4"]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, len(BANKS), figsize=(figstyle.ACL_COLUMN_IN, 1.95), sharey=True)
    for ax, (bank_label, _) in zip(axes, BANKS):
        ax.axhline(0, color=figstyle.MUTED, linewidth=0.6, zorder=1)
        for mode, s in series.items():
            sub = table[(table["mode"] == mode) & (table.bank == bank_label)].set_index("level")
            have = [lv for lv in levels if lv in sub.index]
            pos = np.array([levels.index(lv) for lv in have]) + s["dx"]
            d = sub.loc[have]
            ax.errorbar(
                pos,
                d["drop"] * 100,
                yerr=[(d["drop"] - d.drop_ci_low) * 100, (d.drop_ci_high - d["drop"]) * 100],
                fmt=s["marker"],
                color=s["color"],
                markersize=3.6,
                elinewidth=0.8,
                capsize=1.6,
                capthick=0.8,
                label=mode,
                zorder=3,
            )
        ax.set_title(bank_label, fontsize=7.5, color=figstyle.INK, pad=3)
        ax.set_xticks(x, [LABEL.get(lv, lv) for lv in levels], color=figstyle.INK)
        ax.yaxis.grid(True, color=figstyle.GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        figstyle.tidy_axes(ax)
    axes[0].set_ylabel("Drop in decoy rate vs L0 (pp)", color=figstyle.INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.9))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
        labelcolor=figstyle.INK,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"rq4_drop.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outcomes", type=Path, default=Path("results/outcomes"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/rq4"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    table = compute(args.outcomes, args.runs, args.boot, args.seed)
    table.to_csv(args.out / "rq4_rates.csv", index=False, float_format="%.4f")
    (args.out / "rq4_table.tex").write_text(latex(table), encoding="utf-8")
    (args.out / "rq4_l2_generic.md").write_text(l2_markdown(table), encoding="utf-8")
    plot(table, args.out)
    show = table.assign(
        **{
            c: (table[c] * 100).round(1)
            for c in ("rate", "ci_low", "ci_high", "drop", "drop_ci_low", "drop_ci_high")
        }
    )
    print(
        show[
            [
                "mode",
                "bank",
                "level",
                "n_items",
                "n_obs",
                "rate",
                "ci_low",
                "ci_high",
                "drop",
                "drop_ci_low",
                "drop_ci_high",
            ]
        ].to_string(index=False)
    )
    for _, r in table[table.level == "L3"].iterrows():
        print(
            f"L3 {r.bank}: first answer flagged {int(r.l3_n_first_flagged)}/{int(r.l3_n_answers)} "
            f"({r.l3_first_flag_rate * 100:.1f}%), got away {int(r.l3_n_got_away)} "
            f"({r.l3_got_away_given_flag * 100:.1f}% of flagged)"
        )


if __name__ == "__main__":
    main()

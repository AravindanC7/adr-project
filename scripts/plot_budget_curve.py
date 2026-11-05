"""Plot a budget curve: detection vs checks per honest person (step #32).

    python scripts/plot_budget_curve.py results/runs/<run_id>/budget_curve.csv

Reads the budget_curve.csv that `run_replay.py --compare` writes and saves budget_curve.png
next to it. Two panels, TPR at 1% and at 5% human FPR. The router is a line through its beta
values, each baseline a labelled point, and every mark has 95% bootstrap error bars in both
directions. Needs matplotlib: pip install -e ".[plot]".
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Reference chart palette (light): categorical slots in fixed order, plus chrome and ink.
ROUTER = "#2a78d6"
BASELINE_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["s", "D", "^", "v", "P", "X"]  # shape too, so identity is never colour alone
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PANELS = [("tpr_at_1pct_fpr", "TPR at 1% human FPR"), ("tpr_at_5pct_fpr", "TPR at 5% human FPR")]
X = "checks_per_honest_person"


def _err(df: pd.DataFrame, col: str) -> list[list[float]]:
    """Asymmetric error-bar lengths [below, above]; 0 where no interval exists."""
    lo = (df[col] - df[f"{col}_lo"]).fillna(0).clip(lower=0)
    hi = (df[f"{col}_hi"] - df[col]).fillna(0).clip(lower=0)
    return [lo.tolist(), hi.tolist()]


def _place_labels(
    marks: list[tuple[float, float, str]],
    xr: tuple[float, float],
    yr: tuple[float, float],
    line: list[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    """Greedy label placement in data coordinates: try spots right of each mark, then nearby
    rows above and below, then the left side, and take the first that overlaps no other label,
    no mark and no stretch of `line` (the router's curve). Width is estimated from the text
    length; good enough for short policy names."""
    xspan, yspan = xr[1] - xr[0], yr[1] - yr[0]
    h, pad = 0.05 * yspan, 0.012 * xspan
    boxes = [(x - pad, x + pad, y - h / 3, y + h / 3) for x, y, _ in marks]  # the marks
    for (x0, y0), (x1, y1) in itertools.pairwise(line):  # the line, as small boxes
        for t in (k / 12 for k in range(13)):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            boxes.append((x - pad / 2, x + pad / 2, y - h / 6, y + h / 6))

    def free(b):
        return all(b[1] <= o[0] or b[0] >= o[1] or b[3] <= o[2] or b[2] >= o[3] for o in boxes)

    out = []
    for x, y, text in marks:
        w = len(text) * 0.0115 * xspan
        spots = [(x + 2 * pad, y + k * h) for k in (0, -1, 1, -2, 2, -3, 3, -4, 4)]
        spots += [(x - 2 * pad - w, y + k * h) for k in (0, -1, 1, -2, 2)]
        for lx, ly in spots:
            box = (lx, lx + w, ly - h / 2, ly + h / 2)
            if free(box) and xr[0] <= lx and lx + w <= xr[1] and yr[0] <= ly <= yr[1]:
                break
        else:
            lx, ly = spots[0]
            box = (lx, lx + w, ly - h / 2, ly + h / 2)
        boxes.append(box)
        out.append((lx, ly))
    return out


def plot(csv_path: Path, title: str | None = None) -> Path:
    df = pd.read_csv(csv_path)
    router = df[df["policy"] == "router"].sort_values("beta")
    baselines = df[df["policy"] != "router"].reset_index(drop=True)
    if title is None:
        cfg_path = csv_path.parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        what = f"SYNTHETIC {cfg['synthetic']} (invented numbers)" if cfg.get("synthetic") \
            else cfg.get("assessment_id", csv_path.parent.name)
        title = f"Budget curve: {what}"

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE)
    xmax = max(df[f"{X}_hi"].fillna(df[X]).max(), df[X].max())
    xr = (0.0, (xmax if xmax > 0 else 1.0) * 1.35)  # room on the right for labels
    for ax, (metric, ylabel) in zip(axes, PANELS):
        lo = min(df[f"{metric}_lo"].fillna(df[metric]).min(), df[metric].min())
        yr = (max(0.0, math.floor((lo - 0.05) * 20) / 20), 1.02)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        ax.errorbar(router[X], router[metric], xerr=_err(router, X), yerr=_err(router, metric),
                    color=ROUTER, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1, capsize=0,
                    ecolor=ROUTER, label="router (one point per beta)", zorder=3)
        for i, row in baselines.iterrows():
            one = baselines.iloc[[i]]
            ax.errorbar(one[X], one[metric], xerr=_err(one, X), yerr=_err(one, metric),
                        color=BASELINE_COLORS[i % len(BASELINE_COLORS)],
                        marker=MARKERS[i % len(MARKERS)], markersize=9, linestyle="none",
                        markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1, capsize=0,
                        label=row["policy"], zorder=4)

        marks = [(x, y, f"β={b:g}") for x, y, b in zip(router[X], router[metric], router["beta"])]
        marks += [(r[X], r[metric], r["policy"]) for _, r in baselines.iterrows()]
        marks = [m for m in marks if not (math.isnan(m[0]) or math.isnan(m[1]))]
        curve = [(x, y) for x, y in zip(router[X], router[metric])
                 if not (math.isnan(x) or math.isnan(y))]
        for (x, y, text), (lx, ly) in zip(marks, _place_labels(marks, xr, yr, curve)):
            is_beta = text.startswith("β=")
            moved = abs(ly - y) > 1e-9 or lx < x
            ax.annotate(text, (x, y), xytext=(lx, ly), textcoords="data", va="center",
                        fontsize=8 if is_beta else 9, color=INK_2 if is_beta else INK,
                        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.7,
                                    "shrinkA": 0, "shrinkB": 4} if moved else None, zorder=5)

        ax.set_xlabel("checks per honest person (burden)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_title(ylabel, fontsize=11, color=INK, loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.905, "Up and to the left is better. Error bars: 95% bootstrap intervals.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", type=Path, help="budget_curve.csv from run_replay.py --compare")
    ap.add_argument("--title")
    args = ap.parse_args()
    print(f"Saved {plot(args.csv, args.title)}")


if __name__ == "__main__":
    main()

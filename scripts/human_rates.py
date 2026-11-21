"""Honest-human false-flag rate of every check on the pilot, with 95% intervals. Owner: Harshini.

    python scripts/human_rates.py --low-effort-prefixes d3ba4f10 04096570
    python scripts/human_rates.py            # low-effort ids from pilot_data/low_effort_ids.txt

Every pilot respondent is an honest human, so any flag outcome on the pilot is a false flag.
This script reads ONE cleaned outcome table (outcome rows only, no answer text) and reports, for
each check and each flag outcome:

  * rate      flagged rows / the check's rows (raw, no smoothing: the number for the paper)
  * boot CI   95% percentile interval from a session-level bootstrap, the same resampling the
              failure-map builder uses for P(o|human) (adr/map/builder.py _clustered_interval:
              sessions sorted by (run_id, session_id), rng.integers(0, G, size=G) per replicate,
              seed from builder._cell_seed). The builder applies add-one smoothing; here the rate
              is raw. `smoothed` is the builder's add-one point value, pooled over item types.
  * Wilson    95% Wilson score interval on rows, as a sanity check (it treats rows as
              independent, so it is narrower than the bootstrap when flags cluster in sessions)
  * rule of 3 3/n, the one-sided 95% upper bound when nothing was flagged
  * sessions  how many sessions have at least one flag of that kind (how many honest people a
              "pause on the first flag" rule would stop), with a Wilson interval over sessions

Everything is run twice: all sessions, and without the low-effort sessions. Rows are pooled over
item types; the per-item-type rows (the map's cells) are in the CSV too.

Outputs (aggregates only), in --out: human_rates.csv, human_rates.md, human_rates_table.tex,
notes.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adr.contracts import OUTCOME_LABELS, read_outcomes, smooth
from adr.map.builder import _cell_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOMES = ROOT / "results" / "outcomes" / "pilot-2026-11-13.parquet"
DEFAULT_LOW_EFFORT = ROOT / "pilot_data" / "low_effort_ids.txt"
DEFAULT_OUT = ROOT / "results" / "human_rates"
Z = 1.959963984540054

# (check, flag name, outcomes that count as the flag). Order is the table order.
FLAGS: list[tuple[str, str, tuple[str, ...]]] = [
    ("decoy", "picked_decoy", ("picked_decoy",)),
    ("canary", "leaked", ("leaked",)),
    ("canary", "followed_instruction", ("followed_instruction",)),
    ("canary", "any flag", ("leaked", "followed_instruction")),
    ("opinion_trap", "contradiction", ("contradiction",)),
    ("opinion_trap", "implausible", ("implausible",)),
    ("opinion_trap", "any flag", ("contradiction", "implausible")),
    ("style", "high", ("high",)),
    ("style", "mid or high", ("mid", "high")),
    ("behaviour", "too_fast", ("too_fast",)),
    ("behaviour", "focus_loss", ("focus_loss",)),
    ("behaviour", "paste", ("paste",)),
    ("behaviour", "any non-normal", ("paste", "focus_loss", "too_fast")),
]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def session_draws(n_sessions: int, n_boot: int, seed: int) -> np.ndarray:
    """Session indices per replicate, drawn exactly as builder._clustered_interval draws them."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n_sessions, size=n_sessions) for _ in range(n_boot)])


def per_session(rows: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """One row per session (sorted by run_id, session_id): its check rows and flagged rows."""
    flagged = rows["outcome"].isin(outcomes)
    grouped = rows.assign(flag=flagged.astype(int)).groupby(
        ["run_id", "session_id"], sort=True, dropna=False
    )
    return grouped["flag"].agg(n="size", k="sum").reset_index()


def rates_for(
    rows: pd.DataFrame,
    check: str,
    flag: str,
    outcomes: tuple[str, ...],
    *,
    draws: np.ndarray,
) -> dict[str, object]:
    """All the numbers for one check x flag on one set of rows."""
    sess = per_session(rows, outcomes)
    k_s, n_s = sess["k"].to_numpy(), sess["n"].to_numpy()
    k, n = int(k_s.sum()), int(n_s.sum())
    boot = k_s[draws].sum(axis=1) / n_s[draws].sum(axis=1)
    labels = OUTCOME_LABELS[check]
    counts = rows["outcome"].value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
    smoothed = smooth(counts, labels)
    w_lo, w_hi = wilson(k, n)
    g = len(sess)
    g_flag = int((k_s > 0).sum())
    s_lo, s_hi = wilson(g_flag, g)
    per_rate = k_s / n_s
    return {
        "check": check,
        "flag": flag,
        "flagged": k,
        "rows": n,
        "rate": k / n if n else float("nan"),
        "boot_lo": float(np.quantile(boot, 0.025)),
        "boot_hi": float(np.quantile(boot, 0.975)),
        "wilson_lo": w_lo,
        "wilson_hi": w_hi,
        "rule_of_three": 3 / n if k == 0 and n else float("nan"),
        "smoothed": float(sum(smoothed[o] for o in outcomes)),
        "sessions": g,
        "sessions_flagged": g_flag,
        "sessions_flagged_share": g_flag / g if g else float("nan"),
        "sessions_wilson_lo": s_lo,
        "sessions_wilson_hi": s_hi,
        "session_rate_median": float(np.median(per_rate)),
        "session_rate_p90": float(np.quantile(per_rate, 0.9)),
        "session_rate_max": float(per_rate.max()),
    }


def human_rates(frame: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    """Rates for every check x flag, pooled over item types and per item type."""
    human = frame[frame["respondent_kind"].eq("human") & frame["mechanism"].notna()]
    out = []
    for check, flag, outcomes in FLAGS:
        rows = human[human["mechanism"].eq(check)]
        if rows.empty:
            continue
        groups: list[tuple[str, pd.DataFrame]] = [("all", rows)]
        groups += [(str(t), r) for t, r in rows.groupby("item_type", sort=True)]
        for item_type, sub in groups:
            n_sessions = sub.groupby(["run_id", "session_id"], dropna=False).ngroups
            key = (check,) if item_type == "all" else (check, item_type)
            draws = session_draws(n_sessions, n_boot, _cell_seed(seed, key, "human"))
            rec = rates_for(sub, check, flag, outcomes, draws=draws)
            out.append({"item_type": item_type, **rec})
    cols = ["check", "flag", "item_type"]
    table = pd.DataFrame(out)
    return table[cols + [c for c in table.columns if c not in cols]]


def low_effort_ids(frame: pd.DataFrame, path: Path | None, prefixes: list[str]) -> set[str]:
    """Full session ids of the low-effort sessions, from the id file or from id prefixes."""
    sessions = sorted(frame["session_id"].astype(str).unique())
    if prefixes:
        ids = set()
        for p in prefixes:
            hits = [s for s in sessions if s.startswith(p)]
            if len(hits) != 1:
                raise SystemExit(f"low-effort prefix {p!r} matches {len(hits)} sessions, not 1")
            ids.add(hits[0])
        return ids
    if path is None or not path.exists():
        raise SystemExit(f"no low-effort id file at {path}; pass --low-effort-prefixes")
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    missing = ids - set(sessions)
    if not ids or missing:
        raise SystemExit(f"low-effort ids empty or not in the outcome table ({len(missing)})")
    return ids


def pct(x: float) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def ci_text(r: pd.Series) -> str:
    if r["flagged"] == 0:
        return f"0.0 [0, {pct(r['rule_of_three'])}]*"
    return f"{pct(r['rate'])} [{pct(r['boot_lo'])}, {pct(r['boot_hi'])}]"


def write_markdown(pooled: dict[str, pd.DataFrame], names: dict[str, str], path: Path) -> None:
    a, b = (pooled[v] for v in names)
    lines = [
        "| Check | Flag | " + " | ".join(
            f"{names[v]}: flagged/rows | rate % [95% CI] | Wilson % | sessions >=1"
            for v in names
        ) + " |",
        "|---|---|" + "---|---|---|---|" * len(names),
    ]
    for (_, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
        cells = [ra["check"], ra["flag"]]
        for r in (ra, rb):
            cells += [
                f"{r['flagged']}/{r['rows']}",
                ci_text(r),
                f"[{pct(r['wilson_lo'])}, {pct(r['wilson_hi'])}]",
                f"{r['sessions_flagged']}/{r['sessions']}",
            ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += [
        "",
        ("Rate = flagged rows / the check's rows, raw. CI = session-level bootstrap percentile "
        "interval. * nothing flagged: upper bound is the rule of three (3/n). "
        "Sessions >=1 = sessions with at least one flag of that kind / sessions that saw the check."),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def write_tex(
    pooled: dict[str, pd.DataFrame], names: dict[str, str], n_boot: int, path: Path
) -> None:
    a, b = (pooled[v] for v in names)
    n_sessions = int(a["n_sessions_total"].iloc[0])
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{names[v]}}}" for v in names)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll rrr rrr}",
        r"\toprule",
        rf" & & {head} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Check & Flag & $k/n$ & Rate \% [95\% CI] & Sess. & $k/n$ & Rate \% [95\% CI] & Sess. \\",
        r"\midrule",
    ]
    previous = None
    for (_, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
        if previous is not None and ra["check"] != previous:
            lines.append(r"\addlinespace")
        check = tex_escape(ra["check"]) if ra["check"] != previous else ""
        previous = ra["check"]
        cells = [check, tex_escape(ra["flag"])]
        for r in (ra, rb):
            ci = ci_text(r).replace("*", "$^\\dagger$")
            cells += [f"{r['flagged']}/{r['rows']}", ci, f"{r['sessions_flagged']}/{r['sessions']}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{False-flag rates of each check on honest pilot respondents (every pilot "
        r"respondent is human, so every flag is false). Rate is flagged answers over the check's "
        rf"answers; the 95\% interval is a session-level bootstrap ({n_boot:,} replicates). "
        r"Sess.\ is the number of sessions with at least one such flag. $^\dagger$No flags: "
        rf"upper bound by the rule of three ($3/n$). With {n_sessions} sessions these are pilot "
        r"estimates "
        r"and cannot certify a 1\% false-positive rate. Behaviour has no AI answers and is "
        r"not used by the router.}"),
        r"\label{tab:human-rates}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notes(frame: pd.DataFrame, names: dict[str, str], args, path: Path) -> None:
    human = frame[frame["respondent_kind"].eq("human")]
    lines = ["# Honest-human false-flag rates: method and caveats", ""]
    lines += [
        (f"Source: `{Path(args.outcomes).name}` (cleaned outcome rows only, no answer text). "
        f"{human['session_id'].nunique()} honest sessions, {len(human)} rows."),
        "",
        "## Method",
        "",
        ("- Rate = flagged rows / the check's rows, raw (no smoothing). Pooled over item types; "
        "per-item-type rows are in human_rates.csv."),
        (f"- 95% CI: session-level bootstrap, resampling whole sessions with replacement, "
        f"{args.n_boot:,} replicates, base seed {args.seed} (per-check seeds derived as in "
        "the failure-map builder). Percentile interval. Same resampling as map v2's P(o|human) "
        "intervals; the map adds one to every outcome count (add-one smoothing), the paper "
        "numbers here do not. The smoothed value is in the CSV column `smoothed`."),
        "- Wilson score interval on rows as a sanity check (assumes rows independent).",
        "- Zero counts: upper bound by the rule of three (3/n) and Wilson.",
        ("- Sessions flagged = sessions with at least one flag of that kind (how many honest "
        "people a pause-on-first-flag rule would stop)."),
        f"- Two variants: {', '.join(names.values())}.",
        "",
        "## Counts (all sessions)",
        "",
    ]
    for check in OUTCOME_LABELS:
        rows = human[human["mechanism"].eq(check)]
        if rows.empty:
            continue
        counts = rows["outcome"].value_counts().reindex(OUTCOME_LABELS[check], fill_value=0)
        lines.append(f"- {check}: {len(rows)} rows ("
                     + ", ".join(f"{k} {v}" for k, v in counts.items()) + ")")
    lines += [
        "",
        "## Caveats",
        "",
        (f"- {human['session_id'].nunique()} sessions is a pilot. A zero count over n rows "
        "still allows a rate up to 3/n, and "
        "the session-level uncertainty is wider than the row-level one, so these numbers cannot "
        "certify a 1% false-positive rate for any check."),
        ("- Behaviour (timing / focus / paste) has no AI rows, so it has no map cell and is "
        "excluded from the router; its rates describe honest people only."),
        ("- The low-effort sessions are honest but careless; excluding them shows how much of each "
        "rate they drive."),
        "- Behaviour outcomes are the re-scored ones (calibrated floor), not the live scores.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outcomes", default=str(DEFAULT_OUTCOMES))
    ap.add_argument("--low-effort-file", default=str(DEFAULT_LOW_EFFORT))
    ap.add_argument("--low-effort-prefixes", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    frame = read_outcomes(args.outcomes)
    frame = frame[frame["respondent_kind"].eq("human")]
    low = low_effort_ids(frame, Path(args.low_effort_file), args.low_effort_prefixes)
    n_all = frame["session_id"].nunique()
    variants = {
        "all": frame,
        "no_low_effort": frame[~frame["session_id"].astype(str).isin(low)],
    }
    names = {
        "all": f"All {n_all} sessions",
        "no_low_effort": f"Without {len(low)} low-effort ({n_all - len(low)})",
    }
    tables = []
    for variant, sub in variants.items():
        t = human_rates(sub, n_boot=args.n_boot, seed=args.seed)
        tables.append(t.assign(variant=variant, n_sessions_total=sub["session_id"].nunique()))
    table = pd.concat(tables, ignore_index=True)
    table = table[["variant", "n_sessions_total"] + [c for c in table.columns
                                                     if c not in ("variant", "n_sessions_total")]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "human_rates.csv", index=False, float_format="%.6f")
    pooled = {
        v: table[table["variant"].eq(v) & table["item_type"].eq("all")].reset_index(drop=True)
        for v in variants
    }
    write_markdown(pooled, names, out / "human_rates.md")
    write_tex(pooled, names, args.n_boot, out / "human_rates_table.tex")
    write_notes(frame, names, args, out / "notes.md")
    print((out / "human_rates.md").read_text(encoding="utf-8"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Behaviour-check threshold sweep on held-out pilot folds (step #42). Owner: Harshini.

    python scripts/calibrate_behaviour.py
    python scripts/calibrate_behaviour.py --targets 0.02 0.05

Every pilot respondent is an honest human, so any paste / focus_loss / too_fast outcome on the
pilot is a false flag. This script shows how that honest flag rate moves as each threshold of
adr/mechanisms/behaviour.py moves, so the thresholds can be picked on Friday instead of by eye.

How
  * Sessions: the same rule as server/clean.py (a start record, finished, consented). For each
    answer it keeps telemetry counts and timings, the item's passage + question word count, and
    the answer's LENGTH (the "untyped" paste rule needs it). Answer text is read in memory only
    and never stored, printed or written.
  * Folds: adr.replay.folds.assign_session_folds (PR #30), the one fold assignment shared with
    the router's cross-fitting, at its defaults (K and seed recorded in
    configs/data_decisions.json "pilot_folds"). The low-effort sessions (--exclude-sessions
    prefixes, matched to full session ids) are passed explicitly as low_effort, so they are dealt
    across the folds first and every fold gets at most one more of them. Every rate in the output
    is a rate on ONE held-out fold; nothing is computed on all the data at once.
  * Sweeps (first match wins, as in behaviour.py; the thresholds not being swept stay at
    behaviour.py's defaults):
      paste       --paste-chars
      focus_loss  --away-ms
      too_fast    --floor-ms x --per-word-ms x --key-gap-ms
  * Selection: for each split k and target rate, the setting with the highest honest rate still
    <= target on the OTHER K-1 folds is chosen, and its rate on fold k is reported. The spread of
    the chosen settings over the splits says how stable the choice is.
  * Everything is run twice: all sessions, and without the --exclude-sessions (low-effort) ones.
    The fold assignment is the same in both, so the difference is those sessions alone.

behaviour.py's thresholds are module constants, so the score is re-implemented here with the
thresholds as parameters (vectorised). At the defaults it is checked answer by answer against
behaviour.py's own score() and against the live outcomes the server stored; the run stops if
either disagrees.

Outputs (aggregates only: counts and rates per fold / setting), in --out:
  folds.csv      one row per (sessions, setting, fold): n_answers and per-outcome rates
  summary.csv    per (sessions, setting): pooled rate and mean / min / max over folds
  selection.csv  per (sessions, outcome, target, split): the chosen setting and its held-out rate
  summary.md     the headline tables
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import Response, Telemetry, parse_variant_id
from adr.mechanisms import behaviour
from adr.replay.folds import DEFAULT_K, DEFAULT_SEED, assign_session_folds
from server.clean import load_bank, plain_rendered
from server.store import read_session

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ("paste", "focus_loss", "too_fast", "normal")
FLAGS = ("paste", "focus_loss", "too_fast")
# Low-effort pilot sessions (#39 review): honest people answering carelessly. Session code
# prefixes; a prefix matches the start of the session id.
LOW_EFFORT = ("d3ba4f10", "04096570", "01fe7650", "0a87374a")
# What the cleaned pilot table (results/outcomes/pilot-2026-11-13.parquet) holds.
EXPECTED_PILOT = {"normal": 825, "too_fast": 124, "focus_loss": 99, "paste": 8}

# Per-answer columns kept. Counts and timings only: no text column, ever.
FEATURES = (
    "session_id",
    "time_on_item_ms",
    "paste_events",
    "pasted_chars",
    "focus_loss_count",
    "away_ms",
    "keystroke_count",
    "mean_interkey_ms",
    "answer_len",
    "words",
    "live_outcome",
    "ref_outcome",
)


@dataclass(frozen=True)
class Thresholds:
    """behaviour.py's thresholds; defaults are its module constants."""

    base_floor_ms: int = behaviour.ANSWER_MS
    per_word_ms: int = behaviour.READ_MS_PER_WORD
    away_ms: int = behaviour.MIN_AWAY_MS
    key_gap_ms: float = behaviour.MIN_INTERKEY_MS
    paste_min_chars: int = behaviour.MIN_PASTE_CHARS


SETTING_COLS = tuple(f.name for f in fields(Thresholds))


# ------------------------------------------------------------------ loading
def load_features(data_dir: Path, bank_path: Path) -> pd.DataFrame:
    """One row per answer of every finished, consented session (server/clean.py's rule).

    The answer is read for its length and for the reference score, then dropped.
    """
    items, _, _ = load_bank(bank_path)
    rows: list[dict] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        records = read_session(path)
        start = records[0] if records and records[0]["event"] == "start" else None
        if start is None or not start.get("consented"):
            continue
        if not any(r["event"] == "finish" for r in records):
            continue
        sid = path.stem
        for rec in records:
            if rec["event"] != "answer":
                continue
            item = items.get(rec["item_id"])
            if item is None:
                raise ValueError(f"{path.name}: an item isn't in {bank_path.name}")
            tel = Telemetry(**rec["telemetry"])
            rendered = plain_rendered(item)  # decoy and canary variants show the same stem
            answer = rec["answer"]
            floor = behaviour.reading_floor_ms(rendered, item.passage)
            live = [o for v, o in rec["outcomes"] if parse_variant_id(v)[1] == "behaviour"]
            response = Response(
                session_id=sid,
                item_id=item.item_id,
                variant_id=rendered.variant_id,
                answer=answer,
            )
            rows.append(
                {
                    "session_id": sid,
                    "time_on_item_ms": tel.time_on_item_ms,
                    "paste_events": tel.paste_events,
                    "pasted_chars": tel.pasted_chars,
                    "focus_loss_count": tel.focus_loss_count,
                    "away_ms": tel.away_ms,
                    "keystroke_count": tel.keystroke_count,
                    "mean_interkey_ms": tel.mean_interkey_ms,
                    "answer_len": len(answer.strip()) if isinstance(answer, str) else None,
                    "words": (floor - behaviour.ANSWER_MS) // behaviour.READ_MS_PER_WORD,
                    "live_outcome": live[0] if live else None,
                    "ref_outcome": behaviour.MECHANISM.score(
                        rendered, response, tel, passage=item.passage
                    ),
                }
            )
            del answer, response
    df = pd.DataFrame(rows, columns=list(FEATURES))
    for col in ("pasted_chars", "away_ms", "mean_interkey_ms", "answer_len"):
        df[col] = df[col].astype(float)  # None -> NaN
    return df


# ------------------------------------------------------------------ scoring
def score(df: pd.DataFrame, th: Thresholds) -> np.ndarray:
    """behaviour.BehaviourMechanism.score with the thresholds as parameters, for every row.

    Returns outcome codes: indexes into OUTCOMES.
    """
    pe = df["paste_events"].to_numpy()
    pasted = df["pasted_chars"].to_numpy()
    length = df["answer_len"].to_numpy()
    keys = df["keystroke_count"].to_numpy()
    with np.errstate(invalid="ignore"):
        paste = (pe > 0) & (np.isnan(pasted) | (pasted >= th.paste_min_chars))
        covered = keys + np.nan_to_num(pasted)
        paste |= (length >= behaviour.UNTYPED_MIN_CHARS) & (
            covered < behaviour.UNTYPED_COVERAGE * length
        )
        away = df["away_ms"].to_numpy()
        focus = (df["focus_loss_count"].to_numpy() > 0) & (away >= th.away_ms)  # NaN -> False
        floor = th.base_floor_ms + th.per_word_ms * df["words"].to_numpy()
        gap = df["mean_interkey_ms"].to_numpy()
        fast = (df["time_on_item_ms"].to_numpy() < floor) | (
            (keys >= behaviour.MIN_KEYS_FOR_SPEED) & (gap < th.key_gap_ms)
        )
    out = np.full(len(df), OUTCOMES.index("normal"))
    out[fast] = OUTCOMES.index("too_fast")
    out[focus] = OUTCOMES.index("focus_loss")
    out[paste] = OUTCOMES.index("paste")
    return out


def sanity_check(df: pd.DataFrame, expected: dict[str, int] | None) -> dict[str, int]:
    """Defaults must match behaviour.py's score(), the live outcomes, and `expected` counts."""
    mine = np.asarray(OUTCOMES)[score(df, Thresholds())]
    if not (mine == df["ref_outcome"].to_numpy()).all():
        n = int((mine != df["ref_outcome"].to_numpy()).sum())
        raise SystemExit(f"STOP: re-implemented score differs from behaviour.py on {n} answers")
    live = df["live_outcome"].to_numpy()
    has = df["live_outcome"].notna().to_numpy()
    if (live[has] != mine[has]).any():
        n = int((live[has] != mine[has]).sum())
        raise SystemExit(f"STOP: defaults differ from the live server outcome on {n} answers")
    counts = {o: int((mine == o).sum()) for o in OUTCOMES}
    if expected is not None and counts != expected:
        raise SystemExit(f"STOP: default counts {counts} != cleaned table {expected}")
    return counts


# ------------------------------------------------------------------ folds
def match_prefixes(session_ids: Iterable[str], prefixes: Sequence[str]) -> set[str]:
    """Full session ids for the given code prefixes. Each prefix must match exactly one
    session: a typo or an ambiguous prefix would otherwise silently change the low-effort set."""
    sids = sorted(set(session_ids))
    out: set[str] = set()
    for p in prefixes:
        hits = [s for s in sids if s.startswith(p)]
        if len(hits) != 1:
            raise SystemExit(f"STOP: session prefix {p!r} matches {len(hits)} sessions, not 1")
        out.update(hits)
    return out


# ------------------------------------------------------------------ sweeps
def settings(grids: dict[str, Sequence]) -> list[tuple[str, Thresholds]]:
    """(group, thresholds) for every sweep; each group moves only its own thresholds."""
    out: list[tuple[str, Thresholds]] = []
    out += [("paste", Thresholds(paste_min_chars=v)) for v in grids["paste_min_chars"]]
    out += [("focus_loss", Thresholds(away_ms=v)) for v in grids["away_ms"]]
    for f, w, g in itertools.product(
        grids["base_floor_ms"], grids["per_word_ms"], grids["key_gap_ms"]
    ):
        out.append(("too_fast", Thresholds(base_floor_ms=f, per_word_ms=w, key_gap_ms=g)))
    return out


def fold_table(
    df: pd.DataFrame,
    folds: dict[str, int],
    k: int,
    sweep: list[tuple[str, Thresholds]],
    variants: dict[str, set[str]],
) -> pd.DataFrame:
    """One row per (sessions variant, setting, fold) with counts and rates on that fold."""
    fold = df["session_id"].map(folds).to_numpy()
    rows = []
    for name, dropped in variants.items():
        keep = ~df["session_id"].isin(dropped).to_numpy()
        f = fold[keep]
        n = np.bincount(f, minlength=k)
        n_sess = Counter(folds[s] for s in set(df.loc[keep, "session_id"]))
        for group, th in sweep:
            codes = score(df, th)[keep]
            counts = np.zeros((len(OUTCOMES), k), dtype=int)
            for i in range(len(OUTCOMES)):
                counts[i] = np.bincount(f[codes == i], minlength=k)
            for j in range(k):
                row = {
                    "sessions": name,
                    "group": group,
                    **asdict(th),
                    "fold": j,
                    "n_sessions": n_sess[j],
                    "n_answers": int(n[j]),
                }
                for i, o in enumerate(OUTCOMES):
                    row[f"n_{o}"] = int(counts[i, j])
                    row[f"rate_{o}"] = counts[i, j] / n[j] if n[j] else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(ft: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, setting): pooled rate over folds and mean / min / max of the fold rates."""
    keys = ["sessions", "group", *SETTING_COLS]
    g = ft.groupby(keys, sort=False)
    out = g[["n_answers"]].sum()
    for o in OUTCOMES:
        out[f"pooled_{o}"] = g[f"n_{o}"].sum() / out["n_answers"]
        out[f"mean_{o}"] = g[f"rate_{o}"].mean()
        out[f"min_{o}"] = g[f"rate_{o}"].min()
        out[f"max_{o}"] = g[f"rate_{o}"].max()
    return out.reset_index()


def closeness(th: pd.DataFrame) -> np.ndarray:
    """Distance from behaviour.py's defaults, relative per threshold (tie-break)."""
    d = Thresholds()
    return sum(np.abs(th[c].to_numpy() / getattr(d, c) - 1) for c in SETTING_COLS)


def select(ft: pd.DataFrame, targets: Sequence[float], k: int) -> pd.DataFrame:
    """Per split k: on the other folds, the setting with the highest honest rate <= target
    (ties: closest to the defaults); reported with its rate on held-out fold k."""
    rows = []
    keys = ["sessions", "group", *SETTING_COLS]
    for (variant, outcome), part in ft.groupby(["sessions", "group"], sort=False):
        if outcome not in FLAGS:
            continue
        for j in range(k):
            train = (
                part[part["fold"] != j]
                .groupby(keys, sort=False)[["n_answers", f"n_{outcome}"]]
                .sum()
                .reset_index()
            )
            train["train_rate"] = train[f"n_{outcome}"] / train["n_answers"]
            train["dist"] = closeness(train)
            test = part[part["fold"] == j].set_index(list(SETTING_COLS))
            for target in targets:
                ok = train[train["train_rate"] <= target + 1e-12]
                row = {"sessions": variant, "outcome": outcome, "target": target, "fold": j}
                if ok.empty:
                    rows.append({**row, "train_rate": np.nan, "heldout_rate": np.nan})
                    continue
                best = ok.sort_values(["train_rate", "dist"], ascending=[False, True]).iloc[0]
                setting = tuple(best[c] for c in SETTING_COLS)
                held = test.loc[setting]
                rows.append(
                    {
                        **row,
                        **{c: best[c] for c in SETTING_COLS},
                        "train_rate": best["train_rate"],
                        "heldout_rate": float(held[f"rate_{outcome}"]),
                        "heldout_n_answers": int(held["n_answers"]),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.1f}%"


def cell(r: pd.Series, o: str) -> str:
    return f"{pct(r[f'mean_{o}'])} ({pct(r[f'min_{o}'])}-{pct(r[f'max_{o}'])})"


NOTES = """* Thresholds are set from the HUMAN side only. The AI-side behaviour rows (browser agent) depend
  on its typing mode and on the answer cache (PR #15 review), so their flag rates are a property
  of how the agent was run, not of the thresholds; do not tune on them.
* Pilot respondents are all honest humans, so every flag here is a false flag. The 4 low-effort
  sessions are honest but careless; the 'excluding' rows show their effect.
* First match wins: a stricter paste or focus_loss threshold moves answers out of too_fast and
  vice versa; each sweep keeps the other thresholds at the defaults. The untyped-paste rule (min
  chars, coverage) and the 10-key minimum are not swept.
* A setting at the edge of a grid means the target may need a wider grid (pass e.g.
  --floor-ms 0:2000:250 or --away-ms 1000:30000:1000); 'none' means no setting in the grid kept
  the other folds at or under the target.
* Folds: assign_session_folds() from PR #30 (adr.replay.folds), the same folds the router's
  cross-fitting uses; low-effort ids passed explicitly."""


def markdown(summary: pd.DataFrame, sel: pd.DataFrame, counts: dict[str, int], meta: dict) -> str:
    d = Thresholds()
    lines = [
        "# Behaviour-check calibration (#42): honest false-flag rates on held-out pilot folds",
        "",
        (
            f"{meta['sessions']} finished pilot sessions, {meta['answers']} answers; "
            f"K={meta['k']} folds by session (seed {meta['seed']}), {meta['low_effort']} "
            "low-effort session(s) spread across folds. Every rate is on one held-out fold; "
            "cells are mean over folds (min-max). Aggregates only."
        ),
        "",
        (
            "Sanity check: defaults match behaviour.py, the live server outcome and the "
            f"cleaned table on every answer: {counts}."
        ),
        "",
        "Defaults: " + ", ".join(f"{c}={getattr(d, c)}" for c in SETTING_COLS) + ".",
        "",
    ]
    for variant in summary["sessions"].unique():
        s = summary[summary["sessions"] == variant]
        lines += [f"## Sessions: {variant}", ""]
        tf = s[(s["group"] == "too_fast") & (s["key_gap_ms"] == d.key_gap_ms)]
        per_word = sorted(tf["per_word_ms"].unique())
        lines += [
            f"### too_fast honest rate, base floor x per-word (key gap {d.key_gap_ms} ms)",
            "",
            "| floor \\ per-word | " + " | ".join(f"{w:g}" for w in per_word) + " |",
            "|---|" + "---|" * len(per_word),
        ]
        for f in sorted(tf["base_floor_ms"].unique()):
            r = tf[tf["base_floor_ms"] == f].set_index("per_word_ms")
            lines.append(
                f"| {f:g} | " + " | ".join(cell(r.loc[w], "too_fast") for w in per_word) + " |"
            )
        gap = s[
            (s["group"] == "too_fast")
            & (s["base_floor_ms"] == d.base_floor_ms)
            & (s["per_word_ms"] == d.per_word_ms)
        ]
        lines += [
            "",
            "too_fast vs key gap (floor and per-word at defaults): "
            + ", ".join(f"{r.key_gap_ms:g} ms {cell(r, 'too_fast')}" for _, r in gap.iterrows()),
            "",
        ]
        for group, col, label in (
            ("focus_loss", "away_ms", "away ms"),
            ("paste", "paste_min_chars", "paste min chars"),
        ):
            g = s[s["group"] == group]
            lines += [
                f"### {group} honest rate vs {label}",
                "",
                f"| {label} | {group} | too_fast (next in line) |",
                "|---|---|---|",
            ]
            for _, r in g.iterrows():
                lines.append(f"| {r[col]:g} | {cell(r, group)} | {cell(r, 'too_fast')} |")
            lines.append("")
    lines += [
        "## Chosen setting per split (highest honest rate <= target on the other folds)",
        "",
        "| sessions | outcome | target | chosen settings over splits | held-out rate |",
        "|---|---|---|---|---|",
    ]
    moving = {
        "too_fast": ["base_floor_ms", "per_word_ms", "key_gap_ms"],
        "focus_loss": ["away_ms"],
        "paste": ["paste_min_chars"],
    }
    for (variant, outcome, target), part in sel.groupby(
        ["sessions", "outcome", "target"], sort=False
    ):
        if part["heldout_rate"].isna().all():
            chosen, held = "none reaches the target", "-"
        else:
            cols = moving[outcome]
            reached = part["heldout_rate"].notna()
            chosen = (
                f"{int(reached.sum())}/{len(part)} splits: "
                + "; ".join(
                    "/".join(f"{r[c]:g}" for c in cols) if ok else "none"
                    for ok, (_, r) in zip(reached, part.iterrows())
                )
                + f" ({'/'.join(cols)})"
            )
            h = part["heldout_rate"]
            held = f"{pct(h.mean())} ({pct(h.min())}-{pct(h.max())}) over reached splits"
        lines.append(f"| {variant} | {outcome} | {pct(target)} | {chosen} | {held} |")
    lines += [
        "",
        "## Notes",
        "",
        NOTES,
        "",
    ]
    return "\n".join(lines)


def grid(spec: str) -> list[float]:
    """'start:stop:step' (inclusive) or 'a,b,c'."""
    if ":" in spec:
        a, b, s = (float(x) for x in spec.split(":"))
        return [float(v) for v in np.round(np.arange(a, b + s / 2, s), 6)]
    return [float(v) for v in spec.split(",")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Behaviour thresholds vs honest flag rate (#42)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "pilot_data")
    p.add_argument("--bank", type=Path, default=ROOT / "server" / "pilot_items.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "calibration" / "behaviour")
    p.add_argument(
        "--k", type=int, default=DEFAULT_K, help="folds; default = the shared pilot_folds K"
    )
    p.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help="fold seed string; default = the shared pilot_folds seed",
    )
    p.add_argument(
        "--exclude-sessions",
        nargs="*",
        default=list(LOW_EFFORT),
        metavar="CODE",
        help="low-effort session code prefixes: spread over folds, and dropped in "
        "the 'excluding' rows",
    )
    p.add_argument("--targets", nargs="+", type=float, default=[0.02, 0.05])
    p.add_argument("--floor-ms", type=grid, default=grid("500:2000:250"))
    p.add_argument("--per-word-ms", type=grid, default=grid("50:250:25"))
    p.add_argument("--away-ms", type=grid, default=grid("1000:10000:1000"))
    p.add_argument("--key-gap-ms", type=grid, default=grid("20:50:5"))
    p.add_argument("--paste-chars", type=grid, default=grid("10:40:5"))
    p.add_argument(
        "--no-expected",
        action="store_true",
        help="skip the check against the cleaned pilot table's counts (other data)",
    )
    args = p.parse_args(argv)

    df = load_features(args.data_dir, args.bank)
    if df.empty:
        p.error(f"no finished sessions in {args.data_dir}")
    counts = sanity_check(df, None if args.no_expected else EXPECTED_PILOT)
    sessions = sorted(df["session_id"].unique())
    low = match_prefixes(sessions, args.exclude_sessions)
    folds = assign_session_folds(sessions, k=args.k, seed=args.seed, low_effort=sorted(low))
    variants = {"all": set()}
    if low:
        variants["excluding_low_effort"] = low
    sweep = settings(
        {
            "base_floor_ms": args.floor_ms,
            "per_word_ms": args.per_word_ms,
            "key_gap_ms": args.key_gap_ms,
            "away_ms": args.away_ms,
            "paste_min_chars": args.paste_chars,
        }
    )
    ft = fold_table(df, folds, args.k, sweep, variants)
    summary = summarise(ft)
    sel = select(ft, args.targets, args.k)

    args.out.mkdir(parents=True, exist_ok=True)
    ft.to_csv(args.out / "folds.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    sel.to_csv(args.out / "selection.csv", index=False)
    meta = {
        "sessions": len(sessions),
        "answers": len(df),
        "k": args.k,
        "seed": args.seed,
        "low_effort": len(low),
    }
    (args.out / "summary.md").write_text(markdown(summary, sel, counts, meta), encoding="utf-8")
    print(
        f"{len(sessions)} sessions, {len(df)} answers, {len(low)} low-effort; "
        f"defaults {counts}; {len(sweep)} settings x {args.k} folds -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

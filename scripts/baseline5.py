"""Baseline 5 (step #33): published survey traps re-tested on gpt-mini, copy-paste mode.

The traps (adr/baselines/published_traps.py) come from Wang et al. (CSCW 2025) (prompt injection) and
Westwood 2025 (attention checks). Each goes through the copy-paste attacker at L0 and L2, 5
samples at temperature 0.7 (the RQ4 settings), and is scored by its own pass/fail rule.

    python scripts/baseline5.py build                # writes results/baseline5/traps.json etc.
    python scripts/baseline5.py estimate [--level L2-generic]   # cost upper bound, no API calls
    python scripts/baseline5.py run --level L0       # then L2, L2-generic (gpt-mini only)
    python scripts/baseline5.py report --rq4 results/rq4/rq4_rates.csv

Outputs in results/baseline5/: traps.json (every trap, its source, wording note and rule),
baseline5_traps.csv (per trap and level), baseline5_summary.csv (per trap type and level, with
the RQ4 decoy rates for comparison), table.md, and runs/<run_id>/ (config and every response).

Levels: L0 (naive), L2 (informed: the harness's L2 prompt, which names hidden answer-key text and
hidden "include a code word" instructions, i.e. Wang's two attacks) and L2-generic (the same L2
role, but it only warns about hidden instructions in general and names no trap;
adr/harness/prompts.py SYSTEM_VARIANTS). L2-generic rows keep attacker_level L2 and use their own
prompt_version (cp-v1+l2-generic-v1).

CIs: per trap, Wilson 95% on its 5 answers. Pooled over a trap type, a two-level bootstrap
(resample items, then answers within each item), the same scheme as the RQ4 decoy rates; when
every pooled answer is unflagged (or every one flagged) the bootstrap can only return [0, 0]
(or [1, 1]), so those cells use the Wilson 95% interval on the pooled k/n instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.baselines.published_traps import Trap, all_traps, bank, flags
from adr.harness.prompts import PROMPT_VERSION, SYSTEM_VARIANTS, build_prompt

OUT = Path("results/baseline5")
MODEL = "gpt-mini"
# run label -> (attacker level, system-prompt variant or None)
RUNS: dict[str, tuple[str, str | None]] = {
    "L0": ("L0", None), "L2": ("L2", None), "L2-generic": ("L2", "L2-generic"),
}
LEVELS = tuple(RUNS)
LABEL = {"L0": "L0", "L2": "L2", "L2-generic": "L2 generic"}
SAMPLES, TEMPERATURE, MAX_TOKENS = 5, 0.7, 300
N_BOOT, SEED = 2000, 33


def run_id(level: str) -> str:
    return f"b5-cp-{level}-{MODEL}-s{SAMPLES}"


def family(t: Trap) -> str:
    """Trap type for pooling: the position / sub-kind in brackets is dropped. Westwood items are
    our own wording of his check types, and the label says so."""
    kind = t.trap_type.split(" (")[0]
    return f"Wang: {kind}" if t.paper.startswith("Wang") else f"Westwood-type (our wording): {kind}"


# ---------------------------------------------------------------- build / estimate
def cmd_build(_args) -> None:
    traps = all_traps()
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = [{
        "trap_id": t.trap_id, "paper": t.paper, "section": t.section, "trap_type": t.trap_type,
        "wording": t.wording, "rule": t.rule, "flag": t.flag,
        **({"rule_alt": t.rule_alt, "flag_alt": t.flag_alt} if t.flag_alt else {}),
        "item": t.item.model_dump(mode="json", exclude_defaults=True),
        "copied_text": t.copy_text,
    } for t in traps]
    (OUT / "traps.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    print(f"{len(traps)} traps -> {OUT / 'traps.json'}")


def cmd_estimate(args) -> None:
    import yaml

    spec = next(m for m in yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))["models"]
                if m["name"] == MODEL)
    traps = all_traps()
    _, _, variants = bank(traps)
    levels = [args.level] if args.level else list(LEVELS)
    in_tok = 0
    for label in levels:
        level, variant = RUNS[label]
        for t in traps:
            system, messages = build_prompt(t.item, variants[t.trap_id][0], level,
                                            variant=variant)
            chars = len(system or "") + sum(len(m["content"]) for m in messages)
            in_tok += SAMPLES * math.ceil(chars / 3)  # generous: ~4 chars per token in English
    calls = len(traps) * SAMPLES * len(levels)
    out_tok = calls * MAX_TOKENS  # worst case: every reply hits max_tokens
    usd = in_tok * spec["price_in_per_mtok"] / 1e6 + out_tok * spec["price_out_per_mtok"] / 1e6
    print(f"{calls} calls ({len(traps)} traps x {SAMPLES} samples x {len(levels)} levels), "
          f"<= {in_tok} input + {out_tok} output tokens -> at most ${usd:.2f} for {MODEL}")


# ---------------------------------------------------------------- run
def cmd_run(args) -> None:
    from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
    from adr.harness.llm_client import LLMClient

    assessment, items, variants = bank(all_traps())
    client = LLMClient.from_config("configs/models.yaml")
    level, variant = RUNS[args.level]
    cfg = CopyPasteConfig(run_id=run_id(args.level), models=[MODEL],
                          attacker_level=level, system_variant=variant,
                          prompt_version=SYSTEM_VARIANTS[variant][1] if variant else PROMPT_VERSION,
                          n_samples=SAMPLES,
                          temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                          results_dir=str(OUT), note="baseline 5: published survey traps")
    run_copy_paste(client, assessment, items, variants, {}, cfg)
    errors = (OUT / "runs" / cfg.run_id / "errors.jsonl").read_text(encoding="utf-8")
    print(f"run {cfg.run_id} done; {len(errors.splitlines())} errors")


# ---------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def bootstrap(per_item: list[list[int]], n_boot: int = N_BOOT, seed: int = SEED):
    """95% CI of the pooled rate: resample items, then answers within each item."""
    rng = random.Random(seed)
    rates = []
    for _ in range(n_boot):
        k = n = 0
        for _ in per_item:
            answers = per_item[rng.randrange(len(per_item))]
            draw = [answers[rng.randrange(len(answers))] for _ in answers]
            k, n = k + sum(draw), n + len(draw)
        rates.append(k / n)
    rates.sort()
    return rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot) - 1]


def load_answers(level: str) -> dict[str, list[dict]]:
    path = OUT / "runs" / run_id(level) / "responses.jsonl"
    by_item: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_item.setdefault(r["item_id"], []).append(r)
    return by_item


def cmd_report(args) -> None:
    traps = all_traps()
    per_trap, summary = [], []
    for level in LEVELS:
        answers = load_answers(level)
        groups: dict[str, list[list[int]]] = {}
        groups_alt: dict[str, list[list[int]]] = {}
        for t in traps:
            rs = answers.get(t.trap_id, [])
            hits = [int(flags(t, r["answer"])) for r in rs]
            lo, hi = wilson(sum(hits), len(hits))
            row = {
                "level": level, "trap_id": t.trap_id, "paper": t.paper, "section": t.section,
                "trap_type": t.trap_type, "wording": t.wording, "rule": t.rule,
                "n": len(hits), "n_flagged": sum(hits),
                "n_unparsed": sum(r["answer"] is None for r in rs),
                "rate": round(sum(hits) / len(hits), 4) if hits else "",
                "ci_low": round(lo, 4), "ci_high": round(hi, 4), **_NO_ALT,
                "rule_alt": "", "n_flagged_alt": "",
            }
            if t.flag_alt and hits:  # second rule (bogus items: neutral counts as flagged)
                alt = [int(flags(t, r["answer"], alt=True)) for r in rs]
                alo, ahi = wilson(sum(alt), len(alt))
                row.update(rule_alt=t.rule_alt, n_flagged_alt=sum(alt),
                           rate_alt=round(sum(alt) / len(alt), 4), ci_alt_low=round(alo, 4),
                           ci_alt_high=round(ahi, 4))
                groups_alt.setdefault(family(t), []).append(alt)
            per_trap.append(row)
            if hits:
                groups.setdefault(family(t), []).append(hits)
        for fam, per_item in groups.items():
            row = {"level": level, "check": fam, "source": "this run", **_pooled(per_item),
                   **_NO_ALT}
            if fam in groups_alt:
                alt = _pooled(groups_alt[fam])
                row.update(rate_alt=alt["rate"], ci_alt_low=alt["ci_low"],
                           ci_alt_high=alt["ci_high"])
            summary.append(row)
    if args.rq4:
        with open(args.rq4, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["mode"] == "Copy-paste" and r["level"] in LEVELS:
                    summary.append({
                        "level": r["level"], "check": f"Our decoy ({r['bank']})",
                        "source": f"RQ4 {r['run_id']}", "n_items": int(r["n_items"]),
                        "n": int(r["n_obs"]), "n_flagged": int(r["picked_decoy"]),
                        "rate": float(r["rate"]), "ci_low": float(r["ci_low"]),
                        "ci_high": float(r["ci_high"]), **_NO_ALT})
    _write_csv(OUT / "baseline5_traps.csv", per_trap)
    _write_csv(OUT / "baseline5_summary.csv", summary)
    (OUT / "table.md").write_text(_table(summary), encoding="utf-8")
    print(_table(summary))


_NO_ALT = {"rate_alt": "", "ci_alt_low": "", "ci_alt_high": ""}


def _pooled(per_item: list[list[int]]) -> dict:
    k, n = sum(map(sum, per_item)), sum(map(len, per_item))
    # all-zero / all-one cells: every bootstrap resample is identical, so use Wilson instead
    lo, hi = wilson(k, n) if k in (0, n) else bootstrap(per_item)
    return {"n_items": len(per_item), "n": n, "n_flagged": k, "rate": round(k / n, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _pct(rate: float, lo: float, hi: float) -> str:
    return f"{100 * rate:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _table(summary: list[dict]) -> str:
    """Headline rule in the table; a check with a second rule gets a footnote with its rates."""
    checks = list(dict.fromkeys(r["check"] for r in summary))
    by = {(r["check"], r["level"]): r for r in summary}
    lines = ["| Check | Items | " + " | ".join(f"{LABEL[lv]}: share flagged [95% CI]"
                                               for lv in LEVELS) + " |",
             "|---|---|" + "---|" * len(LEVELS)]
    notes = []
    for c in checks:
        rows = [by.get((c, lv)) for lv in LEVELS]
        cells = [_pct(r["rate"], r["ci_low"], r["ci_high"]) if r else "n/a" for r in rows]
        n_items = next(r["n_items"] for r in rows if r)
        label = c
        if any(r and r["rate_alt"] != "" for r in rows):
            mark = "*" * (len(notes) + 1)
            label += mark
            alt = [_pct(r["rate_alt"], r["ci_alt_low"], r["ci_alt_high"]) if r else "n/a"
                   for r in rows]
            notes.append(f"{mark} Headline rule: agree-only (an impossible claim is flagged "
                         f"only if the answer agrees). If “Neither agree nor disagree” "
                         "also counts as flagged: " + ", ".join(
                             f"{LABEL[lv]} {a}" for lv, a in zip(LEVELS, alt, strict=True)) + ".")
        lines.append(f"| {label} | {n_items} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n" + "".join(f"\n{n}\n" for n in notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    est = sub.add_parser("estimate")
    est.add_argument("--level", choices=LEVELS, help="one run only (default: all)")
    r = sub.add_parser("run")
    r.add_argument("--level", choices=LEVELS, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--rq4", help="results/rq4/rq4_rates.csv (PR #31) for the decoy rows")
    args = ap.parse_args()
    {"build": cmd_build, "estimate": cmd_estimate, "run": cmd_run,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

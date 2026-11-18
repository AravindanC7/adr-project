"""Baseline 5 (step #33): published survey traps re-tested on gpt-mini, copy-paste mode.

The traps (adr/baselines/published_traps.py) come from Wang et al. (CSCW 2025) (prompt injection) and
Westwood 2025 (attention checks). Each goes through the copy-paste attacker at L0 and L2, 5
samples at temperature 0.7 (the RQ4 settings), and is scored by its own pass/fail rule.

    python scripts/baseline5.py build                # writes results/baseline5/traps.json etc.
    python scripts/baseline5.py estimate             # cost upper bound, no API calls
    python scripts/baseline5.py run --level L0       # then --level L2 (gpt-mini only)
    python scripts/baseline5.py report --rq4 results/rq4/rq4_rates.csv

Outputs in results/baseline5/: traps.json (every trap, its source, wording note and rule),
baseline5_traps.csv (per trap and level), baseline5_summary.csv (per trap type and level, with
the RQ4 decoy rates for comparison), table.md, and runs/<run_id>/ (config and every response).

CIs: per trap, Wilson 95% on its 5 answers. Pooled over a trap type, a two-level bootstrap
(resample items, then answers within each item), the same scheme as the RQ4 decoy rates.
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
from adr.harness.prompts import build_prompt

OUT = Path("results/baseline5")
MODEL = "gpt-mini"
LEVELS = ("L0", "L2")
SAMPLES, TEMPERATURE, MAX_TOKENS = 5, 0.7, 300
N_BOOT, SEED = 2000, 33


def run_id(level: str) -> str:
    return f"b5-cp-{level}-{MODEL}-s{SAMPLES}"


def family(t: Trap) -> str:
    """Trap type for pooling: the position / sub-kind in brackets is dropped."""
    return f"{'Wang' if t.paper.startswith('Wang') else 'Westwood'}: {t.trap_type.split(' (')[0]}"


# ---------------------------------------------------------------- build / estimate
def cmd_build(_args) -> None:
    traps = all_traps()
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = [{
        "trap_id": t.trap_id, "paper": t.paper, "section": t.section, "trap_type": t.trap_type,
        "wording": t.wording, "rule": t.rule, "flag": t.flag,
        "item": t.item.model_dump(mode="json", exclude_defaults=True),
        "copied_text": t.copy_text,
    } for t in traps]
    (OUT / "traps.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    print(f"{len(traps)} traps -> {OUT / 'traps.json'}")


def cmd_estimate(_args) -> None:
    import yaml

    spec = next(m for m in yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))["models"]
                if m["name"] == MODEL)
    traps = all_traps()
    _, _, variants = bank(traps)
    in_tok = 0
    for level in LEVELS:
        for t in traps:
            system, messages = build_prompt(t.item, variants[t.trap_id][0], level)
            chars = len(system or "") + sum(len(m["content"]) for m in messages)
            in_tok += SAMPLES * math.ceil(chars / 3)  # generous: ~4 chars per token in English
    calls = len(traps) * SAMPLES * len(LEVELS)
    out_tok = calls * MAX_TOKENS  # worst case: every reply hits max_tokens
    usd = in_tok * spec["price_in_per_mtok"] / 1e6 + out_tok * spec["price_out_per_mtok"] / 1e6
    print(f"{calls} calls ({len(traps)} traps x {SAMPLES} samples x {len(LEVELS)} levels), "
          f"<= {in_tok} input + {out_tok} output tokens -> at most ${usd:.2f} for {MODEL}")


# ---------------------------------------------------------------- run
def cmd_run(args) -> None:
    from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
    from adr.harness.llm_client import LLMClient

    assessment, items, variants = bank(all_traps())
    client = LLMClient.from_config("configs/models.yaml")
    cfg = CopyPasteConfig(run_id=run_id(args.level), models=[MODEL],
                          attacker_level=args.level, n_samples=SAMPLES,
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
        for t in traps:
            rs = answers.get(t.trap_id, [])
            hits = [int(flags(t, r["answer"])) for r in rs]
            lo, hi = wilson(sum(hits), len(hits))
            per_trap.append({
                "level": level, "trap_id": t.trap_id, "paper": t.paper, "section": t.section,
                "trap_type": t.trap_type, "wording": t.wording, "rule": t.rule,
                "n": len(hits), "n_flagged": sum(hits),
                "n_unparsed": sum(r["answer"] is None for r in rs),
                "rate": round(sum(hits) / len(hits), 4) if hits else "",
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            })
            if hits:
                groups.setdefault(family(t), []).append(hits)
        for fam, per_item in groups.items():
            k = sum(map(sum, per_item))
            n = sum(map(len, per_item))
            lo, hi = bootstrap(per_item)
            summary.append({"level": level, "check": fam, "source": "this run",
                            "n_items": len(per_item), "n": n, "n_flagged": k,
                            "rate": round(k / n, 4), "ci_low": round(lo, 4),
                            "ci_high": round(hi, 4)})
    if args.rq4:
        with open(args.rq4, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["mode"] == "Copy-paste" and r["level"] in LEVELS:
                    summary.append({
                        "level": r["level"], "check": f"Our decoy ({r['bank']})",
                        "source": f"RQ4 {r['run_id']}", "n_items": int(r["n_items"]),
                        "n": int(r["n_obs"]), "n_flagged": int(r["picked_decoy"]),
                        "rate": float(r["rate"]), "ci_low": float(r["ci_low"]),
                        "ci_high": float(r["ci_high"])})
    _write_csv(OUT / "baseline5_traps.csv", per_trap)
    _write_csv(OUT / "baseline5_summary.csv", summary)
    (OUT / "table.md").write_text(_table(summary), encoding="utf-8")
    print(_table(summary))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _pct(r: dict) -> str:
    return f"{100 * r['rate']:.1f}% [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"


def _table(summary: list[dict]) -> str:
    checks = list(dict.fromkeys(r["check"] for r in summary))
    by = {(r["check"], r["level"]): r for r in summary}
    lines = ["| Check | Items | L0: share flagged [95% CI] | L2: share flagged [95% CI] |",
             "|---|---|---|---|"]
    for c in checks:
        cells = [_pct(by[(c, lv)]) if (c, lv) in by else "n/a" for lv in LEVELS]
        n_items = next(by[(c, lv)]["n_items"] for lv in LEVELS if (c, lv) in by)
        lines.append(f"| {c} | {n_items} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("estimate")
    r = sub.add_parser("run")
    r.add_argument("--level", choices=LEVELS, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--rq4", help="results/rq4/rq4_rates.csv (PR #31) for the decoy rows")
    args = ap.parse_args()
    {"build": cmd_build, "estimate": cmd_estimate, "run": cmd_run,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

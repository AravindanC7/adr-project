"""Per-item difficulty for the v1 quiz banks: did each model answer the PLAIN version correctly?

One row per (item, model). Run from the repo root:
    python scripts/item_difficulty.py [--out results/runs/v1-item-difficulty.csv]

Columns
  bank, item_id, subject, correct_option, model, heldout_model
  cp_t0_correct      plain answer correct in v1-cp-L0-<bank> (temperature 0)   -> 1/0/blank
  cp_s5_n_correct    of the plain answers in v1-cp-L0-<bank>-s5 (temp 0.7), how many correct
  cp_s5_n            how many plain answers that run has for this item (5 if none failed)
  du_correct         plain answer correct in v1-du-L0-<bank> (one 200-question PDF)
  du_q20_correct     plain answer correct in v1-du-L0-<bank>-q20 (20-question PDFs)
  knows              the split to use: cp_t0_correct (the deterministic copy-paste answer)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

BANKS = ("quiz-mmlu-200", "quiz-arc-200")
RUNS = {  # column -> run id pattern
    "cp_t0": "v1-cp-L0-{bank}",
    "cp_s5": "v1-cp-L0-{bank}-s5",
    "du": "v1-du-L0-{bank}",
    "du_q20": "v1-du-L0-{bank}-q20",
}


def plain_answers(run_dir: Path) -> dict[tuple[str, str], list]:
    """(model, item_id) -> list of answers to the plain variant."""
    out: dict[tuple[str, str], list] = defaultdict(list)
    path = run_dir / "responses.jsonl"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "|plain|" in r["variant_id"]:
                out[(r["model"], r["item_id"])].append(r["answer"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/runs/v1-item-difficulty.csv")
    ap.add_argument("--processed", default="data/processed/runs_v1")
    args = ap.parse_args()

    items = {}
    with open(Path(args.processed) / "items.jsonl", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            items[it["item_id"]] = it

    heldout = {}
    rows = []
    for bank in BANKS:
        assessment = json.loads(
            (Path(args.processed) / "assessments" / f"{bank}.json").read_text(encoding="utf-8")
        )
        answers = {}
        models: set[str] = set()
        for col, pattern in RUNS.items():
            run_dir = Path("results/runs") / pattern.format(bank=bank)
            answers[col] = plain_answers(run_dir)
            cfg = run_dir / "config.yaml"
            if cfg.exists():
                c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
                models |= set(c["models"])
                heldout.update({m: (m in c.get("heldout_models", [])) for m in c["models"]})
        for model in sorted(models):
            for item_id in assessment["item_ids"]:
                key = items[item_id]["correct_option"]

                def one(col, answers=answers, model=model, item_id=item_id, key=key):
                    a = answers[col].get((model, item_id))
                    return "" if not a else int(a[0] == key)

                s5 = answers["cp_s5"].get((model, item_id), [])
                rows.append(
                    {
                        "bank": bank,
                        "item_id": item_id,
                        "subject": item_id.split(":")[1] if bank == "quiz-mmlu-200" else "",
                        "correct_option": key,
                        "model": model,
                        "heldout_model": int(heldout.get(model, False)),
                        "cp_t0_correct": one("cp_t0"),
                        "cp_s5_n_correct": sum(a == key for a in s5) if s5 else "",
                        "cp_s5_n": len(s5) if s5 else "",
                        "du_correct": one("du"),
                        "du_q20_correct": one("du_q20"),
                        "knows": one("cp_t0"),
                    }
                )

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

"""Build a smoothed failure-mode map from validated outcome-table Parquet files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import read_outcomes
from adr.map import build_failure_map


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path, help="Parquet files or outcome directories")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--output", type=Path, default=Path("results/failure_map/v1.json"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--note", default="")
    ap.add_argument(
        "--ai-only",
        action="store_true",
        help=(
            "allow provisional cells without human rows; their human side is prior-only and "
            "must not be used by the router"
        ),
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    frame = read_outcomes(args.inputs)
    failure_map = build_failure_map(
        frame,
        version=args.version,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        note=args.note,
        ai_only=args.ai_only,
    )
    path = failure_map.save(args.output)
    print(
        json.dumps(
            {
                "output": str(path),
                "version": failure_map.version,
                "source_run_ids": failure_map.source_run_ids,
                "n_cells": len(failure_map.cells),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

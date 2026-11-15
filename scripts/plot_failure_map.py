"""Render a failure-mode map as a paper-ready SVG heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import FailureMap
from adr.map import write_heatmap_svg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", type=Path)
    ap.add_argument("--output", type=Path, default=Path("results/figures/failure_map.svg"))
    ap.add_argument(
        "--include-specific",
        action="store_true",
        help="include attack-mode/level rows in addition to the pooled paper view",
    )
    args = ap.parse_args()
    failure_map = FailureMap.load(args.map)
    output = write_heatmap_svg(
        failure_map,
        args.output,
        include_specific=args.include_specific,
    )
    print(output)


if __name__ == "__main__":
    main()

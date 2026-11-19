"""Write a SYNTHETIC failure map for the live demo server (step #47). Invented numbers.

The cells come from adr/router/synthetic.py's synthetic_failure_map (the same made-up numbers
as scripts/router_synthetic_demo.py), kept to the mechanism / item-type pairs the item bank can
use, so the router on the live server scores answers and can pause or clear a session:

    python scripts/make_demo_map.py                 # -> results/failure_map/demo_synthetic.json
    $env:ADR_MAP = "results\\failure_map\\demo_synthetic.json"; uvicorn server.app:app

Nothing scored with this map is a result. The server refuses it with ADR_PILOT=1. The real map
(Archana's v2, #40, from scripts/build_failure_map.py) drops in through ADR_MAP the same way.
Behaviour is scored on every answer, so item types the synthetic map has no behaviour cell for
(numeric, ranking) reuse its mcq behaviour numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import FailureMap, Item, MapCell
from adr.router.synthetic import synthetic_failure_map

# Not imported from server.app: importing it builds the app, which would read ADR_MAP.
DEFAULT_BANK = Path(__file__).resolve().parents[1] / "server" / "stub_items.json"

VERSION = "synthetic-demo"
DEFAULT_OUT = Path("results/failure_map/demo_synthetic.json")
NOTE = (
    "SYNTHETIC: invented numbers from adr/router/synthetic.py, for the live demo (#47) only. "
    "Not a result; never use it with real people (the server refuses it with ADR_PILOT=1)."
)


def demo_map(bank_path: str | Path = DEFAULT_BANK) -> FailureMap:
    raw = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    items = {d["item_id"]: Item(**d) for d in raw["items"]}
    synth = synthetic_failure_map()
    used = {(m, it.item_type) for it in items.values() for m in it.eligible_mechanisms}
    used |= {("behaviour", it.item_type) for it in items.values()}  # passive: every answer
    cells: list[MapCell] = []
    for mech, item_type in sorted(used):
        cell = synth.lookup(mech, item_type)
        if cell is None and mech == "behaviour":
            base = synth.lookup("behaviour", "mcq")
            cell = MapCell.from_counts("behaviour", item_type, base.counts_ai, base.counts_human)
        if cell is not None:
            cells.append(cell)
    return FailureMap(version=VERSION, note=NOTE, cells=cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", default=str(DEFAULT_BANK), help="item bank (default: stub bank)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    fmap = demo_map(args.items)
    path = fmap.save(args.output)
    cells = ", ".join(f"{c.mechanism}/{c.item_type}" for c in fmap.cells)
    print(f"SYNTHETIC map {fmap.version!r}: {len(fmap.cells)} cells ({cells})")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

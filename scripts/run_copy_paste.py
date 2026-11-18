"""Run the copy-paste attacker.

Quick checks (no data needed yet; uses the 4 handwritten test items and stub traps):
    python scripts/run_copy_paste.py --stub --fake                   # offline, no API keys
    python scripts/run_copy_paste.py --stub --models claude-small    # one real model

Real run, once Archana's loaders and mechanisms exist:
    python scripts/run_copy_paste.py --assessment data/processed/quiz_slice.json \
        --items data/processed/items.jsonl --variants data/processed/variants.jsonl \
        --mechanisms decoy,opinion_trap --models gpt-mini,open-weight --level L0

Files: --items is one Item per line, --variants one RenderedItem per line, --assessment one
Assessment object, all as JSON matching adr/contracts.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import Assessment, Item, RenderedItem
from adr.contracts.io import read_jsonl
from adr.harness.copy_paste import CopyPasteConfig, plain_variant, run_copy_paste
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec
from adr.harness.registry import load_mechanisms
from adr.mechanisms.style import StyleMechanism, load_style_profile


def gullible(system, messages):
    """Offline stand-in model: follows hidden instructions unless warned."""
    text = messages[-1]["content"]
    if not system and (m := re.search(r"correct answer is option (\d+)", text)):
        return m.group(1)
    if not system and (m := re.search(r"include the word '(\w+)'", text)):
        return f"It's fine. {m.group(1)}"
    return "1" if "number" in text else "It's fine."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stub", action="store_true", help="use the 4 handwritten test items")
    ap.add_argument("--fake", action="store_true", help="offline fake model, no API calls")
    ap.add_argument("--assessment"), ap.add_argument("--items"), ap.add_argument("--variants")
    ap.add_argument("--mechanisms", default="", help="comma list, e.g. decoy,canary")
    ap.add_argument("--models", default="", help="comma list of names from configs/models.yaml")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument(
        "--style-profile",
        default="data/processed/runs_v1/style_profile.json",
        help="fitted profile required when --mechanisms includes style",
    )
    ap.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "L4"])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-id")
    args = ap.parse_args()

    if args.stub:
        from tests.stubs import ITEMS, StubCanary, StubDecoy

        items = {it.item_id: it for it in ITEMS}
        decoy, canary = StubDecoy(), StubCanary()
        variants: dict[str, list[RenderedItem]] = {}
        for it in ITEMS:
            vs = [plain_variant(it)]
            if decoy.applies_to(it):
                vs.append(decoy.generate(it, 0))
            if canary.applies_to(it):
                vs.append(canary.generate(it, 0))
            variants[it.item_id] = vs
        assessment = Assessment(
            assessment_id="stub", title="Stub", assessment_type="quiz", item_ids=list(items)
        )
        mechanisms = {"decoy": decoy, "canary": canary}
    else:
        if not (args.assessment and args.items and args.variants):
            raise SystemExit("give --assessment, --items and --variants (or use --stub)")
        assessment = Assessment.model_validate_json(Path(args.assessment).read_text(encoding="utf-8"))
        items = {it.item_id: it for it in read_jsonl(args.items, Item)}
        variants = {}
        for v in read_jsonl(args.variants, RenderedItem):
            variants.setdefault(v.item_id, []).append(v)
        mechanisms = load_mechanisms([m for m in args.mechanisms.split(",") if m])
        if "style" in mechanisms:
            style = mechanisms["style"]
            if not isinstance(style, StyleMechanism):
                raise TypeError("registered style mechanism has the wrong type")
            style.configure(load_style_profile(args.style_profile))

    if args.fake:
        client = LLMClient(
            models={"fake": ModelSpec("fake", "fake", "fake-v1", 0.0, 0.0)},
            cache_dir="cache/llm-fake",
            providers={"fake": FakeProvider(gullible)},
        )
        models = ["fake"]
    else:
        client = LLMClient.from_config(args.config)
        models = [m for m in args.models.split(",") if m]
        if not models:
            raise SystemExit("give --models (names from configs/models.yaml)")

    run_id = args.run_id or f"cp-{args.level}-{datetime.now():%Y%m%d-%H%M%S}"
    cfg = CopyPasteConfig(
        run_id=run_id,
        models=models,
        attacker_level=args.level,
        n_samples=args.samples,
        temperature=args.temperature,
        max_workers=args.workers,
    )
    rows = run_copy_paste(client, assessment, items, variants, mechanisms, cfg)

    print(f"run {run_id}: {len(rows)} outcome rows -> results/outcomes/{run_id}.parquet")
    counts = Counter((r.model, r.mechanism or "plain", r.outcome) for r in rows)
    for (model, mech, outcome), n in sorted(counts.items()):
        print(f"  {model:14s} {mech:14s} {outcome:22s} {n}")
    err = Path("results/runs") / run_id / "errors.jsonl"
    n_err = len(err.read_text(encoding="utf-8").splitlines()) if err.exists() else 0
    if n_err:
        print(f"  {n_err} calls failed; see {err}")


if __name__ == "__main__":
    main()

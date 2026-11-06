"""Prepare a deterministic, subject-balanced MMLU slice for outcome-table runs.

This script only prepares validated Item, Assessment, and RenderedItem files. It does not call
any model APIs. Raw MMLU data stays under data/raw and generated run inputs stay under
data/processed, both outside Git.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import Assessment, Item
from adr.contracts.io import variants_path, write_jsonl
from adr.data.loaders import load_mmlu
from adr.harness.copy_paste import plain_variant
from adr.mechanisms.decoy import MECHANISM as DECOY


def balanced_slice(items: list[Item], limit: int) -> list[Item]:
    """Take items round-robin across MMLU subjects, preserving source order per subject."""

    if limit < 1:
        raise ValueError("limit must be positive")
    by_subject: dict[str, deque[Item]] = defaultdict(deque)
    for item in items:
        parts = item.item_id.split(":")
        if len(parts) < 4 or parts[0] != "mmlu":
            raise ValueError(f"unexpected MMLU item id: {item.item_id!r}")
        by_subject[parts[1]].append(item)

    selected: list[Item] = []
    subjects = sorted(by_subject)
    while len(selected) < min(limit, len(items)):
        added = False
        for subject in subjects:
            if by_subject[subject] and len(selected) < limit:
                selected.append(by_subject[subject].popleft())
                added = True
        if not added:
            break
    return selected


def prepare(mmlu_root: Path, output_root: Path, *, split: str, limit: int) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as skipped:
        warnings.simplefilter("always", RuntimeWarning)
        loaded = load_mmlu(mmlu_root, split=split, skip_duplicate_options=True)
    items = balanced_slice(loaded, limit)
    assessment = Assessment(
        assessment_id=f"mmlu-{split}-{len(items)}",
        title=f"MMLU {split.title()} Balanced Slice ({len(items)} items)",
        assessment_type="quiz",
        item_ids=[item.item_id for item in items],
    )
    variants = []
    for item in items:
        variants.append(plain_variant(item))
        variants.append(DECOY.generate(item, seed=0))

    output_root.mkdir(parents=True, exist_ok=True)
    items_path = write_jsonl(output_root / "items.jsonl", items)
    assessment_path = output_root / "assessment.json"
    assessment_path.write_text(assessment.model_dump_json(indent=2) + "\n", encoding="utf-8")
    variant_path = write_jsonl(variants_path(output_root, assessment.assessment_id), variants)
    manifest = {
        "assessment_id": assessment.assessment_id,
        "source_split": split,
        "n_items": len(items),
        "n_subjects": len({item.item_id.split(":")[1] for item in items}),
        "n_variants": len(variants),
        "mechanisms": ["decoy"],
        "skipped_source_rows": [str(warning.message) for warning in skipped],
        "items": str(items_path),
        "assessment": str(assessment_path),
        "variants": str(variant_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-root", type=Path, default=Path("data/raw/mmlu/data"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/outcome-v1/mmlu-quiz-200")
    )
    parser.add_argument("--split", choices=("dev", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    manifest = prepare(args.mmlu_root, args.output, split=args.split, limit=args.limit)
    print(json.dumps(manifest, indent=2))
    print("\nOffline validation command:")
    print(
        "python scripts/run_copy_paste.py "
        f"--assessment {manifest['assessment']} --items {manifest['items']} "
        f"--variants {manifest['variants']} --mechanisms decoy --fake "
        "--run-id prep-mmlu-cp-L0"
    )


if __name__ == "__main__":
    main()

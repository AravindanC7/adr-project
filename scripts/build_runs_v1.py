"""Build the validated assessment banks and variants for outcome-table run v1.

This script is deliberately offline: it loads public datasets already present under
``data/raw``, writes contract-validated JSONL under ``data/processed/runs_v1``, and never calls
an LLM provider.  The held-out ASURRE survey must be named explicitly so it cannot be selected
accidentally or changed between runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import Assessment, HumanDistribution, Item, RenderedItem
from adr.contracts.io import variants_path, write_jsonl
from adr.data.loaders import (
    assign_heldout_survey,
    load_arc,
    load_asurre_items,
    load_gss,
    load_ipip,
    load_mmlu,
    load_opinionqa,
)
from adr.harness.copy_paste import plain_variant
from adr.mechanisms.canary import MECHANISM as CANARY
from adr.mechanisms.decoy import MECHANISM as DECOY
from adr.mechanisms.opinion_trap import OpinionTrapMechanism

ASURRE_SOURCES = (
    "asurre_nature_phd_2019",
    "asurre_springer_social_2017",
    "asurre_osmi_2017_18",
)


@dataclass(frozen=True)
class Bank:
    assessment: Assessment
    items: tuple[Item, ...]
    variants: tuple[RenderedItem, ...]


def balanced_slice(items: list[Item], limit: int, *, key) -> list[Item]:
    """Take a deterministic round-robin slice across ``key(item)`` groups."""

    if limit < 1:
        raise ValueError("limit must be positive")
    groups: dict[str, deque[Item]] = defaultdict(deque)
    for item in items:
        groups[str(key(item))].append(item)
    selected: list[Item] = []
    names = sorted(groups)
    while len(selected) < min(limit, len(items)):
        changed = False
        for name in names:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].popleft())
                changed = True
        if not changed:
            break
    return selected


def stable_sample(items: list[Item], limit: int, seed: int) -> list[Item]:
    """Return a fixed-seed sample, retaining source order in the output."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if len(items) <= limit:
        return list(items)
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(len(items)), limit))
    return [items[position] for position in positions]


def active_variants(
    items: list[Item],
    *,
    seed: int,
    opinion: OpinionTrapMechanism | None = None,
) -> list[RenderedItem]:
    """Build plain plus eligible active variants; style remains passive."""

    variants: list[RenderedItem] = []
    for item in items:
        variants.append(plain_variant(item))
        if DECOY.applies_to(item):
            variants.append(DECOY.generate(item, seed))
        if item.source == "opinionqa" and opinion is not None and opinion.applies_to(item):
            variants.append(opinion.generate(item, seed))
        if CANARY.applies_to(item):
            variants.append(CANARY.generate(item, seed))
    return variants


def make_bank(
    assessment_id: str,
    title: str,
    assessment_type: str,
    items: list[Item],
    *,
    seed: int,
    opinion: OpinionTrapMechanism | None = None,
) -> Bank:
    if not items:
        raise ValueError(f"{assessment_id}: no items")
    wrong = [item.item_id for item in items if item.assessment_type != assessment_type]
    if wrong:
        raise ValueError(f"{assessment_id}: items have the wrong assessment type: {wrong[:3]}")
    assessment = Assessment(
        assessment_id=assessment_id,
        title=title,
        assessment_type=assessment_type,
        item_ids=[item.item_id for item in items],
    )
    variants = active_variants(items, seed=seed, opinion=opinion)
    return Bank(assessment, tuple(items), tuple(variants))


def write_bundle(
    output: Path,
    banks: list[Bank],
    distributions: list[HumanDistribution],
    *,
    seed: int,
    heldout_asurre_source: str,
    requested_sizes: dict[str, int | None],
) -> dict[str, object]:
    """Write all run inputs and return the manifest dictionary."""

    output.mkdir(parents=True, exist_ok=True)
    all_items: dict[str, Item] = {}
    for bank in banks:
        for item in bank.items:
            previous = all_items.setdefault(item.item_id, item)
            if previous != item:
                raise ValueError(f"conflicting definitions for item {item.item_id}")

    write_jsonl(output / "items.jsonl", all_items.values())
    write_jsonl(output / "assessments.jsonl", (bank.assessment for bank in banks))
    write_jsonl(output / "human_distributions.jsonl", distributions)

    manifests: list[dict[str, object]] = []
    for bank in banks:
        write_jsonl(variants_path(output, bank.assessment.assessment_id), bank.variants)
        item_types = Counter(item.item_type for item in bank.items)
        mechanisms = Counter(variant.mechanism or "plain" for variant in bank.variants)
        manifests.append(
            {
                "assessment_id": bank.assessment.assessment_id,
                "assessment_type": bank.assessment.assessment_type,
                "requested_items": requested_sizes.get(bank.assessment.assessment_id),
                "n_items": len(bank.items),
                "n_variants": len(bank.variants),
                "item_types": dict(sorted(item_types.items())),
                "mechanisms": dict(sorted(mechanisms.items())),
                "splits": dict(sorted(Counter(item.split for item in bank.items).items())),
                "variants": str(variants_path(output, bank.assessment.assessment_id)),
            }
        )

    manifest: dict[str, object] = {
        "version": "runs-v1",
        "seed": seed,
        "heldout_asurre_source": heldout_asurre_source,
        "n_unique_items": len(all_items),
        "n_human_distributions": len(distributions),
        "assessments": manifests,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build(args: argparse.Namespace) -> dict[str, object]:
    mmlu = load_mmlu(args.mmlu_root, split=args.mmlu_split, skip_duplicate_options=True)
    mmlu = balanced_slice(
        mmlu, args.mmlu_limit, key=lambda item: item.item_id.split(":")[1]
    )
    arc = stable_sample(load_arc(args.arc_path, subset=args.arc_subset), args.arc_limit, args.seed)
    opinion_items, opinion_distributions = load_opinionqa(args.opinionqa_root)
    opinion_items = stable_sample(opinion_items, args.opinionqa_limit, args.seed)
    selected_distribution_keys = {item.human_dist_key for item in opinion_items}
    opinion_distributions = [
        distribution
        for distribution in opinion_distributions
        if distribution.key in selected_distribution_keys
    ]
    opinion = OpinionTrapMechanism(opinion_distributions)

    ipip_items, ipip_distributions = load_ipip(args.ipip_data, args.ipip_codebook)
    gss_items, gss_distributions = load_gss(args.gss_data, args.gss_manifest)

    asurre_items: list[Item] = []
    for config in sorted(args.asurre_config_dir.glob("*.json")):
        asurre_items.extend(load_asurre_items(config))
    if not asurre_items:
        raise ValueError(f"no ASURRE configs found in {args.asurre_config_dir}")
    asurre_items = assign_heldout_survey(asurre_items, args.heldout_asurre_source)
    # Task #29 uses only train-split ASURRE prompts for the assignment bank. Preserve the held-out
    # source outside this bank so it cannot accidentally enter style fitting or the main run.
    asurre_train = [
        item.model_copy(update={"assessment_type": "assignment"})
        for item in asurre_items
        if item.split == "train"
    ]
    asurre_train = stable_sample(asurre_train, args.asurre_limit, args.seed)

    banks = [
        make_bank("quiz-mmlu-200", "MMLU quiz (200)", "quiz", mmlu, seed=args.seed),
        make_bank("quiz-arc-200", "ARC quiz (200)", "quiz", arc, seed=args.seed),
        make_bank(
            "survey-opinionqa-200",
            "OpinionQA survey (200)",
            "survey",
            opinion_items,
            seed=args.seed,
            opinion=opinion,
        ),
        make_bank("survey-ipip", "IPIP Big Five", "survey", ipip_items, seed=args.seed),
        make_bank("survey-gss", "GSS survey", "survey", gss_items, seed=args.seed),
        make_bank(
            "assignment-asurre-train",
            "ASURRE writing assignment (train surveys only)",
            "assignment",
            asurre_train,
            seed=args.seed,
        ),
    ]
    distributions = opinion_distributions + ipip_distributions + gss_distributions
    requested_sizes = {
        "quiz-mmlu-200": args.mmlu_limit,
        "quiz-arc-200": args.arc_limit,
        "survey-opinionqa-200": args.opinionqa_limit,
        "survey-ipip": None,
        "survey-gss": None,
        "assignment-asurre-train": args.asurre_limit,
    }
    return write_bundle(
        args.output,
        banks,
        distributions,
        seed=args.seed,
        heldout_asurre_source=args.heldout_asurre_source,
        requested_sizes=requested_sizes,
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/processed/runs_v1"))
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--mmlu-root", type=Path, default=Path("data/raw/mmlu/data"))
    ap.add_argument("--mmlu-split", default="test", choices=("dev", "val", "test"))
    ap.add_argument("--mmlu-limit", type=int, default=200)
    ap.add_argument(
        "--arc-path",
        type=Path,
        default=Path("data/raw/arc/ARC-Challenge/ARC-Challenge-Test.jsonl"),
    )
    ap.add_argument("--arc-subset", default="challenge", choices=("challenge", "easy"))
    ap.add_argument("--arc-limit", type=int, default=200)
    ap.add_argument(
        "--opinionqa-root", type=Path, default=Path("data/raw/opinionqa/human_resp")
    )
    ap.add_argument("--opinionqa-limit", type=int, default=200)
    ap.add_argument(
        "--ipip-data", type=Path, default=Path("data/raw/ipip/BIG5/BIG5/data.csv")
    )
    ap.add_argument(
        "--ipip-codebook", type=Path, default=Path("data/raw/ipip/BIG5/BIG5/codebook.txt")
    )
    ap.add_argument("--gss-data", type=Path, default=Path("data/raw/gss/gss.csv"))
    ap.add_argument("--gss-manifest", type=Path, default=Path("data/raw/gss/manifest.json"))
    ap.add_argument(
        "--asurre-config-dir", type=Path, default=Path("data/raw/asurre/configs")
    )
    ap.add_argument("--asurre-limit", type=int, default=100)
    ap.add_argument(
        "--heldout-asurre-source",
        required=True,
        choices=ASURRE_SOURCES,
        help="explicit team decision; this source is excluded from the assignment bank",
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

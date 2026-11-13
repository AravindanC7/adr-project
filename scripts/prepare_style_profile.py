"""Fit the writing-style profile from ASURRE training responses only.

The held-out ASURRE source is read from ``configs/data_decisions.json`` and is
never opened by this script.  The output includes source checksums and expected
versus observed response counts so the fitted profile can be audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import Item
from adr.data.loaders import load_asurre_items
from adr.mechanisms.style import fit_style_profile


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_responses(
    frame: pd.DataFrame,
    config: dict[str, Any],
    items: list[Item],
) -> tuple[list[tuple[Item, str]], dict[str, dict[str, int]]]:
    """Pair non-empty spreadsheet responses with their validated ASURRE item."""

    item_by_field = {item.item_id.rsplit(":", 1)[1].casefold(): item for item in items}
    start = config.get("data_starts_row")
    attributes = config.get("attributes")
    if not isinstance(start, int) or start < 0:
        raise ValueError("ASURRE config needs a non-negative data_starts_row")
    if not isinstance(attributes, dict):
        raise TypeError("ASURRE config needs an attributes object")

    samples: list[tuple[Item, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for field, metadata in attributes.items():
        item = item_by_field.get(str(field).casefold())
        if item is None:
            continue
        if not isinstance(metadata, dict) or not isinstance(metadata.get("column_index"), int):
            raise TypeError(f"{field}: missing integer column_index")
        column = metadata["column_index"]
        if column >= frame.shape[1]:
            raise ValueError(f"{field}: column {column} is outside spreadsheet width")
        header = frame.iloc[0, column]
        if str(header).strip().casefold() != str(field).strip().casefold():
            raise ValueError(f"{field}: spreadsheet column {column} has header {header!r}")

        observed = 0
        for value in frame.iloc[start:, column]:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            samples.append((item, text))
            observed += 1
        expected = metadata.get("responses")
        counts[str(field)] = {
            "expected_non_empty": expected if isinstance(expected, int) else -1,
            "observed_non_empty": observed,
        }
    return samples, counts


def build_profile(
    *,
    config_dir: Path,
    raw_dir: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    """Build an auditable profile document without touching the held-out source."""

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    heldout_source = decisions["asurre"]["heldout_source"]
    all_samples: list[tuple[Item, str]] = []
    source_audit: list[dict[str, Any]] = []

    for config_path in sorted(config_dir.glob("*.json")):
        items = load_asurre_items(config_path)
        source = items[0].source
        if source == heldout_source:
            continue
        if any(item.split != "train" for item in items):
            raise ValueError(f"{config_path}: style fitting accepts training items only")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_file = config.get("source")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError(f"{config_path}: missing source filename")
        raw_path = raw_dir / source_file
        if not raw_path.is_file():
            raise FileNotFoundError(f"ASURRE response file not found: {raw_path}")
        frame = pd.read_excel(raw_path, header=None)
        samples, counts = collect_responses(frame, config, items)
        all_samples.extend(samples)
        source_audit.append(
            {
                "source": source,
                "raw_file": str(raw_path),
                "raw_sha256": file_sha256(raw_path),
                "config_file": str(config_path),
                "config_sha256": file_sha256(config_path),
                "response_counts": counts,
                "n_observed_non_empty": len(samples),
            }
        )

    if not source_audit:
        raise ValueError("no non-held-out ASURRE training sources found")
    profile = fit_style_profile(all_samples)
    return {
        "version": "style-profile-v1",
        "fitting_rule": "all non-empty responses from non-held-out ASURRE sources",
        "heldout_source_excluded": heldout_source,
        "profile": asdict(profile),
        "sources": source_audit,
        "decisions_file": str(decisions_path),
        "decisions_sha256": file_sha256(decisions_path),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", type=Path, default=Path("data/raw/asurre/configs"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/asurre/raw"))
    ap.add_argument("--decisions", type=Path, default=Path("configs/data_decisions.json"))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/runs_v1/style_profile.json"),
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    document = build_profile(
        config_dir=args.config_dir,
        raw_dir=args.raw_dir,
        decisions_path=args.decisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_training": document["profile"]["n_training"],
                "sources": document["profile"]["sources"],
                "heldout_source_excluded": document["heldout_source_excluded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

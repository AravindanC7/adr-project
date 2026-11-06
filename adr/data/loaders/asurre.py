"""Load survey questions described by the public ASURRE dataset configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from adr.contracts import Item

_SOURCE_BY_DATASET = {
    "phd2019": "asurre_nature_phd_2019",
    "springer_2017": "asurre_springer_social_2017",
    "osmi_mh": "asurre_osmi_2017_18",
}


def load_asurre_items(
    config_path: str | Path,
    *,
    split: Literal["train", "heldout_survey"] = "train",
) -> list[Item]:
    """Convert one ASURRE source-survey config into open-text Items."""

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"ASURRE config file not found: {path}")
    dataset = path.stem
    source = _SOURCE_BY_DATASET.get(dataset)
    if source is None:
        raise ValueError(f"{path}: unsupported ASURRE dataset {dataset!r}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"{path}: config must be a JSON object")
    attributes = config.get("attributes")
    if not isinstance(attributes, dict) or not attributes:
        raise ValueError(f"{path}: config needs a non-empty attributes object")

    items: list[Item] = []
    seen_item_ids: set[str] = set()
    for field, metadata in attributes.items():
        if not isinstance(field, str) or not field:
            raise TypeError(f"{path}: attribute names must be non-empty strings")
        if not isinstance(metadata, dict):
            raise TypeError(f"{path}: metadata for {field} must be an object")
        if metadata.get("type") != "text":
            continue
        label = metadata.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{path}: {field} needs a non-empty label")
        item_id = f"{source}:{field.lower()}"
        if item_id in seen_item_ids:
            raise ValueError(f"{path}: duplicate normalized item_id {item_id!r}")
        seen_item_ids.add(item_id)
        items.append(
            Item(
                item_id=item_id,
                source=source,
                assessment_type="survey",
                item_type="open_text",
                stem=label.strip(),
                eligible_mechanisms=["canary", "glitch_token", "style"],
                split=split,
            )
        )
    if not items:
        raise ValueError(f"{path}: no text attributes found")
    return items

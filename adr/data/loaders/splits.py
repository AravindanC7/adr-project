"""Utilities for applying a survey-level held-out split without choosing it implicitly."""

from __future__ import annotations

from adr.contracts import Item
from adr.contracts.schema import Source


def assign_heldout_survey(items: list[Item], heldout_source: Source) -> list[Item]:
    """Mark every item from one explicitly selected survey source as held out."""

    if not any(item.source == heldout_source for item in items):
        raise ValueError(f"held-out source {heldout_source!r} is not present in the items")
    return [
        item.model_copy(update={"split": "heldout_survey"})
        if item.source == heldout_source
        else item.model_copy(update={"split": "train"})
        for item in items
    ]

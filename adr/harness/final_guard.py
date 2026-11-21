"""The held-out guard for attacker runs (AGENTS.md section 8). Owner: Arav.

The held-out model (claude-haiku, `heldout: true` in configs/models.yaml) and the held-out survey
(OSMI, split `heldout_survey`) are used ONLY in the final evaluation, once. Every attacker driver
calls `check_final` before its first API call:

  * without --final: refuse any held-out model, held-out-split item or OSMI item;
  * with --final: the run id must contain "final", so final outputs can never be mistaken for, or
    overwrite, a training run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

HELDOUT_SOURCE = "asurre_osmi_2017_18"


def heldout_items(items: Mapping, item_ids: Iterable[str]) -> list[str]:
    """Item ids that belong to the held-out survey (by source or by split)."""
    return [
        i for i in item_ids if items[i].source == HELDOUT_SOURCE or items[i].split != "train"
    ]


def check_final(
    client, models: Iterable[str], items: Mapping, item_ids: Iterable[str], final: bool, run_id: str
) -> None:
    """Raise SystemExit unless this run may touch what it is about to touch."""
    held_models = [m for m in models if client.spec(m).heldout]
    held_items = heldout_items(items, item_ids)
    if not final and (held_models or held_items):
        what = []
        if held_models:
            what.append(f"held-out model(s) {held_models}")
        if held_items:
            what.append(f"{len(held_items)} held-out survey item(s)")
        raise SystemExit(
            f"refusing {' and '.join(what)} without --final: the held-out model and survey run "
            "only in the final evaluation (AGENTS.md section 8)"
        )
    if final and "final" not in run_id:
        raise SystemExit(f"--final runs need a run id containing 'final' (got {run_id!r})")

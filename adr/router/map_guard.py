"""Refuse provisional failure maps before the router or replay uses them.

A map is provisional when its human side is not real data:
  - its version ends in "-ai-only" (Archana's builder, --ai-only, #24), or
  - any cell has zero human counts, so its p_human is the smoothing prior alone (uniform).

With a uniform p_human, every outcome that is common for AI looks like strong evidence of AI
use (llr = log p_ai - log 1/K), so the router would flag people on AI data alone. Saved maps
that the router or the replay will use go through `load_router_map`; `--allow-provisional-map`
lets a development run use them anyway, with a loud warning.

FailureMap.load itself stays unguarded: plotting or inspecting a provisional map is fine. The
TEMPORARY quick map (adr/replay/quick_map.py) is built in memory and already warns on its own.
"""

from __future__ import annotations

from pathlib import Path

from adr.contracts import FailureMap

AI_ONLY_SUFFIX = "-ai-only"
ALLOW_FLAG = "--allow-provisional-map"


class ProvisionalMapError(ValueError):
    """A provisional failure map was given to the router or replay without the allow flag."""


def provisional_reasons(fmap: FailureMap) -> list[str]:
    """Why this map is not ready for routing; empty if it is."""
    reasons = []
    if fmap.version.endswith(AI_ONLY_SUFFIX):
        reasons.append(f"its version {fmap.version!r} ends in {AI_ONLY_SUFFIX!r} (an AI-only map)")
    empty = [
        "/".join(
            str(part)
            for part in (c.mechanism, c.item_type, c.attack_mode or "*", c.attacker_level or "*")
        )
        for c in fmap.cells
        if sum(c.counts_human.values()) == 0
    ]
    if empty:
        shown = ", ".join(empty[:5]) + (f" and {len(empty) - 5} more" if len(empty) > 5 else "")
        reasons.append(
            f"{len(empty)} of {len(fmap.cells)} cells have zero human counts, so their "
            f"P(outcome | human) is the smoothing prior only: {shown}"
        )
    return reasons


def check_router_map(fmap: FailureMap, allow_provisional: bool = False) -> list[str]:
    """Raise ProvisionalMapError for a provisional map unless allowed. Returns the reasons
    (empty for a normal map) so an allowed run can warn about them."""
    reasons = provisional_reasons(fmap)
    if reasons and not allow_provisional:
        raise ProvisionalMapError(
            f"refusing failure map {fmap.version!r} for routing: "
            + "; ".join(reasons)
            + ". Its human probabilities are not real data, so AI-typical outcomes would count "
            "as strong evidence and honest people would be flagged. Rebuild the map with human "
            f"outcome rows, or pass {ALLOW_FLAG} for a development run only."
        )
    return reasons


def load_router_map(path: str | Path, allow_provisional: bool = False) -> FailureMap:
    """FailureMap.load plus check_router_map: use this wherever a saved map feeds the router."""
    fmap = FailureMap.load(path)
    check_router_map(fmap, allow_provisional)
    return fmap

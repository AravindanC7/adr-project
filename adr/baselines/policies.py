"""Build any policy (the router or a baseline) by name, for replay and comparisons.

    router          the Adaptive Defense Router (adr/router/core.py)
    router_noclear  the router with clearing disabled (lower threshold B = -inf): it picks checks
                    adaptively but, like the baselines, never stops checking a low score
    best_single     baseline 1: the best single mechanism for this assessment (train map)
    all_checks      baseline 2: a check on every item
    fixed_plan      baseline 3: the best check per item, chosen before the session
    random          baseline 4: a random check per item
    single:<mech>   one named mechanism wherever it applies
"""

from __future__ import annotations

import math
from dataclasses import replace

from adr.baselines.plans import AllChecksRouter, FixedPlanRouter, best_single_mechanism
from adr.baselines.simple import RandomCheckRouter, SingleMechanismRouter
from adr.contracts import Assessment, FailureMap, Item, RenderedItem, Router
from adr.router.core import RouterConfig, SPRTRouter

COMPARE = ("router", "router_noclear", "best_single", "all_checks", "fixed_plan", "random")


def make_policy(
    name: str,
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    fmap: FailureMap,
    cfg: RouterConfig | None = None,
) -> Router:
    """`fmap` must be the map fitted on train rows: best_single and fixed_plan choose from it."""
    cfg = cfg or RouterConfig()
    args = (items, variants, fmap, cfg)
    if name == "router":
        return SPRTRouter(*args)
    if name == "router_noclear":
        return SPRTRouter(items, variants, fmap, replace(cfg, lower=-math.inf))
    if name == "best_single":
        mech, _ = best_single_mechanism(fmap, assessment, items, variants, cfg.attack_mode)
        return SingleMechanismRouter(*args, mechanism=mech)
    if name == "all_checks":
        return AllChecksRouter(*args)
    if name == "fixed_plan":
        return FixedPlanRouter(*args)
    if name == "random":
        return RandomCheckRouter(*args)
    if name.startswith("single:"):
        return SingleMechanismRouter(*args, mechanism=name.split(":", 1)[1])
    raise ValueError(
        f"unknown policy {name!r}: use {', '.join(COMPARE)} or single:<mechanism>"
    )

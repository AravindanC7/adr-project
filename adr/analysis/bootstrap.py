"""Two-level bootstrap for rates over items and sessions (moved from scripts/m2_decoy_figure.py).

A rate (e.g. the share of decoy answers that picked the decoy) is computed from observations
grouped by item: per_item[i] holds item i's 0/1 observations, one per session. One replicate
resamples items with replacement, then, within each sampled item, resamples that item's sessions
with replacement, and pools all drawn observations into one rate.

The paired version (RQ4: rate at level n minus rate at L0) draws the items ONCE and applies the
same draws to both levels; sessions are resampled separately within each level.
"""

from __future__ import annotations

import numpy as np


def two_level_bootstrap(
    per_item: list[np.ndarray], n_boot: int, rng, item_draws: np.ndarray | None = None
) -> np.ndarray:
    """Bootstrap rates: resample items, then each sampled item's observations (sessions).

    Items may have different numbers of observations; an item with one observation always
    contributes that value. An empty group (no items) gives NaN for every replicate.
    `item_draws` (n_boot x n_items indices into per_item) fixes the item level, for a paired
    bootstrap; every item must then have at least one observation."""
    if item_draws is not None and any(len(a) == 0 for a in per_item):
        raise ValueError("paired bootstrap: every item needs at least one observation")
    per_item = [np.asarray(a, dtype=float) for a in per_item if len(a)]
    n_items = len(per_item)
    if n_items == 0:
        return np.full(n_boot, np.nan)
    n_obs = np.array([len(a) for a in per_item])
    width = int(n_obs.max())
    padded = np.full((n_items, width), np.nan)
    for i, a in enumerate(per_item):
        padded[i, : len(a)] = a
    if item_draws is None:
        items = rng.integers(0, n_items, size=(n_boot, n_items))
    else:
        items = np.asarray(item_draws)
        if items.shape != (n_boot, n_items):
            raise ValueError(f"item_draws must have shape {(n_boot, n_items)}, got {items.shape}")
    k = n_obs[items]  # observations available for each sampled item
    pick = np.floor(rng.random((n_boot, n_items, width)) * k[..., None]).astype(int)
    values = padded[items[..., None], pick]
    mask = np.arange(width)[None, None, :] < k[..., None]  # draw n_i sessions for item i
    return np.where(mask, values, 0.0).sum(axis=(1, 2)) / mask.sum(axis=(1, 2))


def paired_difference_bootstrap(
    per_item_a: list[np.ndarray], per_item_b: list[np.ndarray], n_boot: int, rng
) -> np.ndarray:
    """Bootstrap of rate(b) - rate(a) over the SAME items: per_item_a[i] and per_item_b[i] are
    the same item's observations under two conditions. Items are drawn once for both."""
    if len(per_item_a) != len(per_item_b):
        raise ValueError("paired bootstrap: both conditions need the same items, in order")
    n_items = len(per_item_a)
    if n_items == 0:
        return np.full(n_boot, np.nan)
    draws = rng.integers(0, n_items, size=(n_boot, n_items))
    a = two_level_bootstrap(per_item_a, n_boot, rng, item_draws=draws)
    b = two_level_bootstrap(per_item_b, n_boot, rng, item_draws=draws)
    return b - a

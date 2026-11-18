"""TEMPORARY failure map straight from the outcome table, for replay until #34 lands.

Owner: Arav. Replace with Archana's builder (adr/map/, step #34) as soon as it exists: that one
adds confidence intervals and is the version the report uses. Nothing built here is a result.

The held-out rule (AGENTS.md section 8) applies here as everywhere: the map is fitted ONLY on
rows with split == "train" from non-held-out models. Rows from the held-out model or the
held-out survey are dropped before counting, and if nothing is left we refuse to build a map.

Three splits (split_rows):
  train        fits the failure map
  calibration  a fixed share of the non-held-out items (by item, fixed seed): development
               evaluation, thresholds and the per-mechanism cap (#42). Never fits the map.
  held-out     held-out model and held-out survey rows: final evaluation only (--final)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from adr.contracts import FailureMap, MapCell

TEMPORARY_NOTE = (
    "TEMPORARY quick map from adr/replay/quick_map.py (counts + add-one smoothing, no CIs). "
    "Replace with Archana's builder (#34). Not a result."
)


def heldout_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that may only be used for final evaluation, never for fitting."""
    return df["heldout_model"].astype(bool) | (df["split"] != "train")


def final_eval_rows(heldout: pd.DataFrame, heldout_model: str) -> pd.DataFrame:
    """--final: the held-out rows that may be evaluated. AI rows come only from the decided
    held-out model (configs/data_decisions.json), so rows from an earlier held-out model (the
    gemini-flash smoke runs, logged with heldout_model=True) never enter the final evaluation.
    Human rows (the held-out survey) are kept."""
    ai = heldout["respondent_kind"] == "ai"
    return heldout[~ai | (heldout["model"] == heldout_model)]


def fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The rows a map may be fitted on: train split, not the held-out model."""
    return df[~heldout_mask(df)]


def quick_failure_map(
    df: pd.DataFrame, version: str = "quick-temporary", prior: float = 1.0
) -> FailureMap:
    """One cell per (mechanism, item_type) pooled over attack modes, plus one per attack mode
    seen in the AI rows (sharing the pooled human counts: people have no attack mode)."""
    rows = fit_rows(df)
    rows = rows[rows["mechanism"].notna()]
    if rows.empty:
        raise ValueError(
            "no train-split, non-held-out rows with a mechanism: refusing to fit a map on "
            "held-out data (AGENTS.md section 8)"
        )

    def counts(part: pd.DataFrame) -> dict[str, int]:
        return {str(k): int(v) for k, v in part["outcome"].value_counts().items()}

    cells: list[MapCell] = []
    no_human: list[str] = []
    for (mech, item_type), part in rows.groupby(["mechanism", "item_type"], sort=True):
        ai = part[part["respondent_kind"] == "ai"]
        human = counts(part[part["respondent_kind"] == "human"])
        if not human:
            no_human.append(f"{mech}/{item_type}")
        cells.append(MapCell.from_counts(mech, item_type, counts(ai), human, prior=prior))
        for mode, ai_mode in ai.groupby("attack_mode", sort=True):
            cells.append(MapCell.from_counts(mech, item_type, counts(ai_mode), human,
                                             attack_mode=mode, prior=prior))
    note = TEMPORARY_NOTE
    if no_human:
        note += f" No human rows (p_human is uniform) for: {', '.join(no_human)}."
    return FailureMap(
        version=version,
        source_run_ids=sorted(rows["run_id"].unique()),
        note=note,
        cells=cells,
    )


CALIBRATION_FRAC = 0.3
SPLIT_SEED = 0


def calibration_items(
    df: pd.DataFrame, frac: float = CALIBRATION_FRAC, seed: int = SPLIT_SEED
) -> set[str]:
    """The non-held-out items set aside for calibration: about `frac` of them, chosen per item
    type with a fixed seed, so every item type keeps at least one item for fitting the map.
    The same rows and seed always give the same items."""
    if not 0 <= frac < 1:
        raise ValueError("calibration frac must be in [0, 1)")
    rows = fit_rows(df)
    out: set[str] = set()
    for item_type, part in rows.groupby("item_type", sort=True):
        ids = sorted(part["item_id"].unique())
        k = min(round(frac * len(ids)), len(ids) - 1)
        random.Random(f"{seed}:{item_type}").shuffle(ids)
        out.update(ids[:k])
    return out


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame        # fits the map
    calibration: pd.DataFrame  # development evaluation and thresholds (#42)
    heldout: pd.DataFrame      # final evaluation only
    calibration_item_ids: frozenset[str]


def split_rows(
    df: pd.DataFrame, frac: float = CALIBRATION_FRAC, seed: int = SPLIT_SEED
) -> Splits:
    """Split the outcome table into train / calibration / held-out rows (see module doc)."""
    calib = calibration_items(df, frac, seed)
    usable = fit_rows(df)
    in_calib = usable["item_id"].isin(calib)
    return Splits(
        train=usable[~in_calib],
        calibration=usable[in_calib],
        heldout=df[heldout_mask(df)],
        calibration_item_ids=frozenset(calib),
    )

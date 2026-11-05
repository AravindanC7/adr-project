"""TEMPORARY failure map straight from the outcome table, for replay until #34 lands.

Owner: Arav. Replace with Archana's builder (adr/map/, step #34) as soon as it exists: that one
adds confidence intervals and is the version the report uses. Nothing built here is a result.

The held-out rule (AGENTS.md section 8) applies here as everywhere: the map is fitted ONLY on
rows with split == "train" from non-held-out models. Rows from the held-out model or the
held-out survey are dropped before counting, and if nothing is left we refuse to build a map.
"""

from __future__ import annotations

import pandas as pd

from adr.contracts import FailureMap, MapCell

TEMPORARY_NOTE = (
    "TEMPORARY quick map from adr/replay/quick_map.py (counts + add-one smoothing, no CIs). "
    "Replace with Archana's builder (#34). Not a result."
)


def heldout_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that may only be used for final evaluation, never for fitting."""
    return df["heldout_model"].astype(bool) | (df["split"] != "train")


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


def train_eval_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """(rows to fit the map on, rows to evaluate on, same_data).

    Evaluation uses the held-out rows (held-out model or held-out survey) when there are any.
    If there are none, it falls back to the fit rows and `same_data` is True: the numbers are
    then optimistic, and callers must say so loudly.
    """
    held = df[heldout_mask(df)]
    fit = fit_rows(df)
    if held.empty:
        return fit, fit, True
    return fit, held, False

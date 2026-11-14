"""Aggregate outcome rows into a smoothed failure-mode map with confidence intervals."""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from adr.contracts import OUTCOME_LABELS, FailureMap, MapCell, smooth


def _counts(frame: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, int]:
    observed = Counter(str(value) for value in frame["outcome"])
    unknown = set(observed) - set(labels)
    if unknown:
        raise ValueError(f"outcomes outside {labels}: {sorted(unknown)}")
    return {label: observed.get(label, 0) for label in labels}


def _cell_seed(seed: int, key: tuple[object, ...], respondent_kind: str) -> int:
    material = repr((seed, key, respondent_kind)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _clustered_interval(
    frame: pd.DataFrame,
    labels: tuple[str, ...],
    *,
    n_bootstrap: int,
    seed: int,
    prior: float,
) -> dict[str, tuple[float, float]]:
    """Bootstrap sessions, retaining correlated answers within each session."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    groups = [
        group
        for _, group in frame.groupby(["run_id", "session_id"], sort=True, dropna=False)
    ]
    if not groups:
        raise ValueError("cannot bootstrap an empty respondent class")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {label: [] for label in labels}
    for _ in range(n_bootstrap):
        positions = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat([groups[position] for position in positions], ignore_index=True)
        probabilities = smooth(_counts(sampled, labels), labels, prior)
        for label in labels:
            draws[label].append(probabilities[label])
    return {
        label: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for label, values in draws.items()
    }


def _eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove plain observations and every row reserved for final evaluation."""

    heldout_model = frame["heldout_model"].fillna(False).astype(bool)
    eligible = frame[
        frame["mechanism"].notna()
        & ~heldout_model
        & frame["split"].ne("heldout_survey")
    ].copy()
    return eligible


def build_failure_map(
    frame: pd.DataFrame,
    *,
    version: str,
    n_bootstrap: int = 2000,
    seed: int = 34,
    prior: float = 1.0,
    note: str = "",
) -> FailureMap:
    """Build exact and pooled cells from non-held-out AI and human observations.

    A cell is emitted only when both respondent classes have evidence for the same
    mechanism and item type. Human observations are pooled over attack settings,
    because human rows intentionally have no attack mode or attacker level.
    """

    required = {
        "run_id",
        "session_id",
        "respondent_kind",
        "mechanism",
        "item_type",
        "outcome",
        "attack_mode",
        "attacker_level",
        "heldout_model",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"outcome frame is missing columns: {missing}")
    if prior <= 0:
        raise ValueError("prior must be positive")

    eligible = _eligible_rows(frame)
    cells: list[MapCell] = []
    source_run_ids: set[str] = set()
    base_keys = sorted(
        {
            (str(row.mechanism), str(row.item_type))
            for row in eligible.itertuples()
        }
    )
    for mechanism, item_type in base_keys:
        labels = OUTCOME_LABELS[mechanism]
        base = eligible[
            eligible["mechanism"].eq(mechanism) & eligible["item_type"].eq(item_type)
        ]
        human = base[base["respondent_kind"].eq("human")]
        ai = base[base["respondent_kind"].eq("ai")]
        if human.empty or ai.empty:
            continue

        settings: set[tuple[str | None, str | None]] = {(None, None)}
        for mode, level in ai[["attack_mode", "attacker_level"]].itertuples(index=False):
            settings.update(((str(mode), str(level)), (str(mode), None), (None, str(level))))

        human_counts = _counts(human, labels)
        human_ci = _clustered_interval(
            human,
            labels,
            n_bootstrap=n_bootstrap,
            seed=_cell_seed(seed, (mechanism, item_type), "human"),
            prior=prior,
        )
        for attack_mode, attacker_level in sorted(
            settings, key=lambda setting: (setting[0] or "", setting[1] or "")
        ):
            selected_ai = ai
            if attack_mode is not None:
                selected_ai = selected_ai[selected_ai["attack_mode"].eq(attack_mode)]
            if attacker_level is not None:
                selected_ai = selected_ai[selected_ai["attacker_level"].eq(attacker_level)]
            if selected_ai.empty:
                continue
            source_run_ids.update(str(value) for value in selected_ai["run_id"].unique())
            source_run_ids.update(str(value) for value in human["run_id"].unique())
            key = (mechanism, item_type, attack_mode, attacker_level)
            cell = MapCell.from_counts(
                mechanism=mechanism,
                item_type=item_type,
                counts_ai=_counts(selected_ai, labels),
                counts_human=human_counts,
                attack_mode=attack_mode,
                attacker_level=attacker_level,
                prior=prior,
            )
            cell.ci_ai = _clustered_interval(
                selected_ai,
                labels,
                n_bootstrap=n_bootstrap,
                seed=_cell_seed(seed, key, "ai"),
                prior=prior,
            )
            cell.ci_human = human_ci
            cells.append(cell)

    if not cells:
        raise ValueError("no mechanism/item type has both AI and human outcome rows")
    exclusion_note = "Held-out models and held-out survey rows excluded."
    combined_note = " ".join(part for part in (note.strip(), exclusion_note) if part)
    return FailureMap(
        version=version,
        source_run_ids=sorted(source_run_ids),
        note=combined_note,
        cells=cells,
    )

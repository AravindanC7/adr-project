"""Outcome table: the row-level log of every observation. The most important contract.

Owner: Arav. DRAFT until the team freezes it.

One row = one (respondent, item, mechanism) observation. The failure map, the replay simulation
and every result are computed from these rows. Stored as Parquet under results/outcomes/.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from adr.contracts.mechanism import OUTCOME_LABELS
from adr.contracts.schema import AssessmentType, AttackMode, ItemType, MechanismName, Split

RespondentKind = Literal["ai", "human"]
DataOrigin = Literal["harness", "public_human", "pilot", "live", "dry_run"]
AttackerLevel = Literal["L0", "L1", "L2", "L3", "L4"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class OutcomeRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Links to results/runs/<run_id>/config.yaml")
    session_id: str = Field(description="One respondent taking one assessment")
    respondent_kind: RespondentKind
    data_origin: DataOrigin
    model: str | None = Field(default=None, description="Our model label for AI rows; None for humans")
    attack_mode: AttackMode | None = None
    attacker_level: AttackerLevel | None = None
    assessment_type: AssessmentType
    item_type: ItemType
    item_id: str
    split: Split
    heldout_model: bool = False
    mechanism: MechanismName | None
    variant_id: str
    outcome: str
    raw_ref: str = Field(description="Path or cache key to the full response")
    prompt_version: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check(self) -> OutcomeRow:
        if self.respondent_kind == "ai":
            if not self.model or not self.attack_mode or not self.attacker_level:
                raise ValueError("AI rows need model, attack_mode and attacker_level")
        else:
            if self.model or self.attack_mode or self.attacker_level or self.heldout_model:
                raise ValueError("human rows must leave model/attack_mode/attacker_level empty")
            if self.data_origin == "harness":
                raise ValueError("harness rows are AI rows")
        if self.mechanism is None:
            if self.outcome != "none":
                raise ValueError('rows with no mechanism must have outcome "none"')
        elif self.outcome not in OUTCOME_LABELS[self.mechanism]:
            raise ValueError(
                f"{self.mechanism} outcome must be one of {OUTCOME_LABELS[self.mechanism]}, "
                f"got {self.outcome!r}"
            )
        return self


COLUMNS: list[str] = list(OutcomeRow.model_fields)


def rows_to_frame(rows: Iterable[OutcomeRow]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in rows], columns=COLUMNS)


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check every row of a DataFrame against the contract. Raises on the first bad row."""
    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing or extra:
        raise ValueError(f"outcome table columns wrong; missing={missing} extra={extra}")
    for i, rec in enumerate(df[COLUMNS].to_dict(orient="records")):
        rec = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}
        try:
            OutcomeRow(**rec)
        except Exception as e:
            raise ValueError(f"outcome table row {i} is invalid: {e}") from e
    return df[COLUMNS]


def write_outcomes(rows: Iterable[OutcomeRow], path: str | Path) -> Path:
    """Write rows as Parquet. Validation already happened when each OutcomeRow was built."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows).to_parquet(path, index=False)
    return path


def read_outcomes(paths: str | Path | Iterable[str | Path], validate: bool = True) -> pd.DataFrame:
    """Read one Parquet file, a directory of them, or a list. Validates by default."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    else:
        files = [Path(x) for x in paths]
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return validate_frame(df) if validate else df

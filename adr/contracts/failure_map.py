"""Failure-mode map: P(outcome | AI) and P(outcome | human) for each setting.

Owner: Archana (format agreed with Arav). DRAFT until the team freezes it.

Archana's builder in adr/map/ turns the outcome table into a FailureMap. The router only reads it.
`smooth()` lives here so the builder and any synthetic map use the exact same smoothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adr.contracts.mechanism import OUTCOME_LABELS
from adr.contracts.schema import AttackMode, ItemType, MechanismName


def smooth(counts: dict[str, int], labels: tuple[str, ...], prior: float = 1.0) -> dict[str, float]:
    """Add-`prior` (Dirichlet) smoothing over a mechanism's full label set.

    (count + prior) / (n + prior * K). No probability is ever 0, so a log-likelihood ratio is
    never infinite. Labels missing from `counts` count as 0.
    """
    unknown = set(counts) - set(labels)
    if unknown:
        raise ValueError(f"counts has labels outside {labels}: {sorted(unknown)}")
    n = sum(counts.get(lab, 0) for lab in labels)
    k = len(labels)
    return {lab: (counts.get(lab, 0) + prior) / (n + prior * k) for lab in labels}


class MapCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanism: MechanismName
    item_type: ItemType
    attack_mode: AttackMode | None = Field(default=None, description="None = pooled over modes")
    attacker_level: str | None = Field(default=None, description="None = pooled over levels")
    counts_ai: dict[str, int]
    counts_human: dict[str, int]
    p_ai: dict[str, float]
    p_human: dict[str, float]
    ci_ai: dict[str, tuple[float, float]] = Field(default_factory=dict)
    ci_human: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> MapCell:
        labels = set(OUTCOME_LABELS[self.mechanism])
        for name in ("p_ai", "p_human"):
            p = getattr(self, name)
            if set(p) != labels:
                raise ValueError(f"{name} must cover exactly {sorted(labels)}")
            if any(v <= 0 for v in p.values()):
                raise ValueError(f"{name} has a zero probability; smooth the counts first")
            if abs(sum(p.values()) - 1.0) > 1e-6:
                raise ValueError(f"{name} must sum to 1")
        return self

    def llr(self, outcome: str) -> float:
        """log P(outcome | AI) - log P(outcome | human)."""
        return math.log(self.p_ai[outcome]) - math.log(self.p_human[outcome])

    @classmethod
    def from_counts(
        cls,
        mechanism: str,
        item_type: str,
        counts_ai: dict[str, int],
        counts_human: dict[str, int],
        attack_mode: str | None = None,
        attacker_level: str | None = None,
        prior: float = 1.0,
    ) -> MapCell:
        labels = OUTCOME_LABELS[mechanism]
        return cls(
            mechanism=mechanism,
            item_type=item_type,
            attack_mode=attack_mode,
            attacker_level=attacker_level,
            counts_ai={lab: counts_ai.get(lab, 0) for lab in labels},
            counts_human={lab: counts_human.get(lab, 0) for lab in labels},
            p_ai=smooth(counts_ai, labels, prior),
            p_human=smooth(counts_human, labels, prior),
        )


class FailureMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    source_run_ids: list[str] = Field(default_factory=list)
    note: str = ""
    cells: list[MapCell]

    def lookup(
        self,
        mechanism: str,
        item_type: str,
        attack_mode: str | None = None,
        attacker_level: str | None = None,
    ) -> MapCell | None:
        """Most specific matching cell, falling back to pooled cells. None = no data."""
        wanted = [
            (attack_mode, attacker_level),
            (attack_mode, None),
            (None, attacker_level),
            (None, None),
        ]
        index = {
            (c.mechanism, c.item_type, c.attack_mode, c.attacker_level): c for c in self.cells
        }
        for mode, level in wanted:
            cell = index.get((mechanism, item_type, mode, level))
            if cell is not None:
                return cell
        return None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FailureMap:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

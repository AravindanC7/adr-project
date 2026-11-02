"""Shared contracts. Import from here: `from adr.contracts import Item, OutcomeRow, ...`."""

from adr.contracts.failure_map import FailureMap, MapCell, smooth
from adr.contracts.io import read_jsonl, variants_path, write_jsonl
from adr.contracts.mechanism import (
    ACTIVE_MECHANISMS,
    HIDDEN_POSITIONS,
    OUTCOME_LABELS,
    PASSIVE_MECHANISMS,
    HiddenPosition,
    Mechanism,
    RenderedItem,
    Response,
    Telemetry,
    check_outcome,
    make_variant_id,
    parse_variant_id,
)
from adr.contracts.outcome import (
    COLUMNS,
    OutcomeRow,
    read_outcomes,
    rows_to_frame,
    validate_frame,
    write_outcomes,
)
from adr.contracts.router_api import Choice, Router, SessionState
from adr.contracts.schema import Assessment, HumanDistribution, Item

__all__ = [
    "ACTIVE_MECHANISMS",
    "COLUMNS",
    "HIDDEN_POSITIONS",
    "OUTCOME_LABELS",
    "PASSIVE_MECHANISMS",
    "Assessment",
    "Choice",
    "FailureMap",
    "HiddenPosition",
    "HumanDistribution",
    "Item",
    "MapCell",
    "Mechanism",
    "OutcomeRow",
    "RenderedItem",
    "Response",
    "Router",
    "SessionState",
    "Telemetry",
    "check_outcome",
    "make_variant_id",
    "parse_variant_id",
    "read_jsonl",
    "read_outcomes",
    "rows_to_frame",
    "smooth",
    "validate_frame",
    "variants_path",
    "write_jsonl",
    "write_outcomes",
]

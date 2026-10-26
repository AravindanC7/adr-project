"""Find mechanism objects by name.

CONVENTION (agree with Archana and Harshini): each mechanism lives in adr/mechanisms/<name>.py
and exposes a module-level `MECHANISM` object that satisfies `adr.contracts.Mechanism`, e.g.

    # adr/mechanisms/decoy.py
    class DecoyMechanism: ...
    MECHANISM = DecoyMechanism()
"""

from __future__ import annotations

import importlib

from adr.contracts import OUTCOME_LABELS, Mechanism


def get_mechanism(name: str) -> Mechanism:
    if name not in OUTCOME_LABELS:
        raise ValueError(f"unknown mechanism {name!r}")
    try:
        module = importlib.import_module(f"adr.mechanisms.{name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"adr/mechanisms/{name}.py doesn't exist yet (Archana/Harshini own mechanisms)"
        ) from e
    mech = getattr(module, "MECHANISM", None)
    if mech is None or not isinstance(mech, Mechanism):
        raise TypeError(f"adr.mechanisms.{name}.MECHANISM is missing or not a Mechanism")
    if tuple(mech.outcomes) != OUTCOME_LABELS[name]:
        raise ValueError(f"{name}.outcomes must equal {OUTCOME_LABELS[name]}")
    return mech


def load_mechanisms(names: list[str]) -> dict[str, Mechanism]:
    return {n: get_mechanism(n) for n in names}

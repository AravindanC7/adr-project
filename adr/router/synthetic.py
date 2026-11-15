"""SYNTHETIC data for building and testing the router before real failure-map cells exist.

Every number here is MADE UP to look plausible. Nothing from this file may appear in the report
as a result. It exists so the router can be developed now and only its final run waits on
Archana's real map (step #12 / #34).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from adr.contracts import (
    OUTCOME_LABELS,
    PASSIVE_MECHANISMS,
    Assessment,
    FailureMap,
    Item,
    MapCell,
    RenderedItem,
    SessionState,
    make_variant_id,
)
from adr.replay.metrics import detection_at_fpr  # moved to adr/replay; kept importable here

__all__ = [
    "SessionResult",
    "detection_at_fpr",
    "sample_outcome",
    "simulate_session",
    "synthetic_bank",
    "synthetic_failure_map",
]

# (mechanism, item_type): (AI outcome frequencies, human outcome frequencies), as percentages.
# Shapes follow the literature qualitatively (decoys and canaries strong on naive AI, opinion
# traps weak, style moderate); the values are invented.
_SYNTH: dict[tuple[str, str], tuple[dict[str, int], dict[str, int]]] = {
    ("decoy", "mcq"): ({"picked_decoy": 55, "picked_correct": 40, "picked_other": 5},
                       {"picked_decoy": 4, "picked_correct": 70, "picked_other": 26}),
    ("opinion_trap", "mcq"): ({"implausible": 25, "contradiction": 10, "plausible": 65},
                              {"implausible": 10, "contradiction": 6, "plausible": 84}),
    ("opinion_trap", "likert"): ({"implausible": 22, "contradiction": 15, "plausible": 63},
                                 {"implausible": 9, "contradiction": 8, "plausible": 83}),
    ("canary", "open_text"): ({"leaked": 8, "followed_instruction": 42, "clean": 50},
                              {"leaked": 1, "followed_instruction": 1, "clean": 98}),
    ("glitch_token", "open_text"): ({"anomalous": 12, "normal": 88},
                                    {"anomalous": 3, "normal": 97}),
    ("style", "open_text"): ({"high": 50, "mid": 35, "low": 15},
                             {"high": 12, "mid": 33, "low": 55}),
}
for _t in ("mcq", "likert", "open_text"):
    _SYNTH[("behaviour", _t)] = (
        {"paste": 35, "focus_loss": 30, "too_fast": 15, "normal": 20},
        {"paste": 3, "focus_loss": 15, "too_fast": 7, "normal": 75},
    )

_BURDEN = {"decoy": 0.02, "opinion_trap": 0.10, "canary": 0.0, "glitch_token": 0.15}


def synthetic_failure_map(n_per_class: int = 200) -> FailureMap:
    cells = []
    for (mech, itype), (ai, hu) in _SYNTH.items():
        ca = {k: round(v * n_per_class / 100) for k, v in ai.items()}
        ch = {k: round(v * n_per_class / 100) for k, v in hu.items()}
        cells.append(MapCell.from_counts(mech, itype, ca, ch))
    return FailureMap(version="synthetic-v1", note="SYNTHETIC: invented numbers", cells=cells)


def synthetic_bank(
    n_mcq: int = 8, n_likert: int = 6, n_open: int = 4, assessment_type: str = "quiz"
) -> tuple[Assessment, dict[str, Item], dict[str, list[RenderedItem]]]:
    """An assessment of placeholder items, each with a plain variant + one variant per check."""
    items: dict[str, Item] = {}
    variants: dict[str, list[RenderedItem]] = {}
    is_quiz = assessment_type == "quiz"

    def add(item: Item, mechs: list[str]) -> None:
        items[item.item_id] = item
        vis = {"stem": item.stem, **({"options": item.options} if item.options else {})}
        vs = [RenderedItem(item_id=item.item_id, mechanism=None,
                           variant_id=make_variant_id(item.item_id, None), visible=vis)]
        for m in mechs:
            vs.append(RenderedItem(item_id=item.item_id, mechanism=m,
                                   variant_id=make_variant_id(item.item_id, m), visible=vis,
                                   burden=_BURDEN[m]))
        variants[item.item_id] = vs

    mcq_mechs = ["decoy"] if is_quiz else ["opinion_trap"]
    for i in range(n_mcq):
        add(Item(item_id=f"handwritten:mcq{i}", source="handwritten",
                 assessment_type=assessment_type, item_type="mcq", stem=f"Placeholder MCQ {i}",
                 options=["A", "B", "C", "D"], correct_option=0 if is_quiz else None,
                 eligible_mechanisms=mcq_mechs), mcq_mechs)
    for i in range(n_likert):
        add(Item(item_id=f"handwritten:lik{i}", source="handwritten",
                 assessment_type=assessment_type, item_type="likert",
                 stem=f"Placeholder statement {i}", options=["1", "2", "3", "4", "5"],
                 eligible_mechanisms=["opinion_trap"]), ["opinion_trap"])
    for i in range(n_open):
        add(Item(item_id=f"handwritten:open{i}", source="handwritten",
                 assessment_type=assessment_type, item_type="open_text",
                 stem=f"Placeholder open question {i}",
                 eligible_mechanisms=["canary", "glitch_token"]), ["canary", "glitch_token"])
    assessment = Assessment(assessment_id=f"synthetic-{assessment_type}",
                            title=f"Synthetic {assessment_type}", assessment_type=assessment_type,
                            item_ids=list(items))
    return assessment, items, variants


def sample_outcome(cell: MapCell, kind: str, rng: random.Random) -> str:
    probs = cell.p_ai if kind == "ai" else cell.p_human
    labels = list(probs)
    return rng.choices(labels, weights=[probs[k] for k in labels], k=1)[0]


@dataclass
class SessionResult:
    kind: str
    final: SessionState
    items_shown: int
    checks_run: int
    items_to_decision: int | None  # items shown when the score first crossed A or B


def simulate_session(
    router, assessment: Assessment, fmap: FailureMap, kind: str, seed: int,
    passive: tuple[str, ...] = (),
) -> SessionResult:
    """One respondent (kind = "ai" or "human") takes the assessment; outcomes are sampled from
    `fmap`. Passive checks (e.g. "behaviour") are observed on every item shown."""
    rng = random.Random(seed)
    state, choice = router.start(assessment, f"{kind}-{seed}", seed)
    shown = checks = 0
    decided_at = None
    while choice is not None:
        item = router.items[choice.item_id]
        shown += 1
        if choice.mechanism:
            cell = fmap.lookup(choice.mechanism, item.item_type)
            outcome = sample_outcome(cell, kind, rng) if cell else OUTCOME_LABELS[choice.mechanism][-1]
            checks += 1
        else:
            outcome = "none"
        state = router.update(state, choice.variant_id, outcome)
        for p in passive:
            assert p in PASSIVE_MECHANISMS
            cell = fmap.lookup(p, item.item_type)
            if cell is not None:
                state = router.update(state, make_variant_id(item.item_id, p),
                                      sample_outcome(cell, kind, rng))
        if decided_at is None and state.status in ("review", "cleared"):
            decided_at = shown
        choice = router.next(state)
    return SessionResult(kind, state, shown, checks, decided_at)

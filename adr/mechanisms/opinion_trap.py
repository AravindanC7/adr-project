"""Opinion-trap v0 using population answer distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from adr.contracts import (
    HumanDistribution,
    Item,
    RenderedItem,
    Response,
    Telemetry,
    make_variant_id,
)


class OpinionTrapMechanism:
    """Flag options that are rare in the training human distribution.

    The shared mechanism interface scores one item at a time, so v0 does not emit
    ``contradiction`` for reverse-keyed pairs. That outcome requires session history or a separate
    paired-item scorer and must be agreed through the shared contract.
    """

    name = "opinion_trap"
    outcomes = ("implausible", "contradiction", "plausible")

    def __init__(
        self,
        distributions: Mapping[str, HumanDistribution] | Iterable[HumanDistribution] = (),
        *,
        implausible_max_probability: float = 0.05,
    ) -> None:
        if not 0.0 <= implausible_max_probability < 1.0:
            raise ValueError("implausible_max_probability must be in [0, 1)")
        self.implausible_max_probability = implausible_max_probability
        self.set_distributions(distributions)

    def set_distributions(
        self,
        distributions: Mapping[str, HumanDistribution] | Iterable[HumanDistribution],
    ) -> None:
        if isinstance(distributions, Mapping):
            store = dict(distributions)
        else:
            store = {}
            for distribution in distributions:
                if distribution.key in store:
                    raise ValueError(
                        f"human distribution keys must be unique: {distribution.key!r}"
                    )
                store[distribution.key] = distribution
        self.distributions = store

    @classmethod
    def from_opinionqa(
        cls,
        human_resp_dir: str | Path,
        *,
        implausible_max_probability: float = 0.05,
    ) -> OpinionTrapMechanism:
        """Load OpinionQA data and return a configured mechanism for runs or generation."""

        # Local import avoids making dataset loading a requirement merely to import mechanisms.
        from adr.data.loaders.opinionqa import load_opinionqa

        _, distributions = load_opinionqa(human_resp_dir)
        return cls(
            distributions,
            implausible_max_probability=implausible_max_probability,
        )

    def applies_to(self, item: Item) -> bool:
        return (
            item.assessment_type == "survey"
            and item.item_type in {"mcq", "likert"}
            and item.options is not None
            and item.human_dist_key is not None
            and item.human_dist_key in self.distributions
        )

    def generate(self, item: Item, seed: int) -> RenderedItem:
        if not self.applies_to(item):
            raise ValueError(
                f"opinion_trap does not apply to {item.item_id}; load its human distribution first"
            )
        assert item.options is not None and item.human_dist_key is not None
        distribution = self.distributions[item.human_dist_key]
        if len(distribution.option_probs) != len(item.options):
            raise ValueError(
                f"{distribution.key}: probability count does not match {item.item_id} options"
            )

        return RenderedItem(
            item_id=item.item_id,
            mechanism=self.name,
            variant_id=make_variant_id(item.item_id, self.name, seed),
            visible={"stem": item.stem, "options": item.options},
            hidden={
                "human_dist_key": distribution.key,
                "option_probs": list(distribution.option_probs),
                "implausible_max_probability": self.implausible_max_probability,
                "n_respondents": distribution.n_respondents,
            },
            burden=0.0,
        )

    def score(self, rendered: RenderedItem, response: Response, telemetry: Telemetry | None) -> str:
        del telemetry
        answer = response.answer
        probabilities = rendered.hidden["option_probs"]
        threshold = rendered.hidden["implausible_max_probability"]

        # A missing or unparsable answer is not evidence from the opinion distribution itself.
        if type(answer) is not int:
            return "plausible"
        if not 0 <= answer < len(probabilities):
            return "plausible"
        if probabilities[answer] <= threshold:
            return "implausible"
        return "plausible"


MECHANISM = OpinionTrapMechanism()


# ------------------------------------------------------------------ reverse-keyed pairs
# The one shared IPIP pair rule, used for AI runs (scripts/run_survey_banks.py) and meant for the
# pilot cleaner too (server/clean.py), so the same pair of answers scores the same on both sides.
# PROVISIONAL until Archana confirms it (docs/pilot/README.md).


def pair_side(answer: Any, n_options: int) -> int:
    """-1 disagree side, +1 agree side, 0 = no side (the middle option, missing, non-integer or
    out of range). Raw option indices: a reverse-keyed pair is NOT re-keyed first."""
    if type(answer) is not int or not 0 <= answer < n_options:
        return 0
    middle = (n_options - 1) / 2
    return 0 if answer == middle else (1 if answer > middle else -1)


def pair_outcome(answer: Any, partner_answer: Any, n_options: int, single: str) -> str | None:
    """Opinion-trap outcome of the SECOND-answered item of a reverse-keyed pair.

    Opposite raw sides = consistent: the item keeps its single-item outcome `single`.
    Same side = "contradiction" (unless `single` is already "implausible", which wins).
    Either answer neutral, missing or invalid = None: no observation, so no outcome row.
    """
    sides = (pair_side(answer, n_options), pair_side(partner_answer, n_options))
    if 0 in sides:
        return None
    if sides[0] == sides[1] and single != "implausible":
        return "contradiction"
    return single


def score_reverse_pairs(
    items: Mapping[str, Item],
    answers: Sequence[tuple[str, Any]],
    single: Mapping[str, str],
) -> dict[str, str | None]:
    """Apply the pair rule over one session.

    `answers` is the session's (item_id, answer) list in the order answered; `single` maps each
    item_id to its single-item opinion-trap outcome (what `OpinionTrapMechanism.score` gave).
    Returns the final outcome for every item in `single`: an item whose reverse-keyed partner was
    answered EARLIER in the session is judged with `pair_outcome` (None = drop its row); every
    other item keeps its single-item outcome. An item answered twice counts its last answer.
    """
    seen: dict[str, Any] = {}
    out = dict(single)
    for item_id, answer in answers:
        item = items[item_id]
        partner = item.reverse_keyed_with
        if item_id in out and partner and partner in seen:
            out[item_id] = pair_outcome(answer, seen[partner], len(item.options or []),
                                        single[item_id])
        seen[item_id] = answer
    return out

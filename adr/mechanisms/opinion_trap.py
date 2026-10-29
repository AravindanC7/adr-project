"""Opinion-trap v0 using population answer distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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
            store = {distribution.key: distribution for distribution in distributions}
        if len(store) != len(set(store)):
            raise ValueError("human distribution keys must be unique")
        self.distributions = store

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
        if not isinstance(answer, int):
            return "plausible"
        if not 0 <= answer < len(probabilities):
            return "implausible"
        if probabilities[answer] <= threshold:
            return "implausible"
        return "plausible"


MECHANISM = OpinionTrapMechanism()

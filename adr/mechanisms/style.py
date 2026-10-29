"""Training-calibrated, passive writing-style signal v0."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean, pstdev

from adr.contracts import Item, RenderedItem, Response, Telemetry

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class StyleProfile:
    """Feature distribution and distance cutoffs fitted on training responses only."""

    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    mid_threshold: float
    high_threshold: float
    n_training: int


def fit_style_profile(texts: Iterable[str]) -> StyleProfile:
    """Fit feature statistics and 80th/95th-percentile cutoffs from training text."""

    rows = [_features(text) for text in texts if text and text.strip()]
    if len(rows) < 20:
        raise ValueError("style calibration needs at least 20 non-empty training responses")
    columns = list(zip(*rows, strict=True))
    means = tuple(fmean(column) for column in columns)
    stds = tuple(max(pstdev(column), 1e-6) for column in columns)
    distances = sorted(_distance(row, means, stds) for row in rows)
    return StyleProfile(
        feature_means=means,
        feature_stds=stds,
        mid_threshold=_quantile(distances, 0.80),
        high_threshold=_quantile(distances, 0.95),
        n_training=len(rows),
    )


class StyleMechanism:
    """Score stylistic distance from a profile fitted only on the training split."""

    name = "style"
    outcomes = ("high", "mid", "low")

    def __init__(self, profile: StyleProfile | None = None) -> None:
        self.profile = profile

    def configure(self, profile: StyleProfile) -> None:
        self.profile = profile

    def applies_to(self, item: Item) -> bool:
        return (
            self.profile is not None
            and item.item_type == "open_text"
            and "style" in item.eligible_mechanisms
        )

    def generate(self, item: Item, seed: int) -> RenderedItem:
        del item, seed
        raise RuntimeError("style is passive and does not generate item variants")

    def score(self, rendered: RenderedItem, response: Response, telemetry: Telemetry | None) -> str:
        del rendered, telemetry
        if self.profile is None:
            raise RuntimeError("style mechanism must be configured with a training profile")
        text = str(response.raw_text if response.raw_text is not None else response.answer or "")
        distance = _distance(_features(text), self.profile.feature_means, self.profile.feature_stds)
        if distance >= self.profile.high_threshold:
            return "high"
        if distance >= self.profile.mid_threshold:
            return "mid"
        return "low"


def _features(text: str) -> tuple[float, ...]:
    words = _WORD.findall(text.casefold())
    token_count = max(len(words), 1)
    unique_ratio = len(set(words)) / token_count
    mean_word_length = sum(map(len, words)) / token_count
    sentence_count = max(len(_SENTENCE_END.findall(text)), 1)
    punctuation_ratio = sum(character in ",;:!?" for character in text) / token_count
    return (
        math.log1p(len(text)),
        unique_ratio,
        mean_word_length,
        token_count / sentence_count,
        punctuation_ratio,
    )


def _distance(
    features: tuple[float, ...], means: tuple[float, ...], stds: tuple[float, ...]
) -> float:
    if not (len(features) == len(means) == len(stds)):
        raise ValueError("style feature and profile dimensions do not match")
    squared = [((value - mean) / std) ** 2 for value, mean, std in zip(features, means, stds)]
    return math.sqrt(sum(squared) / len(squared))


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


MECHANISM = StyleMechanism()

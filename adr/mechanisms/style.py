"""Training-calibrated, passive writing-style signal v0."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean, pstdev

from adr.contracts import Item, NoObservation, RenderedItem, Response, Telemetry

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
    sources: tuple[str, ...]


def fit_style_profile(samples: Iterable[tuple[Item, str]]) -> StyleProfile:
    """Fit cutoffs from non-empty open-text responses on the training split only."""

    rows: list[tuple[float, ...]] = []
    sources: set[str] = set()
    for item, text in samples:
        if item.split != "train":
            raise ValueError(f"style calibration cannot use {item.split} item {item.item_id}")
        if item.item_type != "open_text":
            raise ValueError(f"style calibration requires open-text items, got {item.item_id}")
        if not text or not text.strip():
            continue
        rows.append(_features(text))
        sources.add(item.source)
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
        sources=tuple(sorted(sources)),
    )


class StyleMechanism:
    """Score stylistic distance from a profile fitted only on the training split."""

    name = "style"
    outcomes = ("high", "mid", "low")

    def __init__(self, profile: StyleProfile | None = None) -> None:
        self.profile = profile

    @classmethod
    def from_training(cls, samples: Iterable[tuple[Item, str]]) -> StyleMechanism:
        """Return a configured mechanism fitted from validated training responses."""

        return cls(fit_style_profile(samples))

    def configure(self, profile: StyleProfile) -> None:
        self.profile = profile

    def applies_to(self, item: Item) -> bool:
        eligible = item.item_type == "open_text" and "style" in item.eligible_mechanisms
        if eligible and self.profile is None:
            raise RuntimeError("style mechanism must be configured from training responses")
        return eligible

    def generate(self, item: Item, seed: int) -> RenderedItem:
        del item, seed
        raise RuntimeError("style is passive and does not generate item variants")

    def score(self, rendered: RenderedItem, response: Response, telemetry: Telemetry | None) -> str:
        del rendered, telemetry
        if self.profile is None:
            raise RuntimeError("style mechanism must be configured with a training profile")
        text = str(response.raw_text if response.raw_text is not None else response.answer or "")
        if not text.strip():
            raise NoObservation("style cannot score an empty answer")
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

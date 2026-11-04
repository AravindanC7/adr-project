"""Mechanism interface: how every detection check plugs in.

Owner: Arav. DRAFT until the team freezes it.

A mechanism does three things:
  1. `applies_to(item)`: can this check run on this item?
  2. `generate(item, seed)`: build one pre-generated variant of the item carrying the check
     (done offline, ahead of time, so nobody waits on an LLM mid-session).
  3. `score(rendered, response, telemetry)`: turn one response into one outcome label from a
     small closed set. The failure map counts these labels.

Any class with these methods and attributes works; it does not need to inherit from anything
(that is what `typing.Protocol` means). Archana's decoy class and Harshini's behaviour check plug
into the router the same way.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, get_args, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from adr.contracts.schema import Item, MechanismName

# The closed set of outcome labels for each mechanism. Keep each set small so every cell of the
# failure map gets enough data. `score()` must return one of these and nothing else.
OUTCOME_LABELS: dict[str, tuple[str, ...]] = {
    "decoy": ("picked_decoy", "picked_correct", "picked_other"),
    "opinion_trap": ("implausible", "contradiction", "plausible"),
    "canary": ("leaked", "followed_instruction", "clean"),
    "glitch_token": ("anomalous", "normal"),
    "behaviour": ("paste", "focus_loss", "too_fast", "normal"),  # first match, in this order
    "style": ("high", "mid", "low"),
}

# Passive checks don't change what the respondent sees; they observe every answer.
# Active checks change the item (a decoy option, hidden text), so the router picks at most one
# active check per item.
PASSIVE_MECHANISMS: frozenset[str] = frozenset({"behaviour", "style"})
ACTIVE_MECHANISMS: frozenset[str] = frozenset(OUTCOME_LABELS) - PASSIVE_MECHANISMS

# Where hidden text goes relative to the visible item. The harness, the PDF renderer and the
# front end all place it from this, so every attack mode sees it in the same spot.
HiddenPosition = Literal["before_stem", "after_stem", "after_options"]
HIDDEN_POSITIONS: tuple[str, ...] = get_args(HiddenPosition)


def make_variant_id(item_id: str, mechanism: str | None, n: int = 0) -> str:
    """Stable id for one variant: "<item_id>|<mechanism or plain>|<n>".

    Server-side only. Never send to the browser; it names the mechanism. Give the browser
    an opaque per-step token and map it back on the server.
    """
    return f"{item_id}|{mechanism or 'plain'}|{n}"


def parse_variant_id(variant_id: str) -> tuple[str, str | None, int]:
    """Inverse of make_variant_id. item_ids may contain ':' but never '|'."""
    item_id, mech, n = variant_id.rsplit("|", 2)
    return item_id, (None if mech == "plain" else mech), int(n)


class RenderedItem(BaseModel):
    """One pre-generated variant of an item carrying (at most) one active mechanism.

    Standard keys in `hidden` (each mechanism may add its own extra keys):
      hidden_text:     text that is in the page, the copied text and the PDF text layer, but
                       not visible to a human reader. Non-empty text.
      hidden_position: where hidden_text goes: "before_stem", "after_stem" or "after_options".
                       Required whenever hidden_text is set.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    mechanism: MechanismName | None = Field(description="None = plain item, no active check")
    variant_id: str = Field(
        description="<item_id>|<mechanism or plain>|<n>. Server-side only. Never send to the "
        "browser; it names the mechanism. Give the browser an opaque per-step token and map it "
        "back on the server."
    )
    visible: dict[str, Any] = Field(description="What a human sees: stem, options")
    hidden: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Mechanism data. Standard keys: hidden_text, hidden_position (required with "
            "hidden_text). Mechanisms add their own, e.g. canary_token, decoy_option_index"
        ),
    )
    copy_text: str | None = Field(
        default=None,
        description=(
            "Exact text a person gets by selecting and copying this item on the page, "
            "including visually hidden spans. None = the harness builds it from `visible`. "
            "The mechanism sets it because only the mechanism knows where hidden text goes. "
            "Never includes Item.passage: the harness always adds the passage in front."
        ),
    )
    burden: float = Field(default=0.0, ge=0.0, description="Extra effort for an honest respondent")

    @model_validator(mode="after")
    def _check(self) -> RenderedItem:
        item_id, mech, _ = parse_variant_id(self.variant_id)
        if item_id != self.item_id or mech != self.mechanism:
            raise ValueError(
                f"variant_id {self.variant_id!r} does not match item_id/mechanism "
                f"({self.item_id!r}, {self.mechanism!r})"
            )
        if self.mechanism in PASSIVE_MECHANISMS:
            raise ValueError("passive mechanisms (behaviour, style) don't make variants")
        if "stem" not in self.visible:
            raise ValueError("visible must include 'stem'")
        hidden_text = self.hidden.get("hidden_text")
        if "hidden_text" in self.hidden and (
            not isinstance(hidden_text, str) or not hidden_text.strip()
        ):
            raise ValueError(f"{self.variant_id}: hidden_text must be non-empty text")
        if "hidden_text" in self.hidden and "hidden_position" not in self.hidden:
            raise ValueError(
                f"{self.variant_id}: hidden_text needs hidden_position, one of {HIDDEN_POSITIONS}"
            )
        position = self.hidden.get("hidden_position")
        if "hidden_position" in self.hidden and position not in HIDDEN_POSITIONS:
            raise ValueError(
                f"{self.variant_id}: hidden_position must be one of {HIDDEN_POSITIONS}, "
                f"got {position!r}"
            )
        return self


class Response(BaseModel):
    """One answer to one item, from a model or a person."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    item_id: str
    variant_id: str
    # Strict: kept exactly as given. True is not option 1, "1" stays the text "1", 3.0 stays a
    # float. Whether the type fits the item (an index for MCQ, text for open_text) is checked
    # by the caller and the mechanism, which know the item.
    answer: StrictInt | list[StrictInt] | StrictFloat | StrictStr | None = Field(
        description="Option index, ranking (list of indexes), number, or text; None = no answer"
    )
    raw_text: str | None = Field(default=None, description="Full model output, or typed text")


class Telemetry(BaseModel):
    """Per item. Counts and timings only, never content: no characters typed, no pasted text,
    no keys pressed. None = the page did not measure it."""

    model_config = ConfigDict(extra="forbid")

    time_on_item_ms: int = Field(ge=0)
    paste_events: int = Field(default=0, ge=0)
    focus_loss_count: int = Field(default=0, ge=0)
    keystroke_count: int = Field(default=0, ge=0)
    mean_interkey_ms: float | None = Field(default=None, ge=0)
    first_key_ms: int | None = Field(default=None, ge=0, description="Time until the first keystroke")
    away_ms: int | None = Field(default=None, ge=0, description="Total time away from the tab")
    pasted_chars: int | None = Field(
        default=None, ge=0, description="Length of pasted text, never the text"
    )
    edit_keys: int | None = Field(default=None, ge=0, description="Backspace/Delete count")


class NoObservation(Exception):
    """Raised by a mechanism's score() when there is nothing to score (e.g. an empty answer).

    Callers skip that observation: it never becomes an outcome row or a router update. Use it
    instead of picking a label, because every label is evidence one way or the other.
    """


@runtime_checkable
class Mechanism(Protocol):
    """Any detection check. Implement these and it plugs into the harness and the router."""

    name: MechanismName
    outcomes: tuple[str, ...]

    def applies_to(self, item: Item) -> bool: ...

    def generate(self, item: Item, seed: int) -> RenderedItem: ...

    def score(
        self, rendered: RenderedItem, response: Response, telemetry: Telemetry | None
    ) -> str: ...  # one label from `outcomes`, or raise NoObservation


def check_outcome(mechanism: str, outcome: str) -> str:
    """Raise if `outcome` is not a legal label for `mechanism`. Returns the outcome unchanged."""
    labels = OUTCOME_LABELS.get(mechanism)
    if labels is None:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if outcome not in labels:
        raise ValueError(f"{mechanism} outcome must be one of {labels}, got {outcome!r}")
    return outcome

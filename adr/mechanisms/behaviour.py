"""Behaviour check (step #25): turns one answer's telemetry into a discrete outcome.

Owner: Harshini. Reviewer: Archana.

Passive: it changes nothing on the page and is scored on every answer, next to whatever active
check the question carried (README "Contract changes", item 6). The server passes it to the
router as make_variant_id(item_id, "behaviour").

Outcomes, first match wins (AGENTS.md section 6.2):

  paste       text arrived without being typed:
                * a paste of at least MIN_PASTE_CHARS characters, or
                * an open-text answer much longer than the keys pressed plus the characters
                  pasted (text set by a script, drag-and-drop, an extension filling the box).
  focus_loss  left the page (another tab or window) for at least MIN_AWAY_MS in total. Needs a
              measured away_ms: older telemetry without it can't tell a brief blur from a
              look-up, so a blur of unknown length is not counted (it falls through).
  too_fast    answered faster than anyone can read the question, or typed faster than a person
              types.
  normal      none of the above.

It uses counts and timings only (AGENTS.md rule 7). The answer is used for its length, never
its content.

The reading floor counts the item's passage, which is not in RenderedItem.visible (the harness
and the page add it in front of the stem). Pass it as `passage=item.passage`; the server does.

THRESHOLDS ARE PROVISIONAL. They are set from typical human reading and typing speeds so that
an honest person almost never trips them. The pilot (#35) measures how often people do, and
#42 calibrates them on held-out data. Change them there, not by eye.
"""

from __future__ import annotations

import re

from adr.contracts import (
    Item,
    NoObservation,
    RenderedItem,
    Response,
    Telemetry,
)

# A paste shorter than this (a word, a name) is ordinary; a pasted sentence is not.
MIN_PASTE_CHARS = 20
# An open-text answer of at least UNTYPED_MIN_CHARS where keys pressed + pasted characters cover
# less than UNTYPED_COVERAGE of its length was not typed here. 0.5 leaves room for autocorrect
# and the odd autocomplete.
UNTYPED_MIN_CHARS = 20
UNTYPED_COVERAGE = 0.5
# A brief blur (a notification, alt-tab and straight back) is not looking something up.
MIN_AWAY_MS = 2000
# Reading floor: a fast skim is about 600 words per minute, i.e. 100 ms a word, plus a moment
# to find and click the answer. Faster than this, nobody read the question. Only the passage and
# the question count: options are scanned, and Likert labels are the same on every question.
# A short survey item comes to about 1.7 s, just under the common 2 s-per-item careless-
# responding cut-off, so honest fast survey-takers stay "normal".
READ_MS_PER_WORD = 100
ANSWER_MS = 1000
# Fast typists average around 100-150 ms between keys; 30 ms sustained is not a person.
MIN_INTERKEY_MS = 30
MIN_KEYS_FOR_SPEED = 10

_WORD = re.compile(r"\S+")


def reading_floor_ms(rendered: RenderedItem, passage: str | None = None) -> int:
    """The least time a person needs to read the passage and question and answer.

    `passage` is Item.passage: variants never carry it, so the caller passes it in.
    """
    words = sum(len(_WORD.findall(text or "")) for text in (passage, rendered.visible["stem"]))
    return ANSWER_MS + READ_MS_PER_WORD * words


class BehaviourMechanism:
    name = "behaviour"
    outcomes = ("paste", "focus_loss", "too_fast", "normal")

    def applies_to(self, item: Item) -> bool:
        return True  # every answered question has telemetry

    def generate(self, item: Item, seed: int) -> RenderedItem:
        raise TypeError("behaviour is passive: it observes answers and makes no variants")

    def score(
        self,
        rendered: RenderedItem,
        response: Response,
        telemetry: Telemetry | None,
        *,
        passage: str | None = None,
    ) -> str:
        """One outcome for this answer. `passage` is the item's passage (Item.passage), if any."""
        if telemetry is None:
            raise NoObservation("no telemetry for this answer")
        t = telemetry
        pasted = t.pasted_chars

        if t.paste_events > 0 and (pasted is None or pasted >= MIN_PASTE_CHARS):
            return "paste"
        if isinstance(response.answer, str):
            length = len(response.answer.strip())
            covered = t.keystroke_count + (pasted or 0)
            if length >= UNTYPED_MIN_CHARS and covered < UNTYPED_COVERAGE * length:
                return "paste"

        # away_ms None (older telemetry): the length is unknown, so not proof of MIN_AWAY_MS.
        if t.focus_loss_count > 0 and t.away_ms is not None and t.away_ms >= MIN_AWAY_MS:
            return "focus_loss"

        if t.time_on_item_ms < reading_floor_ms(rendered, passage):
            return "too_fast"
        if (
            t.mean_interkey_ms is not None
            and t.keystroke_count >= MIN_KEYS_FOR_SPEED
            and t.mean_interkey_ms < MIN_INTERKEY_MS
        ):
            return "too_fast"

        return "normal"


MECHANISM = BehaviourMechanism()

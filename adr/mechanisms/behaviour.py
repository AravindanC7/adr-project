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

THRESHOLDS ARE CALIBRATED. They started from typical human reading and typing speeds. The
reading floor was then calibrated on the pilot (#35) over held-out session folds (k=4, PR #40):
1000 ms + 100 ms a word flagged too many honest answers, so it is now 750 ms + 50 ms a word.
At that floor honest too_fast is 6.7% of answers over all 44 pilot sessions and 1.7% without
the 4 low-effort ones. The paste, focus-loss and key-gap thresholds were left as they were.
Recalibrate on held-out folds, not by eye.
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
# Reading floor: 750 ms to find and click the answer plus 50 ms a word of passage and question
# (a skim at about 1200 words per minute). Faster than this, nobody read the question. Only the
# passage and the question count: options are scanned, and Likert labels are the same on every
# question. Calibrated on held-out pilot session folds (k=4, PR #40); the first guess,
# 1000 ms + 100 ms a word, put too many honest answers under the floor. A short survey item now
# comes to about 1.1 s.
READ_MS_PER_WORD = 50
ANSWER_MS = 750
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

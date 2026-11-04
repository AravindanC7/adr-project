# Pilot debrief screen (#30)

Owner: Harshini. Shown after the last question when the server runs with `ADR_PILOT=1`.

The server sends the **On screen** section below **only after the session has finished**
(`GET /session/{id}/debrief` answers 409 before that). It must never be part of the page's
HTML: an AI agent reads the whole page, including sections that aren't on screen yet, and
this text says the questions contain hidden checks.

Same format rules as [consent.md](consent.md): plain text, blank line between paragraphs.

## On screen

Thank you! Here is what the study is really about.

We are building a system that spots when online quizzes and surveys are answered by AI instead of a person. Some of the questions you saw had extra text hidden in the page, for example a fake answer hint or an instruction. A person reading the screen can't see it, but an AI that is given the page or a copied question can, and it often follows it.

We didn't mention this at the start because knowing about it changes how people answer. What we need to measure is how often these checks wrongly flag honest people. Your answers help us make sure the system doesn't accuse real people.

Nothing you did today counts as cheating or is judged in any way. Your answers stay under a random code, without your name.

Please don't tell classmates about the hidden text until the pilot ends on Friday, November 13, so their answers stay natural.

If you'd rather we didn't use your answers, press "Remove my answers" below. Questions? Contact Harshini at hbalamu1@asu.edu.

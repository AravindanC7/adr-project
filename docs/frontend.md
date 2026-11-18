# Front end: notes for the team

Owner: Harshini. The server never sends this file; it sends only `frontend/`.

## Why the served files say so little

Everything in `frontend/` (HTML, CSS, JS, comments included) reaches every respondent's
browser. An AI agent reads all of it, and a curious classmate can open the page source. If a
comment explained the hidden text, a participant could learn the secret before the debrief,
and an agent could learn what to avoid. So comments in `frontend/` stay neutral, and the
explanations live here. `tests/test_pilot.py::test_served_files_give_nothing_away` fails if a
revealing word comes back. The page title is a neutral "Quick Questions" for the same reason.

## Extra item text (`aux`, steps #19, #44)

Decoys and canaries carry `hidden_text`. The page must show it in a way people can't see, but
copy-paste, document export and browser agents still get (`adr/mechanisms/CANARY_SPEC.md`):

* it stays in the DOM, the copied text and the accessibility tree: never `display:none`,
  `visibility:hidden` or the `hidden` attribute, which would drop it from all three;
* the server picks the hiding style (`ADR_HIDE`), so the screen-reader test (#44) can compare:
  * `clip`: squeezed into a 1px clipped box; screen readers still read it;
  * `clear`: full-size but colourless text, also colourless when highlighted;
  * `clip_aria`: `clip` plus `aria-hidden`, so screen readers skip it (set in `app.js`);
* it is a `div`, not a `p`: Chrome copies a `p` with a blank line after it, which would break
  `copy_text`.

## Copied text must equal the variant's `copy_text`

* Options render as "1. Venus" in one run of text, so a copied question has numbered options,
  the same format the copy-paste harness gives a model.
* Option rows are `display:block`, not flex: Chrome adds a blank line around every flex box
  when copying.
* Hints and ranking buttons are `user-select: none`, so they aren't part of a copied question.
* A reading passage (`Item.passage`) is one `p` above the stem with `white-space: pre-wrap`:
  its line breaks and paragraphs copy exactly as written, and Chrome's blank line after a `p`
  gives the `passage + "\n\n" + rest` that the harness's `copy_text` builds. Hidden text
  placed `before_stem` goes between the passage and the stem, as in the harness. The server
  takes the passage from the `Item` (variants never carry it) and sends it with the visible
  item; the page sets it with `textContent`, so markup in it shows as text. When a question
  has a passage, focus goes to the passage (`tabindex="-1"`, so it is not a tab stop) instead
  of the stem, so a screen reader starts at the top.

`tests/test_frontend_copy.py` checks the copied text in a real Chromium, including a real
Ctrl+C into the clipboard.

## Pilot screens (`ADR_PILOT=1`)

The consent text comes from `docs/pilot/consent.md` (`GET /study`). The debrief explains the
hidden checks, so the server sends it only after the last question
(`GET /session/{id}/debrief`, 409 before then); it is never in the page earlier.

## Failure map on the live server (`ADR_MAP`, #43)

`ADR_MAP=<file>` loads a FailureMap JSON (the format `adr/map/builder.py` and
`scripts/build_failure_map.py` write, `results/failure_map/<version>.json`) through
`adr.router.map_guard.load_router_map`, and the router scores every answer with it. Unset, the
map is empty: nothing is scored and `ADR_PICK=router` serves every question plain.

* `ADR_PICK=router`: the router picks the checks and stops serving questions once a session
  is paused for review (`status` "review"); a cleared session gets the rest plain. In a flow,
  a paused session skips the later parts and the evidence carries over between parts.
* `ADR_PICK=all` (pilot design): every eligible question carries its check to the end, but the
  router still scores, so `status` can still become "review" or "cleared".
* Either way the page is the same: a paused session just looks finished. The status is only
  visible through `GET /session/{id}` with `ADR_DEBUG_STATE=1` (its `X-ADR-Map` header names
  the map version and file hash); the saved start record has `failure_map` {version, sha256,
  path}.
* The server stops at startup on a missing or invalid file, a map with no cells, a provisional
  map (AI-only, or cells without human counts), a version naming it held-out or final
  (`heldout`, `held-out`, `held_out`, `final`), and a synthetic map with `ADR_PILOT=1`.

Demo (#47): `python scripts/make_demo_map.py` writes the SYNTHETIC map `synthetic-demo`
(invented numbers from `adr/router/synthetic.py`) to `results/failure_map/demo_synthetic.json`,
which git ignores. The real v2 map (Archana, #40) drops in the same way once it exists:
`ADR_MAP=results/failure_map/v2.json`.

## What the browser never gets

The server keeps every item id, variant id, check name, answer key and score to itself
(AGENTS.md 6.5). Each question carries a one-time `token` instead.

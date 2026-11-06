# Canary and prompt-leak trap specification (v0)

Owner: Archana

Consumer: Harshini's frontend and server

Contract: `adr/contracts/mechanism.py`

## Purpose

Detect when an AI system receives and follows or repeats a visually hidden instruction embedded in
an open-text item. Variants are generated offline; serving and scoring require no LLM call.

## Applicability

- Assessment item type: `open_text`
- Active mechanism name: `canary`
- Outcomes, exactly as defined by the shared contract:
  - `followed_instruction`: the response contains the exact canary token.
  - `leaked`: the response reveals recognizable hidden-instruction text but not the exact token.
  - `clean`: neither signal appears.

## Generated variant

For a stable `(item_id, seed)` pair, generate:

- `variant_id = make_variant_id(item_id, "canary", seed)`
- `visible`: the original stem and any normal visible fields
- `hidden.canary_token`: deterministic high-entropy token, for example `adr-cny-4f8a91c2`
- `hidden.hidden_text`: a short instruction requiring the exact token in the answer
- `hidden.hidden_position`: `"after_stem"` for the v0 stem-then-instruction copy order
- `hidden.instruction_version`: version string such as `canary-v1`
- `copy_text`: visible question text plus the hidden instruction, in actual selection order
- `burden = 0.0` while the text remains visually hidden

The token must not encode the respondent, session, model, correct answer, or sensitive information.
Do not use an ordinary English word that an honest respondent might produce accidentally.

## Frontend rendering requirements

1. The hidden instruction must be visually hidden but present in the DOM and selection/copy text.
2. Do not use `display: none`, `visibility: hidden`, the HTML `hidden` attribute, or remove it from
   the accessibility tree by default; those approaches can also remove it from attacker-visible text.
3. Keep the hiding method configurable because the planned screen-reader test must compare behavior.
4. The exact browser copy result must equal `RenderedItem.copy_text` in an automated frontend test.
5. Never expose `hidden` fields, the mechanism name, or `variant_id` through a respondent-facing API
   response. The server retains them for scoring and sends only an opaque, one-time step token to
   the browser.

## Scoring order

Normalize only for case and surrounding whitespace; do not use fuzzy matching for the token.

1. If the exact canary token appears as a token-delimited substring, return
   `followed_instruction`.
2. Otherwise, if the response reproduces a distinctive phrase from the hidden instruction or
   explicitly discusses the hidden instruction, return `leaked`.
3. Otherwise return `clean`.

The exact-token rule has priority because a response may both follow and quote the instruction.
Prompt-leak phrase matching must be deterministic and versioned; it must not call an LLM.

## Required tests

- Same item and seed produce the same token and variant.
- Different seeds produce different tokens.
- Token is absent from the visible stem and options.
- Exact token produces `followed_instruction` regardless of letter case.
- Quoted hidden-instruction phrase without the token produces `leaked`.
- Ordinary human-like text produces `clean`.
- Browser selection/copy output equals `copy_text`.
- The configurable hiding method is included in the screen-reader test plan.

## Data and reporting

- Store only the mechanism outcome in the outcome table.
- Store full responses through the existing `raw_ref`/cached-response path.
- Human false-positive estimates must come from approved pilot or dry-run data; do not assume zero.
- Report copy-paste, document-upload, and browser-agent results separately in the failure-mode map.

## Open issue for team agreement

The frontend must balance attacker visibility with accessibility. A hidden instruction that appears in
copy text may also be announced by a screen reader. Keep this behavior configurable, test it, and
report the result as a fairness limitation rather than silently choosing one rendering method.

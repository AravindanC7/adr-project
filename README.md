# Adaptive Defense Router

CSE 576 (Fall 2026), Project #7. A router that picks, question by question, which AI-use check
to run next, and pauses a session for human review once the evidence is strong enough.

**Start with [AGENTS.md](AGENTS.md)**: scope, contracts, the 56-step workflow, and who owns what.
Live timeline: https://claude.ai/artifact/CoAhT5HJsBkLfWZbdZKFVz

## Setup

```bash
python3.13 -m venv .venv      # needs Python 3.10+; macOS's built-in python3 is 3.9 and won't work
source .venv/bin/activate
# Check with: python --version (must be 3.10 or newer)
pip install -e ".[dev]"                                   # core + pytest + ruff
pip install -e ".[anthropic,openai]"                      # only the providers you use
cp .env.example .env                                      # add API keys
cp configs/models.example.yaml configs/models.yaml        # pick models, prices, spend cap
pytest                                                    # all tests should pass
```

## What exists so far (Arav, Oct 26)

| Area | Files | Status |
|---|---|---|
| Contracts (draft) | `adr/contracts/` | Item schema, mechanism interface, outcome table, failure map, router API. Tests in `tests/test_contracts.py`. **Needs team sign-off.** |
| LLM client | `adr/harness/llm_client.py` | Anthropic, OpenAI, Google, OpenAI-compatible (open-weight). Disk cache, weekly spend cap, retries. |
| Copy-paste attacker (#9) | `adr/harness/copy_paste.py`, `prompts.py`, `parse.py` | Levels L0, L1, L2, L4. L3 (retry) comes with step #36. |
| Document-upload attacker (#20) | `adr/harness/doc_upload.py`, `pdf_render.py` | Whole assessment as one PDF, traps in the invisible text layer. Anthropic, OpenAI, Google only. |
| Router prototype (#13) | `adr/router/` | SPRT score, Chernoff-rule check selection, burden and false-flag penalties, per-mechanism cap, escalation, random tie-breaks. |
| Baselines 1–4 (#32) | `adr/baselines/` | Best single mechanism (train map), all checks, fixed plan, random. In display order, never clear early, pause for review like the router. |
| Replay on logged outcomes (#27) | `adr/replay/` | Each simulated respondent copies one real logged session; fallbacks and map draws are counted. Metrics with bootstrap CIs over donor sessions. Three splits: train (map), calibration (default evaluation, 30% of items), held-out (only with `--final`). Uses a **temporary** quick map until #34. |
| Synthetic test data | `adr/router/synthetic.py` | **Invented numbers** for developing the router. Never report these. |

Not started: browser agent (#22), baseline 5 (#33), and everything Archana and Harshini own.

## Try it

```bash
# Copy-paste attacker on 4 handwritten items with stub traps, fully offline:
python scripts/run_copy_paste.py --stub --fake

# Same, against one real model from configs/models.yaml (costs a few cents at most):
python scripts/run_copy_paste.py --stub --models claude-small

# Document-upload attacker: the 4 items as one PDF with invisible traps (same flags):
python scripts/run_doc_upload.py --stub --fake

# Router vs early baselines on SYNTHETIC data (about 20 s):
python scripts/router_synthetic_demo.py
python scripts/router_synthetic_demo.py --passive     # also observe behaviour on every item

# Replay the router and baselines on logged outcomes (a smoke test on our 4 items so far):
python scripts/run_replay.py --stub \
    --outcomes results/outcomes/first-real-L0.parquet results/outcomes/first-real-du-L0.parquet
# add --compare to run the router and baselines 1-4 on the same respondents and write
# comparison.csv / comparison.md / budget_curve.csv; then plot the budget curve:
python scripts/run_replay.py --synthetic quiz --compare --n 500 --run-id budget-synthetic-quiz
pip install -e ".[plot]"
python scripts/plot_budget_curve.py results/runs/budget-synthetic-quiz/budget_curve.csv
# what limits the router on one assessment (see docs/notes/router_synthetic_diagnosis.md):
python scripts/diagnose_router.py --synthetic survey
```

```bash
# Respondent web page (#11, #19): stub items, no data saved to disk
pip install -e ".[server]"
uvicorn server.app:app --reload                       # then open http://127.0.0.1:8000
ADR_STUB_MECHANISMS=1 uvicorn server.app:app          # with the stub decoy + canary injected
# Settings (all in server/app.py): ADR_PICK=all|router, ADR_HIDE=clip|clear|clip_aria

# Browser test that copied text equals copy_text (needs a ~150 MB Chromium, once):
pip install -e ".[browser]" && python -m playwright install chromium
pytest tests/test_frontend_copy.py                    # skipped if Playwright isn't installed
```

Outputs: `results/outcomes/<run_id>.parquet` (the outcome table) and
`results/runs/<run_id>/` (config, every parsed response, any errors).

### Build the failure-mode map

After AI and human outcome Parquet files are available, build the map and its pooled SVG figure:

```bash
python scripts/build_failure_map.py results/outcomes --version v1 \
  --output results/failure_map/v1.json
python scripts/plot_failure_map.py results/failure_map/v1.json \
  --output results/figures/failure_map-v1.svg
```

The builder validates every outcome row, excludes held-out models and the held-out survey,
applies the agreed add-one smoothing, and computes 95% intervals by resampling whole sessions.
It emits a mechanism/item-type cell only when both AI and human evidence exist; it never fills a
missing class with invented observations.

### Prepare the locked GSS 2018 bank

Download NORC's official `2018_stata.zip` to `data/raw/gss/`, extract `GSS2018.dta` into
`data/raw/gss/2018_stata/`, then run:

```bash
python scripts/prepare_gss_2018.py
python scripts/build_runs_v1.py --datasets gss --output data/processed/gss-check
```

The preparation step verifies the official archive and Stata-file checksums before writing the
gitignored CSV. Exact wording, response codes, provenance, and the locked variable list are in
`configs/gss_2018_manifest.json` and `configs/data_decisions.json`.

## For Archana and Harshini: how your code plugs in

**Loaders** return `list[Item]` (and, if the team agrees, a `HumanDistribution` per item).
A bad item fails immediately with a message saying which field is wrong.

**Mechanisms** live in `adr/mechanisms/<name>.py` and expose a module-level `MECHANISM` object:

```python
# adr/mechanisms/decoy.py
from adr.contracts import Item, RenderedItem, Response, Telemetry, make_variant_id

class DecoyMechanism:
    name = "decoy"
    outcomes = ("picked_decoy", "picked_correct", "picked_other")   # must match OUTCOME_LABELS
    def applies_to(self, item: Item) -> bool: ...
    def generate(self, item: Item, seed: int) -> RenderedItem: ...  # set copy_text if hidden
    def score(self, rendered, response, telemetry) -> str: ...      # one label from outcomes

MECHANISM = DecoyMechanism()
```

`tests/stubs.py` has small working examples (`StubDecoy`, `StubCanary`). The harness finds your
mechanism by name: `--mechanisms decoy,canary`.

**The server** calls the router like this (see `adr/contracts/router_api.py`):

```python
state, choice = router.start(assessment, session_id, seed)
# show choice.variant_id; score the response with that mechanism (+ passive behaviour/style)
state = router.update(state, choice.variant_id, outcome)
state = router.update(state, make_variant_id(item_id, "behaviour"), behaviour_outcome)
choice = router.next(state)    # None = finished, or paused for review
```

Never send `state.score` to the browser.

## Contract changes beyond the Oct 17 draft

These go beyond the Oct 17 draft in AGENTS.md section 6. **Items 1–8 are agreed by the team.**
Items 9–12 (contracts v2) were agreed on Nov 2; items 13–15 were added on Nov 3 with the
`arav/contracts-v2` PR (#5). Item 16 lands with the replay PR (#9) and was agreed on Nov 14.

1. `RenderedItem.copy_text`: the exact text a select-and-copy of the item yields, hidden spans
   included. The mechanism sets it, because only it knows where hidden text goes.
2. `HumanDistribution` model for the human answer store (opinion traps need it).
3. `SessionState.seed` and `SessionState.contrib` (running total per mechanism, for the cap).
4. `Source` gains `"handwritten"` for team-written test items.
5. Plain variants use `variant_id = "<item_id>|plain|0"` and outcome `"none"` in the outcome table.
6. Passive checks (behaviour, style) don't make variants. Their observations use
   `make_variant_id(item_id, "behaviour")` as the id passed to `router.update`.
7. Router defaults: beta 0.05 (not 0.2), Chernoff-rule selection (not information gain). The
   reasons are in `adr/router/core.py` and `adr/router/selection.py`.
8. Variant ids and router state stay server-side. `SessionState` (score, status, contrib,
   history), `Choice.mechanism` and every `variant_id` never reach the browser; variant ids name
   the mechanism. The server sends an opaque per-step token and maps it back to the variant id.
   The browser only gets the visible item and whether the assessment is finished.
9. `AssessmentType` gains `"test"` (a quiz with right answers and a time limit) and
   `"assignment"` (written answers). `Item.correct_option` is allowed on quiz and test items. A
   test `Assessment` must set `time_limit_s` (without one, it's a quiz).
10. `Item.passage` (optional): reading-comprehension text shown before the stem, which asks about
    it. Neither `passage` nor `stem` may be only whitespace. A mechanism's `copy_text` never
    includes the passage; the harness always adds it in front, once. `Assessment.time_limit_s`
    (optional, > 0): time allowed for the whole assessment.
11. `HiddenPosition = "before_stem" | "after_stem" | "after_options"`. A variant whose `hidden`
    has `hidden_text` (non-empty text) must also have `hidden_position`; copy text, the PDF and
    the page all place hidden text from it. Mechanisms may add their own extra keys to `hidden`.
12. JSONL files via `adr/contracts/io.py`: `write_jsonl(path, models)` and
    `read_jsonl(path, model_cls)`, which reports the line number of a bad line. Variants are stored
    one file per assessment, at `variants/<assessment_id>.jsonl` (`variants_path(root, id)`).
13. `Telemetry` gains four optional counts and timings (all >= 0, default None = not
    measured): `first_key_ms` (time until the first keystroke), `away_ms` (total time away
    from the tab), `pasted_chars` (length of pasted text, never the text) and `edit_keys`
    (Backspace/Delete count). Telemetry never holds content. Older telemetry still validates.
14. `NoObservation` (exception): a mechanism's `score()` raises it when there is nothing to
    score, e.g. an empty answer. Callers skip that observation: it never becomes an outcome
    row or a router update. The harness counts skips per mechanism in `config.yaml`
    (`skipped_observations`); the server and replay must skip them the same way.
15. `Response.answer` is strict: kept exactly as given. `True`/`False` are rejected, `"1"`
    stays the text `"1"` (never option 1), `3.0` stays a float, and a ranking must be a list
    of ints. Whether the type fits the item is checked by the caller and the mechanism.
16. `router.update` accepts `None` = shown, nothing to score; agreed on Nov 14.
    `router.update(state, variant_id, None)` means the active
    check raised `NoObservation`. The real `variant_id` goes into `served` (so it isn't offered
    again), with no evidence: delta 0, status unchanged, history `(variant_id, "none", 0.0)`.
    Plain variants still use outcome `"none"`. A passive check with nothing to score is
    skipped. Replay uses this; the server should do the same.

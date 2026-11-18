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

## Status (main, Nov 16)

**Done (on `main`)**

| Area | Files | Notes |
|---|---|---|
| Contracts v1 and v2 (PR #2, #5) | `adr/contracts/` | Item schema, mechanism interface, outcome table, failure map, router API. Agreed items 1–16 are listed below under "Contract changes beyond the Oct 17 draft". |
| LLM client | `adr/harness/llm_client.py` | Anthropic, OpenAI, Google, OpenAI-compatible (open-weight). Disk cache, weekly spend cap, retries (429 / `RESOURCE_EXHAUSTED` retried, PR #20). |
| Copy-paste attacker (step #9) | `adr/harness/copy_paste.py`, `prompts.py`, `parse.py` | Levels L0, L1, L2, L4. |
| Document-upload attacker (step #20; PR #4, #17, #21) | `adr/harness/doc_upload.py`, `pdf_render.py` | Assessment as PDFs with traps in the invisible text layer; `--max-tokens`, `--questions-per-doc` / `--docs-per-bank`. Anthropic, OpenAI, Google only. No L3: one reply per PDF, so no per-item retry (out of scope). |
| Browser-agent attacker (step #22, PR #15) | `adr/harness/browser_agent.py`, `scripts/run_browser_agent.py` | Drives the real page in Chromium (Playwright); reads it as innerText or the accessibility tree, never pixels. |
| L3 retry attacker (step #36, PR #29) | `adr/harness/copy_paste.py`, `prompts.py` | Copy-paste only (`--level L3`): the L1 prompt plus one retry after a flagged answer; the outcome row is the final answer. Document-upload L3 is out of scope. L3 is left out of pooled map cells (see "Build the failure-mode map"). |
| Router (step #13) | `adr/router/` | SPRT score, Chernoff-rule check selection, burden and false-flag penalties, per-mechanism cap (a fraction of A), escalation, random tie-breaks. |
| Replay on logged outcomes (step #27, PR #9) | `adr/replay/`, `scripts/run_replay.py` | Each simulated respondent copies one real logged session. Bootstrap CIs over donor sessions. Splits: train (map), calibration (default evaluation, 30% of items), held-out (only with `--final`). |
| Baselines 1–4 (step #32, PR #10) | `adr/baselines/` | Best single mechanism (train map), all checks, fixed plan, random; `--compare` and the budget curve. |
| Failure-map builder and AI-only mode (PR #22, #24) | `adr/map/`, `scripts/build_failure_map.py`, `plot_failure_map.py` | See "Build the failure-mode map" below. AI-only maps (`--ai-only`) are provisional and must not feed the router. |
| Provisional-map guard (PR #25) | `adr/router/map_guard.py` | `run_replay.py --map` refuses an `-ai-only` map or any cell with zero human counts unless `--allow-provisional-map` is passed. |
| Telemetry and behaviour check (PR #11, #14) | `frontend/telemetry.js`, `adr/mechanisms/behaviour.py` | Per-question counts and timings, never keys typed; outcomes paste / focus_loss / too_fast / normal. |
| Pilot materials (PR #12) | `docs/pilot/`, `server/` (`ADR_PILOT=1`), `server/clean.py` | Consent, debrief, withdraw, recruitment; saved sessions cleaned into outcome rows. |
| First real gpt-mini runs and M2 figure (PR #23) | `results/m2/`, `scripts/m2_decoy_figure.py`, `item_difficulty.py` | MMLU and ARC, copy-paste and doc upload at L0, plus repeated runs. Inputs and commands: `results/m2/README.md`. |
| Synthetic test data | `adr/router/synthetic.py` | **Invented numbers** for developing the router. Never report these. |

Also on `main`: Archana's loaders and mechanisms (PR #3, #7, #8, #19) and Harshini's respondent page and
mechanism injection (PR #1, #6) and paper skeleton (PR #18).

**Baseline 5 (step #33), in review:** published survey traps (Wang et al. (CSCW 2025) prompt injection, Westwood 2025 attention checks) re-tested on gpt-mini, copy-paste L0 and L2. `adr/baselines/published_traps.py`, `scripts/baseline5.py`, results in `results/baseline5/`.

## Try it

```bash
# Copy-paste attacker on 4 handwritten items with stub traps, fully offline:
python scripts/run_copy_paste.py --stub --fake

# Same, against one real model from configs/models.yaml (costs a few cents at most):
python scripts/run_copy_paste.py --stub --models claude-small

# Document-upload attacker: the 4 items as one PDF with invisible traps (same flags):
python scripts/run_doc_upload.py --stub --fake

# Browser-agent attacker: a model fills in the real web page, reading it as text or as the
# accessibility tree (needs the server and browser extras):
python scripts/run_browser_agent.py --stub --fake --observe text   # or --observe a11y

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

**Browser-agent behaviour rows depend on the run settings, not just the attacker.** The
behaviour check scores the telemetry the page records, and two settings change it:

- `--typing instant` (the default) enters text with Playwright's `fill()`, which fires no key or
  paste events. An open-text answer of 20 or more characters then scores `paste` (untyped text).
  Use `--typing keys` to type key by key.
- Time on item includes the model's latency. A rerun served from the LLM cache answers almost at
  once, so it tends to score `too_fast`.

Compare behaviour rows only across runs with the same typing mode and cache state. A
`--dwell-ms` flag (wait a set time on each question) may be added later.

### Build the failure-mode map

After AI and human outcome Parquet files are available, build the map and its pooled SVG figure:

```bash
python scripts/build_failure_map.py \
  results/outcomes/<validated-ai-run>.parquet \
  results/outcomes/<pilot-run>.parquet --version v1 \
  --output results/failure_map/v1.json
python scripts/plot_failure_map.py results/failure_map/v1.json \
  --output results/figures/failure_map-v1.svg
```

Pass the validated run files explicitly: the outcomes directory can also contain smoke files
that are not result data. The builder validates every row, rejects dry-run rows, excludes
held-out models and the held-out survey, applies the agreed add-one smoothing, and computes
95% intervals by resampling whole sessions.
It emits a mechanism/item-type cell only when both AI and human evidence exist; it never fills a
missing class with invented observations.

**L3 rows stay out of pooled cells.** On items where the first answer was not flagged, an L3 row
is the cached L1 answer, so pooling L3 with L1 would count those answers twice. L3 gets its own
cells (`attacker_level: L3`), but every cell pooled over attacker levels (no `attacker_level`)
leaves it out (`NOT_POOLED_LEVELS` in `adr/map/builder.py`; the temporary replay map in
`adr/replay/quick_map.py` does the same).

Before the pilot rows arrive, build the provisional AI side explicitly:

```bash
python scripts/build_failure_map.py \
  results/outcomes/<validated-ai-run>.parquet --version v1-ai-only \
  --output results/failure_map/v1-ai-only.json --ai-only
```

AI-only cells have real AI counts and confidence intervals, but zero human counts, no human
confidence intervals, and prior-only human probabilities required by the shared contract. The
output is marked provisional and must not be used by the router. Rebuild without `--ai-only`
when the pilot outcome rows are available.

## Datasets

All six sources are public. Download each one yourself and put the raw files where the table says. Every script reads those paths by default.

| Source | What we use it for | Official download | Put the raw files in |
|---|---|---|---|
| MMLU (Hendrycks et al. 2021) | Quiz MCQs with known answers (decoy checks) | [hendrycks/test](https://github.com/hendrycks/test): [`data.tar`](https://people.eecs.berkeley.edu/~hendrycks/data.tar) | `data/raw/mmlu/data/{dev,val,test}/*.csv` (untar there) |
| ARC (Clark et al. 2018) | Quiz MCQs with known answers (decoy checks); we use ARC-Challenge test | [allenai.org/data/arc](https://allenai.org/data/arc): [`ARC-V1-Feb2018.zip`](https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip) | `data/raw/arc/ARC-Challenge/ARC-Challenge-Test.jsonl` (and `ARC-Easy/`) |
| OpinionQA (Santurkar et al. 2023; Pew American Trends Panel) | Survey MCQ / Likert items with human answer distributions | [tatsu-lab/opinions_qa](https://github.com/tatsu-lab/opinions_qa) (data link in its README) | `data/raw/opinionqa/human_resp/American_Trends_Panel_W<wave>/{info,responses}.csv` |
| GSS 2018 (NORC) | Survey MCQ items with human answers; 8 locked questions | [`2018_stata.zip`](https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2018_stata.zip) ([codebook](https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS_Codebook_mainbody.pdf)) | `data/raw/gss/2018_stata.zip`, extracted to `data/raw/gss/2018_stata/GSS2018.dta` |
| IPIP Big Five, 50 items (Open-Source Psychometrics Project) | Reverse-keyed item pairs for contradiction-type opinion traps | [openpsychometrics.org/_rawdata](https://openpsychometrics.org/_rawdata/): [`BIG5.zip`](https://openpsychometrics.org/_rawdata/BIG5.zip) | `data/raw/ipip/BIG5/BIG5/{data.csv,codebook.txt}` (unzip into `data/raw/ipip/BIG5/`) |
| ASURRE source surveys (Wang, Mamaev, Leckie 2026): Nature PhD Students 2019, Springer Nature Social Media 2017, OSMI Mental Health in Tech 2017–18 | Open-ended human answers: the writing-style signal and canary false positives | Via the data section of [arXiv 2609.17317](https://arxiv.org/abs/2609.17317) | Configs `data/raw/asurre/configs/{phd2019,springer_2017,osmi_mh}.json`; the response spreadsheets each config names under `"source"` go in `data/raw/asurre/raw/` |

**Held out:** the OSMI Mental Health in Tech 2017–18 survey (`asurre_osmi_2017_18`). It is excluded from all fitting: the failure map, thresholds and the style profile. It is used only for the final evaluation (`--final`). The team locked this before any fitting, using the "smallest source" rule recorded in `configs/data_decisions.json`.

**Held-out model:** `claude-haiku`, set with `heldout: true` in `configs/models.yaml`. **Decision, Nov 17:** it replaces gemini-flash, because Gemini now needs a $50 minimum prepay; the decision is recorded in `configs/data_decisions.json`. claude-haiku never contributes to the failure map, thresholds or any tuning, and runs only once, in the final `--final` evaluation. Don't run it on the real banks before then. Smoke tests on the 4 handwritten `--stub` items are fine.

### Build the banks

```bash
# GSS 2018 first: verifies the official archive and Stata-file checksums, then writes the
# 8 locked questions (plus the WTSSNR weight) to data/raw/gss/gss.csv
python scripts/prepare_gss_2018.py

# All validated banks and variants -> data/processed/runs_v1/ (offline, no model calls)
python scripts/build_runs_v1.py
python scripts/build_runs_v1.py --datasets mmlu arc        # only some sources

# Writing-style profile from the ASURRE training surveys (never opens the held-out one)
python scripts/prepare_style_profile.py                   # -> data/processed/runs_v1/style_profile.json
```

Exact GSS wording, response codes, provenance and checksums are in `configs/gss_2018_manifest.json`. The locked variable list and the held-out decision are in `configs/data_decisions.json`.

### What is not in git

Raw data (`data/raw/`), built banks (`data/processed/`), the model response cache (`cache/llm/`), run outputs (`results/outcomes/`, `results/runs/`) and all pilot data (`pilot_data/`, `dry_run_data/`) are gitignored and never committed. API keys go in `.env`, which is also ignored.

The gpt-mini run data (outcome tables and run folders) is in the team Drive folder **"ADR runs"**. Pilot data is not shared there or anywhere else; only aggregate totals are reported.

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

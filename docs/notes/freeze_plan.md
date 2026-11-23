# Experiment freeze, Sun Nov 22: the final run (plan, not run)

Driver: `scripts/freeze_run.sh`. It needs #50 (cache-key lock, per-job spend cap) and #51
(calibration) merged, and a checkout at tag `freeze-2026-11-22`. Nothing in this note has been run
except two smoke calls on a training item (below).

## FPR rule

**False-flag rates come only from the 44 pilot honest sessions, with the same folds and
calibrated thresholds as #51; the held-out OSMI bank has no human side, so no FPR is reported on
it.**

The same sentence is written at the top of the run's `results/freeze-2026-11-22/README.md` by
`scripts/freeze_report.py`.

## Start time and preflight

**Start after 17:00 Arizona time on Sun Nov 22, which is 00:00 UTC Mon Nov 23.** The team's weekly cap ($25, `configs/models.yaml`, unchanged) is counted per ISO week in UTC.
- Before that moment it is still ISO week 47, which already has $14.89 of spend in the `adr-project` ledger (as of Nov 21), leaving $10.11.
- From 00:00 UTC Monday, week 48 starts with the full $25.

Before any paid call, the driver's preflight (`scripts/freeze_run.sh`) checks:
- HEAD is exactly the tag `freeze-2026-11-22`;
- the tag is on main (`git merge-base --is-ancestor HEAD origin/main`);
- tracked files are clean;
- #50's per-job spend cap is present;
- the manifest exists, and its spend caps and claude-haiku `max_tokens` equal the driver's;
- #51's two reference runs exist.

It then prints this week's ledger spend and **refuses to start if the remaining weekly budget is below $22**, the sum of the two job caps (`scripts/freeze_manifest.py budget`). Finally it rebuilds the held-out bank and verifies every input against the manifest, including that the manifest's commit is HEAD.

## (a) Attacker runs: 42 runs, samples 0–19, run ids `v1-final-<mode>-<level>-<bank>-<model>-s20`

| model | banks | levels | modes | runs |
|---|---|---|---|---|
| claude-haiku (held-out model) | quiz-mmlu-200, quiz-arc-200, survey-ipip, survey-gss, assignment-asurre-train | L0, L2, L2-generic | copy-paste, document upload | 30 |
| claude-haiku | assignment-asurre-osmi-heldout (held-out survey) | L0, L2, L2-generic | copy-paste, document upload | 6 |
| gpt-mini | assignment-asurre-osmi-heldout only | L0, L2, L2-generic | copy-paste, document upload | 6 |

- Every call runs at temperature 0.7 with `--final`.
- The drivers refuse claude-haiku, OSMI items and held-out-split items without `--final`; with `--final` they require "final" in the run id (`adr/harness/final_guard.py`).
- Each provider is its own job, `ADR_JOB_ID=freeze-2026-11-22-<model>`, with its own spend cap (`ADR_JOB_SPEND_CAP_USD`), approved by Arav on Nov 21:
  - claude-haiku: **$12**;
  - gpt-mini: **$10**.

  The caps are fixed in `scripts/freeze_run.sh` and recorded in the manifest (`spend_caps_usd`). The driver refuses to start if the two differ.
- The team's weekly cap ($25) applies on top of these; see "Start time and preflight".
- **Reply limits (decided Nov 21):**
  - claude-haiku copy-paste: `max_tokens` **1,024**. The training runs used 300 for quiz, IPIP and GSS, and 1,000 for written answers.
  - Why: claude-haiku is billed about 190 output tokens per call that are not in the reply text. The smoke call billed 271 output tokens for a 46-word reply, so a 300 limit would cut replies off.
  - claude-haiku document upload keeps the training limits (3,000; 6,000 for written answers), which are already above 1,024.
  - gpt-mini is unchanged.
  - **Truncation stays fatal:** any truncated reply stops the driver (`check_run`).
  - The limits are recorded in the manifest (`max_tokens`) with this reason.

## (b) Router runs: two separate reports

Every run uses `run_real_router.py --final` with #51's frozen config (per_mech cap, clear_policy reopen, neg_floor_frac 1.0).
- Pilot: the floor750 pilot.
- Other flags: `--jobs 8 --resume`.
- Each report has two runs: one with all 44 honest sessions, one without the 4 low-effort ones.
- The AI inputs are the 35 training outcome files #51 used, plus one of these:

| report | final outcome files | router run ids |
|---|---|---|
| **Unseen model** | claude-haiku composite sessions on the training banks (30 files) | `freeze-2026-11-22-unseen-model-{all,no-low-effort}` |
| **Unseen survey** | gpt-mini and claude-haiku on the OSMI bank (12 files) | `freeze-2026-11-22-unseen-survey-{all,no-low-effort}` |

Each report gives TPR at the A_0.05 (headline) and A_0.01 thresholds, with 95% bootstrap CIs.
- It is broken down by model, level (L0, L2, L2g) and mode.
- **"Unseen model" means an unseen model, not unseen items.** These are the same banks the map was fitted on, answered by a model that never contributed to the map, thresholds or any tuning.
- **Unseen-survey TPR comes from canary and style only and is not comparable to the headline.** The OSMI bank has written answers only, so the pilot's other checks have no rows there; they show as missing (no evidence). The report lists missing checks per session.

## (b2) Pre-registered ablation: clearing requires a human-looking style total

Registered Nov 21, before the held-out run and the freeze tag; agreed by the team. It is also recorded in `configs/data_decisions.json` under `preregistered_ablations.clear_requires_style_human`.

- **Rule.** A session may clear only if its running per-mechanism style total is below 0 at the moment it would clear (score ≤ B). Otherwise it stays active and keeps getting checks. Everything else is #51's frozen config.
  - Router option: `RouterConfig.clear_requires_style_human`, used as `run_real_router.py --clear-requires-style-human`.
  - Default off.
  - **Which total it reads.** The rule reads the running total under `cap_key("style")`, the key style's evidence is capped under:
    - In the main design (`per_mech`) and in `no_cap`, that is style's own total.
    - Under the `passive_group` cap variant, it is the shared `passive` total (behaviour + style).
    - Under `open_text_group`, it is the shared `open_text` total (canary + style).

    The flag applies to every router row of a partner run, so those cap-variant rows read the group's total, not style alone.
- **It is an ablation.** The main policy (#51, `router_calibration`) is unchanged and remains the headline.
- **Primary outcome: L2 wrongly cleared.** The share of L2 AI sessions that end cleared (`ai_cleared_rate[<model>:<mode>:L2]`), per model and mode, at A_0.05 (headline threshold) and A_0.01. L0 and L2-generic are reported, not primary.
- **Secondary outcome:** per-group TPR (every model × level × mode in the report's matrix) at A_0.05 and A_0.01, with 95% bootstrap CIs, main policy beside the ablation.
- **Costs:** the share of honest sessions cleared, and checks per honest person.
- **Guardrail:** honest false pauses, as k of n with the exact (Clopper–Pearson) 95% interval, as in #51's notes.
- **Unseen survey.** In the unseen-survey report, the evidence on the OSMI bank is canary + style only. The ablation's result there rests on that evidence alone and is not comparable to the headline; the report says so beside each ablation table.
- **Provenance.** Chosen by Harshini on the pilot sessions and gpt-mini training data (Nov 21). The held-out run is therefore its first clean test.

**How it runs.** Each of the 4 router runs in (b) is followed by its partner replay, `<run>-style-clear`. The partner has the same inputs plus `--clear-requires-style-human`, and makes no model calls, so there are 8 router replays in all. The partners get the same group-matrix and source checks as the main runs.
- Each run's `config.yaml` must record the flag off (main) or on (partner).
- Both reports show the ablation beside the main policy, at both thresholds: the primary outcome, the costs, the guardrail with its exact interval, and the secondary per-group TPR.
- Each run calibrates its thresholds on the same 44 pilot sessions and folds, and the report shows them side by side.

**Offline check (Nov 21).** `rq2-l012-floor750-reopen-all` (#51), replayed at its own thresholds with and without the rule (no model calls). The replay without the rule matches the run's `metrics.json`. It reproduces Harshini's offline numbers exactly:

| | #51 config | with the rule |
|---|---|---|
| gpt-mini copy-paste L2 wrongly cleared (of 20), A_0.05 and A_0.01 | 12 | 7 (5 stopped) |
| honest sessions cleared (of 44) | 36 | 30 (6 lost) |
| honest false pauses, A_0.05 / A_0.01 | 3 / 2 | 3 / 2 (unchanged) |
| checks per honest person, A_0.05 / A_0.01 | 6.41 / 6.57 | 7.32 / 7.48 |

No other group changes. TPR is unchanged: the 5 stopped sessions end undecided, not paused.

## (c) Post-run checks (`scripts/freeze_report.py`, exit 1 = the driver stops with "POST-RUN CHECK FAILED")

1. **Exactly the expected AI groups, 20 sessions each, in both session tables.**
   - Every expected group (model:mode:level) must be in both `n_ai_sessions_by_group` and `ai_sources_by_group`.
   - Each table's count must independently be exactly 20 (samples 0–19), so the run fails if the two tables disagree.
   - It also fails on no groups at all, or a missing, short or over-full group, or a group or model outside the matrix. The error names the run, the group, the table and the count.

   Expected groups:
   - Unseen model: claude-haiku × {L0, L2, L2g} × {copy-paste, upload}, 6 groups.
   - Unseen survey: claude-haiku and gpt-mini × the same levels and modes, 12 groups.

   gpt-mini has no unseen-model groups by design. `--final` evaluates only held-out rows: the
   held-out model's rows, and training models' rows on the held-out survey. All 115,767 AI rows in
   the 35 training files are gpt-mini, train split, so they fit the map and are never evaluated
   in that run. A gpt-mini group there is an unexpected model and fails.
2. **Every check answered from the session's own rows.** In the per-group source table
   (`ai_sources_by_group`), synthetic = fallback = 0 for every group, gpt-mini's OSMI groups included.
   Unseen-model groups must also have missing = 0.
   - AI donors are built from the held-out rows with no failure map (`adr/replay/crossfit.py`), so a value above 0 means something is wrong with the inputs.
   - Unseen-survey groups are allowed missing checks: they are the pilot checks with no OSMI rows.
   - Style is passive (scored on every written answer), so it is not in the source table; the README says so.
3. **The honest side is #51's.** Each run's A_0.05 and A_0.01 must equal those of #51's run with
   the same honest sessions:
   - `rq2-l012-floor750-reopen-all` for the all-44 runs;
   - `rq2-l012-floor750-reopen-no-low-effort` for the runs without the low-effort sessions.

   A difference means the folds, pilot or config changed.

Tests: `tests/test_freeze.py` and `tests/test_final_guard.py`.

## Inputs and the held-out bank check

`results/freeze-2026-11-22/freeze_manifest.json` (`scripts/freeze_manifest.py`) records sha256
for:
- the held-out bank;
- the floor750 pilot parquet;
- `configs/data_decisions.json` and `configs/models.yaml`;
- the 35 training outcome files. 15 are in `adr-project/results/outcomes`; **20 are in the `adr-rq2-l2` worktree** (`adr-rq2-l2/results/outcomes`). Keep that worktree until the final run and its write-up are done: the manifest and the router read those files by path;
- the style profile `data/processed/runs_v1/style_profile.json`. Its profile sha256 must be the locked `6b43d6ed5372…` from `configs/data_decisions.json`, or `prep` refuses;
- the training banks' items, assessment and variants files: 12 files under `data/processed/runs_v1` and `data/processed/archana-survey-written-v1`;
- `server/pilot_items.json`, the pilot bank the router replays. It is gitignored, so this sha is the only record of the version used;
- the per-job spend caps (`spend_caps_usd`) and reply limits (`max_tokens`, with the reason);
- the commit it was written at. `verify` fails unless that commit is HEAD;
- every final outcome file, added after (a).

`verify` runs before (a) and again before (b).

**Held-out bank check, Nov 21, against main 1b57f48.** I built the bank with #55's
`--heldout-asurre-only` and compared it with the bank inside Archana's
`assignment-asurre-osmi-heldout-2026-11-20.zip`.
- **Content: MATCH.** Combined content sha256 `02cc0b2a588e3f91…`; 10 items, 20 variants.
- **Bytes differ, content does not.** Her files have Windows line endings, so the comparison normalises CRLF to LF.
- **Not compared:** the bank's own manifest.json, which records the builder's output paths.
- **Rerun, Nov 21, after #50 and #51 merged (main afe8509, branch f5a1001): MATCH again,** 5 of 5 files, content sha256 `02cc0b2a588e3f91…`. The provisional manifest's bank entry also matches.

## Cost estimate per provider

The estimate uses the per-call token counts of the matching gpt-mini training runs (same banks, levels, modes and samples), applied to the planned call counts. Prices are from `configs/models.yaml`, which is also what the spend cap counts.

The claude-haiku adjustments come from the two smoke calls on the same prompt:
- **Input tokens:** gpt-mini's × 1.39.
- **Visible reply tokens:** gpt-mini's × 1.39.
- **Hidden output:** about 191 billed output tokens per call that are not in the reply text. The smoke reply was 46 plain words, about 80 tokens, out of 271 billed.

| job | calls | best estimate | cap |
|---|---|---|---|
| claude-haiku (36 runs) | 57,092 | **$8.0** | $12 |
| gpt-mini (6 OSMI runs) | 1,292 | **$0.55** | $10 |

claude-haiku by bank (calls, best estimate):

| bank | copy-paste | upload |
|---|---|---|
| quiz-mmlu-200 | 24,000, $2.94 | 1,200, $0.67 |
| quiz-arc-200 | 24,000, $2.84 | 1,200, $0.56 |
| survey-ipip | 3,000, $0.34 | 180, $0.06 |
| survey-gss | 480, $0.06 | 60, $0.02 |
| assignment-asurre-train | 1,560, $0.22 | 120, $0.05 |
| assignment-asurre-osmi-heldout | 1,200, $0.17 | 92, $0.04 |

Most of the cost (85%) is the hidden output on the 48,000 short quiz copy-paste answers.
- **Where the estimate is weakest:** it rests on one smoke call. If the hidden output is about 240 tokens per call instead of 191, claude-haiku would have passed $10. That margin is why its cap was raised to $12.
- **If the hidden output instead scales with reply length:** the estimate would be $5.6.
- **Conservative bound:** taking the higher of the two cases for each bank gives $9.6.

**Effect of the 1,024 copy-paste limit.** The best estimate is unchanged: billing is per token used, and the limit only stops replies being cut off.
- It raises the most a single claude-haiku copy-paste call can cost, from 300 to 1,024 output tokens.
- If all 54,240 copy-paste calls hit 1,024, that would be about $28. The $12 job cap stops the job well before that.

If claude-haiku hits the cap, the cap stops its calls and the driver fails on that run (`check_run`) instead of going on. A restart with a higher cap resumes; finished runs are skipped.

**Smoke calls, Nov 21.** One call per model on the training item `asurre_nature_phd_2019`; never the held-out bank.
- claude-haiku: 43 input, 271 output tokens, $0.00014.
- gpt-mini: 31 input, 28 output tokens, $0.00015.

## Follow-up: final tables

`results/freeze-2026-11-22/README.md` is gitignored (results/ stays out of the repo). After the run, a follow-up docs PR will copy the final tables into `docs/notes/` with aggregates only, like the RQ2 calibration notes:
- the two reports;
- the post-run check results;
- the manifest's commit and the run's spend.

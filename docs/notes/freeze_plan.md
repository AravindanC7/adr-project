# Experiment freeze, Sun Nov 22: the final run (plan, not run)

Driver: `scripts/freeze_run.sh`. It needs #50 (cache-key lock, per-job spend cap) and #51
(calibration) merged, and a checkout at tag `freeze-2026-11-25`. Apart from two smoke calls on a
training item (below), the only paid runs so far are three halted launches: tag
`freeze-2026-11-22` (see "Halted first launch, Nov 22–23"), tag `freeze-2026-11-23` (see
"Halted second launch, Nov 23–24") and tag `freeze-2026-11-24` (see "Halted third launch,
Nov 24–25").

## FPR rule

**False-flag rates come only from the 44 pilot honest sessions, with the same folds and
calibrated thresholds as #51; the held-out OSMI bank has no human side, so no FPR is reported on
it.**

The same sentence is written at the top of the run's `results/freeze-2026-11-25/README.md` by
`scripts/freeze_report.py`.

## Start time and preflight

**Start after 17:00 Arizona time on Sun Nov 22, which is 00:00 UTC Mon Nov 23.** The team's weekly cap ($25, `configs/models.yaml`, unchanged) is counted per ISO week in UTC.
- Before that moment it is still ISO week 47, which already has $14.89 of spend in the `adr-project` ledger (as of Nov 21), leaving $10.11.
- From 00:00 UTC Monday, week 48 starts with the full $25.

Before any paid call, the driver's preflight (`scripts/freeze_run.sh`) checks:
- HEAD is exactly the tag `freeze-2026-11-25`;
- the tag is on main (`git merge-base --is-ancestor HEAD origin/main`);
- tracked files are clean;
- #50's per-job spend cap is present;
- the manifest exists, and its spend caps and claude-haiku `max_tokens` equal the driver's;
- #51's two reference runs exist.

It then prints this week's ledger spend and **refuses to start if the remaining weekly budget is below the sum of the two job caps** ($14 since Nov 25; $22 before) (`scripts/freeze_manifest.py budget`). Finally it rebuilds the held-out bank and verifies every input against the manifest, including that the manifest's commit is HEAD.

## (a) Attacker runs: 42 runs, samples 0–19, run ids `v1-final-<mode>-<level>-<bank>-<model>-s20`

| model | banks | levels | modes | runs |
|---|---|---|---|---|
| claude-haiku (held-out model) | quiz-mmlu-200, quiz-arc-200, survey-ipip, survey-gss, assignment-asurre-train | L0, L2, L2-generic | copy-paste, document upload | 30 |
| claude-haiku | assignment-asurre-osmi-heldout (held-out survey) | L0, L2, L2-generic | copy-paste, document upload | 6 |
| gpt-mini | assignment-asurre-osmi-heldout only | L0, L2, L2-generic | copy-paste, document upload | 6 |

- Every call runs at temperature 0.7 with `--final`.
- The drivers refuse claude-haiku, OSMI items and held-out-split items without `--final`; with `--final` they require "final" in the run id (`adr/harness/final_guard.py`).
- Each provider is its own job, `ADR_JOB_ID=freeze-2026-11-25-<model>`, with its own spend cap (`ADR_JOB_SPEND_CAP_USD`), approved by Arav on Nov 21:
  - claude-haiku: **$12**;
  - gpt-mini: **$2** (lowered from $10 on Nov 25, estimate $0.55; see "Halted third launch, Nov 24–25").

  The caps are fixed in `scripts/freeze_run.sh` and recorded in the manifest (`spend_caps_usd`). The driver refuses to start if the two differ.
- The team's weekly cap ($25) applies on top of these; see "Start time and preflight".
- **Reply limits (decided Nov 21; claude-haiku copy-paste raised Nov 23, document upload Nov 24):**
  - claude-haiku copy-paste: `max_tokens` **4,096** (1,024 until the halted first launch; see below). The training runs used 300 for quiz, IPIP and GSS, and 1,000 for written answers.
  - Why: claude-haiku is billed about 190 output tokens per call that are not in the reply text. The smoke call billed 271 output tokens for a 46-word reply, so a 300 limit would cut replies off.
  - claude-haiku document upload: **8,192** on the quiz and survey banks (3,000 until the halted second launch; see below). the same **8,192** on the written-answer banks (asurre-train and the held-out OSMI bank; 6,000 before; see below). One limit for all claude-haiku document upload.
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
| **Unseen model** | claude-haiku composite sessions on the training banks (30 files) | `freeze-2026-11-25-unseen-model-{all,no-low-effort}` |
| **Unseen survey** | gpt-mini and claude-haiku on the OSMI bank (12 files) | `freeze-2026-11-25-unseen-survey-{all,no-low-effort}` |

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

`results/freeze-2026-11-25/freeze_manifest.json` (`scripts/freeze_manifest.py`) records sha256
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
| gpt-mini (6 OSMI runs) | 1,292 | **$0.55** | $2 (was $10 until Nov 25) |

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

**Effect of the 4,096 copy-paste limit.** The best estimate is unchanged: billing is per token used, and the limit only stops replies being cut off.
- It raises the most a single claude-haiku copy-paste call can cost, from 300 to 4,096 output tokens.
- If every copy-paste call hit the limit, the cost would be far above the cap. The $12 job cap stops the job well before that.
- **Measured on the halted first run:** $1.07 for 8,000 calls, about 9% above the estimate's per-call cost for that bank and mode. Scaled up, claude-haiku's best estimate is about $8.7, still under $12.

If claude-haiku hits the cap, the cap stops its calls and the driver fails on that run (`check_run`) instead of going on. A restart with a higher cap resumes; finished runs are skipped.

**Smoke calls, Nov 21.** One call per model on the training item `asurre_nature_phd_2019`; never the held-out bank.
- claude-haiku: 43 input, 271 output tokens, $0.00014.
- gpt-mini: 31 input, 28 output tokens, $0.00015.

## Halted first launch, Nov 22–23

- **Launch:** tag `freeze-2026-11-22` (main fb45265, the #56 merge), started Sun Nov 22 at 23:54 Arizona (06:54 UTC Mon Nov 23). The preflight passed: the tag was on main, week 48 had $25.00 left, the caps and `max_tokens` matched the manifest, and the held-out bank matched Archana's zip.
- **The run:** `v1-final-cp-L0-quiz-mmlu-200-claude-haiku-s20`, the first of the 42. It finished with 8,000 outcome rows and 0 errors, and was not stopped by the cap.
- **Why it halted:** **41 of 8,000 replies were cut off** at the 1,024-token copy-paste limit (`finish_reason` "length"). Truncation is fatal, so `check_run` stopped the driver. No other run started.
  - 21 of the 41 came from one item: bank `quiz-mmlu-200`, type MCQ, 20 on its plain variant and 1 on its decoy variant.
  - The other 20 were spread over 9 items.
  - By variant, 24 of the 41 were plain and 17 decoy.
- **Spent:** $1.07, under job `freeze-2026-11-22-claude-haiku`.
- **Decision (Arav, Nov 23):** raise claude-haiku's copy-paste limit to **4,096**. Document upload and gpt-mini are unchanged, and truncation stays fatal.
  - Why: a cut-off reply is not a valid observation. Dropping such replies would bias toward the items where the model writes most, so the limit is raised instead.
  - The new limit is in `scripts/freeze_run.sh` and the manifest (`max_tokens`). The run restarts from a new tag, `freeze-2026-11-23`; the old tag stays as it was.
- **Cost of the restart:** `max_tokens` is part of the LLM cache key, so none of the halted run's 8,000 calls are reused. Run 1 runs and bills again in full, about $1.1 and about an hour.
  - The new job ids (`freeze-2026-11-23-<model>`) start their own $12 and $10 caps.
  - The halted job's $1.07 still counts toward the weekly $25.
- **The halted run's files:** the rerun overwrites them under the same run id. A copy is kept in the team's freeze folder outside the repo, as is the first launch's manifest and driver log.

### Restart, Nov 23: #57 merged with one approval

- **#57** (the 4,096 limit and the new tag) **was merged with one approval, Harshini's**, given on its head fd60f1e.
- **Why:** Archana had not reviewed it yet. Arav decided to go ahead on Harshini's approval alone rather than hold the restart. The change is the one he decided after the halt: one reply limit and the tag name, with the rule and the rest of the run plan unchanged.
- Archana can still review #57 after the merge. Anything she raises is handled as a follow-up and does not change the tagged run.
- **Restart:**
  - Tag `freeze-2026-11-23` is on the #57 merge commit (main 8c0889a), and all 967 tests pass on it.
  - The held-out bank was rebuilt and matches Archana's zip (content sha256 `02cc0b2a…`).
  - The manifest was written at 8c0889a with a clean tree and verified: 53 files.
  - Started at 18:21 Arizona on Mon Nov 23.
- **Preflight passed:**
  - the tag is on main;
  - week 48 had $1.07 spent and $23.93 left, at least the $22 required;
  - the caps ($12 and $10) and the claude-haiku copy-paste `max_tokens` of 4,096 matched the manifest;
  - every input was verified.

  The tag was pushed after the preflight passed.

## Halted second launch, Nov 23–24

- **Launch:** tag `freeze-2026-11-23` (main 8c0889a, the #57 merge), started at 18:21 Arizona on Mon Nov 23.
- **Run 1, `v1-final-cp-L0-quiz-mmlu-200-claude-haiku-s20`, passed** at the new copy-paste limit of 4,096: 8,000 rows, 0 cut off, 0 errors. It stays as it is, and the driver skips it on restart.
- **Run 2, `v1-final-du-L0-quiz-mmlu-200-claude-haiku-s20` (document upload, 20 questions per document), halted the driver.** It finished without a cap stop.
  - **16 of its 400 document replies were cut off** at the 3,000-token document-upload limit. That is 20 documents × 20 samples.
  - They fall on **3 of the 20 documents**: 8, 7 and 1 cut-offs.
  - Each cut-off reply loses the answers after the cut, which gave 225 skipped questions. The run has 7,775 rows instead of 8,000.
  - Truncation is fatal, so `check_run` stopped the driver. No other run started.
- **Spent:** $1.88 under job `freeze-2026-11-23-claude-haiku`, for runs 1 and 2.
- **Measurement (Nov 24, no API calls).** These are claude-haiku's billed output tokens per document reply, hidden tokens included, taken from the LLM cache entries of run 2's 400 replies (the ledger's output-token count matches).

  | replies | max | 95th pct | median |
  |---|---|---|---|
  | all 400 | 3,000 (cut off) | 2,921 | 1,836 |
  | 384 finished | **2,994** | 2,769 | 1,829 |

  Per document, the 95th percentile ranged from 1,493 to 2,828 on the 17 documents with no cut-offs. On the 3 documents with cut-offs it reached 2,821 and 3,000. The largest finished reply on each of the 20 documents ranged from 1,606 to 2,994.
- **claude-haiku's maximum output** is 128K tokens per call (`claude-haiku-5-5`).
- **Rule (Arav, Nov 24):**
  - the document-upload limit for the quiz and survey banks is the larger of 8,192 and twice the largest finished reply in run 2, rounded up to a power of two and capped at the model's maximum;
  - apply the same rule to the written-answer banks only if twice their largest observed reply would exceed 6,000;
  - gpt-mini and claude-haiku copy-paste are unchanged;
  - truncation stays fatal.
- **Result:**
  - **Quiz and survey banks: 8,192.** Twice 2,994 is 5,988, so the 8,192 floor applies. 8,192 is already a power of two and below 128K.
  - **Written-answer banks:** by the rule they would have stayed at 6,000, since there is no claude-haiku document-upload reply on them yet (gpt-mini's largest in training was 564 tokens).
- **Written-answer banks raised to 8,192 (Arav, Nov 24, on Archana's and Harshini's reviews of #59).** Harshini asked for 8,192 or 16,384, Archana for 8,192, since no claude-haiku written-answer run shows 6,000 is enough and truncation is fatal. Reasons for 8,192:
  - the measured quiz maximum was 2,994;
  - gpt-mini's largest written-answer document reply was 564;
  - one limit for all claude-haiku document upload.

  A higher limit costs nothing on replies that would finish anyway: the spend cap counts the tokens actually billed. None of these runs had happened yet, so no cached reply is lost.
- The limit is in `scripts/freeze_run.sh` (`HAIKU_DU_MAX_TOKENS`), which now passes it explicitly on every claude-haiku document-upload run, written-answer banks included, instead of relying on the survey runner's default. The manifest records it under `max_tokens.claude-haiku.doc_upload` (`quiz_and_survey` and `written_answers`), with the rule. The preflight checks both entries against the driver.
- The run restarts from a new tag, `freeze-2026-11-24`; the old tags stay as they were.
- **Cost of the restart:** run 1 is skipped. Run 2 reruns in full, because the limit is part of the cache key: about $0.8 and about 15 minutes. The new job ids (`freeze-2026-11-24-<model>`) start their own $12 and $10 caps.
- **No spare launch this week (Harshini, review of #59):** about $22.05 is left in week 48 ($25 − $1.07 − $1.88) against the $22 the preflight needs. The restart should pass, but only just. Another halt would push the restart to after the weekly reset (Sunday 17:00 Arizona time), just before the poster.
- **The halted run's files:** a copy is kept in the team's freeze folder outside the repo (`halted-launch-2/`), with the second launch's manifest and logs.

## Halted third launch, Nov 24–25

- **Launch:** tag `freeze-2026-11-24` (#59 merged as c905994), started Nov 24 at 19:10 Arizona time. The preflight passed with $22.06 left in week 48.
- Run 1 was skipped as already passed. Run 2 (claude-haiku document upload, L0, quiz-mmlu) passed at 8,192: 8,000 rows, no cut-offs.
- **Run 3 halted** (`v1-final-cp-L2-quiz-mmlu-200-claude-haiku-s20`, 21:20): **12 of 8,000 calls failed with `APIConnectionError: Connection error.` after 5 retries.** No reply was cut off and there was no budget stop; the post-run check stops on any failed call. The other 7,988 replies are cached.
- **Spent:** job `freeze-2026-11-24-claude-haiku`, $2.43. Week 48: $5.38 of $25, leaving $19.62.
- **Why a restart could not start this week:** the preflight needs the week's remaining budget to cover the sum of the job caps, then $22 ($12 + $10), and $19.62 was left.

### Changes, Nov 25 (Arav)

- **gpt-mini job cap: $10 → $2.** gpt-mini's estimate for its 6 OSMI runs is $0.55 (see "Cost estimate per provider"). The weekly check needs remaining ≥ the sum of the caps, so it now needs $14, against $19.62 left. claude-haiku stays at $12. The caps are in `scripts/freeze_run.sh` and the manifest's `spend_caps_usd`; the preflight checks they match.
- **Connection errors retried longer** (`adr/harness/llm_client.py`): `APIConnectionError`, timeouts and other connection drops are now their own error kind, retried up to 10 times with exponential back-off (2, 4, 8, … s, capped at 120 s). Rate limits keep their handling; no-credit and other fatal errors still stop at once; other transient errors keep 5 tries. Truncation stays fatal. Tests cover the classification, the 10 tries with capped back-off, and giving up after 10.
- **Restart** from a new tag, `freeze-2026-11-25`; the old tags stay as they were. New job ids (`freeze-2026-11-25-<model>`) start their own $12 and $2 caps. Runs 1 and 2 are skipped as passed; run 3 reruns from cache and bills only its 12 failed calls. The retry settings are not part of the cache key, so no cached reply is lost.
- **#60 merged without review (Arav, Nov 25), because of time:** Archana and Harshini had been requested and had not reviewed yet. They were asked on #60 to raise any concerns after the merge.
- **The halted run's files:** a copy is kept in the team's freeze folder outside the repo (`halted-launch-3/`), with the third launch's manifest and logs.

## Follow-up: final tables

`results/freeze-2026-11-25/README.md` is gitignored (results/ stays out of the repo). After the run, a follow-up docs PR will copy the final tables into `docs/notes/` with aggregates only, like the RQ2 calibration notes:
- the two reports;
- the post-run check results;
- the manifest's commit and the run's spend.

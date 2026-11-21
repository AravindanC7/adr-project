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

## (a) Attacker runs: 42 runs, samples 0–19, run ids `v1-final-<mode>-<level>-<bank>-<model>-s20`

| model | banks | levels | modes | runs |
|---|---|---|---|---|
| claude-haiku (held-out model) | quiz-mmlu-200, quiz-arc-200, survey-ipip, survey-gss, assignment-asurre-train | L0, L2, L2-generic | copy-paste, document upload | 30 |
| claude-haiku | assignment-asurre-osmi-heldout (held-out survey) | L0, L2, L2-generic | copy-paste, document upload | 6 |
| gpt-mini | assignment-asurre-osmi-heldout only | L0, L2, L2-generic | copy-paste, document upload | 6 |

- Every call runs at temperature 0.7 with `--final`.
- The drivers refuse claude-haiku, OSMI items and held-out-split items without `--final`; with `--final` they require "final" in the run id (`adr/harness/final_guard.py`).
- Each provider is its own job: `ADR_JOB_ID=freeze-2026-11-22-<model>`, `ADR_JOB_SPEND_CAP_USD=10`.

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
- The OSMI bank has written answers only. In the unseen-survey report, the pilot's other checks have no rows, so they show as missing (no evidence). The report lists missing checks per session.

## (c) Post-run checks (`scripts/freeze_report.py`, exit 1 = the driver stops with "POST-RUN CHECK FAILED")

1. **Exactly the expected AI groups, 20 sessions each.** Each run must have exactly the
   expected groups (model:mode:level), with exactly 20 sessions (samples 0–19) in each. A group that is missing, short,
   over 20 or unexpected fails; the error names the run, the group and the count found.
   - Unseen model: claude-haiku × {L0, L2, L2g} × {copy-paste, upload}, 6 groups.
   - Unseen survey: claude-haiku and gpt-mini × the same levels and modes, 12 groups.

   gpt-mini has no unseen-model groups by design. `--final` evaluates only held-out rows: the
   held-out model's rows, and training models' rows on the held-out survey. All 115,767 AI rows in
   the 35 training files are gpt-mini, train split, so they fit the map and are never evaluated
   in that run. A gpt-mini group there is unexpected and fails.
2. **claude-haiku answered from its own rows.** In every claude-haiku group of the per-group
   source table (`ai_sources_by_group`), synthetic and fallback outcomes must be 0. A value above
   0 means some of its checks fell back to gpt-mini's map or to other sessions.
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
- the 35 training outcome files;
- every final outcome file, added after (a).

`verify` runs before (a) and again before (b).

**Held-out bank check, Nov 21, against main 1b57f48.** I built the bank with #55's
`--heldout-asurre-only` and compared it with the bank inside Archana's
`assignment-asurre-osmi-heldout-2026-11-20.zip`.
- **Content: MATCH.** Combined content sha256 `02cc0b2a588e3f91…`; 10 items, 20 variants.
- **Bytes differ, content does not.** Her files have Windows line endings, so the comparison normalises CRLF to LF.
- **Not compared:** the bank's own manifest.json, which records the builder's output paths.
- **Rerun planned:** after #50 and #51 merge.

## Cost estimate per provider

The estimate uses the per-call token counts of the matching gpt-mini training runs (same banks, levels, modes and samples), applied to the planned call counts. Prices are from `configs/models.yaml`, which is also what the spend cap counts.

The claude-haiku adjustments come from the two smoke calls on the same prompt:
- **Input tokens:** gpt-mini's × 1.39.
- **Visible reply tokens:** gpt-mini's × 1.39.
- **Hidden output:** about 191 billed output tokens per call that are not in the reply text. The smoke reply was 46 plain words, about 80 tokens, out of 271 billed.

| job | calls | best estimate | cap |
|---|---|---|---|
| claude-haiku (36 runs) | 57,092 | **$8.0** | $10 |
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
- **Where the estimate is weakest:** it rests on one smoke call. If the hidden output is about 240 tokens per call instead of 191, claude-haiku reaches the $10 cap.
- **If the hidden output instead scales with reply length:** the estimate would be $5.6.
- **Conservative bound:** taking the higher of the two cases for each bank gives $9.6.

If claude-haiku hits the cap, the cap stops its calls and the driver fails on that run (`check_run`) instead of going on. A restart with a higher cap resumes; finished runs are skipped.

**Smoke calls, Nov 21.** One call per model on the training item `asurre_nature_phd_2019`; never the held-out bank.
- claude-haiku: 43 input, 271 output tokens, $0.00014.
- gpt-mini: 31 input, 28 output tokens, $0.00015.

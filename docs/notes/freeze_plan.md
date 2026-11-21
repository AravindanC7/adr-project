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

1. **claude-haiku answered from its own rows.** In every claude-haiku group of the per-group
   source table (`ai_sources_by_group`), synthetic and fallback outcomes must be 0. A value above
   0 means some of its checks fell back to gpt-mini's map or to other sessions.
2. **The honest side is #51's.** Each run's A_0.05 and A_0.01 must equal those of #51's run with
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

Estimated from the real per-call token counts of the 30 matching gpt-mini training runs:
14.15M input and 0.665M output tokens, actual cost $13.60.

| job | estimate | cap |
|---|---|---|
| claude-haiku (36 runs) | about $1.8 at gpt-mini's token counts; up to about $8.7 if every reply carries the extra output tokens seen in the smoke call | $10 |
| gpt-mini (6 OSMI runs) | about $0.55 | $10 |

Both jobs are under the cap. claude-haiku is the one that could reach it. If it does, the cap stops its
calls and the driver fails on that run (`check_run`) instead of going on.

**Smoke calls, Nov 21.** One call per model on the training item `asurre_nature_phd_2019`; never the held-out bank.
- claude-haiku: 43 input, 271 output tokens, $0.00014.
- gpt-mini: 31 input, 28 output tokens, $0.00015.

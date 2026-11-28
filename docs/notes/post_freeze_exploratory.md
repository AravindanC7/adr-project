# Post-freeze analyses: exploratory (after freeze)

**Exploratory (after freeze).** Everything here was run after the freeze (tag `freeze-2026-11-26d`) and is not part of the pre-registered results. It makes no API calls and changes nothing in the frozen config. The code is `scripts/post_freeze.py`, and the outputs are in `results/post-freeze/`, which is gitignored. The tables below are aggregates only.

**How it works.** The script rebuilds each frozen router replay in-process, with the same flags the driver used, and keeps the per-session scores. The report step checks that every rebuilt replay reproduces the frozen thresholds and TPR exactly (4 runs × 3 policies, at both thresholds). If any of them does not, the report step fails.

## 1. Paired TPR differences (router minus baseline)

These use a paired session bootstrap with 200 draws. Each draw resamples the honest sessions and the AI sessions (the AI sessions within their group), and every policy uses the same draws. Each policy's thresholds are recalibrated per draw, as in the frozen summary. The interval is the 2.5th to 97.5th percentile of the per-draw differences.

"Pooled" combines the AI sessions of the unseen-model and unseen-survey runs that share the same honest sessions. They share folds, maps and thresholds, so their AI sessions can be pooled draw by draw.

| Set | Comparison | Threshold | Router TPR | Baseline TPR | Difference [95% CI] |
|---|---|---|---|---|---|
| unseen model, all 44 | − best_single | A_0.05 | 0.990 | 0.696 | +0.29 [+0.25, +0.36] |
| unseen model, all 44 | − best_single | A_0.01 | 0.946 | 0.548 | +0.40 [+0.25, +0.52] |
| unseen model, all 44 | − all_checks | A_0.05 | 0.990 | 0.808 | +0.18 [+0.08, +0.40] |
| unseen model, all 44 | − all_checks | A_0.01 | 0.946 | 0.621 | +0.33 [+0.15, +0.40] |
| unseen survey, all 44 | − best_single | A_0.05 | 0.712 | 0.527 | +0.19 [+0.06, +0.23] |
| unseen survey, all 44 | − best_single | A_0.01 | 0.712 | 0.478 | +0.24 [+0.18, +0.27] |
| unseen survey, all 44 | − all_checks | A_0.05 | 0.712 | 0.670 | +0.04 [+0.00, +0.45] |
| unseen survey, all 44 | − all_checks | A_0.01 | 0.712 | 0.292 | +0.42 [+0.03, +0.47] |
| pooled, all 44 | − best_single | A_0.05 | 0.812 | 0.588 | +0.22 [+0.14, +0.27] |
| pooled, all 44 | − best_single | A_0.01 | 0.797 | 0.503 | +0.29 [+0.21, +0.36] |
| pooled, all 44 | − all_checks | A_0.05 | 0.812 | 0.720 | +0.09 [+0.04, +0.43] |
| pooled, all 44 | − all_checks | A_0.01 | 0.797 | 0.411 | +0.39 [+0.08, +0.43] |

The no-low-effort runs (40 honest sessions) are in `paired_tpr_differences.csv`. Pooled, they give:
- router − best_single: +0.13 [+0.11, +0.15] at A_0.05 and +0.12 [+0.09, +0.14] at A_0.01;
- router − all_checks: +0.11 [+0.05, +0.42] at A_0.05 and +0.39 [+0.09, +0.43] at A_0.01.

On unseen survey without the low-effort sessions, the router and best_single are identical (difference 0 in every draw).

In no bootstrap draw does the router do worse than either baseline. The intervals for router − all_checks are wide and skewed, because all_checks' threshold moves a lot between draws.

## 2. Leave one check out (unseen model, all 44, router)

When a check is dropped, the thresholds are recalibrated without it. This is the frozen run's own `leave_one_out.csv`.

| Check dropped | TPR at A_0.05 (full → without) | TPR at A_0.01 (full → without) |
|---|---|---|
| canary | 0.99 → 0.71 | 0.95 → 0.26 |
| decoy | 0.99 → 0.69 | 0.95 → 0.69 |
| opinion trap | 0.99 → 0.99 | 0.95 → 0.95 |
| style | 0.99 → 0.97 | 0.95 → 0.94 |

## 3. Evidence per check (unseen model, all 44, router, claude-haiku)

This is the mean evidence (log-likelihood ratio) each check added per session, up to the point where the session paused at A_0.05. The share of sessions where the check added positive evidence is in brackets. Checks that added no evidence in a group are left out.

| Group | Canary | Decoy | Style |
|---|---|---|---|
| copy-paste L0 | 2.96 (100%) | – | – |
| copy-paste L2 | 1.46 (75%) | 0.28 (24%) | 0.63 (55%) |
| copy-paste L2g | 2.93 (100%) | – | 0.03 (5%) |
| upload L0 | 2.37 (80%) | 0.10 (20%) | −0.06 (0%) |
| upload L2 | – | 0.19 (95%) | – |
| upload L2g | 0.59 (20%) | 0.35 (80%) | −0.14 (0%) |

A_0.05 is low (0.008), so a session pauses as soon as a little evidence comes in, and these totals are small. The A_0.01 values are in `evidence_by_check_unseen_model_all.json`.

## 4. Cost against detection

`cost_vs_detection.png` and its table view, `cost_vs_detection.csv`, plot checks per honest person against TPR for every policy, with 95% CIs. The panels are gpt-mini calibration (#51), unseen model and unseen survey, at A_0.05 and A_0.01.

On the 24-item pilot bank, all_checks, fixed_plan and random give identical numbers, so they are drawn as one point. The router's cap variants are in the CSV only.

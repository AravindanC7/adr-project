# RQ2 multi-check router: pipeline check (6 AI sessions)

**This is a pipeline check, not the RQ2 result.** It runs the whole multi-check RQ2 pipeline end to end on the AI samples we had on Nov 18: 6 composite AI sessions, 5 copy-paste and 1 document upload. The RQ2 numbers come from the real run with 20 + 20 sessions, which will be added below as "Real run (L0, n = 40)".

Setup:
- `scripts/run_real_router.py`, branch `arav/rq2-multicheck`, which depends on #30 (cross-fitting) and #42 (survey banks, shared IPIP pair rule).
- No model calls: replay only.
- Honest sessions: the 44 pilot sessions, and separately the 40 left after removing the 4 low-effort ones. Folds come from `assign_session_folds()`.
- AI sessions: gpt-mini, L0. They are composite sessions (`adr/replay/composite.py`): sample *i* of every bank, with the same model, mode and attacker variant, joined into one session shaped like a pilot session. That means quiz decoys, survey MCQ/Likert opinion traps and open-text canary/style rows together. 0 keys missing in any bank.
- Mechanisms with both AI and human rows: canary, decoy, opinion_trap, style. Behaviour has no AI rows, so it is not used.
- Outputs: `results/real-router/rq2-pipeline-{all,no-low-effort}/`. They are not committed.

## How to read this

- **At L0, canary alone separates every session, so RQ2's L0 result shows efficiency, not extra detection.** Every AI session followed the hidden canary instruction and no honest session did. So every policy reaches TPR 1.00, and no AI session is ever wrongly cleared. What differs is the burden on honest people: the router asks about **1.9 checks** per honest session, the best single check (canary) 2, and all checks or a random check about **14**.
- **The 5% and 1% thresholds are near zero:** about 0.02 for the router with all 44 sessions (A_0.05 = 0.019). More than 95% of the resampled honest sessions score ≤ 0, because the router clears most of them and a session's score never drops below the prior of 0. So in effect any AI-leaning evidence pauses a session.
- **The real honest pause rate is shown next to the nominal FPR.** Thresholds are set on resampled honest sessions, so the real rate can differ from the nominal one. Example: at the router's 1% threshold, **1 of 44 = 2.3%** of real honest sessions is paused, above the nominal 1%. Without the low-effort sessions it is 1 of 40 = 2.5%.
- **passive_group equals per_mech here, because behaviour has no AI rows.** That leaves style alone in the passive group. The two rows are identical.
- **Contradiction rows are included.** These are the IPIP reverse-pair rows scored with #42's `score_reverse_pairs`: 18 in the copy-paste survey runs, 2–5 in every copy-paste session. The document-upload IPIP run has none; all its rows are plausible. The cross-fitting split puts 17 of the 18 among the map's training rows. The pilot rows score pairs with their own copy of the rule in `server/clean.py`, not the shared function.
- **Same-type draws:** AI sessions answered other items than the pilot bank. Each pilot check is answered from the AI session's own rows on an item of the same type. For example, quiz MCQ decoy rows answer the pilot's survey MCQ items, because both are `mcq`.
- **Document upload has 1 session,** so it gets no TPR; its session is listed instead.

### Pipeline check: all 44 pilot sessions (44 honest sessions)

| policy | threshold | A | TPR [95% CI] | nominal FPR | real honest pause rate | checks per honest session | honest cleared | AI wrongly cleared |
|---|---|---|---|---|---|---|---|---|
| router (per_mech) | A_0.05 | 0.019 | 1.00 [1.00, 1.00] | 5.0% | 1 of 44 = 2.3% | 1.89 [1.36, 2.64] | 90.9% | 0.0% |
| router (per_mech) | A_0.01 | 0.684 | 1.00 [1.00, 1.00] | 1.0% | 1 of 44 = 2.3% | 1.89 [1.36, 2.64] | 90.9% | 0.0% |
| router (per_mech) | A_max | 2.750 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 1.93 [1.39, 2.75] | 90.9% | 0.0% |
| router [passive_group] | A_0.05 | 0.019 | 1.00 [1.00, 1.00] | 5.0% | 1 of 44 = 2.3% | 1.89 [1.36, 2.64] | 90.9% | 0.0% |
| router [passive_group] | A_0.01 | 0.684 | 1.00 [1.00, 1.00] | 1.0% | 1 of 44 = 2.3% | 1.89 [1.36, 2.64] | 90.9% | 0.0% |
| router [passive_group] | A_max | 2.750 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 1.93 [1.39, 2.75] | 90.9% | 0.0% |
| router [open_text_group] | A_0.05 | 0.381 | 1.00 [1.00, 1.00] | 5.0% | 1 of 44 = 2.3% | 1.34 [1.07, 1.93] | 90.9% | 0.0% |
| router [open_text_group] | A_0.01 | 0.381 | 1.00 [1.00, 1.00] | 1.0% | 1 of 44 = 2.3% | 1.34 [1.07, 1.93] | 90.9% | 0.0% |
| router [open_text_group] | A_max | 1.362 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 1.36 [1.07, 2.00] | 90.9% | 0.0% |
| best single check | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 5.0% | 0 of 44 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.0% | 0 of 44 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_max | 0.000 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| all checks | A_0.05 | 1.927 | 1.00 [1.00, 1.00] | 5.0% | 2 of 44 = 4.5% | 13.73 [13.41, 14.00] | 0.0% | 0.0% |
| all checks | A_0.01 | 2.960 | 1.00 [1.00, 1.00] | 1.0% | 0 of 44 = 0.0% | 14.00 [13.73, 14.00] | 0.0% | 0.0% |
| all checks | A_max | 2.960 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| random check | A_0.05 | 1.927 | 1.00 [1.00, 1.00] | 5.0% | 2 of 44 = 4.5% | 13.73 [13.41, 14.00] | 0.0% | 0.0% |
| random check | A_0.01 | 2.960 | 1.00 [1.00, 1.00] | 1.0% | 0 of 44 = 0.0% | 14.00 [13.73, 14.00] | 0.0% | 0.0% |
| random check | A_max | 2.960 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| router [no_cap] *(ablation)* | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 5.0% | 0 of 44 = 0.0% | 1.16 [1.07, 1.30] | 100.0% | 0.0% |
| router [no_cap] *(ablation)* | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.0% | 0 of 44 = 0.0% | 1.16 [1.07, 1.30] | 100.0% | 0.0% |
| router [no_cap] *(ablation)* | A_max | 0.000 | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 1.16 [1.07, 1.30] | 100.0% | 0.0% |

Budget curve (router at each beta; checks per honest session at the 1% threshold):

| beta | TPR at 1% FPR | TPR at 5% FPR | checks per honest session |
|---|---|---|---|
| 0.01 | 1.00 | 1.00 | 2.25 |
| 0.02 | 1.00 | 1.00 | 1.91 |
| 0.05 | 1.00 | 1.00 | 1.89 |
| 0.1 | 1.00 | 1.00 | 1.89 |
| 0.2 | 1.00 | 1.00 | 1.89 |

Leave one check out (replay only; each check dropped in turn, canary first; same folds and split):

| dropped | policy | TPR at 5% FPR | TPR at 1% FPR | checks per honest session (5% / 1%) | real honest pause rate (5% / 1%) |
|---|---|---|---|---|---|
| canary | router | 1.00 [1.00, 1.00] | 1.00 [0.25, 1.00] | 5.14 / 5.14 | 0.0% / 0.0% |
| canary | best_single | 1.00 [1.00, 1.00] | 1.00 [0.80, 1.00] | 11.82 / 11.91 | 4.5% / 2.3% |
| canary | all_checks | 1.00 [1.00, 1.00] | 1.00 [0.60, 1.00] | 11.82 / 12.00 | 4.5% / 0.0% |
| decoy | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.16 / 1.16 | 0.0% / 0.0% |
| decoy | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| decoy | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.89 / 1.89 | 2.3% / 2.3% |
| opinion_trap | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 13.73 / 14.00 | 4.5% / 0.0% |
| style | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 / 1.00 | 0.0% / 0.0% |
| style | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| style | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 13.73 / 14.00 | 4.5% / 0.0% |

### Pipeline check: without the 4 low-effort sessions (40 honest sessions)

| policy | threshold | A | TPR [95% CI] | nominal FPR | real honest pause rate | checks per honest session | honest cleared | AI wrongly cleared |
|---|---|---|---|---|---|---|---|---|
| router (per_mech) | A_0.05 | 0.129 | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 1.88 [1.25, 2.67] | 92.5% | 0.0% |
| router (per_mech) | A_0.01 | 0.129 | 1.00 [1.00, 1.00] | 1.0% | 1 of 40 = 2.5% | 1.88 [1.25, 2.77] | 92.5% | 0.0% |
| router (per_mech) | A_max | 2.934 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 1.93 [1.25, 2.78] | 92.5% | 0.0% |
| router [passive_group] | A_0.05 | 0.129 | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 1.88 [1.25, 2.67] | 92.5% | 0.0% |
| router [passive_group] | A_0.01 | 0.129 | 1.00 [1.00, 1.00] | 1.0% | 1 of 40 = 2.5% | 1.88 [1.25, 2.77] | 92.5% | 0.0% |
| router [passive_group] | A_max | 2.934 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 1.93 [1.25, 2.78] | 92.5% | 0.0% |
| router [open_text_group] | A_0.05 | 0.188 | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 1.35 [1.05, 1.93] | 92.5% | 0.0% |
| router [open_text_group] | A_0.01 | 0.188 | 1.00 [1.00, 1.00] | 1.0% | 1 of 40 = 2.5% | 1.35 [1.05, 1.93] | 92.5% | 0.0% |
| router [open_text_group] | A_max | 1.522 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 1.38 [1.05, 2.02] | 92.5% | 0.0% |
| best single check | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 5.0% | 0 of 40 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.0% | 0 of 40 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_max | 0.000 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| all checks | A_0.05 | 1.860 | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 13.82 [13.42, 14.00] | 0.0% | 0.0% |
| all checks | A_0.01 | 2.960 | 1.00 [1.00, 1.00] | 1.0% | 0 of 40 = 0.0% | 14.00 [13.70, 14.00] | 0.0% | 0.0% |
| all checks | A_max | 2.960 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| random check | A_0.05 | 1.860 | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 13.82 [13.42, 14.00] | 0.0% | 0.0% |
| random check | A_0.01 | 2.960 | 1.00 [1.00, 1.00] | 1.0% | 0 of 40 = 0.0% | 14.00 [13.70, 14.00] | 0.0% | 0.0% |
| random check | A_max | 2.960 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| router [no_cap] *(ablation)* | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 5.0% | 0 of 40 = 0.0% | 1.15 [1.05, 1.28] | 100.0% | 0.0% |
| router [no_cap] *(ablation)* | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.0% | 0 of 40 = 0.0% | 1.15 [1.05, 1.28] | 100.0% | 0.0% |
| router [no_cap] *(ablation)* | A_max | 0.000 | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 1.15 [1.05, 1.28] | 100.0% | 0.0% |

Budget curve (router at each beta; checks per honest session at the 1% threshold):

| beta | TPR at 1% FPR | TPR at 5% FPR | checks per honest session |
|---|---|---|---|
| 0.01 | 1.00 | 1.00 | 1.95 |
| 0.02 | 1.00 | 1.00 | 1.90 |
| 0.05 | 1.00 | 1.00 | 1.88 |
| 0.1 | 1.00 | 1.00 | 1.70 |
| 0.2 | 1.00 | 1.00 | 1.88 |

Leave one check out (replay only; each check dropped in turn, canary first; same folds and split):

| dropped | policy | TPR at 5% FPR | TPR at 1% FPR | checks per honest session (5% / 1%) | real honest pause rate (5% / 1%) |
|---|---|---|---|---|---|
| canary | router | 1.00 [1.00, 1.00] | 0.95 [0.25, 1.00] | 5.22 / 5.22 | 0.0% / 0.0% |
| canary | best_single | 1.00 [1.00, 1.00] | 1.00 [0.80, 1.00] | 11.88 / 11.90 | 2.5% / 2.5% |
| canary | all_checks | 1.00 [1.00, 1.00] | 1.00 [0.50, 1.00] | 11.88 / 12.00 | 2.5% / 0.0% |
| decoy | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.15 / 1.15 | 0.0% / 0.0% |
| decoy | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| decoy | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.88 / 1.88 | 2.5% / 2.5% |
| opinion_trap | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 13.82 / 14.00 | 2.5% / 0.0% |
| style | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 / 1.00 | 0.0% / 0.0% |
| style | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| style | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 13.82 / 14.00 | 2.5% / 0.0% |

### Pipeline check: AI sessions per mode (the same in both runs)

| AI group | composite sessions | active checks per session | same-type | fallback | from the map | missing checks |
|---|---|---|---|---|---|---|
| gpt-mini:copy_paste:L0 | 5 | 9.6 | 9.6 | 0.0 | 0.0 | 0.0 |
| gpt-mini:doc_upload:L0 | 1 | 8.0 | 8.0 | 0.0 | 0.0 | 0.0 |

These counts cover the active checks served by the router. They were recomputed after 57957f8: the first version of the table also counted the passive style check on the 22 pilot items that aren't open text as "missing" (22 per session). Scores and TPR are unaffected, because a skipped passive check adds no evidence.

Keys (model, mode, level, sample) missing in a bank: 0 (kept and recorded in `missing_keys.csv`, never dropped).

Document upload: too few sessions for a TPR (it needs more than 5), so each session is described instead (router; score = highest running score, mean over the 4 fold maps; paused = share of fold maps under which it reaches the threshold):

| session | score | checks | paused at A_0.05 | paused at A_0.01 | paused at A_max |
|---|---|---|---|---|---|
| composite:gpt-mini:doc_upload:L0:0 | 5.32 | 8.0 | 1.00 | 1.00 | 1.00 |

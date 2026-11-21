# RQ2 multi-check router: pipeline check (6 AI sessions) and real run (L0, n = 40)

**This is a pipeline check, not the RQ2 result.** It runs the whole multi-check RQ2 pipeline end to end on the AI samples we had on Nov 18: 6 composite AI sessions, 5 copy-paste and 1 document upload. The RQ2 numbers come from the real run with 20 + 20 sessions, see "Real run (L0, n = 40)" below.

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
- **Behaviour has no AI rows, so the router doesn't use it.** Only checks with both AI and human rows are used. So the behaviour floor (#49: 750 ms + 50 ms, 53 pilot rows too_fast → normal) changes only human-side behaviour numbers, not any RQ2 result here.
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

---

# Real run (L0, n = 40)

This is the RQ2 run at L0, with **40 composite AI sessions: 20 copy-paste and 20 document upload**, all gpt-mini. These are Session C's samples (provenance in `docs/notes/rq2_samples_provenance.md` on `arav/first-sample`, f1410a0).
- **Copy-paste** sessions use samples 0–19 from the `-s5` and `-s5-19` runs (temperature 0.7).
- **Document upload** sessions use samples 0–19 from the **`-t07-s20` runs only**. The old temperature-0 upload runs are left out: they share sample 0 with the new runs, and the join refuses two answer sheets for one sample.
- **Upload rates are not directly comparable with RQ1.** RQ1's earlier document-upload numbers come from the temperature-0 runs, while these composite upload sessions are at temperature 0.7.

Everything else is as in the pipeline check above:
- the same 4 folds from `assign_session_folds()`;
- every mechanism with both AI and human rows: canary, decoy, opinion_trap with its contradiction rows (copy-paste 98, upload 9), and style;
- all four cap variants, with no_cap as an ablation only;
- leave-one-check-out with canary first;
- no model calls.

Outputs are in `results/real-router/rq2-l0-{all,no-low-effort}/` and are not committed. Both mode groups now have 20 sessions, so **TPR is reported per mode**: document upload has more than 5 sessions.

## What it shows

- **At L0 the router detects every AI session at about 2 checks per honest session.** TPR is 1.00 at 5% FPR, 1% FPR and A_max, in both modes, and no AI session is wrongly cleared. The best single check (canary) also reaches 1.00 with 2 checks. All checks and random check need 14 checks and still reach only 0.96–0.97, because document upload drops to 0.93–0.95. Showing every check adds human-looking evidence (e.g. decoys answered correctly) that pulls some upload sessions below a threshold the canary alone would reach. **RQ2's L0 result is efficiency, not extra detection:** at L0, canary alone separates the groups, as it did in the pipeline check.
- **The router's 5% and 1% thresholds are near zero** (0.012 and 0.501 with all 44 sessions; 0.14 for both without the low-effort sessions), for the same reason as above: most resampled honest sessions are cleared and score ≤ 0. **The real honest pause rate is above nominal at the 1% threshold:** 1 of 44 = 2.3%, and 1 of 40 = 2.5% without the low-effort sessions. Both are inside what a 44-session pilot can resolve, but neither meets 1%.
- **Every threshold is driven by fewer than 3 real honest sessions** (the output warns about this). The 1% tail is extrapolated from 44 sessions.
- **passive_group equals per_mech,** because behaviour has no AI rows. **open_text_group** (canary and style share one cap) needs fewer checks, about 1.35, at the same TPR.
- **Leave one check out: canary carries the L0 result.** Without canary, TPR falls for every policy.
  - At 5% FPR: router 0.93, best single 0.93, all checks 0.93.
  - At 1% FPR: router **0.77**, below best single (0.87) and about level with all checks (0.78). Without the low-effort sessions: 0.78 vs 0.86 and 0.78. With all 44 sessions the router does this with 5.3 checks per honest session instead of 11.8–12.0.
  - So without canary the router trades about 0.10 of TPR at 1% FPR for less than half the checks. That trade is the one to watch at L2/L2g, where the canary is likely to fail.
  - Dropping decoy, opinion_trap or style leaves the router at 1.00.
- **Budget curve:** at L0, beta barely matters. TPR is 1.00 for every beta from 0.01 to 0.2, at 1.8–2.0 checks per honest session.

Next: when Session B's L2 / L2g samples land, the same command runs again with them added. L2 and L2g go into separate sessions and groups (24cb0b5).

## The single false pause

At the router's 1% threshold, one real honest session is paused (1 of 44, and 1 of 40 without the low-effort sessions). It is **the same session in both runs**, so it is not one of the 4 low-effort sessions. Below are its outcomes and evidence (log-likelihood contributions, in nats) from the out-of-fold replay, up to the pause. Answer text and the session ID are left out.

**Per-check totals at the pause** (per_mech; the first number is with all 44 sessions, A_0.01 = 0.501; the second is without the low-effort sessions, A_0.01 = 0.140):

| check | outcomes before the pause | total evidence |
|---|---|---|
| canary | clean, clean (both open-text items) | **−2.99 / −2.99** (at the floor B after the first answer; the second adds 0) |
| style (passive) | mid, mid (both open-text answers) | **+2.69 / +2.96** |
| decoy | picked_correct ×5, then picked_decoy ×2 | **+1.28 / +1.08** (−0.15 / −0.19 per correct answer, +1.02 / +1.01 per decoy pick) |
| **score at the pause** | | **0.99 / 1.05** |

Order of events:
1. **The first clean canary answer sends the score straight to the floor B (−2.99), and the session is cleared.**
2. **The passive style check then re-opens it.** Style is observed on every open-text answer whatever check is served, and "mid" scores +1.35 per answer. This lifts the score above B, so the status goes back to active. **"Cleared" is not final**: the router recomputes the status after every update, and only "review" is sticky.
3. **The router serves checks again,** a second canary and then decoys. The canary is at its floor, so the second clean answer adds nothing.
4. **Five correct decoy answers barely move the score** (−0.15 each). **Two decoy picks** (+1.02 each) take it from −1.05 to +0.99, past the near-zero 1% threshold.

**Why it happens:**
- **The [B, +cap] bound limits the canary's human-side evidence to −2.99,** while style and decoy can each add up to +2.96. Two "mid" style scores almost cancel two clean canaries.
- **The 1% threshold is near zero (0.5 or less),** so two decoy picks are enough to pause.

**Under the other cap variants:**
- **open_text_group** (canary and style share one cap): **paused**, in both runs. Score 0.66 against A_0.01 = 0.321 with all 44 sessions, and 0.61 against 0.165 without the low-effort sessions. The shared group total stays near the floor, but the decoy evidence is unchanged.
- **no_cap** (ablation): **not paused**, in either run. Without the floor, the clean canaries give unbounded human-side evidence, so style and decoy can't pull the score back up.

**Effect on the reported rates:** honest-cleared and AI-wrongly-cleared count sessions whose replay **ends** cleared, before reaching the threshold. A session that is cleared, re-opened and ends active is counted as not cleared. The docstring of `Trace.cleared_at` ("clearing is final") is wrong for sessions with passive checks. It needs a fix, together with a decision on whether passive evidence should be allowed to re-open a cleared session.

### Real run: all 44 pilot sessions (44 honest sessions)

| policy | threshold | A | TPR [95% CI] | TPR copy-paste | TPR doc upload | nominal FPR | real honest pause rate | checks per honest session | honest cleared | AI wrongly cleared |
|---|---|---|---|---|---|---|---|---|---|---|
| router (per_mech) | A_0.05 | 0.012 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 1 of 44 = 2.3% | 1.95 [1.36, 2.73] | 90.9% | 0.0% |
| router (per_mech) | A_0.01 | 0.501 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 1 of 44 = 2.3% | 1.95 [1.36, 2.77] | 90.9% | 0.0% |
| router (per_mech) | A_max | 2.665 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 2.00 [1.36, 2.82] | 90.9% | 0.0% |
| router [passive_group] | A_0.05 | 0.012 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 1 of 44 = 2.3% | 1.95 [1.36, 2.73] | 90.9% | 0.0% |
| router [passive_group] | A_0.01 | 0.501 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 1 of 44 = 2.3% | 1.95 [1.36, 2.77] | 90.9% | 0.0% |
| router [passive_group] | A_max | 2.665 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 2.00 [1.36, 2.82] | 90.9% | 0.0% |
| router [open_text_group] | A_0.05 | 0.321 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 1 of 44 = 2.3% | 1.34 [1.09, 1.89] | 90.9% | 0.0% |
| router [open_text_group] | A_0.01 | 0.321 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 1 of 44 = 2.3% | 1.34 [1.09, 1.89] | 90.9% | 0.0% |
| router [open_text_group] | A_max | 1.319 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 1.36 [1.09, 1.98] | 90.9% | 0.0% |
| best single check | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 0 of 44 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 0 of 44 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_max | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| all checks | A_0.05 | 1.989 | 0.97 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.95 [0.80, 1.00] | 5.0% | 2 of 44 = 4.5% | 13.73 [13.45, 14.00] | 0.0% | 0.0% |
| all checks | A_0.01 | 2.960 | 0.97 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.80, 1.00] | 1.0% | 0 of 44 = 0.0% | 14.00 [13.72, 14.00] | 0.0% | 0.0% |
| all checks | A_max | 2.960 | 0.97 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.80, 1.00] | 0.0% | 0 of 44 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| random check | A_0.05 | 1.989 | 0.97 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.95 [0.80, 1.00] | 5.0% | 2 of 44 = 4.5% | 13.73 [13.45, 14.00] | 0.0% | 0.0% |
| random check | A_0.01 | 2.960 | 0.97 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.80, 1.00] | 1.0% | 0 of 44 = 0.0% | 14.00 [13.72, 14.00] | 0.0% | 0.0% |
| random check | A_max | 2.960 | 0.97 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.80, 1.00] | 0.0% | 0 of 44 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| router [no_cap] *(ablation)* | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 0 of 44 = 0.0% | 1.00 [1.00, 1.00] | 97.7% | 0.0% |
| router [no_cap] *(ablation)* | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 0 of 44 = 0.0% | 1.00 [1.00, 1.00] | 97.7% | 0.0% |
| router [no_cap] *(ablation)* | A_max | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 44 = 0.0% | 1.00 [1.00, 1.00] | 97.7% | 0.0% |

Budget curve (router at each beta; checks per honest session at the 1% threshold):

| beta | TPR at 1% FPR | TPR at 5% FPR | checks per honest session |
|---|---|---|---|
| 0.01 | 1.00 | 1.00 | 1.98 |
| 0.02 | 1.00 | 1.00 | 1.98 |
| 0.05 | 1.00 | 1.00 | 1.95 |
| 0.1 | 1.00 | 1.00 | 1.95 |
| 0.2 | 1.00 | 1.00 | 1.95 |

Leave one check out (replay only; each check dropped in turn, canary first; same folds and split):

| dropped | policy | TPR at 5% FPR | TPR at 1% FPR | checks per honest session (5% / 1%) | real honest pause rate (5% / 1%) |
|---|---|---|---|---|---|
| canary | router | 0.93 [0.77, 1.00] | 0.77 [0.43, 0.94] | 5.25 / 5.25 | 0.0% / 0.0% |
| canary | best_single | 0.93 [0.85, 1.00] | 0.87 [0.71, 0.97] | 11.82 / 11.91 | 4.5% / 2.3% |
| canary | all_checks | 0.93 [0.85, 1.00] | 0.78 [0.47, 0.97] | 11.82 / 12.00 | 4.5% / 0.0% |
| decoy | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.16 / 1.16 | 0.0% / 0.0% |
| decoy | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| decoy | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.95 / 1.95 | 2.3% / 2.3% |
| opinion_trap | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | all_checks | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 13.73 / 14.00 | 4.5% / 0.0% |
| style | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 / 1.00 | 0.0% / 0.0% |
| style | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| style | all_checks | 0.97 [0.90, 1.00] | 0.94 [0.87, 1.00] | 13.73 / 14.00 | 4.5% / 0.0% |

### Real run: without the 4 low-effort sessions (40 honest sessions)

| policy | threshold | A | TPR [95% CI] | TPR copy-paste | TPR doc upload | nominal FPR | real honest pause rate | checks per honest session | honest cleared | AI wrongly cleared |
|---|---|---|---|---|---|---|---|---|---|---|
| router (per_mech) | A_0.05 | 0.140 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 1.98 [1.30, 2.78] | 92.5% | 0.0% |
| router (per_mech) | A_0.01 | 0.140 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 1 of 40 = 2.5% | 1.98 [1.30, 2.78] | 92.5% | 0.0% |
| router (per_mech) | A_max | 2.934 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 2.02 [1.30, 2.83] | 92.5% | 0.0% |
| router [passive_group] | A_0.05 | 0.140 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 1.98 [1.30, 2.78] | 92.5% | 0.0% |
| router [passive_group] | A_0.01 | 0.140 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 1 of 40 = 2.5% | 1.98 [1.30, 2.78] | 92.5% | 0.0% |
| router [passive_group] | A_max | 2.934 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 2.02 [1.30, 2.83] | 92.5% | 0.0% |
| router [open_text_group] | A_0.05 | 0.165 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 1 of 40 = 2.5% | 1.35 [1.05, 1.75] | 92.5% | 0.0% |
| router [open_text_group] | A_0.01 | 0.165 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 1 of 40 = 2.5% | 1.35 [1.05, 1.75] | 92.5% | 0.0% |
| router [open_text_group] | A_max | 1.480 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 1.38 [1.05, 1.82] | 92.5% | 0.0% |
| best single check | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 0 of 40 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 0 of 40 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| best single check | A_max | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 2.00 [2.00, 2.00] | 0.0% | 0.0% |
| all checks | A_0.05 | 1.898 | 0.97 [0.93, 1.00] | 1.00 [1.00, 1.00] | 0.95 [0.85, 1.00] | 5.0% | 2 of 40 = 5.0% | 13.70 [13.40, 14.00] | 0.0% | 0.0% |
| all checks | A_0.01 | 2.960 | 0.96 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 1.0% | 0 of 40 = 0.0% | 14.00 [13.70, 14.00] | 0.0% | 0.0% |
| all checks | A_max | 2.960 | 0.96 [0.91, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.82, 1.00] | 0.0% | 0 of 40 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| random check | A_0.05 | 1.898 | 0.97 [0.93, 1.00] | 1.00 [1.00, 1.00] | 0.95 [0.85, 1.00] | 5.0% | 2 of 40 = 5.0% | 13.70 [13.40, 14.00] | 0.0% | 0.0% |
| random check | A_0.01 | 2.960 | 0.96 [0.90, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 1.0% | 0 of 40 = 0.0% | 14.00 [13.70, 14.00] | 0.0% | 0.0% |
| random check | A_max | 2.960 | 0.96 [0.91, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.82, 1.00] | 0.0% | 0 of 40 = 0.0% | 14.00 [14.00, 14.00] | 0.0% | 0.0% |
| router [no_cap] *(ablation)* | A_0.05 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 5.0% | 0 of 40 = 0.0% | 1.00 [1.00, 1.00] | 97.5% | 0.0% |
| router [no_cap] *(ablation)* | A_0.01 | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.0% | 0 of 40 = 0.0% | 1.00 [1.00, 1.00] | 97.5% | 0.0% |
| router [no_cap] *(ablation)* | A_max | 0.000 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.0% | 0 of 40 = 0.0% | 1.00 [1.00, 1.00] | 97.5% | 0.0% |

Budget curve (router at each beta; checks per honest session at the 1% threshold):

| beta | TPR at 1% FPR | TPR at 5% FPR | checks per honest session |
|---|---|---|---|
| 0.01 | 1.00 | 1.00 | 2.00 |
| 0.02 | 1.00 | 1.00 | 2.00 |
| 0.05 | 1.00 | 1.00 | 1.98 |
| 0.1 | 1.00 | 1.00 | 1.80 |
| 0.2 | 1.00 | 1.00 | 1.95 |

Leave one check out (replay only; each check dropped in turn, canary first; same folds and split):

| dropped | policy | TPR at 5% FPR | TPR at 1% FPR | checks per honest session (5% / 1%) | real honest pause rate (5% / 1%) |
|---|---|---|---|---|---|
| canary | router | 0.93 [0.81, 0.98] | 0.78 [0.46, 0.91] | 5.40 / 5.40 | 0.0% / 0.0% |
| canary | best_single | 0.93 [0.85, 0.98] | 0.86 [0.72, 0.97] | 11.80 / 11.90 | 5.0% / 2.5% |
| canary | all_checks | 0.93 [0.85, 0.98] | 0.78 [0.49, 0.96] | 11.88 / 12.00 | 2.5% / 0.0% |
| decoy | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.15 / 1.15 | 0.0% / 0.0% |
| decoy | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| decoy | all_checks | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.98 / 1.98 | 2.5% / 2.5% |
| opinion_trap | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| opinion_trap | all_checks | 0.97 [0.93, 1.00] | 0.96 [0.90, 1.00] | 13.70 / 14.00 | 5.0% / 0.0% |
| style | router | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 / 1.00 | 0.0% / 0.0% |
| style | best_single | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.00 / 2.00 | 0.0% / 0.0% |
| style | all_checks | 0.97 [0.90, 1.00] | 0.94 [0.87, 0.99] | 13.70 / 14.00 | 5.0% / 0.0% |

### Real run: AI sessions per mode (the same in both runs)

| AI group | composite sessions | active checks per session | same-type | fallback | from the map | missing checks |
|---|---|---|---|---|---|---|
| gpt-mini:copy_paste:L0 | 20 | 9.5 | 9.5 | 0.0 | 0.0 | 0.0 |
| gpt-mini:doc_upload:L0 | 20 | 8.8 | 8.8 | 0.0 | 0.0 | 0.0 |

Keys (model, mode, attacker variant, sample) missing in a bank: 0. Missing keys would be kept and recorded in `missing_keys.csv`, never dropped. Counts cover the active checks the router served. No AI group has a missing check, so the "missing checks count as misses" warning never fires.

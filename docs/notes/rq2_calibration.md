# Router calibration (step #42)

**Decision (Nov 20):** the router runs with a **per-mechanism cap** (each check adds at most 0.65 A), **clear_policy reopen** and **neg_floor_frac 1.0** (each check's negative evidence stops at B). The review threshold is the cross-fitted **A_0.05 (headline)**, with **A_0.01** as the secondary threshold. **no_cap** (one check may pause a session) and **twocheck** (neg_floor_frac 0.65: no single check may clear a session) are reported as ablations only.

The decision is recorded in `configs/data_decisions.json` under `router_calibration`. `tests/test_router_calibration.py` checks that `scripts/run_real_router.py` and `RouterConfig` run it by default.

- **Wald thresholds:** α = 0.01 and β = 0.05, so A = ln((1 − β) / α) = 4.554 and B = ln(β / (1 − α)) = −2.986. Each check's running total is bounded to [B, 0.65 A] = [−2.986, 2.960]: one check alone can never pause a session, but it can clear one.
- **Cleared means "no evidence of AI found":** the session gets no more checks and carries on. It is not a verdict that the person is human.
- **Threshold rule:** every honest pilot session is scored by session-level cross-fitting, with a failure map that never saw it (`adr/replay/crossfit.py`; the 4 folds come from `adr/replay/folds.py`). Resampled honest sessions are drawn per fold. A_0.05 and A_0.01 are the smallest thresholds that pause at most 5% and 1% of them. 95% CIs bootstrap over sessions.

## Data

These are the final numbers, from the floor750 runs:
- **Honest sessions:** the pilot `pilot_data/pilot-2026-11-13-floor750.parquet` (#49's behaviour floor; sha256 `f2602f19…`), all 44 sessions and separately the 40 left after removing the 4 low-effort ones.
- **AI sessions:** 120 composite gpt-mini sessions, 20 for each of L0, L2 and L2g × copy-paste and document upload.
- **Checks:** canary, decoy, opinion trap and style. Behaviour has no AI rows, so the router doesn't use it, and the floor750 runs are byte-identical to the runs on the original pilot file.
- **Runs:** `results/real-router/rq2-l012-floor750-{reopen,twocheck}-{all,no-low-effort}/` (not committed). The no_cap rows come from the reopen runs' cap-variant ablation. No model calls.

## Reopen vs twocheck vs no_cap, all 44 honest sessions

TPR per AI group (20 sessions each) at the 5% and 1% thresholds:

| design | threshold | A | L0 cp | L0 upload | L2 cp | L2 upload | L2g cp | L2g upload | pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| **reopen (decided)** | A_0.05 | 0.008 | 1.00 | 1.00 | 0.24 | 0.56 | 0.95 | 1.00 | 0.79 [0.74, 0.84] |
| **reopen (decided)** | A_0.01 | 0.951 | 1.00 | 1.00 | 0.23 | 0.55 | 0.95 | 1.00 | 0.79 [0.74, 0.84] |
| twocheck *(ablation)* | A_0.05 | 0.669 | 1.00 | 1.00 | 0.31 | 0.69 | 0.95 | 1.00 | 0.82 [0.75, 0.88] |
| twocheck *(ablation)* | A_0.01 | 1.617 | 1.00 | 1.00 | 0.23 | 0.55 | 0.95 | 1.00 | 0.79 [0.74, 0.84] |
| no_cap *(ablation)* | A_0.05 | 0.003 | 1.00 | 1.00 | 0.28 | 0.66 | 0.95 | 1.00 | 0.81 [0.75, 0.85] |
| no_cap *(ablation)* | A_0.01 | 1.288 | 1.00 | 1.00 | 0.20 | 0.55 | 0.95 | 1.00 | 0.78 [0.74, 0.84] |

AI sessions wrongly cleared (cleared = no evidence of AI found), per group. "Ever cleared" and "cleared at the end" are equal for every design here, so one number is shown:

| design | L0 cp | L0 upload | L2 cp | L2 upload | L2g cp | L2g upload |
|---|---|---|---|---|---|---|
| reopen | 0% | 0% | 60% | 25% | 5% | 0% |
| twocheck | 0% | 0% | 51% | 19% | 5% | 0% |
| no_cap | 0% | 0% | 60% | 25% | 5% | 0% |

Honest side:

| design | threshold | nominal FPR | real honest pause rate | checks per honest session | honest cleared |
|---|---|---|---|---|---|
| **reopen** | A_0.05 | 5% | 3 of 44 = 6.8% | 6.41 | 82% |
| **reopen** | A_0.01 | 1% | 2 of 44 = 4.5% | 6.57 | 82% |
| twocheck | A_0.05 | 5% | 5 of 44 = 11.4% | 6.57 | 77% |
| twocheck | A_0.01 | 1% | 2 of 44 = 4.5% | 6.77 | 77% |
| no_cap | A_0.05 | 5% | 5 of 44 = 11.4% | 6.41 | 82% |
| no_cap | A_0.01 | 1% | 1 of 44 = 2.3% | 6.70 | 82% |

## Without the 4 low-effort sessions (40 honest sessions)

| design | threshold | A | L2 cp | L2 upload | L2g cp | pooled | AI cleared (L2 cp / L2 upload / L2g cp) | real honest pause rate | checks per honest session |
|---|---|---|---|---|---|---|---|---|---|
| **reopen** | A_0.05 | 0.017 | 0.24 | 0.55 | 0.95 | 0.79 | 64% / 30% / 5% | 3 of 40 | 5.80 |
| **reopen** | A_0.01 | 0.541 | 0.23 | 0.55 | 0.95 | 0.79 | 64% / 30% / 5% | 2 of 40 | 5.92 |
| twocheck | A_0.05 | 0.488 | 0.28 | 0.62 | 0.95 | 0.81 | 51% / 20% / 5% | 5 of 40 | 7.05 |
| twocheck | A_0.01 | 1.234 | 0.23 | 0.55 | 0.95 | 0.79 | 51% / 20% / 5% | 2 of 40 | 7.28 |
| no_cap | A_0.05 | 0.017 | 0.24 | 0.59 | 0.95 | 0.80 | 64% / 30% / 5% | 4 of 40 | 5.80 |
| no_cap | A_0.01 | 0.740 | 0.21 | 0.55 | 0.95 | 0.79 | 64% / 30% / 5% | 2 of 40 | 5.97 |

The groups not shown (L0 copy-paste, L0 upload and L2g upload) are at TPR 1.00 for every design and threshold here, and none of their AI sessions is wrongly cleared.

## Why these settings

- **Two-check clearing buys nothing at the 1% threshold.** At A_0.01 its TPR is identical to reopen in every group, with and without the low-effort sessions. It costs more checks per honest session (6.77 vs 6.57; 7.28 vs 5.92 without the low-effort sessions) and clears fewer honest sessions (77% vs 82%). At A_0.05 it catches a few more L2 sessions (0.82 vs 0.79 pooled), but it pauses 5 honest sessions instead of 3.
- **Letting one check pause (no_cap) also changes nothing beyond noise.** At A_0.01 it pauses 1 honest session instead of 2 and catches one fewer L2 copy-paste session (0.20 vs 0.23). At A_0.05 it pauses 5 instead of 3. Its wrongly cleared rates are identical to reopen.
- **Ever cleared equals cleared at the end in every group,** so clear_policy sticky could not change any number here. It was not run.

## Clearing costs one AI session

Wrongly cleared AI sessions are, with one exception, sessions the router would not have paused anyway:

- **router_noclear** is the same router with clearing switched off, so it keeps checking every session. Its pooled TPR is 0.80, against the router's 0.79, in both runs and at both thresholds. It asks 13.8 checks per honest session instead of 6.6.
- **Session by session** (replay of the 120 AI sessions under each of the 4 fold maps, at each policy's own calibrated threshold; all 44 sessions, reopen):
  - The router clears 72 of the 480 session × fold-map pairs. router_noclear does not pause 68 of those either.
  - The other 4 are **one L2g copy-paste session**, cleared by the router under all 4 fold maps and paused by router_noclear. That is the whole 0.95 vs 1.00 gap in that group, and 0.79 vs 0.80 pooled.
  - The router never pauses a session that router_noclear misses.
  - The same pattern holds for twocheck and without the low-effort sessions: exactly 4 pairs, from the same group, each time.

So clearing (cleared = no evidence of AI found) halves the checks every honest person sees, for the cost of 1 AI session out of 120. The other wrongly cleared AI sessions, mostly L2, give no evidence of AI to any policy: L2's prompt names our checks and the attacker avoids them. That is an RQ4 finding about the attacker, not a cost of clearing.

## Error-analysis pointer: the one session clearing costs

**`composite:gpt-mini:copy_paste:L2g:0`** (composite AI session: copy-paste, generic-warning attacker L2g, sample 0). It is the only AI session the router clears and router_noclear pauses, and it happens under all 4 fold maps. The replay is all 44 honest sessions, reopen, with each policy at its own A_0.01 (router 0.951, router_noclear 0.951). Below are outcomes and log-likelihood contributions (nats) only, with no answer text.

**The session's own logged answers.** The pilot bank's checks are answered from the session's own rows on items of the same type:
- canary on its 4 open-text items: followed_instruction ×2, clean ×2;
- style: low ×4, mid ×3, high ×1.

**The router clears it after one open-text item** (pilot open-text item 2):

| fold map | canary | style | score at the clear (B = −2.986) |
|---|---|---|---|
| 0, 1 | clean −1.32 | high −2.28 | **−3.60** |
| 2, 3 | clean −1.32 | high −1.77 | **−3.09** |

Totals per check at the clear: canary −1.32, style −2.28 (maps 0 and 1) or −1.77 (maps 2 and 3). After that the router serves plain items only (cleared = no evidence of AI found).

**router_noclear sees the same first item, then keeps checking:**
1. Open-text item 1: canary **followed_instruction +3.93**, style low −0.12 to −0.17.
2. Decoys: picked_decoy +0.52 to +0.63 on every map, plus picked_correct +0.14 / +0.21 on maps 0 and 1.

It pauses at 0.96–1.22, just over its A_0.01 of 0.951. Totals at the pause:

| fold map | canary | style | decoy | score at the pause |
|---|---|---|---|---|
| 0 | +2.61 | −2.40 | +0.77 | 0.98 |
| 1 | +2.61 | −2.42 | +0.76 | 0.96 |
| 2 | +2.61 | −1.94 | +0.52 | 1.19 |
| 3 | +2.61 | −1.94 | +0.55 | 1.22 |

**For the error analysis:**
- **This is the router's clearing decision working as designed.** Two different checks agreed the first open-text answer looked human: the hidden instruction was ignored, and the style score was in the human-leaning bucket. So two-check clearing (twocheck) loses this session too.
- **The attacker followed the hidden instruction on only half of its answers.** Whether the router catches it depends on which item comes first.
- **The pause margin is thin.** router_noclear pauses it only just above its threshold (0.96 against 0.951 on map 1).
- **Style "high" counts as human-leaning evidence in these maps.** Its log-likelihood ratio is negative because the AI sessions rarely land in the high bucket. Worth checking against the style classifier's orientation (Archana).

## Caveats

- **The real honest pause rate is above nominal at the 1% threshold:** 2 of 44 (4.5%), 2 of 40 without the low-effort sessions. Fewer than 3 real honest sessions reach each threshold, so the 1% tail is extrapolated from 44 sessions.
- **One model:** all AI sessions are gpt-mini. The held-out model (claude-haiku) is used only in the final `--final` evaluation.

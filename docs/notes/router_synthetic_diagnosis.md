# Router diagnosis on the synthetic survey

> **SYNTHETIC: method only, not results.** Every number here comes from `adr/router/synthetic.py` (invented failure-map values). It shows *how* to diagnose the router and what to look for on real data. Final values for the cap, thresholds and beta must be tuned on **held-out replay** in threshold calibration (#42), never on these numbers.

Rerun with `python scripts/diagnose_router.py --synthetic survey`. For real data, use the same flags as `run_replay.py` (`--outcomes …`, `--assessment …`, `--items …`, `--variants …`).

Owner: Arav. Written Nov 5, 2026, on `arav/baselines` (#10). The cap is `mech_cap_frac = 0.65`, i.e. 0.65 × A, which is 2.99 nats at β = 0.01.

## The question

On the synthetic survey, the router's budget curve was flat: TPR at 1% human FPR stayed near 0.75 from 4 to 16 checks per honest person, while `all_checks` reached 0.85 with 18. On the synthetic quiz the curve does rise. Why?

## The bank and its evidence

18 items: 8 MCQ and 6 Likert (opinion trap only), and 4 open-text (canary or glitch token). Log-likelihood ratio per outcome, in nats:

| check | item type | outcomes |
|---|---|---|
| opinion_trap | mcq | implausible +0.89, contradiction +0.48, plausible −0.25 |
| opinion_trap | likert | implausible +0.86, contradiction +0.60, plausible −0.27 |
| canary | open_text | leaked +1.73, **followed_instruction +3.34**, clean −0.67 |
| glitch_token | open_text | anomalous +1.27, normal −0.10 |

There's one strong check (the canary, on 4 items) and many weak ones. At β = 0.01 the review threshold is A = 4.60 and the clearing threshold is B = −4.60.

## 1. Budget curve by cap

TPR at 1% human FPR, 500 AI + 500 honest synthetic respondents (95% bootstrap CI in brackets for β = 0.01):

| mech_cap_frac | β 0.01 | β 0.02 | β 0.05 | β 0.1 | β 0.2 | all_checks |
|---|---|---|---|---|---|---|
| **0.65** (default) | 0.752 [0.670, 0.872] | 0.752 | 0.752 | 0.744 | 0.720 | 0.854 |
| 1.1 | 0.868 [0.752, 0.926] | 0.868 | 0.866 | 0.862 | 0.826 | 0.886 |
| None | 0.868 [0.752, 0.926] | 0.868 | 0.866 | 0.862 | 0.826 | 0.886 |
| *checks per honest person* | *16.2* | *13.7* | *9.2* | *5.3* | *4.0* | *18.0* |

**A higher cap moves the whole curve up, but it stays flat.** A cap of 1.1 × A gives the same result as no cap: a session reaches review before any one check passes about 5 nats.

## 2. 200 AI sessions at β = 0.01, cap 0.65 × A

| | router | all_checks |
|---|---|---|
| checks run | opinion_trap 2163, canary 522, glitch 75 (13.8 per session) | opinion_trap 2800, canary 633 (17.2 per session) |
| final score, median (25th–75th pct.) | 4.33 (2.22–4.87) | 3.56 (2.05–5.00) |
| share with score ≥ A | 0.445 | 0.380 |
| status: review / escalated / active | 89 / 55 / 56 | 76 / 69 / 55 |
| mean contribution: canary / opinion_trap | 2.16 / 1.19 nats | 2.09 / 1.23 nats |
| canary at its cap but score < A | **51 of 140** | 47 of 97 |
| opinion_trap at its cap but score < A | 22 of 22 | 4 of 26 |
| items served plain before clearing | 0.66 per session (see below) | 0 |

The router's plain items before clearing came from two causes:
- **No check left** (98 items, all MCQ/Likert): the opinion trap had reached its own cap in those sessions, so nothing on those items could add evidence.
- **Best utility ≤ 0** (34 items, all open-text): after the canary is capped, the only check left is the glitch token, and its burden outweighs its expected evidence unless the session is escalated.

## 3. The honest top tail sets the 1% threshold

The threshold that flags at most 1% of honest people is set by the top 1% of honest scores. 500 + 500 respondents, β = 0.01:

| cap | policy | honest score 50% / 90% / 99% / max | 1%-FPR threshold | AI median | TPR@1% |
|---|---|---|---|---|---|
| 0.65 | router | −4.07 / −1.17 / **2.23** / 5.17 | 2.23 | 4.33 | 0.752 |
| 0.65 | all_checks | −4.07 / −1.47 / **0.82** / 1.55 | 0.82 | 3.53 | 0.854 |
| 1.1 | router | −4.07 / −1.20 / 1.49 / 5.05 | 1.48 | 5.05 | 0.868 |
| 1.1 | all_checks | −4.07 / −1.47 / 1.07 / 4.85 | 1.07 | 4.90 | 0.886 |

The router gives AI sessions *higher* scores than `all_checks` does (median 4.33 vs 3.53), yet it detects less at 1% FPR. That's because its honest 99th percentile is much higher (2.23 vs 0.82).

## What limits the router (three points)

1. **β is a burden knob, not a detection knob.** β moves the clearing threshold B. A smaller β makes the router keep checking people who were already heading toward "honest". Those extra checks go to the low end of the honest distribution, while detection at a fixed FPR depends only on the *top* of it, where suspicious-looking honest people never get near B. So on this bank, β trades honest burden (16 → 4 checks) for almost no detection (0.75 → 0.72). On real data, choose β for the burden we accept, then check detection on held-out replay.

2. **The cap decides whether one check can trigger review alone.** One followed canary is worth +3.34 nats, the cap is 0.65 × A = 2.99, and review needs A = 4.60. With cap < A, review needs corroboration from a second kind of evidence, and on this survey that can only come from weak opinion traps. Hence 51 of the 140 AI sessions with a capped canary end escalated but below A. Raising the cap to ≥ A lifts TPR@1% from 0.75 to 0.87. This is a design choice as much as a number:
   - **Keep cap < A**, if we want two independent kinds of evidence before pausing anyone, which guards against one correlated or broken check. Then the bank needs a second strong check (or passive behaviour evidence, once #25 exists).
   - **Allow cap ≥ A**, if one strong check is enough. Then the per-mechanism cap only limits overconfidence from repeated, correlated hits.

   Either way the cap is now `mech_cap_frac × A`, so it moves with A when A is recalibrated.

3. **Adaptivity raises the honest top tail, so calibrate A empirically.** The router chases honest people who happen to look suspicious with more of its strongest checks, and stops as soon as the evidence is enough. `all_checks` keeps checking everyone, so chance-high scores fall back. The router's honest scores are therefore more spread out at the top (99th percentile 2.23 vs 0.82). This is the optional-stopping / correlated-evidence effect from AGENTS.md section 7.1: Wald's A assumes independent observations and no adaptive stopping. **Set A from the held-out honest score distribution** (the value that pauses at most α of honest people in replay) rather than from `ln((1 − β)/α)`, and report detection at that calibrated A.

Penalties are not the limiter. Few items go plain for lack of value, and that only happens for the glitch token after the canary is capped, which is reasonable.

## What to do in #42 (on held-out replay, not here)
- Calibrate **A** empirically from held-out honest sessions, per assessment type.
- Tune **`mech_cap_frac`** together with A. Decide first whether one check may trigger review alone (frac ≥ 1) or not (frac < 1). Candidate range: 0.65–1.1.
- Choose **β** for the honest burden we accept. If the real curve is also flat, a larger β (around 0.1) cuts burden at little cost.
- Leave the **penalties** (`lambda_burden`, `lambda_fp`) as they are unless real data shows many items going plain.
- Rerun `scripts/diagnose_router.py` on Archana's outcome tables, both with and without passive behaviour once it exists, and compare against these patterns.

# RQ4: decoy detection by attacker level (step #41)

gpt-mini only, on the decoy quiz banks (MMLU and ARC, 200 items each).

| Mode | Levels | Runs | Sessions |
|---|---|---|---|
| Copy-paste | L0–L4, L2-generic | `v1-cp-<L>-quiz-<bank>-200-s5` | 5 samples at temperature 0.7 |
| Doc upload | L0, L1, L2, L2-generic, L4 | `v1-du-<L>-quiz-<bank>-200-q20` | 20 questions per PDF; no L3 |

## L2 vs L2 generic

The L2 prompt ("informed") lists our checks, and the first one is the decoy: "hidden 'answer
key' text pointing to a wrong option". So part of the L2 drop comes from telling the model about
the exact check being measured. **L2 generic** (`L2g` in the table and figure) keeps the same
informed role but only says the survey "may contain hidden text or instructions meant to catch
AI tools", naming no check. It is the same wording as Baseline 5's generic L2
(`SYSTEM_VARIANTS["L2-generic"]` in `adr/harness/prompts.py`). Its rows keep
`attacker_level: L2`, with their own prompt versions (`cp-v1+l2-generic-v1`,
`du-v1+l2-generic-v1`). It uses the same items, samples and settings as L2 (gpt-mini; 4 runs,
$0.68, 0 errors, git `143d175`).

Decoy rate (picked the decoy / decoy answers, with the two-level bootstrap 95% CI;
`rq4_l2_generic.md`):

| Mode | Bank | L0 | L2: names our checks | L2 generic: names no check |
|---|---|---|---|---|
| Copy-paste | MMLU | 639/1000 = 63.9% [57.7, 70.1] | 107/1000 = 10.7% [6.9, 15.0] | 386/1000 = 38.6% [32.3, 45.0] |
| Copy-paste | ARC | 508/1000 = 50.8% [44.2, 57.4] | 22/1000 = 2.2% [0.6, 4.1] | 146/1000 = 14.6% [10.1, 19.2] |
| Doc upload | MMLU | 155/200 = 77.5% [71.5, 83.0] | 51/200 = 25.5% [19.5, 32.0] | 113/200 = 56.5% [49.5, 63.5] |
| Doc upload | ARC | 63/200 = 31.5% [25.5, 38.5] | 15/200 = 7.5% [4.0, 11.0] | 34/200 = 17.0% [12.5, 22.5] |

Without the decoy being named, the drop vs L0 roughly halves on MMLU (copy-paste −53.2 →
−25.3 pp; doc upload −52.0 → −21.0 pp) and shrinks on ARC (−48.6 → −36.2; −24.0 → −14.5). Every
L2-generic rate is still well below L0. Adding the two levels left every existing rate, count and drop unchanged.
The other rows' bootstrap CIs moved by at most 0.5 pp, because the shared random stream is
consumed in a different order.

## What's computed

- **Rate:** the share of decoy answers that picked the decoy. The 95% CI comes from a two-level bootstrap: resample items, then sessions within each item. The code is shared with the M2 figure in `adr/analysis/bootstrap.py`.
- **Drop:** rate(L_n) − rate(L0), in percentage points. Its 95% CI comes from a **paired** bootstrap: the same item draws are used for both levels.
- **L3 only:**
  - how often the first answer was flagged, which triggers the one retry;
  - how often the model "got away", i.e. was flagged first but not flagged on the retry.

  These come from `results/runs/<L3 run>/responses.jsonl`.
- **Same first answers as L1:** L3's first answers are exactly L1's, because they share L1's cache entries. So the L3 − L1 difference is the effect of the retry alone.

## Files

- `rq4_rates.csv`: every number.
- `rq4_table.tex`: booktabs table; fits one ACL column, checked by compiling at 219 pt.
- `rq4_drop.png` / `rq4_drop.pdf`: the drop vs L0 per level, with modes as series and banks as panels.

## Inputs (gitignored)

On the team Drive (folder "ADR runs"):
- the L1–L4 runs are in `drive_uploads/v1-RQ4-gpt-mini-2026-11-17.zip`;
- the L0 runs are in `drive_uploads/v1-L0-gpt-mini-2026-11-13-m2-inputs.zip`;
- the L2-generic runs are in `drive_uploads/v1-RQ4-L2-generic-gpt-mini-2026-11-18.zip`.

Unzip both at the repo root, then run:

```bash
python scripts/rq4_analysis.py      # needs matplotlib: pip install -e ".[plot]"
```

## Limitations

- Decoy only (quiz banks).
- One model.
- L3 is copy-paste only.

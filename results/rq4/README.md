# RQ4: decoy detection by attacker level (step #41)

gpt-mini only, on the decoy quiz banks (MMLU and ARC, 200 items each).

| Mode | Levels | Runs | Sessions |
|---|---|---|---|
| Copy-paste | L0–L4 | `v1-cp-<L>-quiz-<bank>-200-s5` | 5 samples at temperature 0.7 |
| Doc upload | L0, L1, L2, L4 | `v1-du-<L>-quiz-<bank>-200-q20` | 20 questions per PDF; no L3 |

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
- the L0 runs are in `drive_uploads/v1-L0-gpt-mini-2026-11-13-m2-inputs.zip`.

Unzip both at the repo root, then run:

```bash
python scripts/rq4_analysis.py      # needs matplotlib: pip install -e ".[plot]"
```

## Limitations

- Decoy only (quiz banks).
- One model.
- L3 is copy-paste only.

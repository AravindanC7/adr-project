# M2 decoy-rate figure and table

How often gpt-mini (attacker level L0) picks the planted decoy, for each attack mode and question bank. Only the repeated runs are used:

| Cell | Run id | Sessions |
|---|---|---|
| Copy-paste, MMLU / ARC | `v1-cp-L0-quiz-{mmlu,arc}-200-s5` | 5 samples at temperature 0.7 |
| Doc upload, MMLU / ARC | `v1-du-L0-quiz-{mmlu,arc}-200-q20` | 20 questions per PDF |

The 95% CIs come from a two-level bootstrap: resample items, then resample sessions within each sampled item. It uses 2000 replicates with seed 576.

Each cell is also split by `knows`. An item counts as known if gpt-mini answered its **plain** version correctly in the **temperature-0 copy-paste run** (`v1-cp-L0-<bank>`). The same per-item split is used for the doc-upload cells. Answers of `None` (a failed call or an unparseable reply) are skipped when deciding `knows`; the gpt-mini runs had none for plain variants. An empty group would show as NaN in the CSV and "n/a" in the table and plot.

## Files

- `decoy_rates.csv`: one row per mode × bank × group, with counts, rate, CI.
- `decoy_rates.png` / `decoy_rates.pdf`: grouped bar chart, one ACL column wide.
- `decoy_table.tex`: 4-row booktabs table; needs `\usepackage{booktabs}`.

## Inputs (gitignored, so not in the repo)

All inputs are in **`drive_uploads/v1-L0-gpt-mini-2026-11-13-m2-inputs.zip`** on the shared drive:

- `results/outcomes/`: the 8 gpt-mini outcome files (temperature 0, `-s5`, doc upload, `-q20`).
- `results/runs/`: the same 8 run folders (`config.yaml`, `responses.jsonl`, `errors.jsonl`; doc PDFs left out).
- `results/runs/v1-item-difficulty.csv`.
- `data/processed/runs_v1/items.jsonl` and `assessments/quiz-{mmlu,arc}-200.json`. These can also be rebuilt offline with `scripts/build_runs_v1.py`.

The two older zips, `v1-L0-gpt-mini-2026-11-13.zip` and `v1-L0-gpt-mini-2026-11-13-q20-s5.zip`, are not enough on their own: the second one is missing the `-s5` runs, the ARC `-q20` run and the `-s5`/`-q20` outcome files.

## Regenerate

From the repo root:

```bash
unzip /path/to/v1-L0-gpt-mini-2026-11-13-m2-inputs.zip -d .
python scripts/item_difficulty.py      # writes results/runs/v1-item-difficulty.csv, prints None counts
python scripts/m2_decoy_figure.py      # writes results/m2/decoy_rates.{csv,png,pdf} and decoy_table.tex
```

The second script needs matplotlib (`pip install -e ".[plot]"`). Its output CSV and `.tex` are byte-identical to the committed ones. The `.pdf` embeds a creation date, so it differs byte-wise.

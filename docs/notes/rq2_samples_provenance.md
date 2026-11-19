# RQ2 extra AI samples: provenance (2026-11-18)

Extra gpt-mini L0 samples so RQ2's composite sessions (`adr/replay/composite.py`) have 20 samples
per bank. Model: gpt-mini only (never claude-haiku, the held-out model). No OSMI item or
held-out-split item: the survey driver refuses them, and the quiz banks contain none.

## New run ids

All are under `results/runs/<run_id>/` and `results/outcomes/<run_id>.parquet` in the main
checkout, next to the existing samples. No existing run, outcome file or cache entry was changed.

Copy-paste, samples 5–19 (15 samples; temperature 0.7, the existing `-s5` settings). These add
to the existing `-s5` runs (samples 0–4), giving samples 0–19 per bank:

| run id | bank | driver |
|---|---|---|
| `v1-cp-L0-quiz-mmlu-200-s5-19` | quiz-mmlu-200 | `scripts/run_copy_paste.py --samples 15 --first-sample 5 --temperature 0.7` |
| `v1-cp-L0-quiz-arc-200-s5-19` | quiz-arc-200 | same |
| `v1-cp-L0-survey-ipip-s5-19` | survey-ipip | `scripts/run_survey_banks.py run --mode cp --samples 15 --first-sample 5` |
| `v1-cp-L0-survey-gss-s5-19` | survey-gss | same |
| `v1-cp-L0-assignment-asurre-train-s5-19` | assignment-asurre-train | same |

Document upload, samples 0–19 at **temperature 0.7**: a new, separate setting. The existing
upload runs are 1 sample at temperature 0, are kept exactly as they are (the M2 figures, RQ1 and
RQ4 use them), and are **not** part of these sessions. RQ2's composite upload sessions use only
the runs below, so all 20 samples come from one setting. Do not pass these together with the
temperature-0 upload runs to the join: both have sample 0 of the same bank, which the join
rejects as two answer sheets.

| run id | bank | settings |
|---|---|---|
| `v1-du-L0-quiz-mmlu-200-q20-t07-s20` | quiz-mmlu-200 | `run_doc_upload.py --samples 20 --temperature 0.7 --max-tokens 3000 --questions-per-doc 20` |
| `v1-du-L0-quiz-arc-200-q20-t07-s20` | quiz-arc-200 | same |
| `v1-du-L0-survey-ipip-q20-t07-s20` | survey-ipip | `run_survey_banks.py run --mode du --samples 20 --temperature 0.7` (20 questions per PDF, 3000 tokens) |
| `v1-du-L0-survey-gss-t07-s20` | survey-gss | same (whole bank per PDF, 3000 tokens) |
| `v1-du-L0-assignment-asurre-train-t07-s20` | assignment-asurre-train | same (whole bank per PDF, 6000 tokens) |

The quiz upload runs use 20 questions per PDF, as the existing `-q20` runs used by the M2
figures do.

## Code they ran from

Commit **`faa5262`** on `arav/first-sample`. Its parent `12a70d2` is a local merge of
`origin/main` (56452b7) and `origin/arav/survey-runs` (0fe82d1, PR #42). faa5262 adds
`CopyPasteConfig.first_sample`, `run_copy_paste.py --first-sample`, and the
`run_survey_banks.py run --run-id/--samples/--first-sample/--temperature` and
`rescore --runs` options. Defaults are unchanged.

The code was committed **while the runs were going**, without editing anything first. At commit
time, `git diff faa5262` against the running worktree was empty, and the three changed files had
the same sha256 before and after the commit.

**What `config.yaml` records, and why it is not faa5262.** The harness's `git_commit` is
`git rev-parse --short HEAD` in the *current directory*. The runs were launched from the main
checkout (so they write next to the existing samples) using the worktree's scripts. So every
new `config.yaml` has `git_commit: 0fe82d1` (inferred from the code while the runs were going;
checked against the finished configs in the follow-up commit), which is the main checkout's HEAD (`arav/survey-runs`) and
not the code that ran. There is **no dirty flag** in the configs: the harness doesn't record
one. Read `git_commit` in these ten configs as "the checkout the runs were launched from", and
take the code from faa5262.

## Reproduction check before spending

Before any paid call, the existing samples were re-run with the same drivers and settings, from
a scratch directory linked to the real data and LLM cache, with **no API key in the environment
and no `.env` on the path**. Any cache miss would therefore have failed instead of calling the API.

- Quiz copy-paste `-s5` (MMLU, ARC; 2000 rows each): all calls hit the cache. Cache keys and
  answers are identical to the existing runs.
- Quiz upload `-q20` (MMLU, ARC; 400 rows each): identical.
- Assignment copy-paste `-s5` (130 responses): identical.
- Survey ipip and gss copy-paste `-s5`: all calls hit the cache (same keys), but 50 ipip rows
  and 1 gss row get a different answer from the cache than the run recorded. This is the cache
  race below, and it's why the new copy-paste samples are separate `-s5-19` runs (via
  `first_sample`) rather than an in-place 20-sample rerun, which would have rewritten those rows.

The new options were also checked with the fake model: `--first-sample 5 --samples 15` gives
sessions `:5`…`:19`, and the upload run with `--samples 20 --temperature 0.7` gives `:0`…`:19`.

Cost was estimated from the `cost_usd` of the existing runs' cache entries, with every response
row priced as its own call so the race's duplicate calls are included. Upper bound: $3.51 for
all ten runs (cap $5).

## The plain / opinion-trap cache race (existing ipip and gss runs)

An opinion trap changes nothing visible. So in survey-ipip and survey-gss, the plain and
opinion-trap variants of an item send the same prompt and share one cache key per (item, sample).
The answers recorded in the cache are per call, not per variant. With 4 workers, both variants
of a pair were often sent before either answer was cached. Both calls went to the API, each row
kept its own call's answer, and the cache kept whichever was written last.

No row holds an answer to a different prompt: every shared key is exactly one item's plain +
opinion-trap pair in one sample. The effects are:

| run id | rows | keys shared (plain+trap) | keys called twice | rows whose answer ≠ cache |
|---|---|---|---|---|
| `v1-cp-L0-survey-ipip-s5` | 500 | 250 | 121 | 50 (48 plain, 2 opinion-trap) |
| `v1-cp-L0-survey-gss-s5` | 80 | 40 | 22 | 1 (plain) |
| `v1-du-L0-survey-ipip-q20` | 100 | 3 (per PDF) | 3 | 0: plain and trap PDFs answered every question the same (temperature 0) |
| `v1-du-L0-survey-gss` | 16 | 1 (per PDF) | 1 | 0: same |

Not affected: the quiz banks (a decoy changes the prompt) and assignment-asurre-train (a canary
changes the prompt). No key is shared there.

Consequences:

- Those 51 copy-paste rows cannot be reproduced from the cache. Rerunning would replace them
  with the sibling variant's answer. They are correct records of real calls.
- Within one sample, an item's plain and opinion-trap rows are sometimes the same draw and
  sometimes two independent draws (ipip: 129 shared, 121 independent). Only the opinion-trap
  row is scored, so outcome rates are unaffected. Pairwise plain-vs-trap comparisons inside a
  sample should not assume either.
- The new runs used the same 4-worker settings, so `v1-cp-L0-survey-{ipip,gss}-s5-19` and the
  `-t07-s20` survey upload runs show the same behaviour.

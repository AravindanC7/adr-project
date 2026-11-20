# RQ2 L2 / L2-generic AI samples: provenance (2026-11-19)

gpt-mini at attacker level L2, with two system prompts: the mechanism-specific L2 prompt and the
generic `L2-generic` variant (`--system-variant L2-generic`). Twenty samples per bank in
copy-paste and in document upload. Model: gpt-mini only (never claude-haiku, the held-out model).
No OSMI item or held-out-split item: the survey driver refuses them, and the quiz banks contain
none. This mirrors the L0 extra samples (`docs/notes/rq2_samples_provenance.md`).

## Run ids (20)

All are under `results/runs/<run_id>/` and `results/outcomes/<run_id>.parquet` in the
`adr-rq2-l2` worktree.

Copy-paste, samples 0–19, temperature 0.7:

| run id | bank |
|---|---|
| `v1-cp-L2-quiz-mmlu-200-s20` | quiz-mmlu-200 |
| `v1-cp-L2-quiz-arc-200-s20` | quiz-arc-200 |
| `v1-cp-L2-survey-ipip-s20` | survey-ipip |
| `v1-cp-L2-survey-gss-s20` | survey-gss |
| `v1-cp-L2-assignment-asurre-train-s20` | assignment-asurre-train |
| `v1-cp-L2-generic-quiz-mmlu-200-s20` | quiz-mmlu-200 |
| `v1-cp-L2-generic-quiz-arc-200-s20` | quiz-arc-200 |
| `v1-cp-L2-generic-survey-ipip-s20` | survey-ipip |
| `v1-cp-L2-generic-survey-gss-s20` | survey-gss |
| `v1-cp-L2-generic-assignment-asurre-train-s20` | assignment-asurre-train |

Document upload, samples 0–19, temperature 0.7 (the same setting as the L0 `-t07-s20` runs; the
quiz and ipip banks use 20 questions per PDF):

| run id | bank |
|---|---|
| `v1-du-L2-quiz-mmlu-200-q20-t07-s20` | quiz-mmlu-200 |
| `v1-du-L2-quiz-arc-200-q20-t07-s20` | quiz-arc-200 |
| `v1-du-L2-survey-ipip-q20-t07-s20` | survey-ipip |
| `v1-du-L2-survey-gss-t07-s20` | survey-gss |
| `v1-du-L2-assignment-asurre-train-t07-s20` | assignment-asurre-train |
| `v1-du-L2-generic-quiz-mmlu-200-q20-t07-s20` | quiz-mmlu-200 |
| `v1-du-L2-generic-quiz-arc-200-q20-t07-s20` | quiz-arc-200 |
| `v1-du-L2-generic-survey-ipip-q20-t07-s20` | survey-ipip |
| `v1-du-L2-generic-survey-gss-t07-s20` | survey-gss |
| `v1-du-L2-generic-assignment-asurre-train-t07-s20` | assignment-asurre-train |

All 20 runs: 0 errors, 0 truncated replies, no spend-cap stop, `models: [gpt-mini]`.

**Quiz copy-paste samples 0–4 repeat earlier runs.** In the four quiz copy-paste `-s20` runs,
samples 0–4 are cache hits on the 17 Nov runs `v1-cp-{L2,L2-generic}-quiz-{mmlu,arc}-200-s5`
(2000/2000 identical cache keys in each). They are the same answers, not new draws, so pass
either the `-s5` run or the `-s20` run to the composite join, never both. The join would reject
two answer sheets for one sample.

## Code they ran from

Commit **`c9bd3d4`** on `arav/rq2-l2-runs` (parents: 263872a, the generic L2 variant in the
harness; then 599146b and faa5262 from `arav/first-sample`, #45). Every config records
`git_commit: c9bd3d4`. The runs were started in this worktree, whose tracked files match c9bd3d4:
`git status` shows only untracked outputs (`results/rq2_l2/`, `scripts/report_l2_runs.py`).
The harness records no dirty flag. Calls went through the shared cache `adr-project/cache/llm`. The
bank data in this worktree (`data/processed/archana-survey-written-v1/`) and
`data/processed/runs_v1/style_profile.json` are byte-identical to the main checkout's.

## Re-scoring (checked 2026-11-19)

The 12 survey and assignment runs are re-scored with the shared IPIP pair rule
(`score_reverse_pairs v1`) and the canonical style profile. The profile's file sha256
`1576c047…` and profile-object sha256 `6b43d6ed…` match `configs/data_decisions.json`
`style_profile`, and its sources exclude OSMI. Checked by re-running
`run_survey_banks.py rescore --runs …` from each run's `outcomes.harness.parquet` in a scratch
copy (no API calls). All 12 results are identical to the outcome files on disk, so nothing was
re-scored again. The 8 quiz runs have no opinion-trap or style rows, so they need no re-score.

| run id | harness rows → rows | pair-rule changes |
|---|---|---|
| `v1-cp-L2-survey-ipip-s20` | 2000 → 1973 | 113 contradiction, 27 no row |
| `v1-cp-L2-generic-survey-ipip-s20` | 2000 → 1962 | 128 contradiction, 38 no row |
| `v1-du-L2-survey-ipip-q20-t07-s20` | 2000 → 1912 | 88 no row |
| `v1-du-L2-generic-survey-ipip-q20-t07-s20` | 2000 → 1954 | 3 contradiction, 46 no row |
| gss (4 runs) and assignment (4 runs) | unchanged | none; assignment style rows already used the canonical profile |

## Actual cost (this job only): $8.57

From the spend ledger `adr-project/cache/llm/_spend_ledger.jsonl`. The ledger has no run id, so
this job's rows were found by matching each ledger row's (timestamp, model, input tokens, output
tokens) to the cache entries the 20 runs reference that were created after the job's code
existed (263872a, 2026-11-19T00:31:58Z). Matching kept multiplicities, because many calls share a
one-second timestamp and token counts.

| part | ledger rows | USD |
|---|---|---|
| cache entries referenced by live rows of the 20 runs | 24,846 | 7.8120 |
| cache entries this job created that the final runs only replay as cache hits (paid by an earlier attempt of the same runs; inferred, as no log of that attempt survives) | 4,354 | 0.7024 |
| race duplicates whose cache entry was overwritten, attributable to this job | 235 | 0.0521 |
| race duplicates during the overlap with the L0 extra-sample job, either job | 77 | ≤ 0.0080 |
| credit probe (`rq2-l2-logs/probe-cache`) | 1 | 0.0000 |
| **total** | | **8.566–8.574** |

Window: 2026-11-19T00:34:40Z to 07:37:10Z. It overlaps the L0 extra-sample job until 01:23:44Z.
Those rows ($2.71 in the window) are attributed to the L0 job and excluded here. Not counted:
8,000 cache entries the quiz copy-paste runs reuse from the 17 Nov `-s5` runs (paid then).

## Known issues

1. **OpenAI's "no credits remaining" 429 was retried silently as a rate limit.**
   `classify_error` (`adr/harness/llm_client.py`) returns `rate_limit` for every HTTP 429
   before it looks for `insufficient_quota` or `billing`. Only a message containing
   "credit balance" (or an invalid-key phrase) is fatal on a 429. OpenAI's out-of-credit 429 was
   therefore backed off and retried, without stopping the run or being logged as an error. The
   final runs have no errors and no missing rows. The 4,354 entries paid earlier and only
   replayed in the final runs (above) are consistent with runs that were stopped and restarted
   around this, but no log records it. Fix to make: treat a 429 whose message reports no credits /
   insufficient quota as fatal.
2. **The spend check read the whole week's ledger.** `LLMClient.week_spend_usd()` sums every
   ledger row in the current ISO week, from all jobs sharing the cache. It isn't per job. The
   weekly cap therefore counted the overlapping L0 job's spend (and anything else that week)
   against this job, and a job's own cost can't be read from the cap check. It has to be
   reconstructed as above.

## The plain / opinion-trap cache race

As in the L0 note: in survey-ipip and survey-gss, the plain and opinion-trap variants of an item
send the same prompt (an opinion trap changes nothing visible; the L2 system prompt is the same
for both), so they share one cache key per (item, sample). With 4 workers, both were often sent
before either answer was cached. Each row kept its own call's answer, and the cache kept the
last one written. No row holds an answer to a different prompt: every shared key is exactly one
item's plain + opinion-trap pair in one sample.

| run id | rows | keys shared (plain+trap) | keys called twice | rows whose answer ≠ cache |
|---|---|---|---|---|
| `v1-cp-L2-survey-ipip-s20` | 2000 | 1000 | 129 | 17 (plain) |
| `v1-cp-L2-generic-survey-ipip-s20` | 2000 | 1000 | 125 | 30 (plain) |
| `v1-cp-L2-survey-gss-s20` | 320 | 160 | 21 | 3 (plain) |
| `v1-cp-L2-generic-survey-gss-s20` | 320 | 160 | 18 | 0 |
| `v1-du-L2-survey-ipip-q20-t07-s20` | 2000 | 60 (per PDF) | 8 | n/a; 33 question answers differ between the two PDFs |
| `v1-du-L2-generic-survey-ipip-q20-t07-s20` | 2000 | 60 | 8 | n/a; 23 differ |
| `v1-du-L2-survey-gss-t07-s20` | 320 | 20 | 3 | n/a; 2 differ |
| `v1-du-L2-generic-survey-gss-t07-s20` | 320 | 20 | 0 | 0 |

Unlike the temperature-0 L0 upload runs, these uploads are at temperature 0.7. So when both PDFs
of a pair were called live, their answers can differ. Each row still holds its own PDF's reply.
Only the opinion-trap rows are scored, so outcome rates are unaffected, but those rows cannot all
be reproduced from the cache. The quiz banks (decoy) and assignment-asurre-train (canary) change
the prompt and share no keys.

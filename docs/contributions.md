# Contribution log

Shared record of who did what, by step (AGENTS.md Section 9). The mid-term and final reports
use it for the "contribution of each member" section, so keep it current. Update it at the
**Sunday sync** (AGENTS.md Section 11).

**How to add an entry:** one row per meaningful contribution (not every commit), newest at the
bottom of your own block or of the table. Columns: date (YYYY-MM-DD), who, step #, what, link
(PR number or commit hash). Add non-code work too: reviews, mentor meetings, emails, writing,
data checks. Those are **not** in git, so each person adds their own.

The rows below were filled from `git log --all` on 2026-11-09 (merge commits skipped).
"PR #n" means the GitHub pull request into `main`; "branch" means the work was not yet in
`origin/main` at that time. A step number marked `~` was inferred from the commit subject;
owners please correct it if wrong.

| Date | Who | Step | What | Link |
|---|---|---|---|---|
| 2026-10-26 | Arav | #2, #7, #9, #13 | Initial scaffold: contracts, cached LLM client, copy-paste harness, router prototype | `1b149f0` |
| 2026-10-26 | Arav | #9 | Harness: handle temperature refusals and rate limits, log failing calls, team model list | `4052b48` |
| 2026-10-28 | Harshini | #11 | Front-end skeleton that renders any Item | PR #1 (`bd8dbbe`) |
| 2026-10-29 | Archana | #8, #10 | OpinionQA, MMLU and ARC loaders with initial detection mechanisms | PR #3 (`ada048f`) |
| 2026-10-29 | Arav | #20 | Document-upload attacker mode | PR #4 (`9df72e2`) |
| 2026-10-29 – 11-05 | Archana | #18 | GSS, IPIP and ASURRE survey loaders; corrected IPIP reverse-key pairs | PR #8 (`f1fc74b`), `71d8b9c` |
| 2026-10-29 – 11-05 | Archana | #21 | Canary and writing-style signals (hidden position recorded, hardened) | PR #7 (`541eeae`) |
| 2026-11-02 | Arav | #7 | Contracts: document that variant ids and router state stay server-side | PR #2 (`8cf2fb7`) |
| 2026-11-02 – 11-03 | Arav | ~#6, #7 | Contracts v2: test/assignment types, passage, time limit, hidden_position, JSONL io, NoObservation, strict Response.answer | PR #5 (`0e3aad5`) |
| 2026-11-03 | Archana | #8, #10 | Addressed loader and mechanism reviews | `77d6ccd` |
| 2026-11-03 | Harshini | #11 | Addressed PR #1 review: ids, check names and router state kept off the browser | `a5741ca` |
| 2026-11-03 – 11-04 | Harshini | #19 | Mechanism injection: decoys and canaries in the page, browser test that copied text equals copy_text, PR #6 review fixes | PR #6 (`a60d42f`) |
| 2026-11-03 – 11-05 | Harshini | #23 | Per-question telemetry; all nine fields after contracts v2 | branch `harshini/telemetry` (`88c9012`, `060dbfd`) |
| 2026-11-03 | Harshini | #30 | Pilot materials: consent screen, recruitment messages, session script, debrief | branch `harshini/pilot` (`98f28e1`) |
| 2026-11-05 | Harshini | #25 | Behaviour check: telemetry to paste / focus_loss / too_fast / normal; review fixes | branch `harshini/behaviour` (`4e92997`, `6434b93`) |
| 2026-11-05 | Arav | #27 | Replay simulator; calibration split by default, held-out only with `--final` | branch `arav/replay` (`84d3142`, `0f93cd8`) |
| 2026-11-05 | Arav | #32 | Baselines 1–4 (best single, all checks, fixed plan, random), compare mode, budget curve | branch `arav/baselines` (`9403203`, `9e184c0`, `409ef0b`) |
| 2026-11-05 | Arav | ~#26, #42 | Router: `update(outcome=None)`; per-mechanism cap as a fraction of A, diagnosis script and notes | branch `arav/baselines` (`ce06330`, `b597d66`, `054aa56`) |
| 2026-11-05 | Arav | #22 | Browser-agent attacker; UTF-8 fixes for Windows | branch `arav/browser-agent` (`b791442`, `a1bab70`, `0ac38cf`) |
| 2026-11-05 | Archana | #16 | Locked gemini-flash as the held-out model in config | branch `archana/runs-v1` (`3362488`) |
| 2026-11-05 – 11-06 | Archana | #29 | MMLU outcome-table slice, provider token prices, validated v1 assessment banks, partial builds | branch `archana/runs-v1` (`04cecf7`, `2efa203`, `8dc73a5`, `12d8c27`) |
| 2026-11-05 | Harshini | ~#24, #30 | Consent wording approved by Prof. Gupta; review fixes (consent wording, no clock times, removal code; subject says "#12") | branch `harshini/pilot` (`99cecee`, `0e5c7c0`) |
| 2026-11-05 | Harshini | ~#35 | Pilot server: every session saved to disk, withdrawal deletes it, ngrok hosting, `ADR_DATA_DIR` | branch `harshini/pilot` (`fba496c`, `c951f89`, `d6d33e4`) |
| 2026-11-05 | Harshini | ~#35 | Pilot flow (quiz then survey, one debrief); pilot bank of 12 quiz MCQ + 12 survey items, Pew Wave 41 questions | branch `harshini/pilot` (`1e15a9d`, `711d2a1`, `b25735d`) |
| 2026-11-08 | Arav | #20 | Doc-upload `--max-tokens`; warn when a reply is cut off | branch `arav/doc-max-tokens` (`a75e84a`) |
| 2026-11-09 | Harshini | ~#30, #35 | Consent: no IRB approval, so class project only, keep only report totals; ~15 min session | branch `harshini/pilot` (`7185165`, `d61bb98`) |
| 2026-11-09 | Harshini | #39 | Pilot cleaning: saved sessions to outcome-table rows | branch `harshini/pilot` (`e63705c`) |
| 2026-11-09 | Harshini | ~#35 | Served front-end files give nothing away; pilot status command | branch `harshini/pilot` (`085de78`, `55bee63`) |

## Not in git (each person adds their own)

| Date | Who | Step | What | Link |
|---|---|---|---|---|
| | | | e.g. kickoff (#1), schema spec (#6), emails (#4), mentor meetings (#5), reviews of others' PRs, mid-term report (#14, #15) | |

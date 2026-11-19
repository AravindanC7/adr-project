# Adaptive Defense Router — Project Context for AI Assistants

> **Read this whole file before writing any code.** It is the single source of truth for scope, formats, ownership and order of work. Three people (and their AI assistants) build pieces that plug into each other, so everyone must work from the same contracts and the same workflow.
>
> **How to use this file:** put it at the root of the shared repo as `AGENTS.md` (and/or `CLAUDE.md`), or paste it at the start of your AI session. Then tell your assistant who you are: *"I am Arav / Archana / Harshini."* Section 12 has instructions specific to each person.
>
> **Live timeline (dates, filters, dependency view):** https://claude.ai/artifact/CoAhT5HJsBkLfWZbdZKFVz
>
> Last updated: Sat Oct 17, 2026. If this file and the timeline page disagree on a date, the timeline page wins; tell Arav so this file gets fixed.

---

## 0. Instructions for the AI assistant (read first)

You are helping one member of a three-person course project team. Follow these rules:

1. **Know whose assistant you are.** Ask your human which team member they are if they haven't said. Work mainly in that person's areas (Section 3). Each person owns their code; the others review it.
2. **Never change the shared contracts alone.** Files under `adr/contracts/` define the formats everyone builds against (Section 6). After they are frozen on **Tue Oct 20**, a change needs agreement from all three humans. If a contract blocks you, stop, explain the problem to your human, and propose the change in writing. Do not work around it with a private format.
3. **Respect the dependency order** (Section 9). Before starting a step, check what it waits on. If an input from another person is not ready, build against a small stub that follows the contract, and say clearly that it is a stub.
4. **Stay in scope** (Section 2). Screenshot or visual-channel attacks are out of scope. Do not add them.
5. **Every LLM call goes through the shared cached client** (`adr/harness/llm_client.py`). Never call a provider API directly anywhere else. Never hard-code API keys; read them from `.env`.
6. **Never tune anything on the held-out model or the held-out survey** (Section 8). That includes prompts, thresholds, smoothing constants and feature choices.
7. **Human data is sensitive.** Do not collect data from real people until the human-subjects decision (step #24) says it is allowed. Telemetry records timings and counts only, never the actual keys typed. Never store names or emails next to responses.
8. **Results go into the outcome table format, saved to shared storage.** Nothing important may live only inside a notebook or on one laptop.
9. **Keep it simple.** This is a 7-week course project with 3 people instead of the usual 4–5. Prefer the simplest version that answers the research question. Check the scope tier (Section 2.2) before adding anything.
10. **When unsure about anything that affects another person's work, ask your human** rather than guessing.

---

## 1. The project in one page

**Course:** CSE 576 Topics in NLP, Fall 2026, Arizona State University. Project #7.
**Mentors:** Prof. Vivek Gupta and Ashish Raj Shekhar (Ashish proposed the project and co-authored DoPE / Integrity Shield).
**Team:** Arav (Aravindan, achidam8@asu.edu), Archana (Ulagarchana, uulagan1@asu.edu), Harshini (hbalamu1@asu.edu).

### Problem
Online surveys, quizzes, tests and take-home assignments can now be answered by an LLM, either by a person copy-pasting or uploading the document, or by a browser agent filling the page in by itself. Existing defenses each use **one** mechanism everywhere: a decoy option, a hidden instruction, a text classifier, or keystroke logging. Each one fails against some kind of attacker, and switching all of them on for every question raises false flags on honest people.

### Our idea
The **Adaptive Defense Router** sits inside the assessment and, **question by question and person by person**, picks which detection check to run next, based on the evidence gathered so far. It keeps a running log-odds score that the respondent is using AI and escalates only as evidence builds:

```
s_t = s_{t-1} + log( P(o_t | AI) / P(o_t | human) )
```

This is Wald's Sequential Probability Ratio Test (SPRT). The probabilities come from a **failure-mode map** we measure first. A low score means carry on; a rising score triggers a harder check; past the upper threshold, the session is **paused for human review**. We never block anyone automatically: even at 1% false positives, 2 of every 200 honest students would be locked out.

### Two parts
1. **Measure** where each check works and where it breaks: the failure-mode map, which answers RQ1.
2. **Build** the router on top of those numbers and compare it with five baselines: RQ2–RQ4.

### Research questions
- **RQ1.** How well does each detection mechanism work across item types, assessment types and AI-use modes? Hit rate on AI, false-flag rate on humans.
- **RQ2.** At a 1% human false-positive rate, does the router beat (a) the best single mechanism and (b) all mechanisms on every item?
- **RQ3.** How many items does the router need to reach a reliable decision?
- **RQ4.** How much does detection drop when the attacker is more careful: warned, aware of the mechanisms, or allowed to retry?

### Headline test
At 1% FPR on honest people, does the router catch more AI use, in fewer questions, than any single check or all of them together?

---

## 2. Scope

### 2.1 In and out

| In scope | Out of scope |
|---|---|
| Surveys first (public human data exists), then quizzes (MMLU/ARC) | Screenshot or vision-only agents and visual traps (**team decision, Oct 2026**; this goes in Limitations and Future Work) |
| Three AI-use modes: **copy-paste**, **document upload**, **browser agent** | Code or math assignments, multilingual assessments, LMS plugins (Future Work) |
| Mechanisms in Section 5 | Automatic blocking of respondents |
| Replay simulation plus a small human pilot and live sessions | Large-scale human studies |

### 2.2 Scope tiers (cut from the right when time is short)

| Must ship | Should ship | Could ship |
|---|---|---|
| Copy-paste and document-upload attacker modes | Browser-agent attacker mode | Glitch-token probe |
| Decoys, opinion traps, canaries, writing-style signal | Behaviour check from telemetry | Live sessions with the router |
| Failure-mode map (RQ1) | Human pilot for false-positive rates | Screen-reader test |
| Router vs all 5 baselines in replay (RQ2, RQ3) | Adaptive attackers (RQ4) | Polished public code and data release |
| Held-out model and held-out survey evaluation | Non-native writer analysis | |

---

## 3. Team, ownership and review

| Area | Owner | Reviewer |
|---|---|---|
| Contracts: outcome table, mechanism interface, router API | Arav | Archana, Harshini |
| Contract: item schema | Harshini + Arav | Archana |
| Dataset loaders, human answer distributions, held-out splits | Archana | Harshini |
| Mechanism library: decoys, opinion traps, canaries, style signal, glitch tokens | Archana | Arav |
| Behaviour-check outcome features | Harshini | Archana |
| Attacker harness (all 3 modes + attacker levels) and shared LLM client | Arav | Archana |
| Large harness runs and the failure-mode map (RQ1) | Archana | Arav |
| Router: SPRT scoring, check selection, escalation, calibration | Arav | Archana |
| Replay simulation, 5 baselines, RQ2–RQ4 metrics | Arav | Harshini |
| Web front end, mechanism injection, telemetry, server | Harshini | Arav |
| Human-subjects question, consent, recruiting, pilot, live sessions | Harshini | All |
| Fairness: screen-reader test (Harshini), non-native writers (Archana) | Harshini, Archana | Arav |
| Mid-term report assembly; poster | Harshini | All |
| Final report integration and submission | Arav | All |

**Code review rule:** every pull request into `main` is reviewed by the listed reviewer. Contract changes need approval from all three.

---

## 4. Course deadlines (fixed)

| Date | What | Points |
|---|---|---|
| **Sun Oct 25** | Milestone 2: mid-term report. Must cover progress since the proposal, challenges, **contribution of each member**, and a timeline to completion | 30 |
| Sun Nov 1 | *Team deadline:* human-subjects decision (pilot or fallback) | — |
| Sun Nov 22 | *Team deadline:* experiment freeze; after this, only bug-fix reruns | — |
| **Nov 25 – Dec 2** | PPT demo and poster to mentor (guidelines also mention Dec 2–4; we plan for the earlier window and are **demo-ready Tue Nov 24**) | part of M3 |
| **Wed Dec 2** | Poster presentation to the class | part of M3 |
| **Fri Dec 4** | Final report (8–10 pages, **ACL format**) and PPT due | 50 |

The course also requires **regular mentor meetings**: a weekly slot with Ashish.

---

## 5. System architecture

The router has five parts:

```
          ┌───────────────────┐
          │ Failure-mode map  │  P(outcome | AI), P(outcome | human) per setting
          └─────────┬─────────┘
                    ▼
┌────────────┐  ┌────────┐  ┌──────────────┐  ┌─────────────────┐
│ Next item  │─▶│ Router │─▶│ Item served  │─▶│ Observe outcome │
│ (schema)   │  │ picks  │  │ with chosen  │  │ decoy hit, paste│
└────────────┘  │ check  │  │ mechanism    │  │ canary leak, ...│
      ▲         └────────┘  └──────────────┘  └────────┬────────┘
      │                                                ▼
┌─────┴───────────────────┐                ┌─────────────────────┐
│ Graded response:        │◀───────────────│ Update score s_t    │
│ continue / harder check │                │ (add log-lik ratio) │
│ / pause for review      │                └─────────────────────┘
└─────────────────────────┘
```

1. **Assessment schema.** Every survey or quiz is a list of JSON items recording type, whether the item has a correct answer, how it is delivered, and which mechanisms it can carry. One front end can then render any assessment.
2. **Mechanism library.** All decoys and traps are **generated ahead of time**, so nobody waits on an LLM call mid-session.

| Mechanism | Applies to | Signal | Known failure |
|---|---|---|---|
| Answer-key decoy (DoPE-style) | Quiz/test MCQ | AI picks a hidden planted wrong option | No correct answer on opinion items; screenshots drop detection to ~2.3% |
| Opinion trap | MCQ, Likert, ranking, opinion | Answer is implausible vs human distribution, or contradicts a reverse-keyed pair | Needs real human data to know what's unusual |
| Canary / prompt-leak trap | Open-text | Answer repeats a hidden word or instruction | Careful agents ignore hidden instructions |
| Glitch-token probe (exploratory) | Text items | Model outputs strange or broken text | Model-family specific |
| Behaviour check | Live completion | Paste, tab switch, typing speed, time per item | Fails if answers are retyped or agents mimic humans |
| Writing-style signal | Short and long text | Sudden change in length, style, fluency | Weak with few written answers; biased against non-native writers |

3. **Failure-mode map.** How often each mechanism fires on AI and on honest humans, in each setting.
4. **Router.** SPRT score plus selection of the most informative next check (Section 7).
5. **Action layer.** Continue, harder check, or pause for human review.

**Note on wording:** the router itself is not "trained." The failure-mode map is *estimated* from data, and the writing-style classifier (if we use one) is *trained*. Keep this distinction in code names and in the report.

---

## 6. Shared contracts (DRAFT — frozen Tue Oct 20)

These are the formats every component reads and writes. The fields below are the **starting draft** for the kickoff on Sun Oct 18. Field names may change in the kickoff; after **Tue Oct 20** they are fixed. The authoritative version is the code in `adr/contracts/`, not this file.

We use **Pydantic v2 models** so that bad data fails loudly at the boundary between components instead of silently breaking someone else's code.

### 6.1 Item schema (`adr/contracts/schema.py`) — owners: Harshini + Arav

```python
from typing import Literal
from pydantic import BaseModel

Source = Literal[
    "opinionqa", "gss", "ipip",
    "asurre_nature_phd_2019", "asurre_springer_social_2017", "asurre_osmi_2017_18",
    "mmlu", "arc",
]
AssessmentType = Literal["survey", "quiz"]
ItemType = Literal["mcq", "likert", "ranking", "numeric", "open_text"]
AttackMode = Literal["copy_paste", "doc_upload", "browser_agent"]
MechanismName = Literal["decoy", "opinion_trap", "canary", "glitch_token", "behaviour", "style"]
Split = Literal["train", "heldout_survey"]

class Item(BaseModel):
    item_id: str                      # "<source>:<native id>", unique and stable
    source: Source
    assessment_type: AssessmentType
    item_type: ItemType
    stem: str                         # question text as shown to a human
    options: list[str] | None = None  # MCQ / Likert / ranking labels, in display order
    correct_option: int | None = None # index into options; quizzes only
    reverse_keyed_with: str | None = None  # IPIP: item_id of the paired reverse-keyed item
    eligible_mechanisms: list[MechanismName]
    human_dist_key: str | None = None # key into the human answer distribution store
    split: Split = "train"

class Assessment(BaseModel):
    assessment_id: str
    title: str
    assessment_type: AssessmentType
    item_ids: list[str]               # display order; the router may reorder
```

### 6.2 Mechanism interface (`adr/contracts/mechanism.py`) — owner: Arav

```python
from typing import Protocol, Any
from pydantic import BaseModel

class RenderedItem(BaseModel):
    """One pre-generated variant of an item carrying one mechanism."""
    item_id: str
    mechanism: MechanismName | None   # None = plain item, no check
    variant_id: str                   # "<item_id>|<mechanism>|<n>", stable
    visible: dict[str, Any]           # what a human sees: stem, options
    hidden: dict[str, Any] = {}       # e.g. hidden_text, canary_token, decoy_option_index
    burden: float = 0.0               # extra effort for an honest respondent (0 = none)

class Response(BaseModel):
    session_id: str
    item_id: str
    variant_id: str
    answer: int | list[int] | float | str | None   # option index, ranking, number, or text
    raw_text: str | None = None       # full model output (AI) or typed text (human)

class Telemetry(BaseModel):
    """Per item. Timings and counts only — never the characters typed."""
    time_on_item_ms: int
    paste_events: int = 0
    focus_loss_count: int = 0
    keystroke_count: int = 0
    mean_interkey_ms: float | None = None

class Mechanism(Protocol):
    name: MechanismName
    outcomes: tuple[str, ...]         # the closed set of labels score() may return
    def applies_to(self, item: Item) -> bool: ...
    def generate(self, item: Item, seed: int) -> RenderedItem: ...   # offline, ahead of time
    def score(self, rendered: RenderedItem, response: Response,
              telemetry: Telemetry | None) -> str: ...               # one label from `outcomes`
```

**Outcome labels (draft).** Keep each set small, so the failure map has enough data per cell.

| Mechanism | Outcome labels |
|---|---|
| decoy | `picked_decoy`, `picked_correct`, `picked_other` |
| opinion_trap | `implausible`, `contradiction`, `plausible` |
| canary | `leaked`, `followed_instruction`, `clean` |
| glitch_token | `anomalous`, `normal` |
| behaviour | `paste`, `focus_loss`, `too_fast`, `normal` (first match in that priority order) |
| style | `high`, `mid`, `low` (classifier score bucket; thresholds set on train split only) |

### 6.3 Outcome table (`adr/contracts/outcome.py`) — owner: Arav

**The most important contract.** The failure map, the replay simulation and every result read it. One row per (respondent, item, mechanism) observation. Stored as Parquet under `results/outcomes/`.

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | Which run produced this row; links to `results/runs/<run_id>/config.yaml` |
| `session_id` | str | One respondent taking one assessment |
| `respondent_kind` | `ai` \| `human` | Ground truth label |
| `data_origin` | `harness` \| `public_human` \| `pilot` \| `live` \| `dry_run` | Where the row came from |
| `model` | str \| null | Exact model id for AI rows; null for humans |
| `attack_mode` | AttackMode \| null | null for humans |
| `attacker_level` | `L0`…`L4` \| null | See Section 7.3; null for humans |
| `assessment_type` | survey \| quiz | |
| `item_type` | ItemType | |
| `item_id` | str | |
| `split` | train \| heldout_survey | |
| `heldout_model` | bool | True if `model` is the held-out model |
| `mechanism` | MechanismName \| null | |
| `variant_id` | str | |
| `outcome` | str | Label from the mechanism's `outcomes` set |
| `raw_ref` | str | Path or key to the full response and cached LLM output |
| `prompt_version` | str \| null | Version tag of the attacker prompt |
| `created_at` | ISO datetime | |

### 6.4 Failure-mode map (`adr/contracts/failure_map.py`) — owner: Archana (format agreed with Arav)

An aggregate of the outcome table, saved as `results/failure_map/<version>.json`.

```python
class MapCell(BaseModel):
    mechanism: MechanismName
    item_type: ItemType
    attack_mode: AttackMode | None     # None = pooled over modes
    attacker_level: str | None         # None = pooled
    counts_ai: dict[str, int]          # outcome -> count
    counts_human: dict[str, int]
    p_ai: dict[str, float]             # smoothed: (count + 1) / (n + K)
    p_human: dict[str, float]
    ci_ai: dict[str, tuple[float, float]]     # 95% interval per outcome
    ci_human: dict[str, tuple[float, float]]
```

**Smoothing:** add-one (a Dirichlet(1) prior) over each mechanism's outcome set, so no probability is ever zero. A zero probability would make a log-likelihood ratio infinite. Human probabilities for content mechanisms (decoy, opinion trap, style) come from public human data. For traps and behaviour, they come from the pilot (or the dry run, labeled preliminary).

### 6.5 Router API (`adr/contracts/router_api.py`) — owner: Arav, consumed by Harshini's server

**Python interface** (used directly by replay and baselines):

```python
class SessionState(BaseModel):
    session_id: str
    assessment_id: str
    score: float = 0.0                 # s_t, log-odds of AI
    served: list[str] = []             # variant_ids already shown
    history: list[tuple[str, str, float]] = []   # (variant_id, outcome, delta)
    status: Literal["active", "escalated", "review", "cleared"] = "active"

class Choice(BaseModel):
    item_id: str
    variant_id: str
    mechanism: MechanismName | None

class Router(Protocol):
    def start(self, assessment: Assessment, session_id: str, seed: int) -> tuple[SessionState, Choice]: ...
    def update(self, state: SessionState, variant_id: str, outcome: str) -> SessionState: ...
    def next(self, state: SessionState) -> Choice | None: ...  # None = assessment finished
```

**HTTP endpoints** (FastAPI, wrapping the same Router):

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/session/start` | `{assessment_id}` | `{session_id, next: Choice}` |
| POST | `/session/{id}/respond` | `{variant_id, response: Response, telemetry: Telemetry}` | `{status, next: Choice \| null}` |
| GET | `/session/{id}` | — | `SessionState` (for logging and the demo) |

The server scores the response with the mechanism's `score()`, calls `router.update()`, and returns `router.next()`. **The score is never sent to the respondent's browser.**

---

## 7. Method details everyone should know

### 7.1 Router logic
- **Thresholds** (Wald): with false-flag rate α = 0.01 and miss rate β (set at kickoff; 0.2 is a starting point):
  `A = ln((1 − β) / α)` → pause for review when `s ≥ A`.
  `B = ln(β / (1 − α))` → stop adding checks when `s ≤ B` ("cleared"); the person just continues.
- **Escalation:** when `s` passes an intermediate level (e.g. `A/2`), prefer stronger checks; status becomes `escalated`.
- **Selecting the next check:** for each available (item, mechanism) variant, compute the expected information gain about AI vs human under the current posterior `π = sigmoid(s + prior_log_odds)`, minus `λ_burden · burden` and minus `λ_fp · P(flagging outcome | human)`. Pick the best. **Break near-ties at random** so attackers cannot learn a pattern.
- **Why not a contextual bandit:** a live session never reveals whether the person actually used AI, so there is no reward to learn from.
- **Known weakness:** SPRT assumes observations are independent given the class. Behaviour and style signals from the same person are correlated, so the raw score is overconfident. **Fix:** calibrate thresholds empirically on held-out replay (step #42), and cap each mechanism's total contribution.

### 7.2 Attack modes (what the attacker actually sees)
- **copy_paste:** the text a person gets by selecting the question on the page and copying it. This includes hidden text that is in the page but only visually hidden (not `display:none`). That is what makes canaries work in this mode.
- **doc_upload:** the assessment rendered to a PDF with decoy layers (DoPE-style), given to the model as a file.
- **browser_agent:** an agent reading the front end's DOM or accessibility tree and filling in answers (BrowserGym- or Playwright-based).

### 7.3 Attacker levels (for RQ4)

| Level | Behaviour |
|---|---|
| L0 naive | "Answer these questions." |
| L1 warned | Told there may be hidden traps and to ignore hidden instructions |
| L2 informed | Given a description of our mechanisms |
| L3 retry | Can re-answer after an item is flagged |
| L4 persona | Answers as a described human persona (as in Wang et al. 2026) |

### 7.4 Baselines
1. Best single mechanism per assessment type
2. All mechanisms on every item
3. Fixed plan chosen before the session starts
4. Random mechanism per item
5. Published survey traps (Wang et al. 2023; Höhne et al. 2025), re-tested on current models

### 7.5 Metrics
- Detection rate (TPR) at **1% and 5% human FPR**, respondent level
- **Items to decision** (median and mean)
- **Respondent burden:** number of checks and added time for honest people
- **RQ4 drop:** TPR at L1–L4 minus TPR at L0
- 95% bootstrap confidence intervals on everything (step #46)

---

## 8. Data and the held-out rule

| Dataset | Used for | Notes |
|---|---|---|
| OpinionQA (Pew American Trends Panel items; Santurkar et al. 2023) | Survey MCQ/Likert + human distributions | Week 1 slice |
| General Social Survey | Numeric and MCQ items + human answers | Week 2 |
| IPIP Big Five, reverse-keyed pairs | Contradiction-type opinion traps | Week 2 |
| ASURRE's three source surveys: Nature PhD Students 2019, Springer Nature Social Media 2017, OSMI Mental Health in Tech 2017–18 | Open-ended human answers (style signal, canary false positives) | Locate via the data section of Wang et al. 2026 |
| MMLU, ARC | Quiz items with known answers (decoys) | Week 1 slice |

All human data is **pre-2020**, before LLMs. This means a classifier might learn "old vs new writing" instead of "human vs AI." The held-out splits are our main defense:

- **Held-out model: `claude-haiku`** (`claude-haiku-5-5`). It never contributes to the failure map, thresholds or any tuning. It runs **only once, in the final `--final` RQ2–RQ4 evaluation**; never run it on the real banks before that.
  - **Decision, Tue Nov 17 (team):** `claude-haiku` replaces `gemini-flash`, because Gemini now needs a $50 minimum prepay. No `claude-haiku` rows existed in `results/outcomes/` when this was decided. Recorded in `configs/data_decisions.json` (`heldout_model`) and set with `heldout: true` in `configs/models.yaml`.
  - History: `gemini-flash` was chosen on **Mon Oct 26** (step #16), before any large run. It is no longer used.
- **Held-out survey:** one ASURRE survey (decided in step #18) is excluded from all fitting.
- Models: the team mixes providers (e.g. one OpenAI, one Anthropic, one Google, one open-weight). The **weekly API spend cap** is set at kickoff. Use the open-weight model for bulk runs where possible.

---

## 9. Workflow order and dependencies (all 56 steps)

Each step lists what it waits on and what it unblocks. **Your step cannot finish before the steps it waits on. Anyone your step unblocks is counting on your date.** If you need to move a date that someone else waits on, tell that person first.

| # | Wk | Step | Owner | Due | Waits on | Unblocks | Flags |
|---|---|---|---|---|---|---|---|
| 1 | 1 | Kickoff: agree contract fields, pick 2 slice models, set spend cap | All | Sun Oct 18 | — | #2, #3, #4, #5, #6, #7 | — |
| 2 | 1 | Repo, folder layout and model-response cache | Arav | Sun Oct 18 | #1 | #9 | — |
| 3 | 1 | Overleaf ACL project and shared contribution log | Harshini | Mon Oct 19 | #1 | #15 | — |
| 4 | 1 | Email Prof. Gupta about human-subjects coverage; ask the TA which demo window applies | Harshini | Mon Oct 19 | #1 | #24 | — |
| 5 | 1 | Book a weekly mentor slot with Ashish | Arav | Mon Oct 19 | #1 | — | — |
| 6 | 1 | Item schema spec frozen | Harshini + Arav | Tue Oct 20 | #1 | #8, #9, #11, #18 | critical |
| 7 | 1 | Outcome-table format, mechanism interface and router API frozen | Arav | Tue Oct 20 | #1 | #9, #10, #13, #17 | critical |
| 8 | 1 | Loaders: OpinionQA and MMLU/ARC subsets with human answer distributions _(Start Sunday on the draft schema fields.)_ | Archana | Wed Oct 21 | #6 | #10 | — |
| 9 | 1 | Copy-paste attacker harness for 2 models _(Test on a few hand-written items until step 8 lands.)_ | Arav | Thu Oct 22 | #2, #6, #7 | #12, #20, #31, #33, #36 | — |
| 10 | 1 | Answer-key decoys (~200 MMLU/ARC) and opinion trap v0 (~200 OpinionQA) | Archana | Thu Oct 22 | #7, #8 | #12, #19, #20 | — |
| 11 | 1 | Front-end skeleton that renders any schema item | Harshini | Fri Oct 23 | #6 | #19, #23 | — |
| 12 | 1 | Mid-term slice run: first failure-map cells | Archana | Fri Oct 23 | #9, #10 | #13, #14 | critical |
| 13 | 1 | SPRT router prototype on the first real cells _(Build it on synthetic cells from Tuesday so only the final run waits on step 12.)_ | Arav | Fri Oct 23 | #7, #12 | #14, #26 | — |
| 14 | 1 | Mid-term sections drafted by their owners | All | Sat Oct 24 | #12, #13 | #15 | — |
| 15 | 1 | Mid-term report assembled and submitted | Harshini | Sun Oct 25 | #3, #14 | #16 | fixed date |
| 16 | 2 | Held-out model locked (Arav leads) | All | Mon Oct 26 | #15 | #29 | — |
| 17 | 2 | Canary and prompt-leak trap spec handed to Harshini | Archana | Tue Oct 27 | #7 | #19, #21 | — |
| 18 | 2 | Loaders: GSS, IPIP Big Five, three ASURRE surveys; held-out survey split | Archana | Wed Oct 28 | #6 | #21, #29 | — |
| 19 | 2 | Mechanism injection in the front end: hidden text, canaries, decoy options | Harshini | Thu Oct 29 | #10, #11, #17 | #22, #28, #35, #44 | critical |
| 20 | 2 | Document-upload attacker mode | Arav | Thu Oct 29 | #9, #10 | #29, #36 | — |
| 21 | 2 | Canary generator and writing-style signal v0 | Archana | Sun Nov 1 | #17, #18 | #29, #37 | — |
| 22 | 2 | Browser-agent attacker mode | Arav | Sun Nov 1 | #19 | #29 | should |
| 23 | 2 | Telemetry: paste, focus/blur, keystroke timing, time per item | Harshini | Sun Nov 1 | #11 | #25, #35 | — |
| 24 | 2 | Human-subjects decision: pilot or dry-run fallback | Harshini | Sun Nov 1 | #4 | #30 | fixed date |
| 25 | 3 | Behaviour-check features as discrete outcomes | Harshini | Wed Nov 4 | #23 | #39 | — |
| 26 | 3 | Router running on real outcome rows | Arav | Wed Nov 4 | #13 | #27, #28 | — |
| 27 | 3 | Replay simulator | Arav | Fri Nov 6 | #26 | #32 | — |
| 28 | 3 | Router API wired into the front end | Harshini | Fri Nov 6 | #19, #26 | #43, #47 | — |
| 29 | 3 | Large harness runs, producing outcome table v1 _(If browser-agent mode (22) is late, run everything else first and add it after.)_ | Archana | Sun Nov 8 | #16, #18, #20, #21, #22 | #32, #34 | critical |
| 30 | 3 | Pilot materials: consent screen, recruitment message, session script | Harshini | Sun Nov 8 | #24 | #35 | — |
| 31 | 3 | Glitch-token probe, 2 days max | Archana | Sun Nov 8 | #9 | — | could |
| 32 | 3 | Baselines 1–4 numbers on outcome table v1 | Arav | Mon Nov 9 | #27, #29 | #38 | — |
| 33 | 4 | Baseline 5: published survey traps re-tested on current models | Arav | Wed Nov 11 | #9 | #38 | — |
| 34 | 4 | Failure-mode map v1 with smoothing and confidence intervals | Archana | Thu Nov 12 | #29 | #37, #38, #40 | critical |
| 35 | 4 | Human pilot | Harshini | Fri Nov 13 | #19, #23, #30 | #39, #43 | should |
| 36 | 4 | Attacker skill levels for RQ4 | Arav | Fri Nov 13 | #9, #20 | #41 | should |
| 37 | 4 | Map heatmap figure; writing-style signal v1 | Archana | Sun Nov 15 | #21, #34 | #45 | — |
| 38 | 4 | First RQ2 and RQ3 results | Arav | Sun Nov 15 | #32, #33, #34 | #41, #42 | — |
| 39 | 4 | Pilot data cleaned into the outcome table | Harshini | Sun Nov 15 | #25, #35 | #40 | — |
| 40 | 5 | Failure-mode map v2 with pilot false-positive rates | Archana | Wed Nov 18 | #34, #39 | #42, #46, #55 | critical |
| 41 | 5 | RQ4 results | Arav | Wed Nov 18 | #36, #38 | #46 | — |
| 42 | 5 | Threshold calibration on held-out replay | Arav | Fri Nov 20 | #38, #40 | #46 | critical |
| 43 | 5 | Live sessions with the router choosing checks | Harshini | Fri Nov 20 | #28, #35 | — | could |
| 44 | 5 | Screen-reader test of the hidden traps | Harshini | Sat Nov 21 | #19 | — | could |
| 45 | 5 | Non-native writer false-flag analysis | Archana | Sun Nov 22 | #37 | — | should |
| 46 | 5 | Bootstrap confidence intervals; experiment freeze | All | Sun Nov 22 | #40, #41, #42 | #47, #48, #49 | fixed date |
| 47 | 6 | Demo build stable on one machine; mentor slot booked | Harshini | Tue Nov 24 | #28, #46 | — | fixed date |
| 48 | 6 | Demo script and PPT v1, each person writing their own slides | All | Tue Nov 24 | #46 | — | — |
| 49 | 6 | Report sections written by their owners | All | Fri Nov 27 | #46 | #50, #51 | — |
| 50 | 6 | Poster draft ready for the mentor | Harshini | Sat Nov 28 | #49 | #53 | fixed date |
| 51 | 6 | Full report integrated, 8–10 pages | Arav | Sun Nov 29 | #49 | #52 | — |
| 52 | 7 | Cross-review: everyone reads the sections they did not write | All | Mon Nov 30 | #51 | #53, #56 | — |
| 53 | 7 | Poster final and printed | Harshini | Tue Dec 1 | #50, #52 | #54 | — |
| 54 | 7 | Poster presentation to the class | All | Wed Dec 2 | #53 | — | fixed date |
| 55 | 7 | Failure-map data release with a README | Archana | Thu Dec 3 | #40 | — | — |
| 56 | 7 | Final report and PPT submitted; code release tagged | Arav | Fri Dec 4 | #52 | — | fixed date |

**Critical path:** #6/#7 (contracts) → #9 + #10 → #12 (mid-term slice) → #19, #20, #22 → **#29 (outcome table v1)** → #34 (map v1) → #38 → #40 (map v2) → #42 (calibration) → #46 (freeze) → report and demo.

---

## 10. Repository layout and conventions

```
adr-project/
├── AGENTS.md                 # this file (also CLAUDE.md)
├── README.md                 # setup and how to run each part
├── pyproject.toml            # Python 3.11; pydantic v2, pandas, pyarrow, fastapi, pytest
├── .env.example              # API key names only, never values
├── adr/
│   ├── contracts/            # FROZEN after Oct 20 — schema, mechanism, outcome, failure_map, router_api
│   ├── data/loaders/         # Archana: one module per source → list[Item] + human distributions
│   ├── mechanisms/           # Archana (decoy, opinion_trap, canary, glitch_token, style); Harshini (behaviour)
│   ├── harness/              # Arav: llm_client.py (cached), copy_paste.py, doc_upload.py, browser_agent.py, attacker_levels.py
│   ├── map/                  # Archana: build failure map from outcome table
│   ├── router/               # Arav: sprt.py, selection.py, escalation.py
│   ├── replay/               # Arav: session simulator
│   └── baselines/            # Arav: the five baselines
├── server/                   # Harshini: FastAPI app wrapping the router + mechanism scoring
├── frontend/                 # Harshini: web UI + telemetry (framework is Harshini's choice)
├── data/                     # gitignored; raw/ and processed/ synced via shared drive
├── cache/llm/                # gitignored; shared LLM response cache
├── results/
│   ├── outcomes/             # outcome table Parquet files
│   ├── failure_map/          # map versions
│   └── runs/<run_id>/        # config.yaml + metrics.json for every experiment
├── notebooks/                # exploration only; nothing final lives here
└── tests/                    # contract tests + unit tests per module
```

**Conventions**
- Python 3.11, type hints everywhere, `ruff` for formatting and lint, `pytest` for tests.
- **One branch per person or feature**, pull requests into `main`, reviewed by the listed reviewer.
- **LLM cache:** every call is keyed by a hash of (model, messages, parameters, prompt_version). Re-running never pays twice. The cache is shared via the drive.
- **Every run gets a `run_id`** and a `config.yaml` recording models, seeds, prompt versions, data splits and the git commit.
- Fixed random seeds for decoy generation, sampling and tie-breaking.
- Contract tests in `tests/test_contracts.py`: each loader's output validates as `Item`; each mechanism's `score()` returns only labels from its `outcomes`; each outcome row validates.
- Commit messages: short imperative summary, e.g. `harness: add doc_upload mode`.

---

## 11. Weekly rhythm

- **Sunday sync (30 min):** check last week's "done when," name blockers, update the **contribution log** (the mid-term grades each member's contribution).
- **Wednesday async status:** three lines each in the group chat: done, next, blocked. Flag a slipping hand-off here, not on Sunday.
- **Weekly mentor check-in** with Ashish: bring one result and one question.
- **Everything lands in shared storage:** model responses, outcome tables, figures.

---

## 12. Per-person instructions

Find your human's section. It lists their steps, what they wait on from others, and what others wait on from them. **Treat every "others wait on you" date as a hard deadline.**

### 12.1 If your human is **Arav**

**Role:** contracts (outcome table, mechanism interface, router API; co-owns item schema), attacker harness and shared LLM client, router, replay simulation, baselines, RQ2–RQ4, final report integration.

**Own steps:** #2, #5, #6, #7, #9, #13, #20, #22, #26, #27, #32, #33, #36, #38, #41, #42, #51, #56

**Day 1 (Sun Oct 18):** create the repo with the layout in Section 10, `pyproject.toml`, `.env.example` and the cached `llm_client.py`. Start writing `adr/contracts/` from Section 6. Mon: review Harshini's schema; book the mentor slot. Tue: freeze contracts.

**Waits on from others:**

| You need | From | Due | For your step |
|---|---|---|---|
| #12 Mid-term slice run: first failure-map cells | Archana | Fri Oct 23 | #13 (Fri Oct 23) |
| #10 Answer-key decoys and opinion trap v0 | Archana | Thu Oct 22 | #20 (Thu Oct 29) |
| #19 Mechanism injection in the front end | Harshini | Thu Oct 29 | #22 (Sun Nov 1) |
| #29 Large harness runs, outcome table v1 | Archana | Sun Nov 8 | #32 (Mon Nov 9) |
| #34 Failure-mode map v1 | Archana | Thu Nov 12 | #38 (Sun Nov 15) |
| #40 Failure-mode map v2 | Archana | Wed Nov 18 | #42 (Fri Nov 20) |

**Others wait on you:**

| Your step | Due | Needed by | For their step |
|---|---|---|---|
| #6 Item schema spec frozen | Tue Oct 20 | Archana | #8 (Wed Oct 21), #18 (Wed Oct 28) |
| #6 Item schema spec frozen | Tue Oct 20 | Harshini | #11 (Fri Oct 23) |
| #7 Outcome table, mechanism interface, router API frozen | Tue Oct 20 | Archana | #10 (Thu Oct 22), #17 (Tue Oct 27) |
| #9 Copy-paste attacker harness | Thu Oct 22 | Archana | #12 (Fri Oct 23), #31 (Sun Nov 8) |
| #20 Document-upload attacker mode | Thu Oct 29 | Archana | #29 (Sun Nov 8) |
| #22 Browser-agent attacker mode | Sun Nov 1 | Archana | #29 (Sun Nov 8) |
| #26 Router running on real outcome rows | Wed Nov 4 | Harshini | #28 (Fri Nov 6) |

**Watch out:** #12 and #13 are both due Fri Oct 23. Build the router on **synthetic** map cells from Tuesday so only the final run depends on Archana's slice.

### 12.2 If your human is **Archana**

**Role:** dataset loaders and human answer distributions, held-out splits, mechanism library (decoys, opinion traps, canaries, writing-style signal, glitch tokens), running the harness at scale, failure-mode map (RQ1), non-native writer analysis, novelty check.

**Own steps:** #8, #10, #12, #17, #18, #21, #29, #31, #34, #37, #40, #45, #55

**Day 1 (Sun Oct 18):** download OpinionQA, MMLU and ARC. Start the OpinionQA loader against the **draft** `Item` fields in Section 6.1, producing `list[Item]` plus a human distribution per item. Expect small field renames before Tue; keep the mapping code in one place so renames are cheap. Mon: finish OpinionQA, start MMLU/ARC.

**Waits on from others:**

| You need | From | Due | For your step |
|---|---|---|---|
| #6 Item schema spec frozen | Harshini + Arav | Tue Oct 20 | #8 (Wed Oct 21), #18 (Wed Oct 28) |
| #7 Outcome table and mechanism interface frozen | Arav | Tue Oct 20 | #10 (Thu Oct 22), #17 (Tue Oct 27) |
| #9 Copy-paste attacker harness | Arav | Thu Oct 22 | #12 (Fri Oct 23), #31 (Sun Nov 8) |
| #20 Document-upload attacker mode | Arav | Thu Oct 29 | #29 (Sun Nov 8) |
| #22 Browser-agent attacker mode | Arav | Sun Nov 1 | #29 (Sun Nov 8) |
| #39 Pilot data cleaned into the outcome table | Harshini | Sun Nov 15 | #40 (Wed Nov 18) |

**Others wait on you:**

| Your step | Due | Needed by | For their step |
|---|---|---|---|
| #10 Answer-key decoys and opinion trap v0 | Thu Oct 22 | Harshini | #19 (Thu Oct 29) |
| #10 Answer-key decoys and opinion trap v0 | Thu Oct 22 | Arav | #20 (Thu Oct 29) |
| #12 Mid-term slice run: first failure-map cells | Fri Oct 23 | Arav | #13 (Fri Oct 23) |
| #17 Canary and prompt-leak trap spec | Tue Oct 27 | Harshini | #19 (Thu Oct 29) |
| #29 Large harness runs, outcome table v1 | Sun Nov 8 | Arav | #32 (Mon Nov 9) |
| #34 Failure-mode map v1 | Thu Nov 12 | Arav | #38 (Sun Nov 15) |
| #40 Failure-mode map v2 | Wed Nov 18 | Arav | #42 (Fri Nov 20) |

**Watch out:** #29 is the most important hand-off of the project. If any attacker mode is late, run the others first and append the late mode afterwards. Time-box the glitch-token probe (#31) to 2 days.

### 12.3 If your human is **Harshini**

**Role:** item schema spec (with Arav), web front end, mechanism injection, telemetry, server, behaviour-check features, human-subjects question, consent, pilot, live sessions, screen-reader test, mid-term report assembly, poster.

**Own steps:** #3, #4, #6, #11, #15, #19, #23, #24, #25, #28, #30, #35, #39, #43, #44, #47, #50, #53

**Day 1 (Sun Oct 18):** write up the item schema spec from the kickoff (start from Section 6.1) and send it to Arav. Mon: email Prof. Gupta about human-subjects coverage for the pilot and the TA about the demo window; set up the Overleaf ACL project and the contribution log. Then start the front-end skeleton that renders any `Item` (MCQ, Likert, open text).

**Waits on from others:**

| You need | From | Due | For your step |
|---|---|---|---|
| #10 Answer-key decoys and opinion trap v0 | Archana | Thu Oct 22 | #19 (Thu Oct 29) |
| #17 Canary and prompt-leak trap spec | Archana | Tue Oct 27 | #19 (Thu Oct 29) |
| #26 Router running on real outcome rows | Arav | Wed Nov 4 | #28 (Fri Nov 6) |

**Others wait on you:**

| Your step | Due | Needed by | For their step |
|---|---|---|---|
| #6 Item schema spec frozen | Tue Oct 20 | Archana | #8 (Wed Oct 21), #18 (Wed Oct 28) |
| #6 Item schema spec frozen | Tue Oct 20 | Arav | #9 (Thu Oct 22) |
| #19 Mechanism injection in the front end | Thu Oct 29 | Arav | #22 (Sun Nov 1) |
| #39 Pilot data cleaned into the outcome table | Sun Nov 15 | Archana | #40 (Wed Nov 18) |

**Watch out:**
- Hidden traps must be *visually* hidden but present in the DOM and in copied text (Section 7.2). Do not use `display:none`, or agents and copy-paste won't see them.
- Screen readers will also read hidden text aloud. That is exactly what step #44 tests, so keep the hiding method configurable.
- Telemetry stores timings and counts only, never the keys typed.
- If the human-subjects answer is not clear by **Sun Nov 1**, switch to the fallback: a team-only dry run with `data_origin = "dry_run"`, numbers labeled preliminary.

---

## 13. Mid-term report (due Sun Oct 25) — what goes in

| Section | Owner |
|---|---|
| Progress since proposal: contracts, loaders, harness, slice results, router prototype | Each owner writes their part; Harshini assembles |
| Preliminary results: first failure-map cells (decoys + opinion traps, copy-paste, 2 models) and a router run | Archana + Arav |
| Challenges: team of 3 vs the expected 4–5 (hence scope tiers), human-subjects question, pre-2020 data confound, API cost | All |
| **Contribution of each member** (from the contribution log) | All |
| Timeline to completion (Section 9, summarized) | Arav |

Results freeze **Fri Oct 23**. Full draft **Sat Oct 24**; each person reads the sections they did not write. Submit **Sun Oct 25**.

## 14. Final report (8–10 pages, ACL format, due Fri Dec 4)

| Section | Owner | Pages |
|---|---|---|
| Abstract, introduction, conclusion; final integration | Arav | ~1 |
| Related work | Archana | ~1 |
| System: schema, front end, telemetry | Harshini | ~1 |
| Mechanisms and failure-mode map (RQ1) | Archana | ~2 |
| Router, replay, baselines (RQ2, RQ3) | Arav | ~2 |
| Adaptive attackers (RQ4) | Arav | ~0.75 |
| Human study, fairness, limitations | Harshini | ~1.5 |

---

## 15. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Human-subjects approval not settled by Nov 1 | Team-only dry run, labeled preliminary; rely on public human data for content signals; live sessions move to "could" |
| Outcome table v1 slips past Nov 8 | Cut to 2 training models and ~200 items per item type. **Never drop the held-out model.** |
| API costs run over | Weekly spend cap; cache everything; open-weight model for bulk runs |
| Router overconfident (correlated signals) | Empirical threshold calibration (#42); cap per-mechanism contribution |
| Screen readers read hidden traps aloud | Test in #44; report the result either way as a fairness finding |
| Guideline demo dates conflict | Ask the TA; be demo-ready Tue Nov 24 regardless |

---

## 16. Key references

- **DoPE** (Shekhar et al. 2026a, arXiv 2601.12505) and **Integrity Shield** (Shekhar et al. 2026b, EACL 2026 demos): decoys in exam documents. From our mentors' group.
- **ASURRE** (Wang, Mamaev, Leckie 2026, arXiv 2609.17317): detecting AI-assisted survey responses; source of the three open-ended surveys.
- **Wang et al. 2023** (arXiv 2306.08833) and **Höhne et al. 2025**: prompt-injection traps in surveys (baseline 5).
- **Westwood 2025** (PNAS): a survey bot passing 99.8% of attention checks.
- **Xu and Malkin 2026** (SOUPS): LLM use rates across crowd platforms.
- **Wald 1945**: SPRT. **van Krimpen-Stoop and Meijer 2001**: sequential person-fit in adaptive testing.
- **Krishna et al. 2023**: paraphrasing evades detectors. **Liang et al. 2023**: detectors biased against non-native writers.
- **Land and Bartolo 2024**: under-trained (glitch) tokens.
- **Nasr et al. 2025; Zhan et al. 2025**: adaptive attacks break static defenses.

## 17. Glossary

- **Mechanism / check:** one detection technique (decoy, opinion trap, canary, glitch token, behaviour, style).
- **Variant:** one pre-generated version of an item carrying one mechanism.
- **Outcome:** the discrete label a mechanism gives for one response.
- **Failure-mode map:** P(outcome | AI) and P(outcome | human) for every mechanism × item type × attack mode × attacker level.
- **Outcome table:** the row-level log of every observation; the map and all results are computed from it.
- **SPRT:** sequential probability ratio test; adds log-likelihood ratios until a threshold is crossed.
- **Held-out model / survey:** data never used for fitting or tuning; only for final evaluation.
- **Replay:** simulated sessions built by sampling real logged outcomes, used to compare the router and baselines offline.

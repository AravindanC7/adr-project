# Adaptive Defense Router

CSE 576 (Fall 2026), Project #7. A router that picks, question by question, which AI-use check
to run next, and pauses a session for human review once the evidence is strong enough.

**Start with [AGENTS.md](AGENTS.md)**: scope, contracts, the 56-step workflow, and who owns what.
Live timeline: https://claude.ai/artifact/CoAhT5HJsBkLfWZbdZKFVz

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -e ".[dev]"                                   # core + pytest + ruff
pip install -e ".[anthropic,openai]"                      # only the providers you use
cp .env.example .env                                      # add API keys
cp configs/models.example.yaml configs/models.yaml        # pick models, prices, spend cap
pytest                                                    # all tests should pass
```

## What exists so far (Arav, Oct 26)

| Area | Files | Status |
|---|---|---|
| Contracts (draft) | `adr/contracts/` | Item schema, mechanism interface, outcome table, failure map, router API. Tests in `tests/test_contracts.py`. **Needs team sign-off.** |
| LLM client | `adr/harness/llm_client.py` | Anthropic, OpenAI, Google, OpenAI-compatible (open-weight). Disk cache, weekly spend cap, retries. |
| Copy-paste attacker (#9) | `adr/harness/copy_paste.py`, `prompts.py`, `parse.py` | Levels L0, L1, L2, L4. L3 (retry) comes with step #36. |
| Router prototype (#13) | `adr/router/` | SPRT score, Chernoff-rule check selection, burden and false-flag penalties, per-mechanism cap, escalation, random tie-breaks. |
| Early baselines | `adr/baselines/simple.py` | Random check per item; single fixed mechanism. |
| Synthetic test data | `adr/router/synthetic.py` | **Invented numbers** for developing the router. Never report these. |

Not started: document-upload attacker (#20), browser agent (#22), replay on real outcomes (#27),
the other baselines, and everything Archana and Harshini own.

## Try it

```bash
# Copy-paste attacker on 4 handwritten items with stub traps, fully offline:
python scripts/run_copy_paste.py --stub --fake

# Same, against one real model from configs/models.yaml (costs a few cents at most):
python scripts/run_copy_paste.py --stub --models claude-small

# Router vs early baselines on SYNTHETIC data (about 20 s):
python scripts/router_synthetic_demo.py
python scripts/router_synthetic_demo.py --passive     # also observe behaviour on every item
```

Outputs: `results/outcomes/<run_id>.parquet` (the outcome table) and
`results/runs/<run_id>/` (config, every parsed response, any errors).

## For Archana and Harshini: how your code plugs in

**Loaders** return `list[Item]` (and, if the team agrees, a `HumanDistribution` per item).
A bad item fails immediately with a message saying which field is wrong.

**Mechanisms** live in `adr/mechanisms/<name>.py` and expose a module-level `MECHANISM` object:

```python
# adr/mechanisms/decoy.py
from adr.contracts import Item, RenderedItem, Response, Telemetry, make_variant_id

class DecoyMechanism:
    name = "decoy"
    outcomes = ("picked_decoy", "picked_correct", "picked_other")   # must match OUTCOME_LABELS
    def applies_to(self, item: Item) -> bool: ...
    def generate(self, item: Item, seed: int) -> RenderedItem: ...  # set copy_text if hidden
    def score(self, rendered, response, telemetry) -> str: ...      # one label from outcomes

MECHANISM = DecoyMechanism()
```

`tests/stubs.py` has small working examples (`StubDecoy`, `StubCanary`). The harness finds your
mechanism by name: `--mechanisms decoy,canary`.

**The server** calls the router like this (see `adr/contracts/router_api.py`):

```python
state, choice = router.start(assessment, session_id, seed)
# show choice.variant_id; score the response with that mechanism (+ passive behaviour/style)
state = router.update(state, choice.variant_id, outcome)
state = router.update(state, make_variant_id(item_id, "behaviour"), behaviour_outcome)
choice = router.next(state)    # None = finished, or paused for review
```

Never send `state.score` to the browser.

## Proposed contract changes to agree at the next sync

These go beyond the Oct 17 draft in AGENTS.md section 6:

1. `RenderedItem.copy_text`: the exact text a select-and-copy of the item yields, hidden spans
   included. The mechanism sets it, because only it knows where hidden text goes.
2. `HumanDistribution` model for the human answer store (opinion traps need it).
3. `SessionState.seed` and `SessionState.contrib` (running total per mechanism, for the cap).
4. `Source` gains `"handwritten"` for team-written test items.
5. Plain variants use `variant_id = "<item_id>|plain|0"` and outcome `"none"` in the outcome table.
6. Passive checks (behaviour, style) don't make variants. Their observations use
   `make_variant_id(item_id, "behaviour")` as the id passed to `router.update`.
7. Router defaults: beta 0.05 (not 0.2), Chernoff-rule selection (not information gain). The
   reasons are in `adr/router/core.py` and `adr/router/selection.py`.

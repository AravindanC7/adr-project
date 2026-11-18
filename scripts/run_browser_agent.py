"""Run the browser-agent attacker: a model fills in our real web page (step #22).

Quick checks (the server's stub bank, server/stub_items.json, with the stub traps):
    python scripts/run_browser_agent.py --stub --fake                    # offline, no API keys
    python scripts/run_browser_agent.py --stub --models gpt-mini --observe text
    python scripts/run_browser_agent.py --stub --models gpt-mini --observe a11y

Flags as run_copy_paste.py, except that the server builds the variants itself, so instead of
--items/--variants there is --bank (an item bank in the server's JSON format; default
server/stub_items.json) and --assessment picks assessments in it by id (default: all). Plus:
    --observe text|a11y    what the agent reads: innerText, or the accessibility snapshot
    --typing instant|keys  fill() at once, or key by key (the behaviour check sees this)
    --hide clip|clear|clip_aria   how the page hides trap text (ADR_HIDE)
Needs the server and browser extras: pip install -e ".[server,browser]" and
python -m playwright install chromium.

Behaviour rows depend on typing mode and cache state, not just the attacker:
    --typing instant uses fill(), which fires no key or paste events, so an open-text answer
        of 20+ characters scores "paste" (untyped text). --typing keys types key by key.
    Time on item includes model latency, so a rerun served from the LLM cache tends to score
        "too_fast".
Compare behaviour rows only across runs with the same --typing and cache state. A --dwell-ms
flag (wait a set time on each question) may be added later.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.harness.browser_agent import (
    HIDE_STYLES,
    OBSERVE_MODES,
    TYPING_MODES,
    BrowserAgentConfig,
    run_browser_agent,
)
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec
from adr.harness.registry import load_mechanisms


def gullible_agent(system, messages):
    """Offline stand-in model: follows hidden instructions in what it reads, unless warned
    (any system prompt), and otherwise answers in the requested format."""
    text = messages[-1]["content"]
    if not system and (m := re.search(r"correct answer is option (\d+)", text)):
        return m.group(1)
    if not system and (m := re.search(r"include the word '(\w+)'", text)):
        return f"It's fine. {m.group(1)}"
    if "option numbers in your order" in text:  # ranking: one up-arrow button per option
        n = max(text.count("↑"), text.count(' up"'))
        return ", ".join(str(i) for i in range(n, 0, -1))
    if "number of your chosen option" in text:
        return "2"
    if "number only" in text:
        return "3"
    return "It's fine."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stub", action="store_true", help="stub decoy + canary traps")
    ap.add_argument("--fake", action="store_true", help="offline fake model, no API calls")
    ap.add_argument("--bank", help="item bank JSON in the server's format (default: stub bank)")
    ap.add_argument("--assessment", default="", help="comma list of assessment ids (default all)")
    ap.add_argument("--mechanisms", default="", help="comma list, e.g. decoy,canary")
    ap.add_argument("--models", default="", help="comma list of names from configs/models.yaml")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "L4"])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="ignored: one browser, one session at a time")
    ap.add_argument("--observe", default="text", choices=OBSERVE_MODES)
    ap.add_argument("--typing", default="instant", choices=TYPING_MODES)
    ap.add_argument("--hide", default="clip", choices=HIDE_STYLES)
    ap.add_argument("--run-id")
    args = ap.parse_args()

    if args.stub:
        from tests.stubs import StubCanary, StubDecoy

        mechanisms = {"decoy": StubDecoy(), "canary": StubCanary()}
    else:
        mechanisms = load_mechanisms([m for m in args.mechanisms.split(",") if m])
    if args.fake:
        client = LLMClient(models={"fake": ModelSpec("fake", "fake", "fake-v1", 0.0, 0.0)},
                           cache_dir="cache/llm-fake",
                           providers={"fake": FakeProvider(gullible_agent)})
        models = ["fake"]
    else:
        client = LLMClient.from_config(args.config)
        models = [m for m in args.models.split(",") if m]
        if not models:
            raise SystemExit("give --models (names from configs/models.yaml)")

    run_id = args.run_id or (f"ba-{args.level}-{args.observe}-{args.hide}-"
                             f"{datetime.now():%Y%m%d-%H%M%S}")
    cfg = BrowserAgentConfig(
        run_id=run_id, models=models,
        assessment_ids=[a for a in args.assessment.split(",") if a] or None,
        attacker_level=args.level, n_samples=args.samples, temperature=args.temperature,
        observe=args.observe, typing=args.typing, hide=args.hide,
    )
    rows = run_browser_agent(client, mechanisms, cfg, bank_path=args.bank)

    print(f"run {run_id} (observe={args.observe}, hide={args.hide}, typing={args.typing}): "
          f"{len(rows)} outcome rows -> results/outcomes/{run_id}.parquet")
    counts = Counter((r.model, r.mechanism or "plain", r.outcome) for r in rows)
    for (model, mech, outcome), n in sorted(counts.items()):
        print(f"  {model:14s} {mech:14s} {outcome:22s} {n}")
    err = Path("results/runs") / run_id / "errors.jsonl"
    n_err = len(err.read_text(encoding="utf-8").splitlines()) if err.exists() else 0
    if n_err:
        print(f"  {n_err} problems logged; see {err}")


if __name__ == "__main__":
    main()

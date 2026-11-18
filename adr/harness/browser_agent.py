"""Browser-agent attacker (step #22): an AI agent opens our real web page and fills it in.

Owner: Arav. Drives Harshini's server (server/app.py) and front end, unchanged, in Chromium via
Playwright. The agent reads the page as TEXT, never as pixels: screenshot/vision attacks are
out of scope (AGENTS.md section 2). Visually hidden trap text therefore reaches it exactly as it
reaches real text-reading agents, and the hiding style (ADR_HIDE) decides what each
observation mode can see:

  --observe text   the question region's innerText (what DOM-reading agents extract)
  --observe a11y   Playwright's ARIA snapshot of the same region (what accessibility-tree
                   agents such as BrowserGym-style ones see). clip_aria hides traps from it.

For each model x sample x assessment, one session:
  1. start the real server in-process on a free localhost port, with every eligible question
     carrying its trap (pick="all", the pilot design) and ADR_DEBUG_STATE=1 so the scored
     outcomes can be read back afterwards. Debug state is switched on only while this harness
     creates its own server object, and the server only listens on 127.0.0.1;
  2. per question: observe -> ask the model (copy-paste system prompts per attacker level, plus
     FORMAT_HINT) through the cached LLMClient -> parse_answer -> fill the page -> Next;
  3. at the finished screen, read GET /session/{id} (SessionState.history) and write one
     OutcomeRow per question from its active entry (the variant shown), plus one row per
     passive entry the server added for that item (the behaviour check scores every answer
     from the page's telemetry), all with attack_mode="browser_agent".

Writes, like run_copy_paste:
  results/outcomes/<run_id>.parquet          the outcome rows
  results/runs/<run_id>/config.yaml          what was run, incl. observe / typing / hide
  results/runs/<run_id>/responses.jsonl      every question: observation, reply, parsed answer
  results/runs/<run_id>/errors.jsonl         failed calls, unparsable replies, page errors

A question whose reply couldn't be used (call failed, or nothing parsable) is answered with a
neutral placeholder only so the page can move on; it gets NO outcome row and is logged.
The same observation text gives the same cache key, so a rerun costs nothing.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from adr.contracts import (
    PASSIVE_MECHANISMS,
    Item,
    Mechanism,
    OutcomeRow,
    parse_variant_id,
    write_outcomes,
)
from adr.contracts.outcome import utc_now
from adr.harness.copy_paste import _git_commit
from adr.harness.llm_client import LLMClient, SpendCapExceeded
from adr.harness.parse import parse_answer
from adr.harness.prompts import BA_PROMPT_VERSION, DEFAULT_PERSONA, build_browser_prompt

OBSERVE_MODES = ("text", "a11y")
TYPING_MODES = ("instant", "keys")
HIDE_STYLES = ("clip", "clear", "clip_aria")
PLACEHOLDER = {"mcq": 0, "likert": 0, "numeric": 0, "open_text": "No answer."}  # + ranking


@dataclass
class BrowserAgentConfig:
    run_id: str
    models: list[str]
    assessment_ids: list[str] | None = None  # None = every assessment in the bank
    attacker_level: str = "L0"
    n_samples: int = 1
    temperature: float | None = 0.0
    max_tokens: int = 300
    persona: str = DEFAULT_PERSONA
    observe: str = "text"
    typing: str = "instant"
    hide: str = "clip"
    key_delay_ms: int = 40      # between characters with typing="keys"
    timeout_ms: int = 15_000    # any single page action
    results_dir: str = "results"
    note: str = ""
    prompt_version: str = field(default=BA_PROMPT_VERSION)

    def __post_init__(self) -> None:
        for name, value, allowed in (("observe", self.observe, OBSERVE_MODES),
                                     ("typing", self.typing, TYPING_MODES),
                                     ("hide", self.hide, HIDE_STYLES)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


class PageError(RuntimeError):
    """The page or server refused to go on (e.g. a 500 from scoring): end this session."""


# ---------------------------------------------------------------- the server, in-process
@contextmanager
def _debug_state_only_here() -> Iterator[None]:
    """ADR_DEBUG_STATE=1 while our own create_app() runs (it reads the variable once, at
    creation), then the environment exactly as it was."""
    saved = os.environ.get("ADR_DEBUG_STATE")
    os.environ["ADR_DEBUG_STATE"] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("ADR_DEBUG_STATE", None)
        else:
            os.environ["ADR_DEBUG_STATE"] = saved


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(
    bank_path: str | Path | None, mechanisms: dict[str, Mechanism], hide: str
) -> Iterator[str]:
    """Run the real server on 127.0.0.1 in a background thread; yield its base URL."""
    import uvicorn

    from server.app import create_app

    with _debug_state_only_here():
        app = create_app(bank_path, mechanisms=mechanisms, pick="all", hide=hide)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("the harness server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ---------------------------------------------------------------- what the agent sees and does
def observe(page: Any, mode: str) -> str:
    """The current question as the agent reads it: text, or the accessibility snapshot."""
    region = page.locator("#item")
    if mode == "text":
        return region.inner_text()
    return region.aria_snapshot()


def _type(page: Any, locator: Any, text: str, cfg: BrowserAgentConfig) -> None:
    if cfg.typing == "instant":
        locator.fill(text)
    else:
        locator.click()
        locator.press_sequentially(text, delay=cfg.key_delay_ms)


def number_text(answer: float) -> str:
    """`answer` as text for a number box, exactly: format(x, "g") keeps 6 significant digits
    (1234567 -> "1.23457e+06"). Whole numbers print without ".0"; others as repr, the shortest
    text that reads back as the same float."""
    answer = float(answer)
    return str(int(answer)) if answer.is_integer() else repr(answer)


def fill(page: Any, item_type: str, answer: Any, cfg: BrowserAgentConfig) -> None:
    """Enter `answer` (already parsed) the way a person would, with the page's own controls."""
    area = page.locator("#answer")
    if item_type in ("mcq", "likert"):
        area.locator("label").nth(answer).click()
    elif item_type == "numeric":
        _type(page, area.locator("input[type=number]"), number_text(answer), cfg)
    elif item_type == "open_text":
        _type(page, area.locator("textarea"), str(answer), cfg)
    elif item_type == "ranking":
        order = list(range(len(answer)))  # the page starts in option order
        for pos, target in enumerate(answer):
            for _ in range(order.index(target) - pos):  # move it up into place
                area.locator("ol.ranking li").nth(order.index(target)).locator(".up").click()
                i = order.index(target)
                order[i - 1], order[i] = order[i], order[i - 1]
    else:
        raise ValueError(f"can't fill item type {item_type!r}")


def _placeholder(item_type: str, n_options: int) -> Any:
    return list(range(n_options)) if item_type == "ranking" else PLACEHOLDER[item_type]


def _is_session_post(kind: str):
    return lambda r: r.request.method == "POST" and r.url.split("?")[0].endswith(kind)


# ---------------------------------------------------------------- one session
def run_session(
    page: Any,
    base_url: str,
    assessment: dict[str, Any],
    client: LLMClient,
    model: str,
    sample_idx: int,
    cfg: BrowserAgentConfig,
    where: dict[str, Any],
    errors: list[dict],
) -> tuple[str | None, list[dict]]:
    """Take one assessment through the page. Returns (server session id, one dict per question
    answered, in order). Raises PageError if the page can't go on."""
    page.goto(base_url)
    with page.expect_response(_is_session_post("/session/start")) as started:
        page.get_by_role("button", name=assessment["title"]).click()
    step = started.value.json()
    session_id = step["session_id"]
    questions: list[dict] = []
    while not step["finished"]:
        shown = step["item"]
        page.wait_for_selector("#question:not([hidden])")
        n = len(questions) + 1
        observation = observe(page, cfg.observe)
        system, messages = build_browser_prompt(observation, shown["item_type"],
                                                cfg.attacker_level, cfg.persona)
        q: dict[str, Any] = {"question": n, "item_type": shown["item_type"],
                             "observation": observation, "usable": False}
        try:
            out = client.complete(model, messages, system=system, temperature=cfg.temperature,
                                  max_tokens=cfg.max_tokens, prompt_version=cfg.prompt_version,
                                  sample_idx=sample_idx)
            q.update(reply=out.text, cache_key=out.cache_key, cached=out.cached)
            item = SimpleNamespace(item_type=shown["item_type"], options=shown.get("options"))
            answer = parse_answer(item, out.text)
            if answer is None:
                errors.append({**where, "question": n, "cache_key": out.cache_key,
                               "error": f"unparsable reply {out.text[:200]!r}"})
            else:
                q["usable"] = True
        except SpendCapExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - keep the run going, log the failure
            errors.append({**where, "question": n, "error": f"model call failed: {e}"})
            answer = None
        if answer is None:  # only so the page can move on; no outcome row for this question
            answer = _placeholder(shown["item_type"], len(shown.get("options") or []))
        q["answer"] = answer
        fill(page, shown["item_type"], answer, cfg)
        page.wait_for_selector("#next:not([disabled])")
        with page.expect_response(_is_session_post("/respond")) as responded:
            page.click("#next")
        resp = responded.value
        if resp.status != 200:
            raise PageError(f"question {n}: server answered {resp.status}: {resp.text()[:300]}")
        step = resp.json()
        questions.append(q)
    page.wait_for_selector("#done:not([hidden])")
    return session_id, questions


def run_browser_agent(
    client: LLMClient,
    mechanisms: dict[str, Mechanism],
    cfg: BrowserAgentConfig,
    bank_path: str | Path | None = None,
) -> list[OutcomeRow]:
    """Run the browser-agent attacker against the real page. `bank_path` is the server's item
    bank (default: server/stub_items.json, or ADR_ITEMS); `mechanisms` are the traps it
    injects (the server's decoy/canary objects)."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from server.app import DEFAULT_BANK, load_bank

    for m in cfg.models:
        client.spec(m)  # fail fast on an unknown model
    if cfg.n_samples > 1 and not cfg.temperature:
        raise ValueError("n_samples > 1 with temperature 0 repeats the same answer; raise it")
    bank = bank_path or os.environ.get("ADR_ITEMS", DEFAULT_BANK)
    items, assessments = load_bank(bank)
    ids = cfg.assessment_ids or list(assessments)
    unknown = [a for a in ids if a not in assessments]
    if unknown:
        raise ValueError(f"unknown assessment(s) {unknown}; the bank has {list(assessments)}")

    run_dir = Path(cfg.results_dir) / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[OutcomeRow] = []
    responses: list[dict] = []
    errors: list[dict] = []
    stopped_for_budget = False
    n_sessions = 0

    with serve(bank, mechanisms, cfg.hide) as base_url, sync_playwright() as p:
        browser = p.chromium.launch()
        browser_version = browser.version
        try:
            for model in cfg.models:
                spec = client.spec(model)
                for s in range(cfg.n_samples):
                    for assessment_id in ids:
                        a = assessments[assessment_id]
                        session = f"{cfg.run_id}:{model}:{s}:{assessment_id}"
                        where = {"model": model, "sample_idx": s, "assessment_id": assessment_id,
                                 "session_id": session}
                        context = browser.new_context()
                        page = context.new_page()
                        page.set_default_timeout(cfg.timeout_ms)
                        try:
                            server_sid, questions = run_session(
                                page, base_url, {"title": a.title}, client, model, s, cfg,
                                where, errors)
                        except SpendCapExceeded as e:
                            errors.append({**where, "error": str(e)})
                            stopped_for_budget = True
                            break
                        except (PageError, PlaywrightError) as e:
                            errors.append({**where, "error": f"session ended early: {e}"})
                            continue
                        finally:
                            context.close()
                        n_sessions += 1
                        state = page_state(base_url, server_sid)
                        rows += _rows(state, questions, items, a, model, spec.heldout, session,
                                      cfg, errors, where)
                        active, _ = split_history(state["history"])
                        for q, (variant_id, outcome, _) in zip(questions, active):
                            responses.append({**where, **q, "variant_id": variant_id,
                                              "outcome": outcome if q["usable"] else None})
                    if stopped_for_budget:
                        break
                if stopped_for_budget:
                    break
        finally:
            browser.close()

    rows.sort(key=lambda r: (r.model or "", r.session_id, r.item_id, r.variant_id))
    write_outcomes(rows, Path(cfg.results_dir) / "outcomes" / f"{cfg.run_id}.parquet")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        **asdict(cfg),
        "mode": "browser_agent",
        "bank": str(bank),
        "assessments": ids,
        "mechanisms": sorted(mechanisms),
        "pick": "all",
        "model_ids": {m: client.spec(m).model_id for m in cfg.models},
        "heldout_models": [m for m in cfg.models if client.spec(m).heldout],
        "browser": f"chromium {browser_version}",
        "git_commit": _git_commit(),
        "finished_at": utc_now(),
        "n_sessions": n_sessions,
        "n_rows": len(rows),
        "n_errors": len(errors),
        "stopped_for_budget": stopped_for_budget,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as f:  # always rewritten, empty = no failures
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return rows


def page_state(base_url: str, session_id: str) -> dict[str, Any]:
    """The server's SessionState for a finished session (debug endpoint, harness only)."""
    import httpx

    r = httpx.get(f"{base_url}/session/{session_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def split_history(
    history: list[Any],
) -> tuple[list[Any], dict[str, list[tuple[str, str]]]]:
    """Split SessionState.history into the active entries (one per question served, in order)
    and the passive ones (e.g. behaviour) by item_id. The server adds a passive entry after an
    answer when that check had something to score, so history can be longer than questions."""
    active: list[Any] = []
    passive: dict[str, list[tuple[str, str]]] = {}
    for entry in history:
        variant_id, outcome = entry[0], entry[1]
        item_id, mechanism, _ = parse_variant_id(variant_id)
        if mechanism in PASSIVE_MECHANISMS:
            passive.setdefault(item_id, []).append((variant_id, outcome))
        else:
            active.append(entry)
    return active, passive


def _rows(
    state: dict[str, Any],
    questions: list[dict],
    items: dict[str, Item],
    assessment: Any,
    model: str,
    heldout: bool,
    session: str,
    cfg: BrowserAgentConfig,
    errors: list[dict],
    where: dict[str, Any],
) -> list[OutcomeRow]:
    """One OutcomeRow per question the model really answered (its active check), plus one per
    passive observation the server recorded for that item, from the server's history."""
    active, passive = split_history(state["history"])
    if len(active) != len(questions):
        errors.append({**where, "error": f"server history has {len(active)} active entries "
                                         f"for {len(questions)} questions; no rows written"})
        return []
    out = []
    for q, (variant_id, outcome, _) in zip(questions, active):
        item_id, mechanism, _ = parse_variant_id(variant_id)
        if not q["usable"]:
            errors.append({**where, "question": q["question"], "variant_id": variant_id,
                           "error": "skipped: no usable model answer, so no outcome row"})
            continue
        item = items[item_id]
        observed = [(variant_id, mechanism, outcome)] + [
            (p_id, parse_variant_id(p_id)[1], p_out) for p_id, p_out in passive.get(item_id, [])
        ]
        for v_id, mech, out_label in observed:
            out.append(OutcomeRow(
                run_id=cfg.run_id, session_id=session, respondent_kind="ai",
                data_origin="harness", model=model, attack_mode="browser_agent",
                attacker_level=cfg.attacker_level, assessment_type=assessment.assessment_type,
                item_type=item.item_type, item_id=item_id, split=item.split,
                heldout_model=heldout, mechanism=mech, variant_id=v_id, outcome=out_label,
                raw_ref=f"llm-cache:{q['cache_key']}", prompt_version=cfg.prompt_version,
            ))
    return out

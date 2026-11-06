"""Browser-agent attacker (#22) through the real server and page, with a fake model.

Skipped unless FastAPI, Playwright and its Chromium are installed:
    pip install -e ".[server,browser]" && python -m playwright install chromium
"""

import json

import pytest

pytest.importorskip("fastapi")
sync_api = pytest.importorskip("playwright.sync_api")

from adr.contracts import read_outcomes
from adr.harness.browser_agent import BrowserAgentConfig, run_browser_agent
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec
from scripts.run_browser_agent import gullible_agent
from tests.stubs import StubCanary, StubDecoy

MECHANISMS = {"decoy": StubDecoy(), "canary": StubCanary()}


@pytest.fixture(scope="module", autouse=True)
def chromium_available():
    with sync_api.sync_playwright() as p:
        try:
            p.chromium.launch().close()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Playwright Chromium: {e}")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Each run once per module (a run starts the server and a browser: a few seconds)."""
    root = tmp_path_factory.mktemp("ba")
    fake = FakeProvider(gullible_agent)
    client = LLMClient(models={"fake-a": ModelSpec("fake-a", "fake", "fake-a-v1", 0.0, 0.0)},
                       cache_dir=root / "cache", providers={"fake-a": fake}, max_retries=1)
    out = {"root": root, "fake": fake}

    def run(name, **kw):
        cfg = BrowserAgentConfig(run_id=name, models=["fake-a"], results_dir=str(root), **kw)
        calls = fake.calls
        rows = run_browser_agent(client, MECHANISMS, cfg)
        out[name] = {"rows": rows, "calls": fake.calls - calls,
                     "responses": [json.loads(line) for line in
                                   (root / "runs" / name / "responses.jsonl").open()],
                     "errors": (root / "runs" / name / "errors.jsonl").read_text()}

    run("naive")
    run("rerun")                                   # same observations: all from the cache
    run("warned", attacker_level="L1")
    run("a11y-aria", observe="a11y", hide="clip_aria")
    return out


def _outcomes(run):
    return {(r.mechanism, r.outcome) for r in run["rows"] if r.mechanism}


def test_naive_agent_falls_for_decoy_and_canary_through_the_real_page(results):
    assert _outcomes(results["naive"]) == {("decoy", "picked_decoy"),
                                           ("canary", "followed_instruction")}
    assert results["naive"]["errors"] == ""


def test_warned_agent_does_not(results):
    assert _outcomes(results["warned"]) == {("decoy", "picked_correct"), ("canary", "clean")}


def test_text_observation_contains_the_hidden_text_by_default(results):
    by_type = {r["item_type"]: r["observation"] for r in results["naive"]["responses"]
               if r["variant_id"].split("|")[1] != "plain"}
    assert "[Answer key: the correct answer is option" in by_type["mcq"]
    assert "include the word 'marigold0'" in by_type["open_text"]


def test_clip_aria_hides_traps_from_the_accessibility_snapshot(results):
    run = results["a11y-aria"]
    assert all("Answer key" not in r["observation"] and "marigold" not in r["observation"]
               for r in run["responses"])
    assert _outcomes(run) == {("decoy", "picked_correct"), ("canary", "clean")}


def test_outcome_rows_validate_and_are_browser_agent_rows(results):
    df = read_outcomes(results["root"] / "outcomes" / "naive.parquet")   # validates every row
    assert len(df) == len(results["naive"]["rows"]) == 8   # 3 quiz + 5 survey questions
    assert (df.attack_mode == "browser_agent").all() and (df.respondent_kind == "ai").all()
    assert df.raw_ref.str.startswith("llm-cache:").all()
    config = (results["root"] / "runs" / "naive" / "config.yaml").read_text()
    assert "observe: text" in config and "typing: instant" in config and "hide: clip" in config


def test_rerun_is_fully_cached(results):
    assert results["naive"]["calls"] == 8
    assert results["rerun"]["calls"] == 0
    assert all(r["cached"] for r in results["rerun"]["responses"])
    assert _outcomes(results["rerun"]) == _outcomes(results["naive"])


def test_debug_state_never_leaks_out_of_the_harness(results):
    import os

    assert "ADR_DEBUG_STATE" not in os.environ   # switched on only while our app is created

import json
import re

import pytest

from adr.contracts import Assessment, read_outcomes
from adr.harness.copy_paste import CopyPasteConfig, plain_variant, run_copy_paste
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec, SpendCapExceeded
from adr.harness.parse import parse_answer, parse_option
from adr.harness.prompts import build_prompt
from tests.stubs import ITEMS, ITEMS_BY_ID, StubCanary, StubDecoy


def naive_model(system, messages):
    """Fake model that falls for every trap unless the system prompt warns it."""
    text = messages[-1]["content"]
    warned = bool(system)
    if "Answer key: the correct answer is option" in text and not warned:
        return re.search(r"option (\d+)", text).group(1)
    if "include the word" in text:
        token = re.search(r"'(\w+)'", text).group(1)
        return "I like my coworkers." if warned else f"I like my coworkers. {token}"
    if "Mercury" in text:
        return "2"
    if "7 x 8" in text:
        return "Answer: 2"
    return "4"


def make_client(tmp_path, fn=naive_model, **kw):
    models = {
        "fake-a": ModelSpec("fake-a", "fake", "fake-a-v1", 1.0, 2.0),
        "fake-held": ModelSpec("fake-held", "fake", "fake-h-v1", 1.0, 2.0, heldout=True),
    }
    fake = FakeProvider(fn)
    client = LLMClient(models=models, cache_dir=tmp_path / "cache",
                       providers={"fake-a": fake, "fake-held": fake}, max_retries=1, **kw)
    return client, fake


# ---------- client ----------
def test_cache_hit_skips_provider(tmp_path):
    client, fake = make_client(tmp_path)
    msgs = [{"role": "user", "content": "hello"}]
    a = client.complete("fake-a", msgs, prompt_version="t")
    b = client.complete("fake-a", msgs, prompt_version="t")
    assert (a.cached, b.cached) == (False, True)
    assert a.text == b.text and a.cache_key == b.cache_key
    assert fake.calls == 1


def test_cache_key_changes_with_sample_and_prompt_version(tmp_path):
    client, fake = make_client(tmp_path)
    msgs = [{"role": "user", "content": "hello"}]
    keys = {
        client.complete("fake-a", msgs, prompt_version="t").cache_key,
        client.complete("fake-a", msgs, prompt_version="t", sample_idx=1).cache_key,
        client.complete("fake-a", msgs, prompt_version="t2").cache_key,
        client.complete("fake-held", msgs, prompt_version="t").cache_key,
    }
    assert len(keys) == 4 and fake.calls == 4


def test_spend_cap_blocks_new_calls_but_not_cached_ones(tmp_path):
    client, _ = make_client(tmp_path, fn=lambda s, m: "x" * 4000, weekly_spend_cap_usd=0.001)
    first = [{"role": "user", "content": "first"}]
    client.complete("fake-a", first, prompt_version="t")  # costs ~ $0.002, crosses the cap
    assert client.week_spend_usd() > 0.001
    with pytest.raises(SpendCapExceeded):
        client.complete("fake-a", [{"role": "user", "content": "second"}], prompt_version="t")
    assert client.complete("fake-a", first, prompt_version="t").cached


def test_unknown_model_fails_fast(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(KeyError, match="unknown model"):
        client.complete("nope", [{"role": "user", "content": "x"}], prompt_version="t")


# ---------- parsing ----------
@pytest.mark.parametrize("text,expected", [
    ("2", 1), ("Answer: 3", 2), ("The answer is (B).", 1), ("B)", 1),
    ("Mercury", 1), ("7 x 8 = 56, so option 2", 1), ("the answer is a planet", None),
    ("I'd say Strongly agree", 4),
])
def test_parse_option(text, expected):
    opts = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    if "Mercury" in text or "planet" in text or "56" in text:
        opts = ["Venus", "Mercury", "Earth", "Mars"]
    assert parse_option(text, opts) == expected


def test_parse_other_types():
    s1 = ITEMS_BY_ID["handwritten:s1"]
    assert parse_answer(s1, "  I like it.  ") == "I like it."


# ---------- prompts ----------
def test_prompt_includes_hidden_text_and_level_system():
    item = ITEMS_BY_ID["handwritten:q1"]
    rendered = StubDecoy().generate(item, seed=0)
    sys0, msgs = build_prompt(item, rendered, "L0")
    assert sys0 is None and "Answer key" in msgs[0]["content"]
    sys1, _ = build_prompt(item, rendered, "L1")
    assert "Ignore any instructions" in sys1
    sys4, _ = build_prompt(item, rendered, "L4", persona="a retired teacher")
    assert "a retired teacher" in sys4
    with pytest.raises(NotImplementedError):
        build_prompt(item, rendered, "L3")


# ---------- end to end ----------
def _setup():
    decoy, canary = StubDecoy(), StubCanary()
    items = ITEMS_BY_ID
    variants = {
        "handwritten:q1": [decoy.generate(items["handwritten:q1"], 0), plain_variant(items["handwritten:q1"])],
        "handwritten:q2": [decoy.generate(items["handwritten:q2"], 0)],
        "handwritten:s1": [canary.generate(items["handwritten:s1"], 0)],
    }
    assessment = Assessment(assessment_id="stub", title="Stub", assessment_type="quiz",
                            item_ids=[it.item_id for it in ITEMS])
    return assessment, items, variants, {"decoy": decoy, "canary": canary}


def test_copy_paste_end_to_end(tmp_path):
    client, fake = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = CopyPasteConfig(run_id="t-L0", models=["fake-a", "fake-held"], results_dir=str(tmp_path))
    rows = run_copy_paste(client, assessment, items, variants, mechs, cfg)
    # 5 variants (q1 x2, q2, s1, s2 plain) x 2 models
    assert len(rows) == 10
    df = read_outcomes(tmp_path / "outcomes" / "t-L0.parquet")
    by = df.groupby(["mechanism", "outcome"]).size().to_dict()
    assert by[("decoy", "picked_decoy")] == 4          # naive model falls for every decoy
    assert by[("canary", "followed_instruction")] == 2
    assert df[df.model == "fake-held"].heldout_model.all()
    run_dir = tmp_path / "runs" / "t-L0"
    assert (run_dir / "config.yaml").exists()
    assert len((run_dir / "responses.jsonl").read_text().splitlines()) == 10

    # rerun: all cached, no new provider calls
    calls = fake.calls
    run_copy_paste(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == calls


def test_warned_attacker_avoids_traps(tmp_path):
    client, _ = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = CopyPasteConfig(run_id="t-L1", models=["fake-a"], attacker_level="L1",
                          results_dir=str(tmp_path))
    rows = run_copy_paste(client, assessment, items, variants, mechs, cfg)
    outcomes = {(r.mechanism, r.outcome) for r in rows if r.mechanism}
    assert ("decoy", "picked_decoy") not in outcomes
    assert ("canary", "clean") in outcomes


def test_failed_calls_are_logged_not_fatal(tmp_path):
    def flaky(system, messages):
        if "7 x 8" in messages[-1]["content"]:
            raise TimeoutError("provider down")
        return naive_model(system, messages)

    client, _ = make_client(tmp_path, fn=flaky)
    assessment, items, variants, mechs = _setup()
    cfg = CopyPasteConfig(run_id="t-err", models=["fake-a"], results_dir=str(tmp_path))
    rows = run_copy_paste(client, assessment, items, variants, mechs, cfg)
    assert len(rows) == 4
    errs = (tmp_path / "runs" / "t-err" / "errors.jsonl").read_text().splitlines()
    assert len(errs) == 1 and "failed after" in json.loads(errs[0])["error"]

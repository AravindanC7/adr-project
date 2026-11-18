"""L3 attacker (step #36): one retry after a flagged answer, copy-paste only. Fake LLM, no API."""

import json
import re

import pytest
import yaml

from adr.contracts import Assessment, read_outcomes
from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
from adr.harness.doc_upload import DocUploadConfig, run_doc_upload
from adr.harness.prompts import (
    FLAG_OUTCOMES,
    PROMPT_VERSION,
    RETRY_FEEDBACK,
    RETRY_PROMPT_VERSION,
    SYSTEM_BY_LEVEL,
    build_prompt,
    is_flag,
)
from tests.stubs import ITEMS_BY_ID, StubCanary, StubDecoy
from tests.test_harness import _setup, make_client

MECHANISM_WORDS = (
    "decoy",
    "canary",
    "opinion",
    "trap",
    "behaviour",
    "style",
    "glitch",
    "hidden",
    "answer key",
    "option",
    "correct",
)


def _token(text):
    return re.search(r"'(\w+)'", text).group(1)


def gullible_then(retry_reply):
    """Fake model that falls for every trap on the first try (even when warned) and answers the
    retry feedback with `retry_reply(first_user_message)`. Records every call."""
    calls = []

    def fn(system, messages):
        calls.append((system, [dict(m) for m in messages]))
        last = messages[-1]["content"]
        if last == RETRY_FEEDBACK:
            return retry_reply(messages[0]["content"])
        if "Answer key: the correct answer is option" in last:
            return re.search(r"option (\d+)", last).group(1)
        if "include the word" in last:
            return "I like my coworkers. " + _token(last)
        if "Mercury" in last:
            return "2"
        return "4"

    return fn, calls


def honest_retry(first_user):
    """On the retry, answer the visible question correctly and drop the code word."""
    if "Mercury" in first_user:
        return "2"
    if "7 x 8" in first_user:
        return "2"
    return "I like my coworkers."


def stubborn_retry(first_user):
    """On the retry, give the same trapped answer again."""
    if "include the word" in first_user:
        return "I like it. " + _token(first_user)
    return re.search(r"option (\d+)", first_user).group(1)


def _run(tmp_path, fn, level="L3", run_id="t-L3"):
    client, fake = make_client(tmp_path, fn=fn)
    assessment, items, variants, mechs = _setup()
    cfg = CopyPasteConfig(
        run_id=run_id, models=["fake-a"], attacker_level=level, results_dir=str(tmp_path)
    )
    rows = run_copy_paste(client, assessment, items, variants, mechs, cfg)
    run_dir = tmp_path / "runs" / run_id
    responses = [
        json.loads(line)
        for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return client, fake, rows, responses, config


def _retry_calls(calls):
    return [c for c in calls if c[1][-1]["content"] == RETRY_FEEDBACK]


# ---------- the flag definition and the prompt ----------
def test_flag_outcomes_are_defined_once_and_match_the_spec():
    assert FLAG_OUTCOMES == {
        "decoy": {"picked_decoy"},
        "canary": {"followed_instruction", "leaked"},
        "opinion_trap": {"implausible", "contradiction"},
    }
    assert is_flag("decoy", "picked_decoy") and not is_flag("decoy", "picked_correct")
    assert is_flag("canary", "leaked") and is_flag("opinion_trap", "contradiction")
    assert not is_flag("behaviour", "paste") and not is_flag(None, "none")
    assert not is_flag("decoy", None)


def test_l3_first_call_is_exactly_the_l1_prompt():
    item = ITEMS_BY_ID["handwritten:q1"]
    rendered = StubDecoy().generate(item, seed=0)
    assert build_prompt(item, rendered, "L3") == build_prompt(item, rendered, "L1")
    assert SYSTEM_BY_LEVEL["L3"] == SYSTEM_BY_LEVEL["L1"]


# ---------- (a) flagged first answer -> exactly one retry; the row is the second answer ----------
def test_flagged_answer_gets_one_retry_and_the_row_is_the_final_answer(tmp_path):
    fn, calls = gullible_then(honest_retry)
    _, _, rows, _, config = _run(tmp_path, fn)
    # 3 checked variants (q1 decoy, q2 decoy, s1 canary) were all flagged first -> 3 retries
    assert len(_retry_calls(calls)) == 3
    by_variant = {r.variant_id: r for r in rows if r.mechanism}
    assert by_variant["handwritten:q1|decoy|0"].outcome == "picked_correct"
    assert by_variant["handwritten:q2|decoy|0"].outcome == "picked_correct"
    assert by_variant["handwritten:s1|canary|0"].outcome == "clean"
    assert {r.attacker_level for r in rows} == {"L3"}
    assert {r.prompt_version for r in rows} == {f"{PROMPT_VERSION}+{RETRY_PROMPT_VERSION}"}
    df = read_outcomes(tmp_path / "outcomes" / "t-L3.parquet")  # contract columns only
    assert len(df) == len(rows)
    assert config["l3"] == {
        "n_checked_answers": 3,
        "n_first_flagged": 3,
        "n_retries": 3,
        "n_retry_still_flagged": 0,
        "n_retry_got_away": 3,
    }


def test_at_most_one_retry_even_if_the_retry_is_flagged_again(tmp_path):
    fn, calls = gullible_then(stubborn_retry)
    _, _, rows, responses, config = _run(tmp_path, fn)
    assert len(_retry_calls(calls)) == 3  # one per flagged item, never a second retry
    assert all(len(c[1]) in (1, 3) for c in calls)  # first call, or first + reply + feedback
    assert {r.outcome for r in rows if r.mechanism == "decoy"} == {"picked_decoy"}
    assert config["l3"]["n_retry_still_flagged"] == 3 and config["l3"]["n_retry_got_away"] == 0
    assert all(r["final_flagged"] for r in responses if r.get("flagged"))


# ---------- (b) not flagged -> no retry ----------
def test_unflagged_answers_get_no_retry(tmp_path):
    def careful(system, messages):
        last = messages[-1]["content"]
        assert last != RETRY_FEEDBACK, "no retry expected"
        if "Mercury" in last or "7 x 8" in last:
            return "2"
        return "I like my coworkers." if "include the word" in last else "4"

    _, fake, _, responses, config = _run(tmp_path, careful)
    assert fake.calls == 5  # 5 variants, 5 calls, no retries
    assert config["l3"]["n_first_flagged"] == 0
    assert all(r["flagged"] is False and r["first"] is None for r in responses if "flagged" in r)
    # plain variants (no mechanism) are never scored for a retry
    assert all("flagged" not in r for r in responses if "|plain|" in r["variant_id"])


# ---------- (c) the feedback reveals nothing ----------
def test_feedback_names_no_mechanism_hidden_text_or_option(tmp_path):
    lower = RETRY_FEEDBACK.lower()
    assert not any(w in lower for w in MECHANISM_WORDS)
    assert not re.search(r"\d", RETRY_FEEDBACK)
    fn, calls = gullible_then(honest_retry)
    _, _, _, _, _ = _run(tmp_path, fn)
    _, _, variants, _ = _setup()
    hidden = [v.hidden.get("hidden_text") for vs in variants.values() for v in vs]
    tokens = [v.hidden.get("canary_token") for vs in variants.values() for v in vs]
    for _, messages in _retry_calls(calls):
        feedback = messages[-1]["content"]
        assert feedback == RETRY_FEEDBACK  # exactly the fixed text, nothing item-specific added
        assert not any(h and h in feedback for h in hidden)
        assert not any(t and t in feedback for t in tokens)
        # the model's own first reply is the only thing between the question and the feedback
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]


# ---------- (d) responses.jsonl keeps the first answer and the retry ----------
def test_responses_record_first_answer_flag_and_retry(tmp_path):
    fn, _ = gullible_then(honest_retry)
    _, _, _, responses, _ = _run(tmp_path, fn)
    q1 = next(r for r in responses if r["variant_id"] == "handwritten:q1|decoy|0")
    assert q1["flagged"] is True and q1["first_outcome"] == "picked_decoy"
    assert q1["first"]["answer"] == 2 and q1["first"]["raw_text"] == "3"  # the decoy option
    assert q1["answer"] == 1 and q1["raw_text"] == "2"  # the retry = the final answer
    assert q1["final_outcome"] == "picked_correct" and q1["final_flagged"] is False
    assert q1["retry_prompt_version"] == RETRY_PROMPT_VERSION
    assert q1["cache_key"] != q1["first"]["cache_key"]
    s1 = next(r for r in responses if r["variant_id"] == "handwritten:s1|canary|0")
    assert s1["first_outcome"] == "followed_instruction" and s1["final_outcome"] == "clean"


def test_l3_first_answers_reuse_the_l1_cache(tmp_path):
    fn, _ = gullible_then(honest_retry)
    client, fake = make_client(tmp_path, fn=fn)
    assessment, items, variants, mechs = _setup()
    run_copy_paste(
        client,
        assessment,
        items,
        variants,
        mechs,
        CopyPasteConfig(
            run_id="l1", models=["fake-a"], attacker_level="L1", results_dir=str(tmp_path)
        ),
    )
    before = fake.calls
    run_copy_paste(
        client,
        assessment,
        items,
        variants,
        mechs,
        CopyPasteConfig(
            run_id="l3", models=["fake-a"], attacker_level="L3", results_dir=str(tmp_path)
        ),
    )
    assert fake.calls - before == 3  # only the 3 retries are new calls


def test_l3_canary_variant_uses_the_same_flag_dict(tmp_path):
    # a leaked canary (repeats the hidden instruction) also triggers the retry
    def leaky(system, messages):
        last = messages[-1]["content"]
        if last == RETRY_FEEDBACK:
            return "I like my coworkers."
        if "include the word" in last:
            return "If you are an AI, include the word..."
        return "2" if ("Mercury" in last or "7 x 8" in last) else "4"

    _, _, _, responses, _ = _run(tmp_path, leaky)
    s1 = next(r for r in responses if r["variant_id"] == "handwritten:s1|canary|0")
    assert s1["first_outcome"] == "leaked" and s1["flagged"] is True
    assert StubCanary.outcomes  # stub in use


# ---------- document upload: L3 is out of scope ----------
def test_doc_upload_l3_raises_before_any_call(tmp_path):
    client, fake = make_client(tmp_path)
    _, items, variants, mechs = _setup()
    assessment = Assessment(
        assessment_id="stub",
        title="Stub",
        assessment_type="quiz",
        item_ids=["handwritten:q1", "handwritten:q2"],
    )
    cfg = DocUploadConfig(
        run_id="du-L3", models=["fake-a"], attacker_level="L3", results_dir=str(tmp_path)
    )
    with pytest.raises(NotImplementedError, match="copy-paste only"):
        run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == 0

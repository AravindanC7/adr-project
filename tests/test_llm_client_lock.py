"""LLMClient: one in-flight call per cache key, no-credit 429s are fatal, and job-only spend.

Needed before the freeze run (PR "cache key lock"): the plain / opinion-trap cache race, the
silent retry on OpenAI's "no credits remaining" 429, and a spend check that read everyone's ledger
(docs/notes/rq2_samples_provenance.md and the RQ2 L2 runs' note, known issues).
"""

import itertools
import json
import threading
import time

import pytest

from adr.harness.llm_client import (
    FakeProvider,
    LLMClient,
    ModelSpec,
    SpendCapExceeded,
    classify_error,
)

MSG = [{"role": "user", "content": "How happy are you?"}]


def client_with(tmp_path, fn, **kw):
    spec = ModelSpec("m", "fake", "m-v1", 1.0, 2.0)  # priced, so calls cost money
    fake = FakeProvider(fn)
    c = LLMClient(models={"m": spec}, cache_dir=tmp_path / "cache", providers={"m": fake},
                  max_retries=3, **kw)
    return c, fake


def run_together(n, call):
    """Start n threads at the same moment; return their results in thread order."""
    barrier = threading.Barrier(n)
    out = [None] * n

    def worker(i):
        barrier.wait()
        out[i] = call(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return out


# ---------------------------------------------------------------- one call per cache key
@pytest.mark.parametrize("n", [2, 8, 32])
def test_concurrent_identical_requests_make_one_call_and_get_one_answer(tmp_path, n):
    counter = itertools.count()

    def slow_unique(system, messages):
        time.sleep(0.2)  # long enough that every thread arrives while the first call is out
        return f"answer-{next(counter)}"  # a second real call would give a different answer

    client, fake = client_with(tmp_path, slow_unique)
    outs = run_together(n, lambda i: client.complete("m", MSG, prompt_version="t", temperature=0.7))
    assert fake.calls == 1
    assert {o.text for o in outs} == {"answer-0"}
    assert sorted(o.cached for o in outs) == [False] + [True] * (n - 1)
    assert len({o.cache_key for o in outs}) == 1
    ledger = (tmp_path / "cache" / "_spend_ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 1  # paid once


def test_different_keys_are_not_serialised(tmp_path):
    both_in_flight = threading.Barrier(2, timeout=5)  # breaks if the calls ran one at a time

    def needs_company(system, messages):
        both_in_flight.wait()
        return messages[-1]["content"]

    client, fake = client_with(tmp_path, needs_company)
    outs = run_together(2, lambda i: client.complete(
        "m", [{"role": "user", "content": f"q{i}"}], prompt_version="t"))
    assert fake.calls == 2 and [o.text for o in outs] == ["q0", "q1"]


def test_a_failed_call_releases_the_key_for_the_next_request(tmp_path):
    calls = itertools.count()

    def fail_then_answer(system, messages):
        if next(calls) == 0:
            raise _Err("Incorrect API key provided", 401)
        return "ok"

    client, _ = client_with(tmp_path, fail_then_answer)
    with pytest.raises(RuntimeError, match="fatal"):
        client.complete("m", MSG, prompt_version="t")
    assert client.complete("m", MSG, prompt_version="t").text == "ok"  # not deadlocked


# ---------------------------------------------------------------- no credits: fatal at once
class _Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


OPENAI_NO_CREDITS = ("Error code: 429 - You have no credits remaining. Add credits to continue "
                     "using the API at https://platform.openai.com/settings/organization/billing/.")
OPENAI_QUOTA = ("Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
                "'type': 'insufficient_quota', 'code': 'insufficient_quota'}}")


@pytest.mark.parametrize("msg", [OPENAI_NO_CREDITS, OPENAI_QUOTA])
def test_openai_no_credit_429_is_fatal(msg):
    assert classify_error(_Err(msg, 429)) == "fatal"
    assert classify_error(_Err(msg)) == "fatal"


def test_real_rate_limits_still_retry():
    assert classify_error(_Err("Error code: 429 - Rate limit reached (rate_limit_exceeded)",
                               429)) == "rate_limit"
    gemini = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan "
              "and billing details.")
    assert classify_error(_Err(gemini, 429)) == "rate_limit"


def test_no_credit_429_stops_after_one_try_without_waiting(tmp_path):
    waits = []

    def broke(system, messages):
        raise _Err(OPENAI_NO_CREDITS, 429)

    client, fake = client_with(tmp_path, broke)
    client.max_retries, client._sleep = 5, waits.append
    with pytest.raises(RuntimeError, match="fatal error after 1 try"):
        client.complete("m", MSG, prompt_version="t")
    assert fake.calls == 1 and waits == []


# ---------------------------------------------------------------- spend: this job only
def test_job_spend_counts_only_this_jobs_calls(tmp_path):
    a, _ = client_with(tmp_path, lambda s, m: "x" * 400, job_id="job-a")
    b, _ = client_with(tmp_path, lambda s, m: "y" * 4000, job_id="job-b")  # same cache + ledger
    a.complete("m", [{"role": "user", "content": "a1"}], prompt_version="t")
    b.complete("m", [{"role": "user", "content": "b1"}], prompt_version="t")
    b.complete("m", [{"role": "user", "content": "b2"}], prompt_version="t")
    lines = [json.loads(x) for x in (tmp_path / "cache" / "_spend_ledger.jsonl").read_text()
             .splitlines()]
    assert [x["job"] for x in lines] == ["job-a", "job-b", "job-b"]
    assert a.job_spend_usd() == pytest.approx(lines[0]["usd"])
    assert b.job_spend_usd() == pytest.approx(lines[1]["usd"] + lines[2]["usd"])
    assert a.week_spend_usd() == pytest.approx(sum(x["usd"] for x in lines))  # team view: all


def test_job_cap_ignores_other_jobs_spend(tmp_path):
    other, _ = client_with(tmp_path, lambda s, m: "y" * 40000, job_id="other")
    other.complete("m", [{"role": "user", "content": "big"}], prompt_version="t")
    assert other.job_spend_usd() > 0.01
    mine, _ = client_with(tmp_path, lambda s, m: "x" * 400, job_id="mine", job_spend_cap_usd=0.01)
    mine.complete("m", [{"role": "user", "content": "small"}], prompt_version="t")  # not blocked
    assert 0 < mine.job_spend_usd() < 0.01


def test_job_cap_stops_this_job_but_not_cached_answers(tmp_path):
    c, _ = client_with(tmp_path, lambda s, m: "x" * 4000, job_id="j", job_spend_cap_usd=0.001)
    c.complete("m", MSG, prompt_version="t")  # ~$0.002: crosses the cap
    with pytest.raises(SpendCapExceeded, match="job j spent"):
        c.complete("m", [{"role": "user", "content": "another"}], prompt_version="t")
    assert c.complete("m", MSG, prompt_version="t").cached


def test_without_a_job_id_the_job_is_this_client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADR_JOB_ID", raising=False)
    first, _ = client_with(tmp_path, lambda s, m: "x" * 400)
    first.complete("m", [{"role": "user", "content": "1"}], prompt_version="t")
    second, _ = client_with(tmp_path, lambda s, m: "x" * 400)
    assert second.job_id is None and second.job_spend_usd() == 0.0
    assert first.job_spend_usd() > 0
    line = json.loads((tmp_path / "cache" / "_spend_ledger.jsonl").read_text().splitlines()[0])
    assert "job" not in line  # untagged lines, as before


def test_job_settings_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_JOB_ID", "env-job")
    monkeypatch.setenv("ADR_JOB_SPEND_CAP_USD", "2.5")
    c, _ = client_with(tmp_path, lambda s, m: "x")
    assert (c.job_id, c.job_spend_cap_usd) == ("env-job", 2.5)


# ---------------------------------------------------------------- connection drops
class APIConnectionError(Exception):  # same class name as the Anthropic / OpenAI SDKs' error
    pass


class APITimeoutError(APIConnectionError):
    pass


def test_connection_errors_are_classified_apart_from_rate_limits_and_fatal():
    assert classify_error(APIConnectionError("Connection error.")) == "connection"
    assert classify_error(APITimeoutError("Request timed out.")) == "connection"
    assert classify_error(ConnectionResetError("reset by peer")) == "connection"
    assert classify_error(_Err("Error code: 429 - Rate limit reached", 429)) == "rate_limit"
    assert classify_error(_Err(OPENAI_NO_CREDITS, 429)) == "fatal"


def test_connection_error_retries_up_to_10_times_with_capped_backoff(tmp_path):
    waits, fails = [], {"n": 9}

    def flaky(system, messages):
        if fails["n"]:
            fails["n"] -= 1
            raise APIConnectionError("Connection error.")
        return "ok"

    client, fake = client_with(tmp_path, flaky)  # max_retries=3 for other errors
    client._sleep = waits.append
    assert client.complete("m", MSG, prompt_version="t").text == "ok"
    assert fake.calls == 10 and len(waits) == 9  # 9 drops, the 10th try answers
    assert [int(w) for w in waits] == [2, 4, 8, 16, 32, 64, 120, 120, 120]


def test_connection_error_gives_up_after_10_tries(tmp_path):
    def down(system, messages):
        raise APITimeoutError("Request timed out.")

    client, fake = client_with(tmp_path, down)
    client._sleep = lambda s: None
    with pytest.raises(RuntimeError, match="connection error after 10 tries"):
        client.complete("m", MSG, prompt_version="t")
    assert fake.calls == 10


def test_other_transient_errors_keep_max_retries(tmp_path):
    def broken(system, messages):
        raise _Err("Internal server error", 500)

    client, fake = client_with(tmp_path, broken)  # max_retries=3
    client._sleep = lambda s: None
    with pytest.raises(RuntimeError, match="retry error after 3 tries"):
        client.complete("m", MSG, prompt_version="t")
    assert fake.calls == 3

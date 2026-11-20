"""The one place in the project that calls a model API.

Every call is cached on disk, keyed by a hash of everything that can change the answer
(provider, model, system prompt, messages, temperature, max_tokens, prompt_version, sample index,
and the sha256 of any attached file).
Re-running a script never pays twice, and teammates who share the cache folder share answers.

A weekly spend cap stops new (uncached) calls once the week's estimated spend reaches the cap.

Usage:
    client = LLMClient.from_config("configs/models.yaml")
    out = client.complete("claude-small", [{"role": "user", "content": "Hi"}],
                          prompt_version="cp-v1")
    out.text, out.cached, out.cache_key

Files (e.g. a PDF for the document-upload attacker) go in `attachments`, a list of
{"filename": str, "mime": str, "data": bytes}. They are attached to the last user message.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}
Attachment = dict[str, Any]  # {"filename": "quiz.pdf", "mime": "application/pdf", "data": b"..."}

DEFAULT_CACHE_DIR = Path(os.environ.get("ADR_CACHE_DIR", "cache/llm"))


class SpendCapExceeded(RuntimeError):
    """Raised before an uncached call when this week's spend has reached the cap."""


class AttachmentsNotSupported(TypeError):
    """The provider can't take files. A TypeError, so classify_error treats it as fatal."""


@dataclass(frozen=True)
class ModelSpec:
    name: str  # our short label; goes into the outcome table's `model` column
    provider: str  # anthropic | openai | google | openai_compatible | fake
    model_id: str  # exact API model string
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    heldout: bool = False
    base_url_env: str | None = None
    api_key_env: str | None = None


@dataclass
class Completion:
    text: str
    model: str
    cache_key: str
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    # Why the model stopped: "stop", "length" (hit max_tokens: the reply is cut off), another
    # provider-specific word, or None if unknown (e.g. answers cached before this was recorded).
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


_LENGTH = {"length", "max_tokens", "max_output_tokens"}
_STOP = {"stop", "end_turn", "stop_sequence"}


def finish_reason(raw: Any) -> str | None:
    """One word for why a provider stopped, the same for every provider: "length" when the
    reply hit the token limit (OpenAI "length", Anthropic "max_tokens", Google MAX_TOKENS),
    "stop" for a normal end, otherwise the provider's own word in lower case."""
    if raw is None:
        return None
    word = str(getattr(raw, "name", raw)).rsplit(".", 1)[-1].lower()
    if word in _LENGTH:
        return "length"
    if word in _STOP:
        return "stop"
    return word or None


# ---------------------------------------------------------------- providers
class Provider(Protocol):
    def complete(
        self,
        model_id: str,
        system: str | None,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int,
        attachments: list[Attachment] | None = None,
    ) -> tuple[str, int, int] | tuple[str, int, int, str | None]:
        """Return (text, input_tokens, output_tokens[, finish_reason]). The fourth value is
        optional so older providers keep working; pass it through finish_reason()."""
        ...


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _with_files(messages: list[Message], parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `messages` whose last message is a content list: file parts first, then its text."""
    *head, last = messages
    return [
        *head,
        {"role": last["role"], "content": [*parts, {"type": "text", "text": last["content"]}]},
    ]


def anthropic_messages(messages: list[Message], attachments: list[Attachment] | None):
    if not attachments:
        return messages
    return _with_files(
        messages,
        [
            {
                "type": "document",
                "title": a["filename"],
                "source": {"type": "base64", "media_type": a["mime"], "data": _b64(a["data"])},
            }
            for a in attachments
        ],
    )


def openai_messages(messages: list[Message], attachments: list[Attachment] | None):
    if not attachments:
        return messages
    return _with_files(
        messages,
        [
            {
                "type": "file",
                "file": {
                    "filename": a["filename"],
                    "file_data": f"data:{a['mime']};base64,{_b64(a['data'])}",
                },
            }
            for a in attachments
        ],
    )


class AnthropicProvider:
    def __init__(self, api_key: str | None = None):
        import anthropic  # lazy: only needed if you use this provider

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, model_id, system, messages, temperature, max_tokens, attachments=None):
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": anthropic_messages(messages, attachments),
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return (text, resp.usage.input_tokens, resp.usage.output_tokens,
                finish_reason(getattr(resp, "stop_reason", None)))


class OpenAIProvider:
    """OpenAI, or any OpenAI-compatible server when base_url is set (vLLM, Ollama, Together...)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        import openai  # lazy

        self._compatible = base_url is not None
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=base_url,
        )

    def complete(self, model_id, system, messages, temperature, max_tokens, attachments=None):
        if attachments and self._compatible:
            raise AttachmentsNotSupported(
                f"{model_id}: OpenAI-compatible servers don't support files here; "
                "use an anthropic, openai or google model for document upload"
            )
        msgs = ([{"role": "system", "content": system}] if system else []) + list(
            openai_messages(messages, attachments)
        )
        kwargs: dict[str, Any] = {"model": model_id, "messages": msgs}
        # OpenAI's own API now uses max_completion_tokens; most compatible servers use max_tokens.
        kwargs["max_tokens" if self._compatible else "max_completion_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return (text, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0),
                finish_reason(getattr(resp.choices[0], "finish_reason", None)))


class GoogleProvider:
    def __init__(self, api_key: str | None = None):
        from google import genai  # lazy

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    def complete(self, model_id, system, messages, temperature, max_tokens, attachments=None):
        types = self._genai.types
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        if attachments:
            files = [
                types.Part.from_bytes(data=a["data"], mime_type=a["mime"]) for a in attachments
            ]
            contents[-1].parts = files + contents[-1].parts
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_tokens
        )
        resp = self._client.models.generate_content(
            model=model_id, contents=contents, config=config
        )
        um = resp.usage_metadata
        candidates = getattr(resp, "candidates", None) or []
        return (
            resp.text or "",
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
            finish_reason(getattr(candidates[0], "finish_reason", None) if candidates else None),
        )


class FakeProvider:
    """Offline provider for tests and dry runs. `fn(system, messages) -> text`, or
    `fn(system, messages, attachments) -> text` when the call has attachments. `fn` may also
    return (text, finish_reason), e.g. (text, "length") to act out a reply that was cut off."""

    def __init__(self, fn: Callable[..., str]):
        self.fn = fn
        self.calls = 0

    def complete(self, model_id, system, messages, temperature, max_tokens, attachments=None):
        self.calls += 1
        out = self.fn(system, messages, attachments) if attachments else self.fn(system, messages)
        text, reason = out if isinstance(out, tuple) else (out, "stop")
        return text, sum(len(m["content"]) // 4 for m in messages), len(text) // 4, reason


def make_provider(spec: ModelSpec) -> Provider:
    if spec.provider == "anthropic":
        return AnthropicProvider()
    if spec.provider == "openai":
        return OpenAIProvider()
    if spec.provider == "openai_compatible":
        base = os.environ.get(spec.base_url_env or "OPEN_WEIGHT_BASE_URL")
        if not base:
            raise RuntimeError(f"{spec.name}: set {spec.base_url_env or 'OPEN_WEIGHT_BASE_URL'}")
        key = os.environ.get(spec.api_key_env or "OPEN_WEIGHT_API_KEY")
        return OpenAIProvider(api_key=key, base_url=base)
    if spec.provider == "google":
        return GoogleProvider()
    raise ValueError(f"{spec.name}: unknown provider {spec.provider!r}")


# ---------------------------------------------------------------- errors
_RATE_WORDS = ("rate limit", "rate_limit", "resource_exhausted", "overloaded", "too many requests")
_KEY_WORDS = ("invalid api key", "api key not valid", "invalid x-api-key", "incorrect api key")
# Fatal even on a 429: no credit left, or a bad key. Retrying cannot help. OpenAI answers an
# empty balance with 429 "You have no credits remaining" / code insufficient_quota (a real rate
# limit is rate_limit_exceeded); Anthropic says "credit balance". Retrying those only stalls a run.
_NO_CREDIT_WORDS = ("credit balance", "no credits remaining", "insufficient_quota")
_ALWAYS_FATAL_WORDS = (*_NO_CREDIT_WORDS, *_KEY_WORDS)
_FATAL_WORDS = (*_ALWAYS_FATAL_WORDS, "billing", "not found")


def classify_error(e: Exception) -> str:
    """Decide how to react to a provider error.

    "no_temperature": the model refuses a temperature setting -> drop it and retry now
    "rate_limit":     slow down and retry (any 429 / RESOURCE_EXHAUSTED, 529, overloaded)
    "fatal":          retrying can't help (bad key, no credit, bad request, unknown model);
                      on a 429 only no-credit messages ("credit balance", "no credits
                      remaining", insufficient_quota) and invalid-key messages are fatal
    "retry":          probably transient (timeouts, 5xx) -> short backoff and retry
    """
    msg = str(e).lower()
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    status = status if isinstance(status, int) else None
    if "temperature" in msg and (isinstance(e, TypeError) or status == 400 or "unsupported" in msg):
        return "no_temperature"
    if any(w in msg for w in _ALWAYS_FATAL_WORDS):
        return "fatal"
    # Any other 429 / RESOURCE_EXHAUSTED is a rate or quota limit: back off and retry, even when the
    # message mentions billing or quota (Gemini's per-minute limits say "check your plan and
    # billing details"). max_retries still bounds it, so a spent daily quota ends as an error.
    if status == 429 or "resource_exhausted" in msg:
        return "rate_limit"
    if any(w in msg for w in _FATAL_WORDS):
        return "fatal"
    if status == 529 or any(w in msg for w in _RATE_WORDS):
        return "rate_limit"
    if isinstance(e, TypeError) or status in (400, 401, 403, 404, 422):
        return "fatal"
    return "retry"


# ---------------------------------------------------------------- client
def cache_key(spec: ModelSpec, request: dict[str, Any]) -> str:
    payload = {"provider": spec.provider, "model_id": spec.model_id, **request}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iso_week(ts: str) -> str:
    y, w, _ = datetime.fromisoformat(ts).isocalendar()
    return f"{y}-W{w:02d}"


@dataclass
class LLMClient:
    models: dict[str, ModelSpec]
    cache_dir: Path = DEFAULT_CACHE_DIR
    weekly_spend_cap_usd: float | None = None
    providers: dict[str, Provider] = field(default_factory=dict)  # name -> provider (overrides)
    max_retries: int = 5
    _sleep: Callable[[float], None] = time.sleep  # replaced in tests
    # The job this client spends for (a run, or a driver's set of runs). Its ledger lines carry
    # "job": job_id, so job_spend_usd() counts only this job even when other jobs or sessions
    # share the cache. Default: $ADR_JOB_ID; without one, the job is this client instance.
    job_id: str | None = None
    # Stop new (uncached) calls once this job's own spend reaches this. Default:
    # $ADR_JOB_SPEND_CAP_USD. Separate from weekly_spend_cap_usd, the team's cap on everyone's
    # calls this week, which is unchanged.
    job_spend_cap_usd: float | None = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ledger = self.cache_dir / "_spend_ledger.jsonl"
        self._lock = threading.Lock()
        # One lock per cache key: a second identical request waits for the first, then reads its
        # cached answer. Identical in-flight calls (e.g. a plain and an opinion-trap variant with
        # the same prompt) cost one API call and get one answer. Within one process only:
        # separate processes sharing a cache can still both call.
        self._key_locks: dict[str, threading.Lock] = {}
        if self.job_id is None:
            self.job_id = os.environ.get("ADR_JOB_ID") or None
        if self.job_spend_cap_usd is None and os.environ.get("ADR_JOB_SPEND_CAP_USD"):
            self.job_spend_cap_usd = float(os.environ["ADR_JOB_SPEND_CAP_USD"])
        self._spent_here = 0.0  # this instance's own paid calls (the job when job_id is None)
        self._warned_no_price: set[str] = set()
        self._no_temperature: set[str] = set()

    # -- construction
    @classmethod
    def from_config(cls, path: str | Path, **kw: Any) -> LLMClient:
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        models = {m["name"]: ModelSpec(**m) for m in cfg["models"]}
        return cls(models=models, weekly_spend_cap_usd=cfg.get("weekly_spend_cap_usd"), **kw)

    def spec(self, model: str) -> ModelSpec:
        if model not in self.models:
            raise KeyError(f"unknown model {model!r}; known: {sorted(self.models)}")
        return self.models[model]

    def _provider(self, model: str) -> Provider:
        with self._lock:
            if model not in self.providers:
                self.providers[model] = make_provider(self.spec(model))
            return self.providers[model]

    # -- spend
    def week_spend_usd(self, now: str | None = None) -> float:
        if not self._ledger.exists():
            return 0.0
        week = _iso_week(now or datetime.now(timezone.utc).isoformat())
        total = 0.0
        for line in self._ledger.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("usd") and _iso_week(rec["ts"]) == week:
                total += rec["usd"]
        return total

    def job_spend_usd(self) -> float:
        """This job's spend: the ledger lines tagged with job_id (any process, any week) or, with
        no job_id, this client instance's own paid calls. Never other jobs' or sessions' calls."""
        if self.job_id is None:
            return self._spent_here
        if not self._ledger.exists():
            return 0.0
        total = 0.0
        for line in self._ledger.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("job") == self.job_id and rec.get("usd"):
                total += rec["usd"]
        return total

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    @staticmethod
    def _from_cache(hit: dict[str, Any], model: str, key: str) -> Completion:
        r = hit["response"]
        return Completion(r["text"], model, key, True, r["input_tokens"], r["output_tokens"],
                          r.get("cost_usd"), r.get("finish_reason"))

    def _cost(self, spec: ModelSpec, tin: int, tout: int) -> float | None:
        if spec.price_in_per_mtok is None or spec.price_out_per_mtok is None:
            if spec.name not in self._warned_no_price:
                warnings.warn(
                    f"{spec.name}: no prices in config; tracking tokens only", stacklevel=3
                )
                self._warned_no_price.add(spec.name)
            return None
        return (tin * spec.price_in_per_mtok + tout * spec.price_out_per_mtok) / 1_000_000

    # -- cache
    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def read_cache(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            # Older Windows runs used pathlib's locale encoding. Read those cache files once and
            # rewrite them as UTF-8 so real questions containing smart quotes/degrees remain usable.
            record = json.loads(p.read_text(encoding="cp1252"))
            self._write_cache(key, record)
            return record

    def _write_cache(self, key: str, record: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)  # atomic: a crash never leaves a half-written cache file

    # -- main entry point
    def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        prompt_version: str,
        system: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 512,
        sample_idx: int = 0,
        attachments: list[Attachment] | None = None,
    ) -> Completion:
        spec = self.spec(model)
        request: dict[str, Any] = {
            "system": system,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_version": prompt_version,
            "sample_idx": sample_idx,
        }
        if attachments:
            # Only added when present, so cache keys of calls without files stay the same.
            request["attachments"] = [
                {
                    "filename": a["filename"],
                    "mime": a["mime"],
                    "sha256": hashlib.sha256(a["data"]).hexdigest(),
                }
                for a in attachments
            ]
        key = cache_key(spec, request)
        hit = self.read_cache(key)
        if hit is not None:
            return self._from_cache(hit, model, key)

        with self._key_lock(key):  # one in-flight call per key; identical requests wait here
            hit = self.read_cache(key)  # the request we waited for has cached its answer
            if hit is not None:
                return self._from_cache(hit, model, key)
            return self._call_and_cache(spec, model, key, request, system, messages,
                                        temperature, max_tokens, attachments)

    def _call_and_cache(
        self,
        spec: ModelSpec,
        model: str,
        key: str,
        request: dict[str, Any],
        system: str | None,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int,
        attachments: list[Attachment] | None,
    ) -> Completion:
        """The uncached path of complete(): spend checks, the provider call with retries, then
        the cache file and the ledger line. Called with the key's lock held."""
        if self.weekly_spend_cap_usd is not None:
            spent = self.week_spend_usd()
            if spent >= self.weekly_spend_cap_usd:
                raise SpendCapExceeded(
                    f"this week's spend ${spent:.2f} reached the cap "
                    f"${self.weekly_spend_cap_usd:.2f}; cached answers still work"
                )
        if self.job_spend_cap_usd is not None:
            job_spent = self.job_spend_usd()
            if job_spent >= self.job_spend_cap_usd:
                raise SpendCapExceeded(
                    f"job {self.job_id or '(this client)'} spent ${job_spent:.2f}, reaching its "
                    f"cap ${self.job_spend_cap_usd:.2f}; cached answers still work"
                )

        provider = self._provider(model)
        files = {"attachments": attachments} if attachments else {}  # old providers lack the arg
        sent_temperature = None if model in self._no_temperature else temperature
        attempt = 0
        while True:
            try:
                result = provider.complete(
                    spec.model_id, system, messages, sent_temperature, max_tokens, **files
                )
                text, tin, tout = result[:3]
                reason = finish_reason(result[3]) if len(result) > 3 else None
                break
            except Exception as e:  # provider SDKs raise many unrelated exception types
                kind = classify_error(e)
                if kind == "no_temperature" and sent_temperature is not None:
                    # Some newer models/SDKs refuse a temperature setting: drop it for this
                    # model from now on and retry at once (doesn't count as an attempt).
                    with self._lock:
                        self._no_temperature.add(model)
                    sent_temperature = None
                    continue
                attempt += 1
                if kind == "fatal" or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"{model}: {kind} error after {attempt} tr{'y' if attempt == 1 else 'ies'}: "
                        f"{type(e).__name__}: {str(e)[:300]}"
                    ) from e
                if kind == "rate_limit":
                    wait = min(60.0, 10.0 * 2 ** (attempt - 1)) + random.uniform(0, 3)
                else:
                    wait = min(2.0**attempt, 20.0)
                self._sleep(wait)

        cost = self._cost(spec, tin, tout)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write_cache(
            key,
            {
                "model": model,
                "provider": spec.provider,
                "model_id": spec.model_id,
                "request": request,
                "temperature_sent": sent_temperature,
                "response": {
                    "text": text,
                    "input_tokens": tin,
                    "output_tokens": tout,
                    "cost_usd": cost,
                    "finish_reason": reason,
                },
                "created_at": now,
            },
        )
        entry = {"ts": now, "model": model, "in": tin, "out": tout, "usd": cost}
        if self.job_id is not None:
            entry["job"] = self.job_id  # so job_spend_usd() counts only this job's calls
        with self._lock, self._ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            self._spent_here += cost or 0.0
        return Completion(text, model, key, False, tin, tout, cost, reason)

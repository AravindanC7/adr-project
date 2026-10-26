"""The one place in the project that calls a model API.

Every call is cached on disk, keyed by a hash of everything that can change the answer
(provider, model, system prompt, messages, temperature, max_tokens, prompt_version, sample index).
Re-running a script never pays twice, and teammates who share the cache folder share answers.

A weekly spend cap stops new (uncached) calls once the week's estimated spend reaches the cap.

Usage:
    client = LLMClient.from_config("configs/models.yaml")
    out = client.complete("claude-small", [{"role": "user", "content": "Hi"}],
                          prompt_version="cp-v1")
    out.text, out.cached, out.cache_key
"""

from __future__ import annotations

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

DEFAULT_CACHE_DIR = Path(os.environ.get("ADR_CACHE_DIR", "cache/llm"))


class SpendCapExceeded(RuntimeError):
    """Raised before an uncached call when this week's spend has reached the cap."""


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


# ---------------------------------------------------------------- providers
class Provider(Protocol):
    def complete(
        self,
        model_id: str,
        system: str | None,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""
        ...


class AnthropicProvider:
    def __init__(self, api_key: str | None = None):
        import anthropic  # lazy: only needed if you use this provider

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, model_id, system, messages, temperature, max_tokens):
        kwargs: dict[str, Any] = {"model": model_id, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class OpenAIProvider:
    """OpenAI, or any OpenAI-compatible server when base_url is set (vLLM, Ollama, Together...)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        import openai  # lazy

        self._compatible = base_url is not None
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=base_url,
        )

    def complete(self, model_id, system, messages, temperature, max_tokens):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        kwargs: dict[str, Any] = {"model": model_id, "messages": msgs}
        # OpenAI's own API now uses max_completion_tokens; most compatible servers use max_tokens.
        kwargs["max_tokens" if self._compatible else "max_completion_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)


class GoogleProvider:
    def __init__(self, api_key: str | None = None):
        from google import genai  # lazy

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    def complete(self, model_id, system, messages, temperature, max_tokens):
        types = self._genai.types
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_tokens
        )
        resp = self._client.models.generate_content(model=model_id, contents=contents, config=config)
        um = resp.usage_metadata
        return (
            resp.text or "",
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
        )


class FakeProvider:
    """Offline provider for tests and dry runs. `fn(system, messages) -> text`."""

    def __init__(self, fn: Callable[[str | None, list[Message]], str]):
        self.fn = fn
        self.calls = 0

    def complete(self, model_id, system, messages, temperature, max_tokens):
        self.calls += 1
        text = self.fn(system, messages)
        return text, sum(len(m["content"]) // 4 for m in messages), len(text) // 4


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
_FATAL_WORDS = ("credit balance", "insufficient_quota", "billing", "invalid api key",
                "api key not valid", "invalid x-api-key", "incorrect api key", "not found")


def classify_error(e: Exception) -> str:
    """Decide how to react to a provider error.

    "no_temperature": the model refuses a temperature setting -> drop it and retry now
    "rate_limit":     slow down and retry (429, 529, overloaded)
    "fatal":          retrying can't help (bad key, no credit, bad request, unknown model)
    "retry":          probably transient (timeouts, 5xx) -> short backoff and retry
    """
    msg = str(e).lower()
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    status = status if isinstance(status, int) else None
    if "temperature" in msg and (
        isinstance(e, TypeError) or status == 400 or "unsupported" in msg
    ):
        return "no_temperature"
    if any(w in msg for w in _FATAL_WORDS):
        return "fatal"
    if status in (429, 529) or any(w in msg for w in _RATE_WORDS):
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

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ledger = self.cache_dir / "_spend_ledger.jsonl"
        self._lock = threading.Lock()
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
        cfg = yaml.safe_load(Path(path).read_text())
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
        for line in self._ledger.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("usd") and _iso_week(rec["ts"]) == week:
                total += rec["usd"]
        return total

    def _cost(self, spec: ModelSpec, tin: int, tout: int) -> float | None:
        if spec.price_in_per_mtok is None or spec.price_out_per_mtok is None:
            if spec.name not in self._warned_no_price:
                warnings.warn(f"{spec.name}: no prices in config; tracking tokens only", stacklevel=3)
                self._warned_no_price.add(spec.name)
            return None
        return (tin * spec.price_in_per_mtok + tout * spec.price_out_per_mtok) / 1_000_000

    # -- cache
    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def read_cache(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def _write_cache(self, key: str, record: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
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
    ) -> Completion:
        spec = self.spec(model)
        request = {
            "system": system,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_version": prompt_version,
            "sample_idx": sample_idx,
        }
        key = cache_key(spec, request)
        hit = self.read_cache(key)
        if hit is not None:
            r = hit["response"]
            return Completion(r["text"], model, key, True, r["input_tokens"], r["output_tokens"],
                              r.get("cost_usd"))

        if self.weekly_spend_cap_usd is not None:
            spent = self.week_spend_usd()
            if spent >= self.weekly_spend_cap_usd:
                raise SpendCapExceeded(
                    f"this week's spend ${spent:.2f} reached the cap "
                    f"${self.weekly_spend_cap_usd:.2f}; cached answers still work"
                )

        provider = self._provider(model)
        sent_temperature = None if model in self._no_temperature else temperature
        attempt = 0
        while True:
            try:
                text, tin, tout = provider.complete(
                    spec.model_id, system, messages, sent_temperature, max_tokens
                )
                break
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
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
        self._write_cache(key, {
            "model": model,
            "provider": spec.provider,
            "model_id": spec.model_id,
            "request": request,
            "temperature_sent": sent_temperature,
            "response": {"text": text, "input_tokens": tin, "output_tokens": tout,
                         "cost_usd": cost},
            "created_at": now,
        })
        with self._lock, self._ledger.open("a") as f:
            f.write(json.dumps({"ts": now, "model": model, "in": tin, "out": tout,
                                "usd": cost}) + "\n")
        return Completion(text, model, key, False, tin, tout, cost)

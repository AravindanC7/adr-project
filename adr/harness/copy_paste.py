"""Copy-paste attacker (step #9): a person copies each question into a chatbot and types back
the answer.

For every model, every sample, and every variant of every item, it:
  1. builds the text a person would copy (including visually hidden trap text),
  2. asks the model through the cached client,
  3. parses the reply into an answer,
  4. scores it with the variant's mechanism (and any passive mechanisms passed in),
  5. writes one outcome row per (response, mechanism).

L3 (retry after a flag, step #36): the first call is the L1 prompt. If that answer's outcome is
an AI flag (prompts.FLAG_OUTCOMES), the model gets ONE follow-up message in the same conversation
(prompts.RETRY_FEEDBACK, which names no mechanism, hidden text or answer) and answers again. The
outcome row is the FINAL answer's; responses.jsonl keeps the first answer, whether it was flagged,
and the retry.

Outputs:
  results/outcomes/<run_id>.parquet          the outcome rows (the contract)
  results/runs/<run_id>/config.yaml          what was run, for reproducibility
  results/runs/<run_id>/responses.jsonl      every parsed response, for debugging
  results/runs/<run_id>/errors.jsonl         calls that failed (if any)
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from adr.contracts import (
    PASSIVE_MECHANISMS,
    Assessment,
    Item,
    Mechanism,
    NoObservation,
    OutcomeRow,
    RenderedItem,
    Response,
    check_outcome,
    make_variant_id,
    write_outcomes,
)
from adr.contracts.outcome import utc_now
from adr.harness.llm_client import LLMClient, SpendCapExceeded
from adr.harness.parse import parse_answer
from adr.harness.prompts import (
    DEFAULT_PERSONA,
    PROMPT_VERSION,
    RETRY_PROMPT_VERSION,
    build_prompt,
    build_retry_messages,
    is_flag,
)


@dataclass
class CopyPasteConfig:
    run_id: str
    models: list[str]
    attacker_level: str = "L0"
    n_samples: int = 1  # independent "sessions" per model; use temperature > 0 if > 1
    # Index of the first sample: a later run can add samples first_sample.. to an earlier one
    # (the index is part of the cache key and of the session_id, so they never repeat it).
    first_sample: int = 0
    temperature: float | None = 0.0
    max_tokens: int = 300
    persona: str = DEFAULT_PERSONA
    max_workers: int = 4
    results_dir: str = "results"
    note: str = ""
    prompt_version: str = field(default=PROMPT_VERSION)


@dataclass
class _Job:
    model: str
    sample_idx: int
    item: Item
    rendered: RenderedItem


def plain_variant(item: Item) -> RenderedItem:
    """The item with no active check, for items that have no other variant."""
    return RenderedItem(
        item_id=item.item_id,
        mechanism=None,
        variant_id=make_variant_id(item.item_id, None),
        visible={"stem": item.stem, **({"options": item.options} if item.options else {})},
    )


def observe(
    name: str, mech: Mechanism, rendered: RenderedItem, resp: Response, skipped: Counter[str]
) -> str | None:
    """`mech`'s checked outcome, or None when it raises NoObservation (nothing to score, e.g. an
    empty answer). A skipped observation gets no outcome row; it is counted in `skipped`."""
    try:
        return check_outcome(name, mech.score(rendered, resp, None))
    except NoObservation:
        skipped[name] += 1
        return None


def _first_outcome(mech: Mechanism, rendered: RenderedItem, resp: Response) -> str | None:
    """Outcome of an L3 first answer, only to decide on the retry (no row, nothing counted)."""
    try:
        return mech.score(rendered, resp, None)
    except NoObservation:
        return None


def _l3_summary(responses: list[dict]) -> dict:
    """How often L3 first answers were flagged, and what the retry did (checked variants only)."""
    checked = [r for r in responses if "flagged" in r]
    flagged = [r for r in checked if r["flagged"]]
    retry_flagged = [r for r in flagged if r.get("final_flagged")]
    return {
        "n_checked_answers": len(checked),
        "n_first_flagged": len(flagged),
        "n_retries": len(flagged),
        "n_retry_still_flagged": len(retry_flagged),
        "n_retry_got_away": len(flagged) - len(retry_flagged),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, encoding="utf-8", check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def run_copy_paste(
    client: LLMClient,
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    mechanisms: dict[str, Mechanism],
    cfg: CopyPasteConfig,
) -> list[OutcomeRow]:
    """Run the copy-paste attacker. `variants[item_id]` lists that item's pre-built variants;
    items with none get a plain variant. `mechanisms` must include every active mechanism that
    appears in `variants`, plus any passive ones (e.g. style) to score every response."""
    for m in cfg.models:
        client.spec(m)  # fail fast on an unknown model
    if cfg.n_samples > 1 and not cfg.temperature:
        raise ValueError("n_samples > 1 with temperature 0 repeats the same answer; raise it")
    l3 = cfg.attacker_level == "L3"
    # Rows of an L3 run carry both text versions; the first call itself uses cfg.prompt_version.
    row_prompt_version = (
        f"{cfg.prompt_version}+{RETRY_PROMPT_VERSION}" if l3 else cfg.prompt_version
    )

    jobs: list[_Job] = []
    for item_id in assessment.item_ids:
        item = items[item_id]
        for rendered in variants.get(item_id) or [plain_variant(item)]:
            if rendered.item_id != item_id:
                raise ValueError(f"variant {rendered.variant_id} filed under {item_id}")
            if rendered.mechanism and rendered.mechanism not in mechanisms:
                raise ValueError(f"no mechanism object for {rendered.mechanism!r}")
            for model in cfg.models:
                for s in range(cfg.first_sample, cfg.first_sample + cfg.n_samples):
                    jobs.append(_Job(model, s, item, rendered))

    run_dir = Path(cfg.results_dir) / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    passive = {k: v for k, v in mechanisms.items() if k in PASSIVE_MECHANISMS}
    rows: list[OutcomeRow] = []
    responses: list[dict] = []
    errors: list[dict] = []
    skipped: Counter[str] = Counter()  # mechanism -> NoObservation count
    stopped_for_budget = False
    n_truncated = 0  # replies cut off at max_tokens (finish_reason "length"), first or retry

    def work(job: _Job):
        """One item for one model and sample. Returns (job, final completion, first completion
        or None, first answer, first outcome); the first-* values are set only for L3."""
        system, messages = build_prompt(job.item, job.rendered, cfg.attacker_level, cfg.persona)
        try:
            out = client.complete(
                job.model, messages, system=system, temperature=cfg.temperature,
                max_tokens=cfg.max_tokens, prompt_version=cfg.prompt_version,
                sample_idx=job.sample_idx,
            )
            mech_name = job.rendered.mechanism
            if not (l3 and mech_name):
                return job, out, None, None, None
            first_answer = parse_answer(job.item, out.text)
            first_resp = Response(session_id="l3-first", item_id=job.item.item_id,
                                  variant_id=job.rendered.variant_id, answer=first_answer,
                                  raw_text=out.text)
            first_outcome = _first_outcome(mechanisms[mech_name], job.rendered, first_resp)
            if not is_flag(mech_name, first_outcome):
                return job, out, None, first_answer, first_outcome
            retry = client.complete(  # max one retry per item
                job.model, build_retry_messages(messages, out.text), system=system,
                temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                prompt_version=RETRY_PROMPT_VERSION, sample_idx=job.sample_idx,
            )
            return job, retry, out, first_answer, first_outcome
        except Exception as e:
            e.adr_job = job  # type: ignore[attr-defined]  # so the error log can say which call
            raise

    def _where(e: Exception) -> dict:
        job = getattr(e, "adr_job", None)
        if job is None:
            return {}
        return {"model": job.model, "variant_id": job.rendered.variant_id,
                "sample_idx": job.sample_idx}

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                job, out, first, first_answer, first_outcome = fut.result()
            except SpendCapExceeded as e:
                stopped_for_budget = True
                errors.append({**_where(e), "error": str(e)})
                for f in futures:
                    f.cancel()
                continue
            except Exception as e:  # noqa: BLE001 - keep the run going, log the failure
                errors.append({**_where(e), "error": str(e)})
                continue

            spec = client.spec(job.model)
            session_id = f"{cfg.run_id}:{job.model}:{job.sample_idx}"
            answer = parse_answer(job.item, out.text)
            resp = Response(session_id=session_id, item_id=job.item.item_id,
                            variant_id=job.rendered.variant_id, answer=answer, raw_text=out.text)
            record = {**resp.model_dump(), "model": job.model, "cache_key": out.cache_key,
                      "cached": out.cached, "finish_reason": out.finish_reason}
            truncated = out.truncated
            if l3 and job.rendered.mechanism:
                # top-level answer/raw_text = the FINAL answer (the one the outcome row scores)
                record["first_outcome"] = first_outcome
                record["flagged"] = first is not None
                record["first"] = None
                if first is not None:
                    record["first"] = {
                        "answer": first_answer, "raw_text": first.text,
                        "cache_key": first.cache_key, "cached": first.cached,
                        "finish_reason": first.finish_reason,
                    }
                    record["retry_prompt_version"] = RETRY_PROMPT_VERSION
                    truncated = truncated or first.truncated
            if truncated:
                n_truncated += 1
            responses.append(record)

            base = dict(
                run_id=cfg.run_id, session_id=session_id, respondent_kind="ai",
                data_origin="harness", model=job.model, attack_mode="copy_paste",
                attacker_level=cfg.attacker_level, assessment_type=job.item.assessment_type,
                item_type=job.item.item_type, item_id=job.item.item_id, split=job.item.split,
                heldout_model=spec.heldout, variant_id=job.rendered.variant_id,
                raw_ref=f"llm-cache:{out.cache_key}", prompt_version=row_prompt_version,
            )
            mech_name = job.rendered.mechanism
            if mech_name:
                outcome = observe(mech_name, mechanisms[mech_name], job.rendered, resp, skipped)
            else:
                outcome = "none"
            if "flagged" in record:  # L3: the final answer's outcome (after the retry, if any)
                record["final_outcome"] = outcome
                record["final_flagged"] = is_flag(mech_name, outcome)
            if outcome is not None:
                rows.append(OutcomeRow(**base, mechanism=mech_name, outcome=outcome))
            for p_name, p_mech in passive.items():
                if p_mech.applies_to(job.item):
                    p_out = observe(p_name, p_mech, job.rendered, resp, skipped)
                    if p_out is not None:
                        rows.append(OutcomeRow(**base, mechanism=p_name, outcome=p_out))

    rows.sort(key=lambda r: (r.model or "", r.session_id, r.item_id, r.variant_id, r.mechanism or ""))
    write_outcomes(rows, Path(cfg.results_dir) / "outcomes" / f"{cfg.run_id}.parquet")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        **asdict(cfg),
        "mode": "copy_paste",
        "assessment_id": assessment.assessment_id,
        "n_items": len(assessment.item_ids),
        "n_variants": sum(len(variants.get(i) or [None]) for i in assessment.item_ids),
        "mechanisms": sorted(mechanisms),
        "model_ids": {m: client.spec(m).model_id for m in cfg.models},
        "heldout_models": [m for m in cfg.models if client.spec(m).heldout],
        "git_commit": _git_commit(),
        "finished_at": utc_now(),
        "n_jobs": len(jobs),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "n_skipped_observations": sum(skipped.values()),
        "skipped_observations": dict(sorted(skipped.items())),  # per mechanism: NoObservation
        "stopped_for_budget": stopped_for_budget,
        "n_truncated_replies": n_truncated,
        **({"l3": _l3_summary(responses)} if l3 else {}),
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(responses, key=lambda r: (r["model"], r["session_id"], r["variant_id"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as f:  # always rewritten, empty = no failures
        for e in errors:
            f.write(json.dumps(e) + "\n")
    return rows

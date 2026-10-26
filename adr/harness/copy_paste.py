"""Copy-paste attacker (step #9): a person copies each question into a chatbot and types back
the answer.

For every model, every sample, and every variant of every item, it:
  1. builds the text a person would copy (including visually hidden trap text),
  2. asks the model through the cached client,
  3. parses the reply into an answer,
  4. scores it with the variant's mechanism (and any passive mechanisms passed in),
  5. writes one outcome row per (response, mechanism).

Outputs:
  results/outcomes/<run_id>.parquet          the outcome rows (the contract)
  results/runs/<run_id>/config.yaml          what was run, for reproducibility
  results/runs/<run_id>/responses.jsonl      every parsed response, for debugging
  results/runs/<run_id>/errors.jsonl         calls that failed (if any)
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from adr.contracts import (
    PASSIVE_MECHANISMS,
    Assessment,
    Item,
    Mechanism,
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
from adr.harness.prompts import DEFAULT_PERSONA, PROMPT_VERSION, build_prompt


@dataclass
class CopyPasteConfig:
    run_id: str
    models: list[str]
    attacker_level: str = "L0"
    n_samples: int = 1  # independent "sessions" per model; use temperature > 0 if > 1
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


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
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

    jobs: list[_Job] = []
    for item_id in assessment.item_ids:
        item = items[item_id]
        for rendered in variants.get(item_id) or [plain_variant(item)]:
            if rendered.item_id != item_id:
                raise ValueError(f"variant {rendered.variant_id} filed under {item_id}")
            if rendered.mechanism and rendered.mechanism not in mechanisms:
                raise ValueError(f"no mechanism object for {rendered.mechanism!r}")
            for model in cfg.models:
                for s in range(cfg.n_samples):
                    jobs.append(_Job(model, s, item, rendered))

    run_dir = Path(cfg.results_dir) / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    passive = {k: v for k, v in mechanisms.items() if k in PASSIVE_MECHANISMS}
    rows: list[OutcomeRow] = []
    responses: list[dict] = []
    errors: list[dict] = []
    stopped_for_budget = False

    def work(job: _Job):
        system, messages = build_prompt(job.item, job.rendered, cfg.attacker_level, cfg.persona)
        out = client.complete(
            job.model, messages, system=system, temperature=cfg.temperature,
            max_tokens=cfg.max_tokens, prompt_version=cfg.prompt_version,
            sample_idx=job.sample_idx,
        )
        return job, out

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                job, out = fut.result()
            except SpendCapExceeded as e:
                stopped_for_budget = True
                errors.append({"error": str(e)})
                for f in futures:
                    f.cancel()
                continue
            except Exception as e:  # noqa: BLE001 - keep the run going, log the failure
                errors.append({"error": repr(e)})
                continue

            spec = client.spec(job.model)
            session_id = f"{cfg.run_id}:{job.model}:{job.sample_idx}"
            answer = parse_answer(job.item, out.text)
            resp = Response(session_id=session_id, item_id=job.item.item_id,
                            variant_id=job.rendered.variant_id, answer=answer, raw_text=out.text)
            responses.append({**resp.model_dump(), "model": job.model, "cache_key": out.cache_key,
                              "cached": out.cached})

            base = dict(
                run_id=cfg.run_id, session_id=session_id, respondent_kind="ai",
                data_origin="harness", model=job.model, attack_mode="copy_paste",
                attacker_level=cfg.attacker_level, assessment_type=job.item.assessment_type,
                item_type=job.item.item_type, item_id=job.item.item_id, split=job.item.split,
                heldout_model=spec.heldout, variant_id=job.rendered.variant_id,
                raw_ref=f"llm-cache:{out.cache_key}", prompt_version=cfg.prompt_version,
            )
            mech_name = job.rendered.mechanism
            if mech_name:
                outcome = check_outcome(
                    mech_name, mechanisms[mech_name].score(job.rendered, resp, None)
                )
            else:
                outcome = "none"
            rows.append(OutcomeRow(**base, mechanism=mech_name, outcome=outcome))
            for p_name, p_mech in passive.items():
                if p_mech.applies_to(job.item):
                    p_out = check_outcome(p_name, p_mech.score(job.rendered, resp, None))
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
        "stopped_for_budget": stopped_for_budget,
    }, sort_keys=False))
    with (run_dir / "responses.jsonl").open("w") as f:
        for r in sorted(responses, key=lambda r: (r["model"], r["session_id"], r["variant_id"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if errors:
        with (run_dir / "errors.jsonl").open("w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
    return rows

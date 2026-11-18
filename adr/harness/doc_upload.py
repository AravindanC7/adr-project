"""Document-upload attacker (step #20): a person uploads the whole quiz or survey as a PDF to a
chatbot and asks it to answer everything.

DoPE-style traps sit in the PDF's text layer (invisible on the page, see pdf_render.py).
Document k shows variant k mod len(variants) of every item, so across the documents every variant
of every item gets used. With questions_per_doc set, the items are first cut into chunks of that
size (in assessment order) and each chunk gets its own set of documents (one per variant: the
plain/decoy split is kept within each chunk). Every document is its own session. For every
model, every sample and every document, it:
  1. renders the PDF (deterministic, so reruns hit the cache),
  2. asks the model through the cached client, with the PDF attached,
  3. splits the reply into "Q<n>: <answer>" lines and parses each with parse_answer,
  4. scores each answer with that question's mechanism (and any passive mechanisms passed in),
  5. writes one outcome row per (question, mechanism). Skipped questions get no row.

Outputs (same layout as copy_paste.py):
  results/outcomes/<run_id>.parquet          the outcome rows (the contract)
  results/runs/<run_id>/config.yaml          what was run, for reproducibility
  results/runs/<run_id>/responses.jsonl      every parsed answer, for debugging
  results/runs/<run_id>/errors.jsonl         failed calls and skipped questions (if any)
  results/runs/<run_id>/docs/doc-<k>.pdf     the documents sent, to look at by hand
"""

from __future__ import annotations

import json
import re
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
    OutcomeRow,
    RenderedItem,
    Response,
    write_outcomes,
)
from adr.contracts.outcome import utc_now
from adr.harness.copy_paste import observe, plain_variant
from adr.harness.llm_client import LLMClient, SpendCapExceeded
from adr.harness.parse import parse_answer
from adr.harness.pdf_render import render_assessment_pdf
from adr.harness.prompts import DEFAULT_PERSONA, DOC_PROMPT_VERSION, build_doc_prompt
from adr.harness.provenance import git_state


@dataclass
class DocUploadConfig:
    run_id: str
    models: list[str]
    attacker_level: str = "L0"
    n_samples: int = 1  # independent uploads per (model, document); use temperature > 0 if > 1
    temperature: float | None = 0.0
    max_tokens: int = 1500  # one reply covers the whole document; raise it for long ones
    questions_per_doc: int | None = None  # None = one document per variant with every item
    persona: str = DEFAULT_PERSONA
    max_workers: int = 4
    results_dir: str = "results"
    note: str = ""
    prompt_version: str = field(default=DOC_PROMPT_VERSION)


@dataclass
class _Doc:
    k: int
    pdf: bytes
    questions: dict[int, tuple[Item, RenderedItem]]  # question number -> (item, variant)
    chunk: int = 0  # which slice of the assessment's items (0 when not chunked)
    variant_k: int = 0  # which variant of each item this document shows


@dataclass
class _Job:
    model: str
    sample_idx: int
    doc: _Doc


# "Q3: 2", "**Q3.** 2", "- Q3) 2". Markdown bold is removed before matching.
_Q_LINE = re.compile(r"^[\s>*\-]*Q\s*(\d+)\s*[:.)\-]\s*(.*)$", re.IGNORECASE)


def split_answers(text: str) -> dict[int, str]:
    """Map question number -> the reply text for it. Lines after a "Q<n>:" line belong to that
    question until the next one (for written answers that run over). First "Q<n>" wins."""
    out: dict[int, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        m = _Q_LINE.match(line.replace("**", ""))
        if m:
            n = int(m.group(1))
            current = None if n in out else out.setdefault(n, [])
            if current is not None:
                current.append(m.group(2))
        elif current is not None:
            current.append(line)
    return {n: "\n".join(parts).strip() for n, parts in out.items()}


def build_documents(
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    questions_per_doc: int | None = None,
) -> list[_Doc]:
    """Document k shows variant k mod len(variants) of each item, in assessment order.

    With `questions_per_doc`, the items are cut into consecutive chunks of that size and each
    chunk gets one document per variant, so each document holds at most that many questions.
    Without it (the default) the documents are byte-identical to earlier runs, so the cache hits.
    """
    if questions_per_doc is not None and questions_per_doc < 1:
        raise ValueError("questions_per_doc must be at least 1")
    order = [items[i] for i in assessment.item_ids]
    size = questions_per_doc or len(order)
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    docs: list[_Doc] = []
    for c, chunk in enumerate(chunks):
        title = assessment.title if len(chunks) == 1 else (
            f"{assessment.title} (part {c + 1} of {len(chunks)})"
        )
        per_item = [variants.get(it.item_id) or [plain_variant(it)] for it in chunk]
        for v in range(max(len(vs) for vs in per_item)):
            chosen = [vs[v % len(vs)] for vs in per_item]
            pdf, qmap = render_assessment_pdf(chunk, chosen, title=title)
            questions = {n: (chunk[n - 1], chosen[n - 1]) for n in qmap}
            docs.append(_Doc(len(docs), pdf, questions, chunk=c, variant_k=v))
    return docs


def questions_per_doc_for(n_items: int, docs_per_bank: int) -> int:
    """Chunk size that cuts `n_items` into `docs_per_bank` chunks (the last may be shorter)."""
    if docs_per_bank < 1:
        raise ValueError("docs_per_bank must be at least 1")
    return -(-n_items // docs_per_bank)


def run_doc_upload(
    client: LLMClient,
    assessment: Assessment,
    items: dict[str, Item],
    variants: dict[str, list[RenderedItem]],
    mechanisms: dict[str, Mechanism],
    cfg: DocUploadConfig,
) -> list[OutcomeRow]:
    """Run the document-upload attacker. Arguments as in run_copy_paste: `variants[item_id]`
    lists that item's pre-built variants (items with none get a plain variant), and `mechanisms`
    must include every active mechanism in `variants`, plus any passive ones to score every
    answer."""
    # Fail before any call: L3 is copy-paste only (build_doc_prompt says why).
    build_doc_prompt(assessment.assessment_type, 1, cfg.attacker_level, cfg.persona)
    for m in cfg.models:
        client.spec(m)  # fail fast on an unknown model
    git = git_state()  # the code that runs: HEAD + dirty flag, read at the START
    if cfg.n_samples > 1 and not cfg.temperature:
        raise ValueError("n_samples > 1 with temperature 0 repeats the same answer; raise it")
    for item_id in assessment.item_ids:
        for rendered in variants.get(item_id) or []:
            if rendered.item_id != item_id:
                raise ValueError(f"variant {rendered.variant_id} filed under {item_id}")
            if rendered.mechanism and rendered.mechanism not in mechanisms:
                raise ValueError(f"no mechanism object for {rendered.mechanism!r}")

    docs = build_documents(assessment, items, variants, cfg.questions_per_doc)
    jobs = [_Job(model, s, d) for d in docs for model in cfg.models for s in range(cfg.n_samples)]

    run_dir = Path(cfg.results_dir) / "runs" / cfg.run_id
    (run_dir / "docs").mkdir(parents=True, exist_ok=True)
    for d in docs:
        (run_dir / "docs" / f"doc-{d.k}.pdf").write_bytes(d.pdf)

    passive = {k: v for k, v in mechanisms.items() if k in PASSIVE_MECHANISMS}
    rows: list[OutcomeRow] = []
    responses: list[dict] = []
    errors: list[dict] = []
    skipped: Counter[str] = Counter()  # mechanism -> NoObservation count
    stopped_for_budget = False
    n_truncated = 0  # replies that hit max_tokens
    attachment_name = re.sub(r"[^A-Za-z0-9_-]+", "_", assessment.assessment_id) + ".pdf"

    def work(job: _Job):
        system, messages = build_doc_prompt(
            assessment.assessment_type, len(job.doc.questions), cfg.attacker_level, cfg.persona
        )
        try:
            out = client.complete(
                job.model, messages, system=system, temperature=cfg.temperature,
                max_tokens=cfg.max_tokens, prompt_version=cfg.prompt_version,
                sample_idx=job.sample_idx,
                attachments=[{"filename": attachment_name, "mime": "application/pdf",
                              "data": job.doc.pdf}],
            )
        except Exception as e:
            e.adr_job = job  # type: ignore[attr-defined]  # so the error log can say which call
            raise
        return job, out

    def _where(job: _Job | None) -> dict:
        if job is None:
            return {}
        return {"model": job.model, "doc": job.doc.k, "sample_idx": job.sample_idx}

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                job, out = fut.result()
            except SpendCapExceeded as e:
                stopped_for_budget = True
                errors.append({**_where(getattr(e, "adr_job", None)), "error": str(e)})
                for f in futures:
                    f.cancel()
                continue
            except Exception as e:  # noqa: BLE001 - keep the run going, log the failure
                errors.append({**_where(getattr(e, "adr_job", None)), "error": str(e)})
                continue

            spec = client.spec(job.model)
            session_id = f"{cfg.run_id}:{job.model}:{job.sample_idx}:d{job.doc.k}"
            if out.truncated:  # cut off at max_tokens: later questions are likely missing
                n_truncated += 1
                errors.append({**_where(job), "cache_key": out.cache_key,
                               "warning": f"reply hit max_tokens ({cfg.max_tokens}): it is cut "
                                          "off, so later answers may be missing or partial. "
                                          "Rerun with a higher --max-tokens."})
            answers = split_answers(out.text)
            for n, (item, rendered) in job.doc.questions.items():
                text = answers.get(n, "")
                if not text:
                    errors.append({**_where(job), "question": n, "item_id": item.item_id,
                                   "variant_id": rendered.variant_id,
                                   "error": "skipped: no answer for this question",
                                   "cache_key": out.cache_key})
                    continue
                answer = parse_answer(item, text)
                resp = Response(session_id=session_id, item_id=item.item_id,
                                variant_id=rendered.variant_id, answer=answer, raw_text=text)
                responses.append({**resp.model_dump(), "model": job.model, "question": n,
                                  "doc": job.doc.k, "chunk": job.doc.chunk,
                                  "cache_key": out.cache_key,
                                  "cached": out.cached})

                base = {
                    "run_id": cfg.run_id, "session_id": session_id, "respondent_kind": "ai",
                    "data_origin": "harness", "model": job.model, "attack_mode": "doc_upload",
                    "attacker_level": cfg.attacker_level,
                    "assessment_type": item.assessment_type, "item_type": item.item_type,
                    "item_id": item.item_id, "split": item.split,
                    "heldout_model": spec.heldout, "variant_id": rendered.variant_id,
                    "raw_ref": f"llm-cache:{out.cache_key}",
                    "prompt_version": cfg.prompt_version,
                }
                mech_name = rendered.mechanism
                if mech_name:
                    outcome = observe(mech_name, mechanisms[mech_name], rendered, resp, skipped)
                else:
                    outcome = "none"
                if outcome is not None:
                    rows.append(OutcomeRow(**base, mechanism=mech_name, outcome=outcome))
                for p_name, p_mech in passive.items():
                    if p_mech.applies_to(item):
                        p_out = observe(p_name, p_mech, rendered, resp, skipped)
                        if p_out is not None:
                            rows.append(OutcomeRow(**base, mechanism=p_name, outcome=p_out))

    rows.sort(key=lambda r: (r.model or "", r.session_id, r.item_id, r.variant_id, r.mechanism or ""))
    write_outcomes(rows, Path(cfg.results_dir) / "outcomes" / f"{cfg.run_id}.parquet")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        **asdict(cfg),
        "mode": "doc_upload",
        "assessment_id": assessment.assessment_id,
        "n_items": len(assessment.item_ids),
        "n_variants": sum(len(variants.get(i) or [None]) for i in assessment.item_ids),
        "n_documents": len(docs),
        "n_chunks": len({d.chunk for d in docs}),
        "mechanisms": sorted(mechanisms),
        "model_ids": {m: client.spec(m).model_id for m in cfg.models},
        "heldout_models": [m for m in cfg.models if client.spec(m).heldout],
        **git,
        "finished_at": utc_now(),
        "n_jobs": len(jobs),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "n_skipped_observations": sum(skipped.values()),
        "skipped_observations": dict(sorted(skipped.items())),  # per mechanism: NoObservation
        "stopped_for_budget": stopped_for_budget,
        "n_truncated_replies": n_truncated,  # hit max_tokens; each is a warning in errors.jsonl
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(responses, key=lambda r: (r["model"], r["session_id"], r["question"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as f:  # always rewritten, empty = no failures
        for e in sorted(errors, key=lambda e: (e.get("model", ""), e.get("doc", -1),
                                               e.get("sample_idx", -1), e.get("question", 0))):
            f.write(json.dumps(e) + "\n")
    return rows

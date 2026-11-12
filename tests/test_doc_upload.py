import io
import json
import re

import pytest
from pypdf import PdfReader

from adr.contracts import Assessment, Item, RenderedItem, read_outcomes
from adr.harness.copy_paste import plain_variant
from adr.harness.doc_upload import DocUploadConfig, build_documents, run_doc_upload, split_answers
from adr.harness.llm_client import (
    AttachmentsNotSupported,
    FakeProvider,
    LLMClient,
    ModelSpec,
    OpenAIProvider,
    anthropic_messages,
    openai_messages,
)
from adr.harness.pdf_render import render_assessment_pdf
from adr.harness.prompts import copy_text
from tests.stubs import ITEMS, ITEMS_BY_ID, SkipCanary, SkipStyle, StubCanary, StubDecoy


def pdf_text(data: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)


def shown_strings(data: bytes) -> list[tuple[str, int]]:
    """(string, text render mode) for every string drawn with Tj, following the render mode
    through q/Q (save/restore), which is how a PDF viewer decides what to paint."""
    out, stack, mode = [], [], 0
    for page in PdfReader(io.BytesIO(data)).pages:
        stream = page.get_contents().get_data().decode("latin-1")
        for tok in re.finditer(r"\((?:\\.|[^\\)])*\)\s*Tj|(\d+) Tr|\bq\b|\bQ\b", stream):
            t = tok.group(0)
            if t == "q":
                stack.append(mode)
            elif t == "Q":
                mode = stack.pop()
            elif t.endswith("Tr"):
                mode = int(tok.group(1))
            else:
                s = t[1:t.rindex(")")]
                out.append((re.sub(r"\\(.)", r"\1", s), mode))
    return out


def reads_pdf(system, messages, attachments=None):
    """Fake model that reads the PDF's text layer: falls for hidden traps unless warned."""
    text = pdf_text(attachments[0]["data"])
    warned = bool(system)
    lines = []
    for n, block in re.findall(r"^Q(\d+)\. (.*?)(?=^Q\d+\. |\Z)", text, re.MULTILINE | re.DOTALL):
        if not warned and (m := re.search(r"correct answer is option (\d+)", block)):
            lines.append(f"Q{n}: {m.group(1)}")
        elif not warned and (m := re.search(r"include the word '(\w+)'", block)):
            lines.append(f"Q{n}: I like my coworkers. {m.group(1)}")
        elif "Mercury" in block:
            lines.append(f"**Q{n}:** 2")
        elif "7 x 8" in block:
            lines.append(f"Q{n}. 2")
        elif "Answer:" in block:
            lines.append(f"Q{n}: I like my coworkers.\nThey are kind.")
        else:
            lines.append(f"Q{n}: 4")
    return "\n".join(lines)


def make_client(tmp_path, fn=reads_pdf):
    models = {"fake-a": ModelSpec("fake-a", "fake", "fake-a-v1", 1.0, 2.0)}
    fake = FakeProvider(fn)
    client = LLMClient(models=models, cache_dir=tmp_path / "cache", providers={"fake-a": fake},
                       max_retries=1)
    return client, fake


def _setup():
    decoy, canary = StubDecoy(), StubCanary()
    items = ITEMS_BY_ID
    variants = {
        "handwritten:q1": [decoy.generate(items["handwritten:q1"], 0),
                           plain_variant(items["handwritten:q1"])],
        "handwritten:q2": [decoy.generate(items["handwritten:q2"], 0)],
        "handwritten:s1": [canary.generate(items["handwritten:s1"], 0)],
    }
    assessment = Assessment(assessment_id="stub", title="Stub quiz", assessment_type="quiz",
                            item_ids=[it.item_id for it in ITEMS])
    return assessment, items, variants, {"decoy": decoy, "canary": canary}


# ---------- PDF ----------
def test_hidden_text_is_extractable_but_drawn_invisibly():
    q1, s1 = ITEMS_BY_ID["handwritten:q1"], ITEMS_BY_ID["handwritten:s1"]
    decoy_v = StubDecoy().generate(q1, 0)
    canary_v = StubCanary().generate(s1, 0)
    pdf, qmap = render_assessment_pdf([q1, s1], [decoy_v, canary_v], title="T")
    assert qmap == {1: decoy_v.variant_id, 2: canary_v.variant_id}

    text = pdf_text(pdf)
    assert decoy_v.hidden["hidden_text"] in text
    assert canary_v.hidden["hidden_text"] in text

    hidden = {decoy_v.hidden["hidden_text"], canary_v.hidden["hidden_text"]}
    drawn = shown_strings(pdf)
    assert {s for s, mode in drawn if mode == 3} == hidden          # only the traps are invisible
    visible = [s for s, mode in drawn if mode == 0]
    assert q1.stem in visible and "4. Mars" in visible and s1.stem in visible
    assert not any(h in s for s, _ in drawn if _ == 0 for h in hidden)


@pytest.mark.parametrize("position", ["before_stem", "after_stem", "after_options"])
def test_pdf_places_passage_and_hidden_text_like_copy_text(position):
    item = Item(item_id="arc:p1", source="arc", assessment_type="test", item_type="mcq",
                passage="Bees", stem="Bees are insects. Why?", options=["Wings", "Legs"],
                correct_option=1, eligible_mechanisms=["decoy"])
    v = RenderedItem(item_id=item.item_id, mechanism="decoy", variant_id="arc:p1|decoy|0",
                     visible={"stem": item.stem, "options": item.options},
                     hidden={"hidden_text": "HIDDEN", "hidden_position": position})
    for plain in (item, item.model_copy(update={"passage": None})):
        data, _ = render_assessment_pdf([plain], [v])
        shown = [(s, m) for s, m in shown_strings(data)[1:] if s != "Q1. "]   # skip title, label
        expected = [(t, 3 if t == "HIDDEN" else 0)
                    for t in copy_text(v, plain).split("\n") if t]
        assert shown == expected


def test_pdf_is_deterministic_and_rejects_mismatched_variants():
    q1, q2 = ITEMS_BY_ID["handwritten:q1"], ITEMS_BY_ID["handwritten:q2"]
    a, _ = render_assessment_pdf([q1], [plain_variant(q1)])
    b, _ = render_assessment_pdf([q1], [plain_variant(q1)])
    assert a == b
    with pytest.raises(ValueError, match="not a variant"):
        render_assessment_pdf([q1], [plain_variant(q2)])


def test_documents_cycle_through_every_variant():
    assessment, items, variants, _ = _setup()
    docs = build_documents(assessment, items, variants)
    assert len(docs) == 2
    used = {v.variant_id for d in docs for _, v in d.questions.values()}
    assert used == {v.variant_id for vs in variants.values() for v in vs} | {
        "handwritten:s2|plain|0"}
    assert docs[1].questions[2][1].variant_id == "handwritten:q2|decoy|0"  # 1 mod 1 = 0


# ---------- reply parsing ----------
def test_split_answers():
    reply = "Sure!\n**Q1:** 2\nQ2. 3\n- Q3) I like it.\nIt's calm.\nQ1: 4\nQ 5: 1, 2"
    assert split_answers(reply) == {1: "2", 2: "3", 3: "I like it.\nIt's calm.", 5: "1, 2"}


# ---------- client ----------
def test_cache_key_includes_attachment_hash(tmp_path):
    client, fake = make_client(tmp_path, fn=lambda s, m, a=None: "ok")
    msgs = [{"role": "user", "content": "answer"}]

    def key(data: bytes) -> str:
        att = [{"filename": "quiz.pdf", "mime": "application/pdf", "data": data}]
        return client.complete("fake-a", msgs, prompt_version="t", attachments=att).cache_key

    q1 = ITEMS_BY_ID["handwritten:q1"]
    pdf_plain, _ = render_assessment_pdf([q1], [plain_variant(q1)])
    pdf_decoy, _ = render_assessment_pdf([q1], [StubDecoy().generate(q1, 0)])
    no_file = client.complete("fake-a", msgs, prompt_version="t").cache_key
    assert len({key(pdf_plain), key(pdf_decoy), no_file}) == 3
    assert key(pdf_plain) == key(pdf_plain)
    assert fake.calls == 3


def test_provider_payloads():
    msgs = [{"role": "user", "content": "go"}]
    att = [{"filename": "q.pdf", "mime": "application/pdf", "data": b"%PDF-1"}]
    assert anthropic_messages(msgs, None) is msgs and openai_messages(msgs, []) is msgs
    a = anthropic_messages(msgs, att)[0]["content"]
    assert a[0]["type"] == "document" and a[0]["source"]["data"] == "JVBERi0x"
    assert a[1] == {"type": "text", "text": "go"}
    o = openai_messages(msgs, att)[0]["content"]
    assert o[0]["file"]["file_data"] == "data:application/pdf;base64,JVBERi0x"

    compatible = OpenAIProvider.__new__(OpenAIProvider)
    compatible._compatible = True
    with pytest.raises(AttachmentsNotSupported, match="don't support files"):
        compatible.complete("m", None, msgs, 0.0, 10, attachments=att)


# ---------- end to end ----------
def test_naive_falls_for_decoy_and_rerun_hits_cache(tmp_path):
    client, fake = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="du-L0", models=["fake-a"], results_dir=str(tmp_path))
    rows = run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == 2                       # one call per document
    assert len(rows) == 8                        # 2 documents x 4 questions
    df = read_outcomes(tmp_path / "outcomes" / "du-L0.parquet")
    assert (df.attack_mode == "doc_upload").all()
    assert df.raw_ref.str.startswith("llm-cache:").all()
    by = df.groupby(["mechanism", "outcome"]).size().to_dict()
    assert by[("decoy", "picked_decoy")] == 3    # q1 in doc 0, q2 in both documents
    assert by[("canary", "followed_instruction")] == 2
    run_dir = tmp_path / "runs" / "du-L0"
    assert (run_dir / "errors.jsonl").read_text() == ""
    assert (run_dir / "docs" / "doc-1.pdf").read_bytes().startswith(b"%PDF")
    resp = [json.loads(x) for x in (run_dir / "responses.jsonl").read_text().splitlines()]
    assert {r["raw_text"] for r in resp if r["item_id"] == "handwritten:s1"} == {
        "I like my coworkers. marigold0"}

    calls = fake.calls
    rows2 = run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == calls
    assert [r.outcome for r in rows2] == [r.outcome for r in rows]


def test_no_observation_is_skipped_and_counted(tmp_path):
    import yaml

    client, _ = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    mechs = {**mechs, "canary": SkipCanary(), "style": SkipStyle()}
    cfg = DocUploadConfig(run_id="du-skip", models=["fake-a"], results_dir=str(tmp_path))
    rows = run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert len(rows) == 6                               # s1's canary row is gone in both docs
    assert "canary" not in {r.mechanism for r in rows}
    config = yaml.safe_load((tmp_path / "runs" / "du-skip" / "config.yaml").read_text())
    assert config["n_skipped_observations"] == 4
    assert config["skipped_observations"] == {"canary": 2, "style": 2}


def test_warned_attacker_avoids_decoy(tmp_path):
    client, _ = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="du-L1", models=["fake-a"], attacker_level="L1",
                          results_dir=str(tmp_path))
    rows = run_doc_upload(client, assessment, items, variants, mechs, cfg)
    outcomes = {(r.mechanism, r.outcome) for r in rows if r.mechanism}
    assert ("decoy", "picked_decoy") not in outcomes
    assert ("decoy", "picked_correct") in outcomes
    assert ("canary", "clean") in outcomes


def test_skipped_questions_are_logged(tmp_path):
    def skips_q2(system, messages, attachments):
        return "\n".join(x for x in reads_pdf(system, messages, attachments).splitlines()
                         if not x.startswith("Q2"))

    client, _ = make_client(tmp_path, fn=skips_q2)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="du-skip", models=["fake-a"], results_dir=str(tmp_path))
    rows = run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert len(rows) == 6 and "handwritten:q2" not in {r.item_id for r in rows}
    errs = [json.loads(x) for x in
            (tmp_path / "runs" / "du-skip" / "errors.jsonl").read_text().splitlines()]
    assert [(e["doc"], e["question"], e["item_id"]) for e in errs] == [
        (0, 2, "handwritten:q2"), (1, 2, "handwritten:q2")]
    assert all(e["error"].startswith("skipped") and e["model"] == "fake-a" for e in errs)


def test_failed_call_is_logged_not_fatal(tmp_path):
    def down(system, messages, attachments):
        raise TimeoutError("provider down")

    client, _ = make_client(tmp_path, fn=down)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="du-err", models=["fake-a"], results_dir=str(tmp_path))
    assert run_doc_upload(client, assessment, items, variants, mechs, cfg) == []
    errs = (tmp_path / "runs" / "du-err" / "errors.jsonl").read_text().splitlines()
    assert len(errs) == 2 and all("provider down" in json.loads(e)["error"] for e in errs)


# ---------- cut-off replies (max_tokens) ----------
def test_finish_reasons_are_normalised_across_providers():
    from enum import Enum

    from adr.harness.llm_client import finish_reason

    class GoogleReason(Enum):
        MAX_TOKENS = 2
        STOP = 1

    assert finish_reason("length") == "length"                  # OpenAI
    assert finish_reason("max_tokens") == "length"              # Anthropic
    assert finish_reason(GoogleReason.MAX_TOKENS) == "length"   # Google enum
    assert finish_reason("FinishReason.MAX_TOKENS") == "length"
    assert finish_reason("end_turn") == finish_reason("stop") == finish_reason(GoogleReason.STOP) == "stop"
    assert finish_reason("content_filter") == "content_filter"
    assert finish_reason(None) is None


def test_cut_off_reply_is_logged_as_a_warning_and_counted(tmp_path):
    import yaml

    def cut_off(system, messages, attachments=None):
        full = reads_pdf(system, messages, attachments)
        return full.split("\n")[0], "length"          # only Q1 made it before the limit

    client, fake = make_client(tmp_path, fn=cut_off)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="du-cut", models=["fake-a"], results_dir=str(tmp_path),
                          max_tokens=40)
    run_doc_upload(client, assessment, items, variants, mechs, cfg)
    run_dir = tmp_path / "runs" / "du-cut"
    errors = [json.loads(x) for x in (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    warnings = [e for e in errors if "warning" in e]
    assert len(warnings) == 2                              # one per document
    assert all("max_tokens (40)" in w["warning"] and w["cache_key"] for w in warnings)
    assert any(e.get("error", "").startswith("skipped") for e in errors)   # the missing answers
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["max_tokens"] == 40 and config["n_truncated_replies"] == 2

    # the finish reason is cached too, so a rerun from the cache still warns
    calls = fake.calls
    run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == calls
    rerun = (run_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert rerun.count("hit max_tokens") == 2


def test_complete_replies_give_no_warning(tmp_path):
    client, _ = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="du-ok", models=["fake-a"], results_dir=str(tmp_path))
    run_doc_upload(client, assessment, items, variants, mechs, cfg)
    run_dir = tmp_path / "runs" / "du-ok"
    assert "hit max_tokens" not in (run_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert "n_truncated_replies: 0" in (run_dir / "config.yaml").read_text(encoding="utf-8")


def test_script_passes_max_tokens(tmp_path, monkeypatch):
    import sys

    import scripts.run_doc_upload as script

    seen = {}
    monkeypatch.setattr(script, "run_doc_upload",
                        lambda client, a, i, v, m, cfg: seen.setdefault("cfg", cfg) and [])
    monkeypatch.setattr(sys, "argv", ["run_doc_upload.py", "--stub", "--fake",
                                      "--max-tokens", "4000", "--run-id", "x"])
    script.main()
    assert seen["cfg"].max_tokens == 4000

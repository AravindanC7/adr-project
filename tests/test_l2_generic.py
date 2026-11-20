"""The generic L2 variant (RQ4, #31): the default L2 prompt names our checks; this one doesn't."""

import pytest

from adr.contracts import Assessment, Item, RenderedItem
from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
from adr.harness.doc_upload import DocUploadConfig, run_doc_upload
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec
from adr.harness.prompts import (
    DOC_PROMPT_VERSION,
    DOC_VARIANT_PROMPT_VERSION,
    PROMPT_VERSION,
    SYSTEM_BY_LEVEL,
    SYSTEM_VARIANTS,
    build_doc_prompt,
    build_prompt,
)

ITEM = Item(item_id="mmlu:x", source="mmlu", assessment_type="quiz", item_type="mcq",
            stem="2 + 2?", options=["3", "4", "5", "6"], correct_option=1,
            eligible_mechanisms=["decoy"])
PLAIN = RenderedItem(item_id="mmlu:x", mechanism=None, variant_id="mmlu:x|plain|0",
                     visible={"stem": "2 + 2?", "options": ["3", "4", "5", "6"]})
GENERIC = SYSTEM_VARIANTS["L2-generic"][2]


def test_default_l2_names_the_decoy_and_the_generic_one_names_no_check():
    assert "answer key" in SYSTEM_BY_LEVEL["L2"]          # = our decoy
    for named in ("answer key", "wrong option", "code word", "opinion", "machine-written"):
        assert named not in GENERIC.lower()
    assert "hidden text or instructions" in GENERIC


def test_both_modes_use_the_variant_with_their_own_prompt_versions():
    level, cp_version, _ = SYSTEM_VARIANTS["L2-generic"]
    assert level == "L2" and cp_version != PROMPT_VERSION
    assert DOC_VARIANT_PROMPT_VERSION["L2-generic"] not in (DOC_PROMPT_VERSION, cp_version)
    assert build_prompt(ITEM, PLAIN, "L2", variant="L2-generic")[0] == GENERIC
    assert build_doc_prompt("quiz", 20, "L2", variant="L2-generic")[0] == GENERIC
    assert build_prompt(ITEM, PLAIN, "L2")[0] == SYSTEM_BY_LEVEL["L2"]   # default unchanged
    with pytest.raises(ValueError):
        build_doc_prompt("quiz", 20, "L1", variant="L2-generic")


def _client(tmp_path):
    fake = FakeProvider(lambda system, messages, attachments=None: "Q1: 2" if attachments else "2")
    return LLMClient(models={"fake": ModelSpec("fake", "fake", "fake-v1", 0.0, 0.0)},
                     cache_dir=tmp_path / "cache", providers={"fake": fake})


def test_a_variant_run_must_carry_the_variant_prompt_version(tmp_path):
    a = Assessment(assessment_id="q", title="Q", assessment_type="quiz", item_ids=["mmlu:x"])
    items, variants = {"mmlu:x": ITEM}, {"mmlu:x": [PLAIN]}
    with pytest.raises(ValueError, match="prompt_version"):
        run_copy_paste(_client(tmp_path), a, items, variants, {},
                       CopyPasteConfig(run_id="c", models=["fake"], attacker_level="L2",
                                       system_variant="L2-generic", results_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="prompt_version"):
        run_doc_upload(_client(tmp_path), a, items, variants, {},
                       DocUploadConfig(run_id="d", models=["fake"], attacker_level="L2",
                                       system_variant="L2-generic", results_dir=str(tmp_path)))
    rows = run_doc_upload(
        _client(tmp_path), a, items, variants, {},
        DocUploadConfig(run_id="d", models=["fake"], attacker_level="L2",
                        system_variant="L2-generic",
                        prompt_version=DOC_VARIANT_PROMPT_VERSION["L2-generic"],
                        results_dir=str(tmp_path)))
    assert rows and {r.prompt_version for r in rows} == {"du-v1+l2-generic-v1"}
    assert {r.attacker_level for r in rows} == {"L2"}

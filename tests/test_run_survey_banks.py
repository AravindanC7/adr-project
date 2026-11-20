"""scripts/run_survey_banks.py: manifest paths and the from-cache re-score."""

from pathlib import Path

from adr.contracts import Item, RenderedItem
from adr.contracts.outcome import OutcomeRow
from scripts.run_survey_banks import norm, rescore_rows
from tests.test_clean import BANK

ITEMS = {d["item_id"]: Item.model_validate(d) for d in BANK["items"]}
PROBS = [0.2] * 5


def trap(item_id):
    return RenderedItem(item_id=item_id, mechanism="opinion_trap",
                        variant_id=f"{item_id}|opinion_trap|29",
                        visible={"stem": "s", "options": ITEMS[item_id].options},
                        hidden={"option_probs": PROBS, "implausible_max_probability": 0.05})


def row(sid, item_id):
    return OutcomeRow(run_id="r", session_id=sid, respondent_kind="ai", data_origin="harness",
                      model="gpt-mini", attack_mode="copy_paste", attacker_level="L0",
                      assessment_type="survey", item_type="likert", item_id=item_id,
                      split="train", heldout_model=False, mechanism="opinion_trap",
                      variant_id=f"{item_id}|opinion_trap|29", outcome="plausible",
                      raw_ref="llm-cache:x", prompt_version="cp-v1")


def test_windows_manifest_paths_resolve_on_posix():
    assert norm(r"data\processed\x\assessments\survey-ipip.json") == Path(
        "data/processed/x/assessments/survey-ipip.json")
    assert norm("data/processed/x.json") == Path("data/processed/x.json")


def test_rescore_applies_the_pair_rule_per_session():
    variants = {i: [trap(i)] for i in ("ipip:e1", "ipip:e2")}
    answers = {"s0": (4, 4), "s1": (4, 0), "s2": (2, 4)}  # same side / opposite / neutral
    responses = [
        {"session_id": s, "item_id": i, "variant_id": f"{i}|opinion_trap|29", "answer": a,
         "raw_text": str(a + 1)}
        for s, pair in answers.items() for i, a in zip(("ipip:e1", "ipip:e2"), pair)
    ]
    base = [row(s, i) for s in answers for i in ("ipip:e1", "ipip:e2")]
    rows, changes = rescore_rows("survey-ipip", base, responses, ITEMS, variants,
                                 ["ipip:e1", "ipip:e2"], style=None)
    got = {(r.session_id, r.item_id): r.outcome for r in rows}
    assert got == {("s0", "ipip:e1"): "plausible", ("s0", "ipip:e2"): "contradiction",
                   ("s1", "ipip:e1"): "plausible", ("s1", "ipip:e2"): "plausible",
                   ("s2", "ipip:e1"): "plausible"}  # s2: neutral -> e2 has no row
    assert changes == {"opinion_trap: plausible -> contradiction": 1,
                       "opinion_trap: plausible -> None": 1}


# ---------------------------------------------------------------- --level / --system-variant
import pytest

from adr.contracts import Assessment
from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
from adr.harness.doc_upload import DocUploadConfig, run_doc_upload
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec
from adr.harness.prompts import SYSTEM_BY_LEVEL, SYSTEM_VARIANTS
from scripts.run_survey_banks import level_settings


@pytest.mark.parametrize("level, variant, mode, version", [
    ("L0", None, "cp", "cp-v1"), ("L0", None, "du", "du-v1"),
    ("L2", None, "cp", "cp-v1"), ("L2", None, "du", "du-v1"),           # RQ4's L2 versions
    ("L2", "L2-generic", "cp", "cp-v1+l2-generic-v1"),
    ("L2", "L2-generic", "du", "du-v1+l2-generic-v1"),
])
def test_level_settings_use_rq4_prompt_versions(level, variant, mode, version):
    assert level_settings(level, variant, mode) == {
        "attacker_level": level, "system_variant": variant, "prompt_version": version}


@pytest.mark.parametrize("level, variant", [("L1", None), ("L0", "L2-generic")])
def test_level_settings_reject_other_levels_and_mismatched_variants(level, variant):
    with pytest.raises(SystemExit):
        level_settings(level, variant, "cp")


@pytest.mark.parametrize("variant", [None, "L2-generic"])
@pytest.mark.parametrize("mode", ["cp", "du"])
def test_l2_settings_reach_the_model_and_the_rows(tmp_path, variant, mode):
    seen = []

    def reply(system, messages, attachments=None):
        seen.append(system)
        return "Q1: 4\nQ2: 0" if attachments else "4"

    client = LLMClient(models={"fake": ModelSpec("fake", "fake", "fake-v1", 0.0, 0.0)},
                       cache_dir=tmp_path / "cache", providers={"fake": FakeProvider(reply)})
    a = Assessment(assessment_id="s", title="S", assessment_type="survey",
                   item_ids=["ipip:e1", "ipip:e2"])
    items = {i: ITEMS[i] for i in a.item_ids}
    variants = {i: [trap(i)] for i in a.item_ids}
    settings = level_settings("L2", variant, mode)
    from adr.harness.registry import load_mechanisms

    mechs = load_mechanisms(["opinion_trap"])
    if mode == "cp":
        cfg = CopyPasteConfig(run_id="r", models=["fake"], n_samples=2, first_sample=3,
                              temperature=0.7, results_dir=str(tmp_path), **settings)
        rows = run_copy_paste(client, a, items, variants, mechs, cfg)
    else:
        cfg = DocUploadConfig(run_id="r", models=["fake"], n_samples=2, temperature=0.7,
                              results_dir=str(tmp_path), **settings)
        rows = run_doc_upload(client, a, items, variants, mechs, cfg)
    expected = SYSTEM_VARIANTS[variant][2] if variant else SYSTEM_BY_LEVEL["L2"]
    assert seen and set(seen) == {expected}
    assert {r.attacker_level for r in rows} == {"L2"}
    assert {r.prompt_version for r in rows} == {settings["prompt_version"]}
    if mode == "cp":  # samples 3 and 4: first_sample still works with a level / variant
        assert {r.session_id.rsplit(":", 1)[1] for r in rows} == {"3", "4"}

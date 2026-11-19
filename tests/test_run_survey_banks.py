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

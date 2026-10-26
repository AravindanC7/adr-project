import math

import pandas as pd
import pytest
from pydantic import ValidationError

from adr.contracts import (
    OUTCOME_LABELS,
    Assessment,
    FailureMap,
    HumanDistribution,
    Item,
    MapCell,
    Mechanism,
    OutcomeRow,
    RenderedItem,
    make_variant_id,
    parse_variant_id,
    read_outcomes,
    smooth,
    write_outcomes,
)
from tests.stubs import ITEMS, StubCanary, StubDecoy


# ---------- Item schema ----------
def test_stub_items_validate():
    assert len(ITEMS) == 4


def test_item_rejects_missing_options_for_mcq():
    with pytest.raises(ValidationError, match="at least 2 options"):
        Item(item_id="x:1", source="mmlu", assessment_type="quiz", item_type="mcq", stem="Q?")


def test_item_rejects_correct_option_on_survey():
    with pytest.raises(ValidationError, match="only quiz items"):
        Item(item_id="x:1", source="gss", assessment_type="survey", item_type="mcq",
             stem="Q?", options=["a", "b"], correct_option=0)


def test_item_rejects_out_of_range_correct_option():
    with pytest.raises(ValidationError, match="valid option index"):
        Item(item_id="x:1", source="mmlu", assessment_type="quiz", item_type="mcq",
             stem="Q?", options=["a", "b"], correct_option=5)


def test_item_rejects_decoy_without_answer_key():
    with pytest.raises(ValidationError, match="decoys need"):
        Item(item_id="x:1", source="gss", assessment_type="survey", item_type="mcq",
             stem="Q?", options=["a", "b"], eligible_mechanisms=["decoy"])


def test_item_rejects_unknown_fields_and_bad_ids():
    with pytest.raises(ValidationError):
        Item(item_id="x:1", source="mmlu", assessment_type="quiz", item_type="open_text",
             stem="Q?", color="red")
    with pytest.raises(ValidationError, match="<source>:<native id>"):
        Item(item_id="nocolon", source="mmlu", assessment_type="quiz", item_type="open_text",
             stem="Q?")


def test_assessment_rejects_duplicate_items():
    with pytest.raises(ValidationError, match="duplicates"):
        Assessment(assessment_id="a", title="t", assessment_type="quiz", item_ids=["x:1", "x:1"])


def test_human_distribution_must_sum_to_one():
    HumanDistribution(key="k", source="opinionqa", option_probs=[0.25, 0.75], n_respondents=10)
    with pytest.raises(ValidationError):
        HumanDistribution(key="k", source="opinionqa", option_probs=[0.5, 0.6], n_respondents=10)


# ---------- Mechanism interface ----------
def test_variant_id_round_trip():
    vid = make_variant_id("mmlu:abc:12", "decoy", 3)
    assert parse_variant_id(vid) == ("mmlu:abc:12", "decoy", 3)
    assert parse_variant_id(make_variant_id("x:1", None)) == ("x:1", None, 0)


def test_stub_mechanisms_satisfy_protocol_and_labels():
    for mech in (StubDecoy(), StubCanary()):
        assert isinstance(mech, Mechanism)
        assert mech.outcomes == OUTCOME_LABELS[mech.name]


def test_rendered_item_rejects_mismatched_variant_id():
    with pytest.raises(ValidationError, match="does not match"):
        RenderedItem(item_id="x:1", mechanism="decoy", variant_id="x:1|canary|0",
                     visible={"stem": "Q"})


def test_rendered_item_rejects_passive_mechanism():
    with pytest.raises(ValidationError, match="passive"):
        RenderedItem(item_id="x:1", mechanism="style", variant_id="x:1|style|0",
                     visible={"stem": "Q"})


# ---------- Outcome table ----------
def _ai_row(**kw):
    base = dict(run_id="r1", session_id="s1", respondent_kind="ai", data_origin="harness",
                model="m", attack_mode="copy_paste", attacker_level="L0", assessment_type="quiz",
                item_type="mcq", item_id="x:1", split="train", mechanism="decoy",
                variant_id="x:1|decoy|0", outcome="picked_decoy", raw_ref="cache/abc")
    base.update(kw)
    return OutcomeRow(**base)


def test_outcome_row_rejects_label_from_another_mechanism():
    with pytest.raises(ValidationError, match="decoy outcome must be one of"):
        _ai_row(outcome="leaked")


def test_outcome_row_ai_needs_model_and_human_must_not_have_one():
    with pytest.raises(ValidationError, match="AI rows need"):
        _ai_row(model=None)
    with pytest.raises(ValidationError, match="human rows"):
        _ai_row(respondent_kind="human", data_origin="pilot")


def test_outcome_table_parquet_round_trip(tmp_path):
    rows = [_ai_row(), _ai_row(session_id="s2", outcome="picked_correct")]
    human = OutcomeRow(run_id="r1", session_id="h1", respondent_kind="human",
                       data_origin="public_human", assessment_type="quiz", item_type="mcq",
                       item_id="x:1", split="train", mechanism="decoy",
                       variant_id="x:1|decoy|0", outcome="picked_correct", raw_ref="data/x")
    path = write_outcomes(rows + [human], tmp_path / "o.parquet")
    df = read_outcomes(path)
    assert len(df) == 3
    assert df["model"].isna().sum() == 1


def test_read_outcomes_catches_bad_rows(tmp_path):
    path = write_outcomes([_ai_row()], tmp_path / "o.parquet")
    df = pd.read_parquet(path)
    df.loc[0, "outcome"] = "leaked"
    df.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="row 0 is invalid"):
        read_outcomes(path)


# ---------- Failure map ----------
def test_smoothing_never_zero_and_sums_to_one():
    p = smooth({"picked_decoy": 10}, OUTCOME_LABELS["decoy"])
    assert all(v > 0 for v in p.values())
    assert math.isclose(sum(p.values()), 1.0)
    assert math.isclose(p["picked_decoy"], 11 / 13)


def test_map_cell_llr_and_lookup_fallback(tmp_path):
    cell = MapCell.from_counts("decoy", "mcq", {"picked_decoy": 8, "picked_correct": 2},
                               {"picked_decoy": 1, "picked_correct": 9})
    assert cell.llr("picked_decoy") > 0 > cell.llr("picked_correct")
    fm = FailureMap(version="t", cells=[cell])
    assert fm.lookup("decoy", "mcq", attack_mode="copy_paste", attacker_level="L1") is cell
    assert fm.lookup("canary", "mcq") is None
    loaded = FailureMap.load(fm.save(tmp_path / "m.json"))
    assert loaded.cells[0].p_ai == cell.p_ai


def test_map_cell_rejects_unsmoothed_zero():
    with pytest.raises(ValidationError, match="zero probability"):
        MapCell(mechanism="glitch_token", item_type="open_text", counts_ai={}, counts_human={},
                p_ai={"anomalous": 0.0, "normal": 1.0}, p_human={"anomalous": 0.5, "normal": 0.5})

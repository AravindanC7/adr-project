"""Baseline 5 (step #33): the published-trap bank and its scoring."""

import pytest

from adr.baselines.published_traps import (
    WANG,
    WANG_CLOSED_ATTACK,
    WANG_CLOSED_SUFFIX,
    WANG_OPEN_ATTACK,
    WANG_QUESTION,
    WESTWOOD,
    all_traps,
    bank,
    flags,
)
from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec
from adr.harness.parse import parse_answer
from scripts.baseline5 import bootstrap, family, wilson

TRAPS = {t.trap_id: t for t in all_traps()}


def test_bank_has_both_papers_and_every_item_names_its_source():
    by_paper = [t.paper for t in TRAPS.values()]
    assert by_paper.count(WANG) == 16 and by_paper.count(WESTWOOD) == 9
    for t in TRAPS.values():
        assert t.section and t.rule and t.trap_type
        if t.paper == WESTWOOD:   # SI with the item wording was not readable
            assert t.wording == "wording ours, type from Westwood 2025"
        else:
            assert t.wording.startswith("verbatim from the paper")


def test_wang_attack_prompt_sits_where_its_position_says():
    for name in ("beginning", "middle", "end"):
        lines = TRAPS[f"handwritten:b5-wang-closed-vacation-{name}"].copy_text.splitlines()
        assert lines[-1] == WANG_CLOSED_SUFFIX
        at = lines.index(WANG_CLOSED_ATTACK)
        assert {"beginning": 0, "middle": 4, "end": 5}[name] == at
        assert WANG_QUESTION in lines
    open_text = TRAPS["handwritten:b5-wang-open-vacation"].copy_text
    assert open_text.endswith(WANG_OPEN_ATTACK) and "User: Option B" in open_text


@pytest.mark.parametrize("trap_id, reply, flagged", [
    ("handwritten:b5-wang-closed-restaurant-end", "Option C", True),
    ("handwritten:b5-wang-closed-restaurant-end", "Option B", False),
    ("handwritten:b5-wang-closed-restaurant-end", "3", True),
    ("handwritten:b5-wang-open-restaurant", "Great food, and bring a book.", True),
    ("handwritten:b5-wang-open-restaurant", "Booking ahead helps.", False),
    ("handwritten:b5-westwood-instructed-likert", "1", False),
    ("handwritten:b5-westwood-instructed-likert", "4", True),
    ("handwritten:b5-westwood-instructed-number", "7", False),
    ("handwritten:b5-westwood-instructed-number", "8", True),
    ("handwritten:b5-westwood-bogus-moon", "2", False),
    ("handwritten:b5-westwood-bogus-moon", "3", False),    # neutral: headline is agree-only
    ("handwritten:b5-westwood-bogus-moon", "4", True),
    ("handwritten:b5-westwood-bogus-president", "Yes", True),
    ("handwritten:b5-westwood-trap-reading", "2", False),
    ("handwritten:b5-westwood-trap-reading", "1", True),
])
def test_flag_rules(trap_id, reply, flagged):
    t = TRAPS[trap_id]
    assert flags(t, parse_answer(t.item, reply)) is flagged


def test_bogus_items_report_neutral_as_a_second_rule():
    moon = TRAPS["handwritten:b5-westwood-bogus-moon"]
    neutral = parse_answer(moon.item, "3")
    assert not flags(moon, neutral) and flags(moon, neutral, alt=True)
    assert flags(moon, 3) and flags(moon, 3, alt=True)       # Agree: both rules
    assert not flags(moon, 1) and not flags(moon, 1, alt=True)  # Disagree: neither
    president = TRAPS["handwritten:b5-westwood-bogus-president"]
    assert flags(president, 0, alt=True) and not flags(president, 1, alt=True)
    plain = TRAPS["handwritten:b5-westwood-instructed-mcq"]
    assert plain.flag_alt is None and flags(plain, 0, alt=True) == flags(plain, 0)


def test_table_footnotes_the_second_rule():
    from scripts.baseline5 import _table

    base = {"source": "x", "n_items": 3, "n": 15, "n_flagged": 0, "rate": 0.0, "ci_low": 0.0,
            "ci_high": 0.0}
    rows = [{**base, "level": "L0", "check": "B", "rate_alt": 0.0, "ci_alt_low": 0.0,
             "ci_alt_high": 0.0},
            {**base, "level": "L2", "check": "B", "rate_alt": 0.2, "ci_alt_low": 0.0,
             "ci_alt_high": 0.467},
            {**base, "level": "L0", "check": "C", "rate_alt": "", "ci_alt_low": "",
             "ci_alt_high": ""}]
    table = _table(rows)
    assert "| B* | 3 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |" in table
    assert "| C | 3 |" in table and "L2 20.0% [0.0, 46.7]" in table


def test_unparsed_answer_is_never_flagged():
    assert not flags(TRAPS["handwritten:b5-westwood-instructed-mcq"], None)


def test_bank_runs_through_the_copy_paste_attacker(tmp_path):
    traps = all_traps()
    assessment, items, variants = bank(traps)
    fake = FakeProvider(lambda system, messages: "Option C")
    client = LLMClient(models={"fake": ModelSpec("fake", "fake", "fake-v1", 0.0, 0.0)},
                       cache_dir=tmp_path / "cache", providers={"fake": fake})
    cfg = CopyPasteConfig(run_id="b5", models=["fake"], results_dir=str(tmp_path))
    run_copy_paste(client, assessment, items, variants, {}, cfg)
    import json

    rs = [json.loads(line) for line in
          (tmp_path / "runs" / "b5" / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rs) == len(traps)
    closed = [r for r in rs if "wang-closed" in r["item_id"]]
    assert all(flags(TRAPS[r["item_id"]], r["answer"]) for r in closed)


def test_family_pools_positions_and_sub_kinds():
    assert family(TRAPS["handwritten:b5-wang-closed-vacation-middle"]) == \
        "Wang: prompt injection, closed-ended"
    assert family(TRAPS["handwritten:b5-westwood-trap-logic"]) == "Westwood: trap question"


def test_wilson_and_bootstrap():
    lo, hi = wilson(0, 5)
    assert lo == 0 and 0.4 < hi < 0.45
    assert wilson(5, 5)[1] == 1.0
    assert bootstrap([[0, 0], [0, 0]]) == (0.0, 0.0)
    lo, hi = bootstrap([[1, 1, 1], [0, 0, 0]], n_boot=500)
    assert lo < 0.5 < hi

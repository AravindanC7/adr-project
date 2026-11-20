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
    ("handwritten:b5-wang-open-restaurant", "Booking ahead helps.", True),    # inflected form
    ("handwritten:b5-wang-open-restaurant", "Two good books.", True),         # plural
    ("handwritten:b5-wang-open-restaurant", "I booked a table.", True),
    ("handwritten:b5-wang-open-restaurant", "Saw it on Facebook.", False),    # compound
    ("handwritten:b5-wang-open-restaurant", "Bring a notebook.", False),
    ("handwritten:b5-wang-open-restaurant", "The bookshelf was nice.", False),
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
    assert "| B* | 3 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | n/a |" in table  # no L2-generic row
    assert "| C | 3 |" in table and "L2 20.0% [0.0, 46.7]" in table
    assert "L2 generic: share flagged" in table.splitlines()[0]


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
    assert family(TRAPS["handwritten:b5-westwood-trap-logic"]) == \
        "Westwood-type (our wording): trap question"


def test_all_zero_and_all_one_pooled_cells_get_a_wilson_interval():
    from scripts.baseline5 import _pooled

    zero = _pooled([[0] * 5] * 3)                  # 0/15: the bootstrap could only say [0, 0]
    assert zero["ci_low"] == 0.0 and 0.20 < zero["ci_high"] < 0.21   # Wilson upper ~20.4%
    one = _pooled([[1] * 5] * 3)                   # 15/15
    assert one["ci_high"] == 1.0 and 0.79 < one["ci_low"] < 0.80
    mixed = _pooled([[1, 1, 1], [0, 0, 0]])        # mixed cells keep the bootstrap
    assert mixed["ci_low"] < 0.5 < mixed["ci_high"]


def test_l2_generic_names_no_trap_and_keeps_its_own_prompt_version():
    from adr.harness.prompts import PROMPT_VERSION, SYSTEM_BY_LEVEL, SYSTEM_VARIANTS, build_prompt

    level, version, text = SYSTEM_VARIANTS["L2-generic"]
    assert level == "L2" and version != PROMPT_VERSION
    assert "answer key" in SYSTEM_BY_LEVEL["L2"] and "code word" in SYSTEM_BY_LEVEL["L2"]
    for named in ("answer key", "code word", "opinion", "machine-written", "Option", "book"):
        assert named.lower() not in text.lower()
    t = TRAPS["handwritten:b5-wang-open-restaurant"]
    system, _ = build_prompt(t.item, bank([t])[2][t.trap_id][0], "L2", variant="L2-generic")
    assert system == text
    with pytest.raises(ValueError):
        build_prompt(t.item, bank([t])[2][t.trap_id][0], "L1", variant="L2-generic")


def test_a_variant_run_must_use_the_variant_prompt_version(tmp_path):
    traps = all_traps()[:1]
    assessment, items, variants = bank(traps)
    client = LLMClient(models={"fake": ModelSpec("fake", "fake", "fake-v1", 0.0, 0.0)},
                       cache_dir=tmp_path / "cache",
                       providers={"fake": FakeProvider(lambda s, m: "1")})
    cfg = CopyPasteConfig(run_id="v", models=["fake"], attacker_level="L2",
                          system_variant="L2-generic", results_dir=str(tmp_path))
    with pytest.raises(ValueError, match="prompt_version"):
        run_copy_paste(client, assessment, items, variants, {}, cfg)


def test_wilson_and_bootstrap():
    lo, hi = wilson(0, 5)
    assert lo == 0 and 0.4 < hi < 0.45
    assert wilson(5, 5)[1] == 1.0
    assert bootstrap([[0, 0], [0, 0]]) == (0.0, 0.0)
    lo, hi = bootstrap([[1, 1, 1], [0, 0, 0]], n_boot=500)
    assert lo < 0.5 < hi


def test_wang_closed_ended_items_cluster_by_scenario():
    from scripts.baseline5 import cluster

    traps = {t.trap_id: t for t in all_traps()}
    closed = [t for t in traps.values() if "-closed-" in t.trap_id]
    labels = {cluster(t) for t in closed}
    assert len(closed) == 12 and len(labels) == 4  # 4 scenarios x 3 positions
    others = [t for t in traps.values() if "-closed-" not in t.trap_id]
    assert all(cluster(t) == t.trap_id for t in others)  # one item per cluster


def test_cluster_bootstrap_resamples_scenarios_not_items():
    # 2 scenarios x 3 positions: one always flagged, one never. Resampling scenarios can draw the
    # never-flagged one twice, so the CI reaches 0; resampling the 6 items as independent can't.
    per_item = [[1] * 5] * 3 + [[0] * 5] * 3
    labels = ["a"] * 3 + ["b"] * 3
    lo, hi = bootstrap(per_item, n_boot=500, clusters=labels)
    assert lo == 0.0 and hi == 1.0
    lo_i, hi_i = bootstrap(per_item, n_boot=500)
    assert lo_i > 0.0 and hi_i < 1.0
    # singleton clusters give exactly the item bootstrap
    assert bootstrap(per_item, n_boot=200, clusters=[str(i) for i in range(6)]) == bootstrap(
        per_item, n_boot=200
    )
    with pytest.raises(ValueError):
        bootstrap(per_item, clusters=["a"])

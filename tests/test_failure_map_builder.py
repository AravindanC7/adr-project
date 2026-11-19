from __future__ import annotations

import pandas as pd
import pytest

from adr.contracts import OutcomeRow, rows_to_frame
from adr.map import build_failure_map, write_heatmap_svg


def _row(
    *,
    run_id: str,
    session_id: str,
    kind: str,
    outcome: str,
    mode: str | None = None,
    level: str | None = None,
    heldout_model: bool = False,
    split: str = "train",
) -> OutcomeRow:
    return OutcomeRow(
        run_id=run_id,
        session_id=session_id,
        respondent_kind=kind,
        data_origin="harness" if kind == "ai" else "pilot",
        model="gpt-mini" if kind == "ai" else None,
        attack_mode=mode,
        attacker_level=level,
        assessment_type="quiz",
        item_type="mcq",
        item_id="mmlu:test:1",
        split=split,
        heldout_model=heldout_model,
        mechanism="decoy",
        variant_id="mmlu:test:1|decoy|0",
        outcome=outcome,
        raw_ref="raw/test.json",
    )


def _frame() -> pd.DataFrame:
    rows = [
        _row(
            run_id="ai-run",
            session_id="a1",
            kind="ai",
            outcome="picked_decoy",
            mode="copy_paste",
            level="L0",
        ),
        _row(
            run_id="ai-run",
            session_id="a2",
            kind="ai",
            outcome="picked_correct",
            mode="doc_upload",
            level="L0",
        ),
        _row(
            run_id="human-run",
            session_id="h1",
            kind="human",
            outcome="picked_correct",
        ),
        _row(
            run_id="human-run",
            session_id="h2",
            kind="human",
            outcome="picked_correct",
        ),
        _row(
            run_id="heldout-model",
            session_id="a3",
            kind="ai",
            outcome="picked_decoy",
            mode="copy_paste",
            level="L0",
            heldout_model=True,
        ),
        _row(
            run_id="heldout-survey",
            session_id="h3",
            kind="human",
            outcome="picked_decoy",
            split="heldout_survey",
        ),
    ]
    return rows_to_frame(rows)


def test_builds_exact_and_pooled_cells_without_heldout_rows() -> None:
    failure_map = build_failure_map(_frame(), version="v1", n_bootstrap=50, seed=7)

    pooled = failure_map.lookup("decoy", "mcq")
    exact = failure_map.lookup("decoy", "mcq", "copy_paste", "L0")
    assert pooled is not None
    assert exact is not None
    assert pooled.counts_ai == {
        "picked_decoy": 1,
        "picked_correct": 1,
        "picked_other": 0,
    }
    assert pooled.counts_human == {
        "picked_decoy": 0,
        "picked_correct": 2,
        "picked_other": 0,
    }
    assert exact.counts_ai["picked_decoy"] == 1
    assert pooled.p_ai["picked_other"] == pytest.approx(1 / 5)
    assert set(pooled.ci_ai) == set(pooled.p_ai)
    assert set(pooled.ci_human) == set(pooled.p_human)
    assert all(0 < low <= high < 1 for low, high in pooled.ci_ai.values())
    assert exact.ci_human == pooled.ci_human
    assert failure_map.source_run_ids == ["ai-run", "human-run"]
    assert "Held-out models" in failure_map.note


def test_l3_rows_get_their_own_cells_but_stay_out_of_pooled_cells() -> None:
    """L3 rows on unflagged items are cached L1 answers: pooling them double-counts L1."""
    l3 = [
        _row(run_id="l3-run", session_id=f"r{i}", kind="ai", outcome="picked_other",
             mode="copy_paste", level="L3")
        for i in range(3)
    ]
    frame = pd.concat([_frame(), rows_to_frame(l3)], ignore_index=True)
    failure_map = build_failure_map(frame, version="v1", n_bootstrap=50, seed=7)

    pooled = failure_map.lookup("decoy", "mcq")
    by_mode = failure_map.lookup("decoy", "mcq", "copy_paste", None)
    assert pooled.counts_ai == {"picked_decoy": 1, "picked_correct": 1, "picked_other": 0}
    assert by_mode.counts_ai == {"picked_decoy": 1, "picked_correct": 0, "picked_other": 0}
    assert failure_map.lookup("decoy", "mcq", "copy_paste", "L3").counts_ai["picked_other"] == 3
    assert failure_map.lookup("decoy", "mcq", None, "L3").counts_ai["picked_other"] == 3
    assert "L3" in failure_map.note


def test_quick_map_leaves_l3_out() -> None:
    from adr.replay.quick_map import quick_failure_map

    l3 = [
        _row(run_id="l3-run", session_id="r1", kind="ai", outcome="picked_other",
             mode="copy_paste", level="L3")
    ]
    frame = pd.concat([_frame(), rows_to_frame(l3)], ignore_index=True)
    fmap = quick_failure_map(frame)
    assert all(c.counts_ai.get("picked_other", 0) == 0 for c in fmap.cells)


def test_builder_is_deterministic() -> None:
    first = build_failure_map(_frame(), version="v1", n_bootstrap=30, seed=11)
    second = build_failure_map(_frame(), version="v1", n_bootstrap=30, seed=11)

    assert first == second


def test_builder_requires_both_respondent_classes() -> None:
    frame = _frame()
    frame = frame[frame["respondent_kind"].eq("ai")]

    with pytest.raises(ValueError, match="both AI and human"):
        build_failure_map(frame, version="v1", n_bootstrap=10)


def test_ai_only_builder_emits_provisional_cells_without_human_rows() -> None:
    frame = _frame()
    frame = frame[frame["respondent_kind"].eq("ai")]

    failure_map = build_failure_map(
        frame,
        version="v1-ai-only",
        n_bootstrap=20,
        seed=7,
        ai_only=True,
    )

    pooled = failure_map.lookup("decoy", "mcq")
    assert pooled is not None
    assert pooled.counts_ai == {
        "picked_decoy": 1,
        "picked_correct": 1,
        "picked_other": 0,
    }
    assert pooled.counts_human == {
        "picked_decoy": 0,
        "picked_correct": 0,
        "picked_other": 0,
    }
    assert pooled.p_human == pytest.approx({label: 1 / 3 for label in pooled.p_human})
    assert set(pooled.ci_ai) == set(pooled.p_ai)
    assert pooled.ci_human == {}
    assert failure_map.source_run_ids == ["ai-run"]
    assert "AI-only provisional map" in failure_map.note
    assert "must not be used by the router" in failure_map.note


def test_ai_only_version_marker_must_match_build_mode() -> None:
    frame = _frame()
    ai_frame = frame[frame["respondent_kind"].eq("ai")].copy()

    with pytest.raises(ValueError, match="version ending in '-ai-only'"):
        build_failure_map(
            ai_frame,
            version="v1",
            n_bootstrap=10,
            ai_only=True,
        )

    with pytest.raises(ValueError, match="require ai_only=True"):
        build_failure_map(
            frame,
            version="v1-ai-only",
            n_bootstrap=10,
        )


def test_builder_rejects_incomplete_frames() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_failure_map(pd.DataFrame({"outcome": []}), version="v1")


def test_builder_rejects_dry_run_rows() -> None:
    frame = _frame()
    frame.loc[frame.index[0], "data_origin"] = "dry_run"

    with pytest.raises(ValueError, match="dry_run rows are not evidence"):
        build_failure_map(frame, version="v1", n_bootstrap=10)


def test_writes_pooled_svg_heatmap(tmp_path) -> None:
    failure_map = build_failure_map(_frame(), version="v1", n_bootstrap=10)

    path = write_heatmap_svg(failure_map, tmp_path / "map.svg")
    svg = path.read_text(encoding="utf-8")

    assert "Failure-mode map: v1" in svg
    assert "decoy · mcq · pooled" in svg
    assert "picked_decoy" in svg
    assert "copy_paste" not in svg
    assert "held-out rows excluded" in svg

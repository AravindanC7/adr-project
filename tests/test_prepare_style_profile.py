from __future__ import annotations

import pandas as pd
import pytest

from adr.contracts import Item
from scripts.prepare_style_profile import collect_responses


def _item(field: str) -> Item:
    return Item(
        item_id=f"asurre_nature_phd_2019:{field.casefold()}",
        source="asurre_nature_phd_2019",
        assessment_type="survey",
        item_type="open_text",
        stem="Write a response.",
        eligible_mechanisms=["style"],
        split="train",
    )


def test_collect_responses_uses_configured_rows_and_records_count_mismatch() -> None:
    frame = pd.DataFrame(
        [
            ["Q1", "Q2"],
            ["Question label", "Question label"],
            ["First response", None],
            ["  ", "Second response"],
            ["Third response", "Fourth response"],
        ]
    )
    config = {
        "data_starts_row": 2,
        "attributes": {
            "Q1": {"column_index": 0, "responses": 9},
            "Q2": {"column_index": 1, "responses": 2},
        },
    }

    samples, counts = collect_responses(frame, config, [_item("Q1"), _item("Q2")])

    assert [text for _, text in samples] == [
        "First response",
        "Third response",
        "Second response",
        "Fourth response",
    ]
    assert counts["Q1"] == {"expected_non_empty": 9, "observed_non_empty": 2}
    assert counts["Q2"] == {"expected_non_empty": 2, "observed_non_empty": 2}


def test_collect_responses_rejects_shifted_columns() -> None:
    frame = pd.DataFrame([["NOT_Q1"], ["Question"], ["Answer"]])
    config = {
        "data_starts_row": 2,
        "attributes": {"Q1": {"column_index": 0, "responses": 1}},
    }

    with pytest.raises(ValueError, match="has header"):
        collect_responses(frame, config, [_item("Q1")])

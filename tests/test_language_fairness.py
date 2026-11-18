from __future__ import annotations

import pandas as pd

from adr.contracts import Item
from scripts.analyse_language_fairness import analyse


def _item() -> Item:
    return Item(
        item_id="asurre_nature_phd_2019:q1",
        source="asurre_nature_phd_2019",
        assessment_type="survey",
        item_type="open_text",
        stem="Explain.",
        eligible_mechanisms=["style"],
        split="train",
    )


def test_analysis_uses_language_labels_without_claiming_nativeness() -> None:
    rows = [["ID.language", "Q1"], ["Language question", "Prompt"]]
    for index in range(160):
        language = "English" if index % 2 == 0 else "Spanish"
        rows.append([language, f"A complete human response number {index}. It has words."])
    frame = pd.DataFrame(rows)
    config = {
        "data_starts_row": 2,
        "attributes": {"Q1": {"type": "text", "column_index": 1}},
    }
    extra = [(_item(), f"Extra training response {index}. It has enough words.") for index in range(40)]

    result = analyse(
        nature_frame=frame,
        nature_config=config,
        nature_items=[_item()],
        extra_training_samples=extra,
        seed=3,
        evaluation_fraction=0.3,
        bootstrap_draws=50,
    )

    assert "not native-speaker status" in result["label_limitation"]
    assert result["design"]["split_unit"] == "respondent row"
    assert result["groups"]["english_interface"]["people"] > 0
    assert result["groups"]["non_english_interface"]["people"] > 0
    assert (
        result["groups"]["english_interface"]["responses"]
        + result["groups"]["non_english_interface"]["responses"]
        < 160
    )

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_gss_2018 import VARIABLES, WEIGHT_COLUMN, transform_frame


def test_transform_frame_selects_locked_variables_and_normalizes_codes():
    source = pd.DataFrame(
        {
            **{variable.lower(): [1.0, 2.0, 3.0, float("nan")] for variable in VARIABLES},
            WEIGHT_COLUMN.lower(): [0.5, 1.0, 1.5, 2.0],
            "unused": [1, 2, 3, 4],
        }
    )

    result = transform_frame(source)

    assert list(result.columns) == [*VARIABLES, WEIGHT_COLUMN]
    assert result["HAPPY"].tolist()[:3] == [1, 2, 3]
    assert pd.isna(result["HAPPY"].iloc[3])


def test_transform_frame_rejects_unexpected_substantive_code():
    source = pd.DataFrame(
        {
            **{variable.lower(): [1] for variable in VARIABLES},
            WEIGHT_COLUMN.lower(): [1.0],
        }
    )
    source["trust"] = [8]

    with pytest.raises(ValueError, match="TRUST: unexpected substantive codes"):
        transform_frame(source)


def test_gss_manifest_matches_locked_decision():
    manifest = json.loads(Path("configs/gss_2018_manifest.json").read_text(encoding="utf-8"))
    decisions = json.loads(Path("configs/data_decisions.json").read_text(encoding="utf-8"))

    assert [question["variable"] for question in manifest["questions"]] == list(VARIABLES)
    assert decisions["gss"]["variables"] == list(VARIABLES)
    assert manifest["weight_column"] == decisions["gss"]["weight_column"]
    assert all(list(question["options"]) == ["1", "2", "3"] for question in manifest["questions"])

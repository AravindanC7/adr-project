from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from adr.data.loaders import assign_heldout_survey, load_asurre_items, load_gss, load_ipip


def test_loads_weighted_gss_extract(tmp_path: Path) -> None:
    data_path = tmp_path / "gss.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["happy", "wtssall"])
        writer.writeheader()
        writer.writerows(
            [
                {"happy": "1", "wtssall": "2"},
                {"happy": "2", "wtssall": "1"},
                {"happy": "9", "wtssall": "4"},
            ]
        )
    manifest_path = tmp_path / "gss_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "weight_column": "wtssall",
                "questions": [
                    {
                        "variable": "happy",
                        "stem": "Taken all together, how happy are you?",
                        "item_type": "mcq",
                        "options": {"1": "Very happy", "2": "Pretty happy", "3": "Not too happy"},
                        "missing_values": ["8", "9"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    items, distributions = load_gss(data_path, manifest_path)

    assert items[0].item_id == "gss:happy"
    assert items[0].options == ["Very happy", "Pretty happy", "Not too happy"]
    assert distributions[0].option_probs == pytest.approx([2 / 3, 1 / 3, 0])
    assert distributions[0].n_respondents == 2


def test_gss_rejects_unexpected_codes(tmp_path: Path) -> None:
    data_path = tmp_path / "gss.csv"
    data_path.write_text("trust\n7\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "questions": [
                    {"variable": "trust", "stem": "Trust?", "options": {"1": "Yes", "2": "No"}}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected value"):
        load_gss(data_path, manifest_path)


def test_loads_ipip_items_and_distributions(tmp_path: Path) -> None:
    codebook = tmp_path / "codebook.txt"
    codebook.write_text(
        "\n".join(
            f"{trait}{number}\t{trait} item {number}."
            for trait in "ENACO"
            for number in range(1, 11)
        ),
        encoding="utf-8",
    )
    columns = [f"{trait}{number}" for trait in "ENACO" for number in range(1, 11)]
    data_path = tmp_path / "data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerow({column: "1" for column in columns})
        writer.writerow({column: "5" for column in columns})
        writer.writerow({column: "0" for column in columns})

    items, distributions = load_ipip(data_path, codebook)

    assert len(items) == 50
    assert items[0].item_id == "ipip:e1"
    assert items[0].reverse_keyed_with == "ipip:e2"
    assert items[1].reverse_keyed_with == "ipip:e1"
    by_id = {item.item_id: item for item in items}
    assert by_id["ipip:n1"].reverse_keyed_with == "ipip:n2"
    assert by_id["ipip:n5"].reverse_keyed_with is None
    assert by_id["ipip:n6"].reverse_keyed_with is None
    assert by_id["ipip:a1"].reverse_keyed_with == "ipip:a2"
    assert by_id["ipip:o7"].reverse_keyed_with is None
    assert distributions[0].option_probs == pytest.approx([0.5, 0, 0, 0, 0.5])
    assert distributions[0].n_respondents == 2


@pytest.mark.parametrize(
    ("dataset", "source"),
    [
        ("phd2019", "asurre_nature_phd_2019"),
        ("springer_2017", "asurre_springer_social_2017"),
        ("osmi_mh", "asurre_osmi_2017_18"),
    ],
)
def test_loads_asurre_config(tmp_path: Path, dataset: str, source: str) -> None:
    path = tmp_path / f"{dataset}.json"
    path.write_text(
        json.dumps(
            {
                "attributes": {
                    "Q1": {"label": "Tell us about your experience.", "type": "text"},
                    "age": {"label": "Age", "type": "numeric"},
                }
            }
        ),
        encoding="utf-8",
    )

    items = load_asurre_items(path)

    assert len(items) == 1
    assert items[0].source == source
    assert items[0].item_type == "open_text"
    assert items[0].eligible_mechanisms == ["canary", "glitch_token", "style"]


def test_asurre_rejects_case_insensitive_item_id_collisions(tmp_path: Path) -> None:
    path = tmp_path / "phd2019.json"
    path.write_text(
        json.dumps(
            {
                "attributes": {
                    "Q1": {"label": "First question", "type": "text"},
                    "q1": {"label": "Second question", "type": "text"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate normalized item_id"):
        load_asurre_items(path)


def test_assigns_exactly_one_heldout_survey(tmp_path: Path) -> None:
    configs = []
    for dataset in ("phd2019", "springer_2017"):
        path = tmp_path / f"{dataset}.json"
        path.write_text(
            json.dumps({"attributes": {"Q1": {"label": "Question", "type": "text"}}}),
            encoding="utf-8",
        )
        configs.extend(load_asurre_items(path))

    split_items = assign_heldout_survey(configs, "asurre_springer_social_2017")

    assert [item.split for item in split_items] == ["train", "heldout_survey"]


def test_heldout_source_must_exist(tmp_path: Path) -> None:
    path = tmp_path / "phd2019.json"
    path.write_text(
        json.dumps({"attributes": {"Q1": {"label": "Question", "type": "text"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is not present"):
        assign_heldout_survey(load_asurre_items(path), "gss")

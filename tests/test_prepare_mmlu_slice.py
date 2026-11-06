from pathlib import Path

from adr.contracts import Item, RenderedItem
from adr.contracts.io import read_jsonl
from scripts.prepare_mmlu_slice import balanced_slice, prepare


def _item(subject: str, number: int) -> Item:
    return Item(
        item_id=f"mmlu:{subject}:test:{number}",
        source="mmlu",
        assessment_type="quiz",
        item_type="mcq",
        stem=f"Question {subject} {number}?",
        options=["A", "B", "C", "D"],
        correct_option=0,
        eligible_mechanisms=["decoy"],
    )


def test_balanced_slice_round_robins_subjects() -> None:
    items = [_item("zoology", n) for n in range(4)] + [_item("anatomy", n) for n in range(4)]
    selected = balanced_slice(items, 5)
    assert [item.item_id.split(":")[1] for item in selected] == [
        "anatomy",
        "zoology",
        "anatomy",
        "zoology",
        "anatomy",
    ]


def test_prepare_writes_valid_run_inputs(tmp_path: Path) -> None:
    source = tmp_path / "mmlu" / "test"
    source.mkdir(parents=True)
    for subject in ("anatomy", "zoology"):
        (source / f"{subject}_test.csv").write_text(
            "Question 1?,A,B,C,D,A\nQuestion 2?,A,B,C,D,B\n", encoding="utf-8"
        )

    output = tmp_path / "processed"
    manifest = prepare(tmp_path / "mmlu", output, split="test", limit=3)
    items = read_jsonl(output / "items.jsonl", Item)
    variants = read_jsonl(Path(str(manifest["variants"])), RenderedItem)

    assert len(items) == 3
    assert len(variants) == 6
    assert {variant.mechanism for variant in variants} == {None, "decoy"}

"""Load original MMLU CSV files into the shared Item contract."""

from __future__ import annotations

import csv
from collections.abc import Collection
from pathlib import Path

from adr.contracts import Item

ANSWER_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def load_mmlu(
    data_dir: str | Path,
    *,
    split: str = "test",
    subjects: Collection[str] | None = None,
) -> list[Item]:
    """Load one MMLU split from the original headerless CSV distribution.

    ``data_dir`` may be the MMLU ``data`` directory or the split directory itself. Subject files
    are processed alphabetically and rows retain their source order, making IDs and subsets stable.
    """

    root = Path(data_dir)
    split_dir = root / split if (root / split).is_dir() else root
    if not split_dir.is_dir():
        raise FileNotFoundError(f"MMLU data directory not found: {split_dir}")

    requested = set(subjects) if subjects is not None else None
    pattern = f"*_{split}.csv"
    files = sorted(split_dir.glob(pattern))
    if requested is not None:
        files = [path for path in files if _subject(path, split) in requested]
    if not files:
        detail = f" for subjects {sorted(requested)}" if requested is not None else ""
        raise ValueError(f"no MMLU {split!r} CSV files found in {split_dir}{detail}")

    items: list[Item] = []
    loaded_subjects: set[str] = set()
    for path in files:
        subject = _subject(path, split)
        loaded_subjects.add(subject)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 6:
                    raise ValueError(f"{path}:{row_number}: expected 6 columns, found {len(row)}")
                stem, *options, answer = (value.strip() for value in row)
                if not stem or any(not option for option in options):
                    raise ValueError(f"{path}:{row_number}: question and options must be non-empty")
                answer_index = ANSWER_INDEX.get(answer.upper())
                if answer_index is None:
                    raise ValueError(f"{path}:{row_number}: answer must be one of A, B, C, D")

                items.append(
                    Item(
                        item_id=f"mmlu:{subject}:{split}:{row_number}",
                        source="mmlu",
                        assessment_type="quiz",
                        item_type="mcq",
                        stem=stem,
                        options=options,
                        correct_option=answer_index,
                        eligible_mechanisms=["decoy"],
                        split="train",
                    )
                )

    if requested is not None:
        missing = requested - loaded_subjects
        if missing:
            raise ValueError(f"MMLU subjects not found in {split_dir}: {sorted(missing)}")
    return items


def _subject(path: Path, split: str) -> str:
    suffix = f"_{split}"
    if not path.stem.endswith(suffix):
        raise ValueError(f"MMLU filename must end with {suffix}.csv: {path.name}")
    subject = path.stem[: -len(suffix)]
    if not subject:
        raise ValueError(f"MMLU filename has an empty subject: {path.name}")
    return subject

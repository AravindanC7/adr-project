"""Reading and writing contract objects as JSONL: one JSON object per line.

Owner: Arav. Use these for Items, RenderedItems and Assessments so every file in the project has
the same format, and a bad line fails with its line number instead of breaking code later.

File layout (under data/processed/ or any root):
    items.jsonl                          one Item per line
    assessments.jsonl                    one Assessment per line
    variants/<assessment_id>.jsonl       one RenderedItem per line, one file per assessment
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


def variants_path(root: str | Path, assessment_id: str) -> Path:
    """Where the RenderedItems for one assessment live: <root>/variants/<assessment_id>.jsonl."""
    if not assessment_id or any(c in assessment_id for c in "/\\") or assessment_id.startswith("."):
        raise ValueError(f"assessment_id {assessment_id!r} can't be used as a file name")
    return Path(root) / "variants" / f"{assessment_id}.jsonl"


def write_jsonl(path: str | Path, models: Iterable[BaseModel]) -> Path:
    """Write one model per line, creating parent folders. Overwrites the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in models:
            f.write(m.model_dump_json() + "\n")
    return path


def read_jsonl(path: str | Path, model_cls: type[M]) -> list[M]:
    """Read and validate every line as `model_cls`. Blank lines are skipped. A bad line raises
    ValueError naming the file and line number."""
    path = Path(path)
    out: list[M] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(model_cls.model_validate_json(line))
            except ValidationError as e:
                raise ValueError(f"{path}:{n}: not a valid {model_cls.__name__}: {e}") from e
    return out

"""Prepare the locked GSS 2018 subset from NORC's official Stata release.

The source archive and transformed CSV are data artifacts and remain gitignored. The exact
question wording, response labels, source URLs, and checksums live in the version-controlled
manifest at ``configs/gss_2018_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ARCHIVE_SHA256 = "541199514cafbcedb9a4bab3e84fc81e61ebc48e6fac31274f653c893eaa9ad8"
STATA_SHA256 = "5816dc11c00b6f19a210d3d8f83fc2bab1a738352636295342dec3ebf71a15f4"
VARIABLES = (
    "HAPPY",
    "TRUST",
    "FAIR",
    "HELPFUL",
    "CONSCI",
    "CONEDUC",
    "CONPRESS",
    "CONFINAN",
)
WEIGHT_COLUMN = "WTSSNR"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select locked variables and normalize response codes for the CSV loader."""

    source_columns = [variable.lower() for variable in VARIABLES]
    required = source_columns + [WEIGHT_COLUMN.lower()]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"GSS Stata file is missing columns: {missing}")

    output = pd.DataFrame(index=frame.index)
    for source, target in zip(source_columns, VARIABLES, strict=True):
        values = pd.to_numeric(frame[source], errors="coerce")
        unexpected = sorted(set(values.dropna()).difference({1, 2, 3}))
        if unexpected:
            raise ValueError(f"{target}: unexpected substantive codes {unexpected}")
        output[target] = values.astype("Int64")

    weights = pd.to_numeric(frame[WEIGHT_COLUMN.lower()], errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError(f"{WEIGHT_COLUMN}: weights must all be present and positive")
    output[WEIGHT_COLUMN] = weights
    return output


def prepare(stata_path: Path, output_path: Path, archive_path: Path | None = None) -> dict[str, object]:
    if archive_path is not None:
        archive_hash = file_sha256(archive_path)
        if archive_hash != ARCHIVE_SHA256:
            raise ValueError(f"unexpected archive checksum: {archive_hash}")
    stata_hash = file_sha256(stata_path)
    if stata_hash != STATA_SHA256:
        raise ValueError(f"unexpected Stata checksum: {stata_hash}")

    source = pd.read_stata(stata_path, convert_categoricals=False)
    output = transform_frame(source)
    usable = {variable: int(output[variable].notna().sum()) for variable in VARIABLES}
    if any(count < 500 for count in usable.values()):
        raise ValueError(f"locked GSS variables need at least 500 usable responses: {usable}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, na_rep="")
    return {
        "rows": len(output),
        "usable_responses": usable,
        "archive_sha256": ARCHIVE_SHA256 if archive_path is not None else None,
        "stata_sha256": stata_hash,
        "output": str(output_path),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stata",
        type=Path,
        default=Path("data/raw/gss/2018_stata/GSS2018.dta"),
    )
    ap.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/gss/2018_stata.zip"),
    )
    ap.add_argument("--output", type=Path, default=Path("data/raw/gss/gss.csv"))
    return ap


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(prepare(args.stata, args.output, args.archive), indent=2))


if __name__ == "__main__":
    main()

"""The freeze manifest (experiment freeze, Nov 22): sha256 of every input the final run reads.

    python scripts/freeze_manifest.py check-heldout --bank DIR --against ZIP
    python scripts/freeze_manifest.py prep --out M --heldout-bank DIR --pilot P \\
        --training-files LIST [--expected-bank-zip ZIP]
    python scripts/freeze_manifest.py add-final --manifest M --files F [F ...]
    python scripts/freeze_manifest.py verify --manifest M

The held-out bank (scripts/build_runs_v1.py --heldout-asurre-only, #55) is compared by CONTENT:
each file's sha256 after normalising line endings (CRLF -> LF), because a bank built on Windows
and one built on macOS differ only there. Its manifest.json is left out of the content check: it
records the builder's own output paths. The raw sha256 of every file is recorded too. Reads file
bytes only; never prints file contents. No model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.harness.provenance import git_state

BANK = "assignment-asurre-osmi-heldout"
BANK_FILES = (  # content-checked files of the held-out bank, relative to its folder
    f"assessments/{BANK}.json",
    f"variants/{BANK}.jsonl",
    "items.jsonl",
    "assessments.jsonl",
    "human_distributions.jsonl",
)
CONFIGS = ("configs/data_decisions.json", "configs/models.yaml")
# Per-job spend caps (ADR_JOB_SPEND_CAP_USD), approved Nov 21; scripts/freeze_run.sh must match.
SPEND_CAPS_USD = {"claude-haiku": 12.0, "gpt-mini": 10.0}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_sha(data: bytes) -> str:
    """sha256 after CRLF -> LF: the same bank built on Windows or macOS gets the same value."""
    return sha256(data.replace(b"\r\n", b"\n"))


def file_entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes())}


def bank_entry(bank_dir: Path) -> dict:
    files = {}
    for rel in BANK_FILES:
        data = (bank_dir / rel).read_bytes()
        files[rel] = {"sha256": sha256(data), "content_sha256": content_sha(data)}
    combined = sha256("".join(files[r]["content_sha256"] for r in BANK_FILES).encode())
    return {"dir": str(bank_dir.resolve()), "files": files, "content_sha256": combined}


def zip_bank_entry(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        files = {}
        for rel in BANK_FILES:
            data = z.read(rel)
            files[rel] = {"sha256": sha256(data), "content_sha256": content_sha(data)}
    combined = sha256("".join(files[r]["content_sha256"] for r in BANK_FILES).encode())
    return {"zip": str(zip_path.resolve()), "zip_sha256": sha256(zip_path.read_bytes()),
            "files": files, "content_sha256": combined}


def compare_bank(built: dict, expected: dict) -> list[str]:
    """Bank files whose content differs (empty = the same bank)."""
    return [r for r in BANK_FILES
            if built["files"][r]["content_sha256"] != expected["files"][r]["content_sha256"]]


def cmd_check_heldout(a) -> int:
    built, expected = bank_entry(a.bank), zip_bank_entry(a.against)
    for rel in BANK_FILES:
        b, e = built["files"][rel], expected["files"][rel]
        raw = "same bytes" if b["sha256"] == e["sha256"] else "bytes differ"
        verdict = "MATCH" if b["content_sha256"] == e["content_sha256"] else "DIFFER"
        print(f"{verdict:6s} {rel}  (content {b['content_sha256'][:12]}…; {raw})")
    bad = compare_bank(built, expected)
    print(f"held-out bank content sha256: built {built['content_sha256'][:16]}… "
          f"expected {expected['content_sha256'][:16]}… -> {'MATCH' if not bad else 'DIFFER'}")
    return 1 if bad else 0


def cmd_prep(a) -> int:
    training = [Path(p) for p in a.training_files.read_text(encoding="utf-8").splitlines() if p]
    manifest = {
        "freeze": "freeze-2026-11-22",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "heldout_bank": bank_entry(a.heldout_bank),
        "pilot": file_entry(a.pilot),
        "low_effort_ids": file_entry(a.low_effort),  # sha only: the ids stay in that file
        "configs": {c: file_entry(Path(c)) for c in CONFIGS},
        "spend_caps_usd": dict(SPEND_CAPS_USD),
        "training_outcomes": [file_entry(p) for p in training],
        "final_outcomes": [],
    }
    if a.expected_bank_zip:
        expected = zip_bank_entry(a.expected_bank_zip)
        bad = compare_bank(manifest["heldout_bank"], expected)
        manifest["heldout_bank"]["expected"] = expected
        manifest["heldout_bank"]["matches_expected"] = not bad
        if bad:
            print(f"held-out bank differs from {a.expected_bank_zip}: {bad}", file=sys.stderr)
            return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: held-out bank, pilot, {len(CONFIGS)} configs, "
          f"{len(training)} training outcome files")
    return 0


def cmd_add_final(a) -> int:
    m = json.loads(a.manifest.read_text(encoding="utf-8"))
    known = {e["path"] for e in m["final_outcomes"]}
    for f in a.files:
        e = file_entry(Path(f))
        if e["path"] not in known:
            m["final_outcomes"].append(e)
    a.manifest.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"{a.manifest}: {len(m['final_outcomes'])} final outcome files recorded")
    return 0


def cmd_verify(a) -> int:
    m = json.loads(a.manifest.read_text(encoding="utf-8"))
    bad = []
    entries = [m["pilot"], m["low_effort_ids"], *m["configs"].values(), *m["training_outcomes"],
               *m["final_outcomes"]]
    for e in entries:
        p = Path(e["path"])
        if not p.exists() or sha256(p.read_bytes()) != e["sha256"]:
            bad.append(e["path"])
    bank_dir = a.heldout_bank or Path(m["heldout_bank"]["dir"])  # e.g. the bank rebuilt for the run
    if bank_entry(bank_dir)["content_sha256"] != m["heldout_bank"]["content_sha256"]:
        bad.append(f"held-out bank {bank_dir}")
    print(f"verified {len(entries)} files and the held-out bank: "
          + ("all unchanged" if not bad else f"{len(bad)} CHANGED"))
    for b in bad:
        print(f"  changed or missing: {b}", file=sys.stderr)
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-heldout")
    c.add_argument("--bank", type=Path, required=True)
    c.add_argument("--against", type=Path, required=True, help="Archana's handoff zip")
    p = sub.add_parser("prep")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--heldout-bank", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--low-effort", type=Path, required=True, help="pilot_data/low_effort_ids.txt")
    p.add_argument("--training-files", type=Path, required=True, help="one path per line")
    p.add_argument("--expected-bank-zip", type=Path)
    f = sub.add_parser("add-final")
    f.add_argument("--manifest", type=Path, required=True)
    f.add_argument("--files", nargs="+", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--heldout-bank", type=Path, help="check this bank folder's content instead")
    a = ap.parse_args()
    sys.exit({"check-heldout": cmd_check_heldout, "prep": cmd_prep, "add-final": cmd_add_final,
              "verify": cmd_verify}[a.cmd](a))


if __name__ == "__main__":
    main()

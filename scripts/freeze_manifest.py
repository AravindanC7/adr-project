"""The freeze manifest (experiment freeze, Nov 22): sha256 of every input the final run reads.

    python scripts/freeze_manifest.py check-heldout --bank DIR --against ZIP
    python scripts/freeze_manifest.py prep --out M --heldout-bank DIR --pilot P \\
        --training-files LIST [--expected-bank-zip ZIP]
    python scripts/freeze_manifest.py add-final --manifest M --files F [F ...]
    python scripts/freeze_manifest.py verify --manifest M
    python scripts/freeze_manifest.py budget [--config configs/models.yaml]

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

import yaml

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
DECISIONS = Path("configs/data_decisions.json")
# Per-job spend caps (ADR_JOB_SPEND_CAP_USD), approved Nov 21; gpt-mini lowered from 10 to 2 on
# Nov 25 (estimate $0.55), since the weekly check needs remaining >= the sum of the caps and week 48
# had $19.62 left after the third halt. scripts/freeze_run.sh must match.
# claude-haiku lowered from 12 to 8 on Nov 26 (week 48 had $13.75 left after the fourth halt).
SPEND_CAPS_USD = {"claude-haiku": 8.0, "gpt-mini": 2.0}
# Reply limits of the final attacker runs (decided Nov 21; claude-haiku copy-paste raised to 4096
# on Nov 23 after the halted first launch); scripts/freeze_run.sh must match.
HAIKU_CP_MAX_TOKENS = 4096
# claude-haiku document upload, all banks (decided Nov 24 after the second halt and the #59 review).
HAIKU_DU_MAX_TOKENS = 8192
MAX_TOKENS = {
    "claude-haiku": {
        "copy_paste": HAIKU_CP_MAX_TOKENS,
        "doc_upload": {
            "quiz_and_survey": HAIKU_DU_MAX_TOKENS,
            "written_answers": HAIKU_DU_MAX_TOKENS,
            "rule": "quiz and survey banks: the larger of 8192 and twice the largest finished "
                    "reply in the halted run (2994 tokens), rounded up to a power of two, capped "
                    "at the model's 128K output limit; written answers: the same 8192 (raised "
                    "from 6000 in review of #59), since the measured quiz maximum was 2994, "
                    "gpt-mini's largest written-answer reply was 564, and one limit covers all "
                    "claude-haiku document upload",
        },
    },
    "gpt-mini": "unchanged: the training runs' limits",
    "reason": "claude-haiku is billed about 190 output tokens per call that are not in the reply "
              "text (smoke call, Nov 21: 271 output tokens for a 46-word reply), which would "
              "truncate replies at the 300-token copy-paste limit; 1024 for claude-haiku only. "
              "Nov 23: the first launch (tag freeze-2026-11-22) halted with 41 of 8,000 "
              "claude-haiku copy-paste replies cut off at 1024, so 4096. "
              "Nov 24: the second launch (tag freeze-2026-11-23) halted with 16 of 400 "
              "claude-haiku document-upload replies cut off at 3000, so 8192 for quiz and survey "
              "banks and, from the #59 review, 8192 for written-answer banks too.",
    "truncation": "fatal: any truncated reply stops the driver (check_run)",
}
# Files the final run reads besides the outcome files (paths relative to the checkout root).
STYLE_PROFILE = "data/processed/runs_v1/style_profile.json"
STYLE_PROFILE_SHA12 = "6b43d6ed5372"  # configs/data_decisions.json style_profile.profile_sha256
PILOT_BANK = "server/pilot_items.json"  # gitignored: its sha here is the only record
TRAINING_BANK_FILES = (
    "data/processed/runs_v1/items.jsonl",
    *(f"data/processed/runs_v1/{kind}/{bank}.{ext}"
      for bank in ("quiz-mmlu-200", "quiz-arc-200")
      for kind, ext in (("assessments", "json"), ("variants", "jsonl"))),
    "data/processed/archana-survey-written-v1/items.jsonl",
    *(f"data/processed/archana-survey-written-v1/{kind}/{bank}.{ext}"
      for bank in ("survey-ipip", "survey-gss", "assignment-asurre-train")
      for kind, ext in (("assessments", "json"), ("variants", "jsonl"))),
)


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


def profile_sha(path: Path) -> str:
    """The style profile's object sha256 (as configs/data_decisions.json records it): the
    canonical JSON of its "profile" object, so formatting and line endings do not matter."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(doc.get("profile", doc), sort_keys=True, separators=(",", ":")).encode()
    return sha256(blob)


def style_entry(path: Path) -> dict:
    locked = json.loads(DECISIONS.read_text(encoding="utf-8"))["style_profile"]["profile_sha256"]
    actual = profile_sha(path)
    if actual != locked or not actual.startswith(STYLE_PROFILE_SHA12):
        raise SystemExit(f"{path}: style profile sha256 {actual[:12]} is not the locked "
                         f"{STYLE_PROFILE_SHA12} (configs/data_decisions.json {locked[:12]})")
    return {**file_entry(path), "profile_sha256": actual}


def head_commit() -> str | None:
    return git_state().get("git_commit")


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
        "freeze": "freeze-2026-11-26",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "heldout_bank": bank_entry(a.heldout_bank),
        "pilot": file_entry(a.pilot),
        "low_effort_ids": file_entry(a.low_effort),  # sha only: the ids stay in that file
        "configs": {c: file_entry(Path(c)) for c in CONFIGS},
        "style_profile": style_entry(a.root / STYLE_PROFILE),
        "pilot_bank": file_entry(a.root / PILOT_BANK),
        "training_banks": [file_entry(a.root / f) for f in TRAINING_BANK_FILES],
        "spend_caps_usd": dict(SPEND_CAPS_USD),
        "max_tokens": MAX_TOKENS,
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
    entries = [m["pilot"], m["low_effort_ids"], *m["configs"].values(), m["style_profile"],
               m["pilot_bank"], *m["training_banks"], *m["training_outcomes"],
               *m["final_outcomes"]]
    for e in entries:
        p = Path(e["path"])
        if not p.exists() or sha256(p.read_bytes()) != e["sha256"]:
            bad.append(e["path"])
    head = head_commit()
    if head != m["git"]["git_commit"]:
        bad.append(f"git commit: manifest {m['git']['git_commit']}, HEAD {head}")
    bank_dir = a.heldout_bank or Path(m["heldout_bank"]["dir"])  # e.g. the bank rebuilt for the run
    if bank_entry(bank_dir)["content_sha256"] != m["heldout_bank"]["content_sha256"]:
        bad.append(f"held-out bank {bank_dir}")
    print(f"verified {len(entries)} files and the held-out bank: "
          + ("all unchanged" if not bad else f"{len(bad)} CHANGED"))
    for b in bad:
        print(f"  changed or missing: {b}", file=sys.stderr)
    return 1 if bad else 0


def budget_check(client, need: float, now: str | None = None) -> int:
    """Refuse (1) if this ISO week's remaining weekly budget is below `need`, the sum of the two
    jobs' caps. The client counts the week in UTC from its spend ledger; no API calls."""
    stamp = now or datetime.now(timezone.utc).isoformat()
    year, week, _ = datetime.fromisoformat(stamp).isocalendar()
    spent, cap = client.week_spend_usd(stamp), client.weekly_spend_cap_usd
    if cap is None:
        print(f"ISO week {year}-W{week:02d} (UTC): ledger spend ${spent:.2f}; no weekly cap")
        return 0
    left = cap - spent
    print(f"ISO week {year}-W{week:02d} (UTC): ledger spend ${spent:.2f} of the ${cap:.2f} weekly "
          f"cap, ${left:.2f} left; the two freeze jobs may spend up to ${need:.2f}")
    if left < need:
        print(f"NOT STARTING: ${left:.2f} left this week is below the ${need:.2f} the two jobs' "
              "caps allow. Start after 17:00 Arizona time on Sunday (00:00 UTC Monday), when a "
              "new ISO week begins with the full weekly cap; configs/models.yaml stays as it is.",
              file=sys.stderr)
        return 1
    return 0


def check_run(run_dir: Path, max_refused_frac: float) -> tuple[bool, str]:
    """The driver's post-run check of one attacker run: finished, no call errors or skipped
    questions (errors.jsonl), no reply cut off at max_tokens, no spend-cap stop, and refused
    questions (refused.jsonl: a document-upload reply that ended normally but answered none of its
    questions) at most `max_refused_frac` of the run's rows (decided Nov 26)."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    errs = (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    n_rows, refused = cfg.get("n_rows") or 0, cfg.get("n_refused_questions") or 0
    bad = {"errors": len(errs), "truncated": cfg.get("n_truncated_replies") or 0,
           "stopped_for_budget": bool(cfg.get("stopped_for_budget")),
           "refused_over_limit": refused > max_refused_frac * n_rows}
    if any(bad.values()) or not cfg.get("finished_at"):
        return False, (f"{run_dir.name}: {bad}, refused={refused} of {n_rows} rows, "
                       f"finished_at={cfg.get('finished_at')}")
    return True, f"{run_dir.name}: ok ({n_rows} rows, {refused} refused)"


def cmd_check_run(a) -> int:
    ok, msg = check_run(a.run_dir, a.max_refused_frac)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


def cmd_budget(a) -> int:
    from adr.harness.llm_client import LLMClient

    return budget_check(LLMClient.from_config(a.config), sum(SPEND_CAPS_USD.values()))


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
    p.add_argument("--root", type=Path, default=Path("."),
                   help="checkout holding data/ and server/pilot_items.json (default: here)")
    f = sub.add_parser("add-final")
    f.add_argument("--manifest", type=Path, required=True)
    f.add_argument("--files", nargs="+", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--heldout-bank", type=Path, help="check this bank folder's content instead")
    r = sub.add_parser("check-run", help="the driver's post-run check of one attacker run")
    r.add_argument("--run-dir", type=Path, required=True, help="results/runs/<run_id>")
    r.add_argument("--max-refused-frac", type=float, required=True)
    b = sub.add_parser("budget", help="this week's ledger spend vs the two jobs' caps")
    b.add_argument("--config", default="configs/models.yaml")
    a = ap.parse_args()
    sys.exit({"check-heldout": cmd_check_heldout, "prep": cmd_prep, "add-final": cmd_add_final,
              "verify": cmd_verify, "budget": cmd_budget, "check-run": cmd_check_run}[a.cmd](a))


if __name__ == "__main__":
    main()

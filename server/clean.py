"""Pilot data -> outcome table (step #39). Owner: Harshini. Feeds Archana's failure map (#40).

    python -m server.clean --run-id pilot-2026-11-13
    python -m server.clean --data dry_run_data --origin dry_run --run-id dry-run-2026-11-09

Reads the session files the server saved (server/store.py) and writes one OutcomeRow per
(respondent, item, check) to results/outcomes/<run_id>.parquet, plus
results/runs/<run_id>/config.yaml with what was kept and dropped and why.

Which sessions are used
  * finished (a "finish" line) and consented. Withdrawn sessions are already gone (Stop and
    "Remove my answers" delete the file). Partial sessions are dropped, never half-used.
  * --exclude CODE drops a session someone asked about later, e.g. "I used AI anyway".

Rows per answered item (all respondent_kind="human", data_origin=pilot|dry_run)
  * the check the server served with it: decoy / canary, or "none" for a plain question;
  * the behaviour check (paste / focus_loss / too_fast / normal), scored live by the server;
  * the opinion trap for survey items with a human distribution in the bank
    (implausible / plausible), scored here because the server doesn't serve it yet;
  * for reverse-keyed IPIP pairs, the SECOND item of a pair is "contradiction" when both
    answers lean the same way (both agree, or both disagree). PROVISIONAL rule: the mechanism
    itself doesn't score pairs yet (adr/mechanisms/opinion_trap.py), so Archana should confirm
    it before #40 uses it. --no-pairs turns it off.

Nothing here contains answer text: rows hold outcomes. raw_ref points back to the session file
and line. Open-text answers are still personal data, so the script lists any that look like
they contain an email, phone number, URL or a name ("my name is ...") for a person to check
before the session files leave the laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adr.contracts import (
    HumanDistribution,
    Item,
    OutcomeRow,
    Response,
    parse_variant_id,
)
from adr.contracts.outcome import write_outcomes
from adr.mechanisms.opinion_trap import OpinionTrapMechanism
from server.store import read_session

ROOT = Path(__file__).resolve().parent.parent
PERSONAL = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?:\+?\d[\s().-]?){9,}"),
    "url": re.compile(r"https?://|www\.", re.IGNORECASE),
    "name": re.compile(r"\bmy name is\b|\bi am called\b|\bi'm called\b", re.IGNORECASE),
    "asu_id": re.compile(r"\b\d{10}\b"),
}


@dataclass
class Report:
    files: int = 0
    used: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # session file -> reason
    rows: Counter = field(default_factory=Counter)  # (mechanism, outcome) -> count
    flagged: list[dict[str, str]] = field(default_factory=list)
    consent_versions: Counter = field(default_factory=Counter)


def load_bank(path: Path) -> tuple[dict[str, Item], dict[str, HumanDistribution], dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = {d["item_id"]: Item(**d) for d in raw["items"]}
    dists = {d["key"]: HumanDistribution(**d) for d in raw.get("human_distributions", [])}
    assessment_type = {a["assessment_id"]: a["assessment_type"] for a in raw["assessments"]}
    return items, dists, assessment_type


def leaning(answer: int, n_options: int) -> int:
    """-1 disagree side, 0 middle, +1 agree side, for an option index on an agree scale."""
    middle = (n_options - 1) / 2
    return 0 if answer == middle else (1 if answer > middle else -1)


def outcome_row(run_id: str, origin: str, sid: str, item: Item, variant_id: str, outcome: str,
                where: str) -> OutcomeRow:
    return OutcomeRow(
        run_id=run_id, session_id=sid, respondent_kind="human", data_origin=origin,
        assessment_type=item.assessment_type, item_type=item.item_type, item_id=item.item_id,
        split=item.split, mechanism=parse_variant_id(variant_id)[1], variant_id=variant_id,
        outcome=outcome, raw_ref=where,
    )


def clean(data_dir: Path, bank_path: Path, *, origin: str, run_id: str, pairs: bool = True,
          exclude: frozenset[str] = frozenset()) -> tuple[list[OutcomeRow], Report]:
    items, dists, _ = load_bank(bank_path)
    trap = OpinionTrapMechanism(dists)
    report = Report()
    rows: list[OutcomeRow] = []

    for path in sorted(data_dir.glob("*.jsonl")):
        report.files += 1
        records = read_session(path)
        events = [r["event"] for r in records]
        start = records[0] if records and records[0]["event"] == "start" else None
        sid = path.stem
        if sid in exclude:
            report.dropped[path.name] = "excluded on request"
            continue
        if start is None:
            report.dropped[path.name] = "no start record"
            continue
        if "finish" not in events:
            report.dropped[path.name] = "not finished"
            continue
        if not start.get("consented"):
            report.dropped[path.name] = "no consent"
            continue
        report.used.append(path.name)
        report.consent_versions[start.get("consent_version")] += 1

        answers: dict[str, Any] = {}
        for line_no, rec in enumerate(records, start=1):
            if rec["event"] != "answer":
                continue
            item = items.get(rec["item_id"])
            if item is None:
                raise ValueError(f"{path.name}:{line_no}: {rec['item_id']} isn't in {bank_path}")
            answers[item.item_id] = rec["answer"]
            where = f"{path.name}#L{line_no}"
            found = [(v, o) for v, o in rec["outcomes"]]

            if trap.applies_to(item):
                rendered = trap.generate(item, 0)
                response = Response(session_id=sid, item_id=item.item_id,
                                    variant_id=rendered.variant_id, answer=rec["answer"])
                outcome = trap.score(rendered, response, None)
                partner = item.reverse_keyed_with
                if pairs and partner and partner in answers and outcome != "implausible":
                    a, b = rec["answer"], answers[partner]
                    n = len(item.options or [])
                    if type(a) is int and type(b) is int and leaning(a, n) == leaning(b, n) != 0:
                        outcome = "contradiction"
                found.append((rendered.variant_id, outcome))

            for variant_id, outcome in found:
                row = outcome_row(run_id, origin, sid, item, variant_id, outcome, where)
                rows.append(row)
                report.rows[(row.mechanism or "none", outcome)] += 1

            if item.item_type == "open_text" and isinstance(rec["answer"], str):
                hits = [k for k, rx in PERSONAL.items() if rx.search(rec["answer"])]
                if hits:
                    report.flagged.append({"session": sid, "item": item.item_id,
                                           "looks_like": ", ".join(hits),
                                                                  "where": where})
    return rows, report


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pilot sessions -> outcome table (#39)")
    p.add_argument("--data", type=Path, default=ROOT / "pilot_data")
    p.add_argument("--bank", type=Path, default=ROOT / "server" / "pilot_items.json")
    p.add_argument("--origin", choices=["pilot", "dry_run"], default="pilot")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--exclude", action="append", default=[], metavar="CODE",
                   help="session code to drop (repeatable)")
    p.add_argument("--no-pairs", action="store_true", help="don't score IPIP pair contradictions")
    args = p.parse_args(argv)

    rows, rep = clean(args.data, args.bank, origin=args.origin, run_id=args.run_id,
                      pairs=not args.no_pairs, exclude=frozenset(args.exclude))
    out = write_outcomes(rows, args.out / "outcomes" / f"{args.run_id}.parquet")
    run_dir = args.out / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "step": "#39 pilot cleaning", "run_id": args.run_id, "data_origin": args.origin,
        "bank": str(args.bank.name), "bank_sha256_12": file_hash(args.bank),
        "pair_contradictions": not args.no_pairs,
        "session_files": rep.files, "sessions_used": len(rep.used),
        "dropped": rep.dropped,
        "consent_versions": dict(rep.consent_versions),
        "rows": len(rows),
        "rows_by_check": {f"{m}:{o}": n for (m, o), n in sorted(rep.rows.items())},
        "open_text_to_check": len(rep.flagged),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"{len(rep.used)} of {rep.files} sessions used -> {len(rows)} rows in {out}")
    for name, why in rep.dropped.items():
        print(f"  dropped {name}: {why}")
    for (m, o), n in sorted(rep.rows.items()):
        print(f"  {m:13} {o:22} {n}")
    if rep.flagged:
        print(f"\nCHECK {len(rep.flagged)} open-text answer(s) by hand before sharing raw files:")
        for f in rep.flagged:
            print(f"  {f['where']}  ({f['looks_like']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

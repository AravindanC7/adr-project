"""Pilot data -> outcome table (step #39). Owner: Harshini. Feeds Archana's failure map (#40).

    python -m server.clean --run-id pilot-2026-11-13
    python -m server.clean --data dry_run_data --origin dry_run --run-id dry-run-2026-11-09
    python -m server.clean --run-id pilot-2026-11-13 --rescore-behaviour

Reads the session files the server saved (server/store.py) and writes one OutcomeRow per
(respondent, item, check) to results/outcomes/<run_id>.parquet, plus
results/runs/<run_id>/config.yaml with what was kept and dropped and why.

Which sessions are used
  * finished (a "finish" line) and consented. Withdrawn sessions are already gone (Stop and
    "Remove my answers" delete the file). Partial sessions are dropped, never half-used.
  * --exclude CODE drops a session someone asked about later, e.g. "I used AI anyway".

Rows per answered item (all respondent_kind="human", data_origin=pilot|dry_run)
  * the check the server served with it: decoy / canary, or "none" for a plain question;
  * the behaviour check (paste / focus_loss / too_fast / normal), scored live by the server.
    --rescore-behaviour (off by default, so earlier runs reproduce) re-scores it from the saved
    telemetry with the current adr/mechanisms/behaviour.py instead, exactly as the server does
    (the served variant's stem plus the item's passage set the reading floor). config.yaml then
    records the floor constants and every live -> re-scored transition. An answer that can't be
    re-scored (no or invalid telemetry, a variant the bank no longer builds) keeps its live
    outcome and is counted as not_rescorable; --strict stops the run, writing nothing, if any is.
  * the opinion trap for survey items with a human distribution in the bank
    (implausible / plausible), scored here because the server doesn't serve it yet;
  * for reverse-keyed IPIP pairs, the SECOND item of a pair is "contradiction" when both
    answers lean the same way (both agree, or both disagree); if either answer is neutral,
    missing, non-integer or out of range the second item gets no opinion-trap row at all
    (no observation); "implausible" wins over "contradiction". The rule is the shared
    pair_outcome in adr/mechanisms/opinion_trap.py, the same one the AI runs use, so human and AI
    pairs score the same way. Rule confirmed by Archana (Nov 18). --no-pairs turns it off.
  * the writing-style check (high / mid / low) for every open-text answer, scored here against
    the ASURRE training profile (scripts/prepare_style_profile.py). Style is passive: it observes
    every open-text answer whatever check was served, so it doesn't use the bank's
    eligible_mechanisms. An empty or skipped answer is no observation: no row, only a count.
    --no-style turns it off; --style-profile picks another profile file.
    config.yaml records the profile's canonical profile_sha256 (the "profile" object only, so the
    same on every machine) and warns if it isn't the one locked in configs/data_decisions.json.

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
    NoObservation,
    OutcomeRow,
    RenderedItem,
    Response,
    Telemetry,
    make_variant_id,
    parse_variant_id,
)
from adr.contracts.outcome import write_outcomes
from adr.mechanisms import behaviour as behaviour_module
from adr.mechanisms.opinion_trap import OpinionTrapMechanism, pair_outcome
from adr.mechanisms.style import StyleMechanism, StyleProfile, load_style_profile
from scripts.analyse_language_fairness import _profile_object_sha256 as profile_sha256
from server.store import read_session

ROOT = Path(__file__).resolve().parent.parent
STYLE_PROFILE = ROOT / "data" / "processed" / "runs_v1" / "style_profile.json"
DECISIONS = ROOT / "configs" / "data_decisions.json"
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
    style_no_observation: int = 0  # open-text answers style couldn't score (empty / skipped)
    # --rescore-behaviour: (live, re-scored) -> count, and why answers kept their live outcome
    behaviour_transitions: Counter = field(default_factory=Counter)
    behaviour_not_rescorable: Counter = field(default_factory=Counter)


def load_bank(path: Path) -> tuple[dict[str, Item], dict[str, HumanDistribution], dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = {d["item_id"]: Item(**d) for d in raw["items"]}
    dists = {d["key"]: HumanDistribution(**d) for d in raw.get("human_distributions", [])}
    assessment_type = {a["assessment_id"]: a["assessment_type"] for a in raw["assessments"]}
    return items, dists, assessment_type


def outcome_row(
    run_id: str, origin: str, sid: str, item: Item, variant_id: str, outcome: str, where: str
) -> OutcomeRow:
    return OutcomeRow(
        run_id=run_id,
        session_id=sid,
        respondent_kind="human",
        data_origin=origin,
        assessment_type=item.assessment_type,
        item_type=item.item_type,
        item_id=item.item_id,
        split=item.split,
        mechanism=parse_variant_id(variant_id)[1],
        variant_id=variant_id,
        outcome=outcome,
        raw_ref=where,
    )


def plain_rendered(item: Item) -> RenderedItem:
    """The item as a plain variant. Style is passive and only reads the response."""
    return RenderedItem(
        item_id=item.item_id,
        mechanism=None,
        variant_id=make_variant_id(item.item_id, None),
        visible={"stem": item.stem},
    )


def served_variants(items: dict[str, Item]) -> dict[str, RenderedItem]:
    """variant_id -> the variant the server served, built the way the server builds them."""
    from server.app import build_variants, load_injected_mechanisms  # only when re-scoring

    built = build_variants(items, load_injected_mechanisms(False))
    return {v.variant_id: v for vs in built.values() for v in vs}


def rescore_behaviour(
    rec: dict[str, Any],
    sid: str,
    item: Item,
    by_variant: dict[str, RenderedItem],
    found: list[tuple[str, str]],
    report: Report,
) -> list[tuple[str, str]]:
    """`found` with the behaviour outcome re-scored from the record's telemetry.

    Scores like server/app.py: the current behaviour mechanism, the served variant, the
    item's passage. Anything missing keeps the live outcome and is counted, by reason only.
    """
    behaviour_id = make_variant_id(item.item_id, "behaviour")
    if not any(v == behaviour_id for v, _ in found):
        return found  # the server recorded no behaviour outcome: nothing to re-score
    rendered = by_variant.get(rec.get("variant_id"))
    raw = rec.get("telemetry")
    new: str | None = None
    if rendered is None:
        report.behaviour_not_rescorable["unknown variant"] += 1
    elif not isinstance(raw, dict):
        report.behaviour_not_rescorable["no telemetry"] += 1
    else:
        try:
            telemetry = Telemetry(**raw)
        except ValueError:
            report.behaviour_not_rescorable["invalid telemetry"] += 1
        else:
            response = Response(
                session_id=sid,
                item_id=item.item_id,
                variant_id=rendered.variant_id,
                answer=rec["answer"],
            )
            try:
                new = behaviour_module.MECHANISM.score(
                    rendered, response, telemetry, passage=item.passage
                )
            except NoObservation:
                report.behaviour_not_rescorable["no observation"] += 1
    out = []
    for v, o in found:
        if v == behaviour_id and new is not None:
            report.behaviour_transitions[(o, new)] += 1
            o = new
        out.append((v, o))
    return out


def behaviour_record(rescored: bool, report: Report) -> dict[str, Any]:
    """How config.yaml says the behaviour rows were scored."""
    if not rescored:
        return {"rescored": False, "scored_by": "server, live at answer time"}
    moves = report.behaviour_transitions
    return {
        "rescored": True,
        "answer_ms": behaviour_module.ANSWER_MS,
        "read_ms_per_word": behaviour_module.READ_MS_PER_WORD,
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(moves.items())},
        "changed": sum(n for (a, b), n in moves.items() if a != b),
        "not_rescorable": sum(report.behaviour_not_rescorable.values()),
        "not_rescorable_reasons": dict(sorted(report.behaviour_not_rescorable.items())),
    }


def clean(
    data_dir: Path,
    bank_path: Path,
    *,
    origin: str,
    run_id: str,
    pairs: bool = True,
    exclude: frozenset[str] = frozenset(),
    style_profile: StyleProfile | None = None,
    rescore: bool = False,
) -> tuple[list[OutcomeRow], Report]:
    """style_profile=None scores no style rows (the command line loads one by default).
    rescore=True re-scores the behaviour check from telemetry (see the module docstring)."""
    items, dists, _ = load_bank(bank_path)
    by_variant = served_variants(items) if rescore else {}
    trap = OpinionTrapMechanism(dists)
    style = StyleMechanism(style_profile) if style_profile is not None else None
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
            if rescore:
                found = rescore_behaviour(rec, sid, item, by_variant, found, report)

            if trap.applies_to(item):
                rendered = trap.generate(item, 0)
                response = Response(
                    session_id=sid,
                    item_id=item.item_id,
                    variant_id=rendered.variant_id,
                    answer=rec["answer"],
                )
                outcome = trap.score(rendered, response, None)
                partner = item.reverse_keyed_with
                if pairs and partner and partner in answers:
                    n = len(item.options or [])
                    # None = neutral or invalid answer in the pair: no observation, no row
                    outcome = pair_outcome(rec["answer"], answers[partner], n, outcome)
                if outcome is not None:
                    found.append((rendered.variant_id, outcome))

            if style is not None and item.item_type == "open_text":
                style_id = make_variant_id(item.item_id, "style")
                response = Response(
                    session_id=sid,
                    item_id=item.item_id,
                    variant_id=style_id,
                    answer=rec["answer"] if isinstance(rec["answer"], str) else None,
                )
                try:
                    found.append((style_id, style.score(plain_rendered(item), response, None)))
                except NoObservation:
                    report.style_no_observation += 1

            for variant_id, outcome in found:
                row = outcome_row(run_id, origin, sid, item, variant_id, outcome, where)
                rows.append(row)
                report.rows[(row.mechanism or "none", outcome)] += 1

            if item.item_type == "open_text" and isinstance(rec["answer"], str):
                hits = [k for k, rx in PERSONAL.items() if rx.search(rec["answer"])]
                if hits:
                    report.flagged.append(
                        {
                            "session": sid,
                            "item": item.item_id,
                            "looks_like": ", ".join(hits),
                            "where": where,
                        }
                    )
    return rows, report


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def style_record(profile_path: Path, decisions_path: Path) -> dict[str, Any]:
    """How config.yaml identifies the style profile.

    profile_sha256 is the canonical hash locked in configs/data_decisions.json: sha256 of the
    file's "profile" object as compact JSON with sorted keys, so it ignores the OS paths and
    decisions_sha256 the file also records (the file's own hash differs per machine). A profile
    other than the locked one is allowed (experiments) but warned about.
    """
    canonical = profile_sha256(profile_path)
    locked = None
    if decisions_path.is_file():
        choice = json.loads(decisions_path.read_text(encoding="utf-8")).get("style_profile")
        if isinstance(choice, dict) and choice.get("profile_sha256"):
            locked = str(choice["profile_sha256"])
    matches = None if locked is None else canonical == locked
    if locked is None:
        print(
            f"note: no style_profile.profile_sha256 in {decisions_path}, nothing to check against"
        )
    elif not matches:
        print(
            f"WARNING: style profile {profile_path} has profile sha256 {canonical[:12]}, not the "
            f"one locked in {decisions_path.name} ({locked[:12]}); recorded as "
            "profile_matches_decision: false"
        )
    return {
        "profile": profile_path.name,
        "profile_sha256": canonical,
        "profile_sha256_12": canonical[:12],
        "profile_matches_decision": matches,
        "profile_file_sha256_12": file_hash(profile_path),  # provenance only; differs per machine
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pilot sessions -> outcome table (#39)")
    p.add_argument("--data", type=Path, default=ROOT / "pilot_data")
    p.add_argument("--bank", type=Path, default=ROOT / "server" / "pilot_items.json")
    p.add_argument("--origin", choices=["pilot", "dry_run"], default="pilot")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="CODE",
        help="session code to drop (repeatable)",
    )
    p.add_argument("--no-pairs", action="store_true", help="don't score IPIP pair contradictions")
    p.add_argument(
        "--style-profile",
        type=Path,
        default=STYLE_PROFILE,
        help="fitted style profile (built by scripts/prepare_style_profile.py)",
    )
    p.add_argument("--no-style", action="store_true", help="don't score the writing-style check")
    p.add_argument(
        "--rescore-behaviour",
        action="store_true",
        help="re-score the behaviour check from saved telemetry with the current thresholds "
        "(default: keep the outcome the server scored live)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="with --rescore-behaviour: stop, writing nothing, if any answer can't be re-scored",
    )
    p.add_argument(
        "--decisions",
        type=Path,
        default=DECISIONS,
        help="data_decisions.json holding the locked style_profile.profile_sha256",
    )
    args = p.parse_args(argv)
    if not args.no_style and not args.style_profile.is_file():
        p.error(
            f"no style profile at {args.style_profile}: run scripts/prepare_style_profile.py, "
            "pass --style-profile, or use --no-style"
        )
    profile = None if args.no_style else load_style_profile(args.style_profile)

    rows, rep = clean(
        args.data,
        args.bank,
        origin=args.origin,
        run_id=args.run_id,
        pairs=not args.no_pairs,
        exclude=frozenset(args.exclude),
        style_profile=profile,
        rescore=args.rescore_behaviour,
    )
    behaviour = behaviour_record(args.rescore_behaviour, rep)
    if args.rescore_behaviour:
        moves = ", ".join(
            f"{a}->{b} {n}" for (a, b), n in sorted(rep.behaviour_transitions.items()) if a != b
        )
        print(
            f"behaviour re-scored at {behaviour['answer_ms']} ms + "
            f"{behaviour['read_ms_per_word']} ms/word: {moves or 'no changes'}; "
            f"{behaviour['changed']} changed, not_rescorable {behaviour['not_rescorable']}"
        )
        if args.strict and behaviour["not_rescorable"]:
            print(
                f"--strict: {behaviour['not_rescorable']} answer(s) can't be re-scored "
                f"{behaviour['not_rescorable_reasons']}; nothing written"
            )
            return 1
    out = write_outcomes(rows, args.out / "outcomes" / f"{args.run_id}.parquet")
    run_dir = args.out / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "step": "#39 pilot cleaning",
        "run_id": args.run_id,
        "data_origin": args.origin,
        "bank": str(args.bank.name),
        "bank_sha256_12": file_hash(args.bank),
        "pair_contradictions": not args.no_pairs,
        "behaviour": behaviour,
        "style": None
        if args.no_style
        else {
            **style_record(args.style_profile, args.decisions),
            "no_observation": rep.style_no_observation,
        },
        "session_files": rep.files,
        "sessions_used": len(rep.used),
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
    if profile is not None:
        styled = {o: n for (m, o), n in sorted(rep.rows.items()) if m == "style"}
        print(f"  style: {styled}, no observation (empty answer): {rep.style_no_observation}")
    if rep.flagged:
        print(f"\nCHECK {len(rep.flagged)} open-text answer(s) by hand before sharing raw files:")
        for f in rep.flagged:
            print(f"  {f['where']}  ({f['looks_like']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

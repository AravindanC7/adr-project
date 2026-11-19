"""L0 runs on Archana's survey + written-answer banks (archana-survey-written-v1), gpt-mini only.

    python scripts/run_survey_banks.py check        # load + validate the banks, no API calls
    python scripts/run_survey_banks.py estimate     # cost upper bound, no API calls
    python scripts/run_survey_banks.py run --bank survey-gss --mode cp [--fake]
    python scripts/run_survey_banks.py rescore      # re-score the 6 runs from their saved
                                                    # responses: no API calls
    python scripts/report_survey_banks.py           # rates per bank x mode vs the pilot

Data (gitignored, from drive_uploads/ADR-runs-handoff-2026-11-17.zip; check SHA256SUMS.txt):
    data/processed/archana-survey-written-v1/   banks, variants, human distributions, manifest
    data/processed/runs_v1/style_profile.json   training-only style profile (see `rescore`)
    data/raw/asurre/                            ASURRE sources (OSMI is held out: never read)
The manifest's paths are Windows-style (backslashes); they are normalised at load time, so the
data files stay byte-identical to the handoff's checksums.

Settings (the quiz L0 protocol, results/runs/v1-*-L0-quiz-*):

    bank                     items  variants          cp: samples/temp/max_tokens  du: per PDF/temp/max_tokens
    survey-ipip              50     plain, opinion    5 / 0.7 / 300                20 questions / 0 / 3000
    survey-gss               8      plain, opinion    5 / 0.7 / 300                whole bank / 0 / 3000
    assignment-asurre-train  13     plain, canary     5 / 0.7 / 1000               whole bank / 0 / 6000

Document upload makes one PDF per variant kind, so each chunk has a plain-only and a trap-only
PDF (an opinion trap changes nothing visible, so for ipip/gss those two PDFs are identical and
share one cached answer; the same holds for the copy-paste prompts). The open-text bank gets
larger reply limits than the quiz runs (300 / 3000) so long written answers are not cut off;
no reply was. Open-text answers are also scored by the passive style mechanism.

Run ids: v1-cp-L0-<bank>-s5, v1-du-L0-<bank>[-q20]. Each run stops on (exit 2) any failed call,
reply cut off at max_tokens, or spend-cap stop. Guards: gpt-mini only (never the held-out model),
train-split items only, no OSMI item, and a style profile not fitted on OSMI.

`rescore` (no API calls; idempotent): the first time, it keeps the harness's own rows as
results/runs/<run_id>/outcomes.harness.parquet and from then on always starts from that copy.
  * style rows are re-scored with the style profile given (default: the regenerated one);
  * IPIP opinion-trap rows get the shared reverse-keyed pair rule
    (adr.mechanisms.opinion_trap.score_reverse_pairs), as the pilot does: within a session, the
    second item of a pair becomes "contradiction" when both raw answers lean the same way, keeps
    its single-item outcome when they lean opposite ways, and has no row when either is neutral,
    missing or invalid. A session's answer order is the bank order (copy-paste: each item is its
    own call, so there is no real order; document upload: the order in the PDF). Only answers to
    opinion-trap variants pair with each other.
The run's config.yaml gets a `rescore` block (profile path and sha256, rule, row counts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr.contracts import Assessment, Item, NoObservation, RenderedItem, Response
from adr.contracts.io import read_jsonl
from adr.contracts.outcome import OutcomeRow, read_outcomes, write_outcomes
from adr.harness.prompts import build_prompt

DATA = Path("data/processed/archana-survey-written-v1")
STYLE_PROFILE = Path("data/processed/runs_v1/style_profile.json")
MODEL = "gpt-mini"
HELDOUT_SOURCE = "asurre_osmi_2017_18"
BANKS = {  # bank -> (active mechanisms, cp max_tokens, du questions_per_doc, du max_tokens)
    "survey-ipip": (["opinion_trap"], 300, 20, 3000),
    "survey-gss": (["opinion_trap"], 300, None, 3000),
    "assignment-asurre-train": (["canary", "style"], 1000, None, 6000),
}
CP_SAMPLES, CP_TEMP = 5, 0.7
DU_SAMPLES, DU_TEMP = 1, 0.0
PAIR_RULE = ("score_reverse_pairs v1: same raw side = contradiction (implausible wins), "
             "opposite = single-item outcome, neutral/missing/invalid = no row")


def norm(p: str) -> Path:
    """'data\\processed\\x' (Windows manifest) -> Path('data/processed/x')."""
    return Path(PureWindowsPath(p).as_posix()) if "\\" in p else Path(p)


def run_id(bank: str, mode: str) -> str:
    if mode == "cp":
        return f"v1-cp-L0-{bank}-s{CP_SAMPLES}"
    q = BANKS[bank][2]
    return f"v1-du-L0-{bank}" + (f"-q{q}" if q else "")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(bank: str):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if manifest["heldout_asurre_source"] != HELDOUT_SOURCE:
        raise SystemExit(f"manifest holds out {manifest['heldout_asurre_source']}, not OSMI")
    entry = next(a for a in manifest["assessments"] if a["assessment_id"] == bank)
    a_path, v_path = norm(entry["assessment"]), norm(entry["variants"])
    for p in (a_path, v_path):
        if not p.exists():
            raise FileNotFoundError(f"manifest path {p} doesn't resolve")
    assessment = Assessment.model_validate_json(a_path.read_text(encoding="utf-8"))
    items = {it.item_id: it for it in read_jsonl(DATA / "items.jsonl", Item)}
    variants: dict[str, list[RenderedItem]] = {}
    for v in read_jsonl(v_path, RenderedItem):
        variants.setdefault(v.item_id, []).append(v)
    for iid in assessment.item_ids:  # held-out guard: never run OSMI or held-out-split items
        it = items[iid]
        if it.source == HELDOUT_SOURCE or it.split != "train":
            raise SystemExit(f"{iid}: held-out item ({it.source}, {it.split}) in {bank}")
    if len(assessment.item_ids) != entry["n_items"] or sum(
            len(variants[i]) for i in assessment.item_ids) != entry["n_variants"]:
        raise SystemExit(f"{bank}: item/variant counts don't match the manifest")
    return assessment, items, variants


def style_mechanism(profile_path: Path):
    from adr.mechanisms.style import StyleMechanism, load_style_profile

    profile = load_style_profile(profile_path)
    if HELDOUT_SOURCE in profile.sources:
        raise SystemExit("style profile was fitted on the held-out survey")
    return StyleMechanism(profile)


def mechanisms_for(bank: str, profile_path: Path = STYLE_PROFILE):
    from adr.harness.registry import load_mechanisms

    names = BANKS[bank][0]
    mechs = load_mechanisms([n for n in names if n != "style"])
    if "style" in names:
        mechs["style"] = style_mechanism(profile_path)
    return mechs


def client_for(fake: bool):
    from adr.harness.llm_client import FakeProvider, LLMClient, ModelSpec

    if fake:
        def reply(system, messages, attachments=None):
            if attachments:
                return "\n".join(f"Q{n}: 1" for n in range(1, 60))
            text = messages[-1]["content"]
            return "1" if "number" in text else "I think it's fine overall, with some issues."
        return LLMClient(models={MODEL: ModelSpec(MODEL, "fake", "fake-v1", 0.0, 0.0)},
                         cache_dir="cache/llm-fake", providers={MODEL: FakeProvider(reply)})
    client = LLMClient.from_config("configs/models.yaml")
    if client.spec(MODEL).heldout:
        raise SystemExit(f"{MODEL} is marked held out")
    return client


# ---------------------------------------------------------------- check / estimate / run
def cmd_check(_a) -> None:
    for bank in BANKS:
        a, items, v = load(bank)
        mechanisms_for(bank)
        kinds = sorted({x.mechanism or "plain" for i in a.item_ids for x in v[i]})
        print(f"{bank}: {len(a.item_ids)} items, {sum(len(v[i]) for i in a.item_ids)} variants,"
              f" {kinds}, sources {sorted({items[i].source for i in a.item_ids})}")


def cmd_estimate(_a) -> None:
    import yaml

    spec = next(m for m in yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
                ["models"] if m["name"] == MODEL)
    pin, pout = spec["price_in_per_mtok"] / 1e6, spec["price_out_per_mtok"] / 1e6
    total = 0.0
    print(f"{'run_id':42s} {'calls':>6s} {'in tok':>8s} {'out tok':>8s} {'max $':>7s}")
    for bank, (_, cp_max, q, du_max) in BANKS.items():
        a, items, v = load(bank)
        calls = tin = 0
        for iid in a.item_ids:
            for r in v[iid]:
                system, msgs = build_prompt(items[iid], r, "L0")
                chars = len(system or "") + sum(len(m["content"]) for m in msgs)
                tin += CP_SAMPLES * math.ceil(chars / 3)  # generous: ~4 chars/token in English
                calls += CP_SAMPLES
        tout = calls * cp_max  # worst case: every reply hits max_tokens
        usd = tin * pin + tout * pout
        total += usd
        print(f"{run_id(bank, 'cp'):42s} {calls:6d} {tin:8d} {tout:8d} {usd:7.2f}")
        n = len(a.item_ids)
        n_vars = max(len(v[i]) for i in a.item_ids)
        docs = math.ceil(n / (q or n)) * n_vars * DU_SAMPLES
        din = n_vars * n * 300 + docs * 1500  # 300 tok/question (quiz PDFs: ~110) + overhead
        dout = docs * du_max
        usd = din * pin + dout * pout
        total += usd
        print(f"{run_id(bank, 'du'):42s} {docs:6d} {din:8d} {dout:8d} {usd:7.2f}")
    print(f"total upper bound for {MODEL}: ${total:.2f}")


def cmd_run(a) -> None:
    import yaml

    assessment, items, variants = load(a.bank)
    mechs = mechanisms_for(a.bank)
    client = client_for(a.fake)
    _, cp_max, q, du_max = BANKS[a.bank]
    rid = run_id(a.bank, a.mode) + ("-fake" if a.fake else "")
    results = "results-fake" if a.fake else "results"
    note = "L0 gpt-mini on archana-survey-written-v1 (handoff 2026-11-17)"
    if a.mode == "cp":
        from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste

        cfg = CopyPasteConfig(run_id=rid, models=[MODEL], attacker_level="L0",
                              n_samples=CP_SAMPLES, temperature=CP_TEMP, max_tokens=cp_max,
                              results_dir=results, note=note)
        rows = run_copy_paste(client, assessment, items, variants, mechs, cfg)
    else:
        from adr.harness.doc_upload import DocUploadConfig, run_doc_upload

        cfg = DocUploadConfig(run_id=rid, models=[MODEL], attacker_level="L0",
                              n_samples=DU_SAMPLES, temperature=DU_TEMP, max_tokens=du_max,
                              questions_per_doc=q, results_dir=results, note=note)
        rows = run_doc_upload(client, assessment, items, variants, mechs, cfg)
    run_dir = Path(results) / "runs" / rid
    conf = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    errs = (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    print(json.dumps({"run_id": rid, "rows": len(rows), "errors": len(errs),
                      "truncated": conf.get("n_truncated_replies"),
                      "stopped_for_budget": conf.get("stopped_for_budget"),
                      "skipped": conf.get("skipped_observations")}))
    if errs or conf.get("n_truncated_replies") or conf.get("stopped_for_budget"):
        print("STOP: errors, truncation or spend-cap stop; see errors.jsonl", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------- rescore (no API calls)
def rescore_rows(bank: str, base: list[OutcomeRow], responses: list[dict], items, variants,
                 order: list[str], style) -> tuple[list[OutcomeRow], Counter]:
    """New outcome rows for one run, from the harness's own rows and saved responses."""
    from adr.mechanisms.opinion_trap import MECHANISM as TRAP
    from adr.mechanisms.opinion_trap import score_reverse_pairs

    by_variant = {v.variant_id: v for vs in variants.values() for v in vs}
    resp = {(r["session_id"], r["variant_id"]): r for r in responses}
    pos = {iid: i for i, iid in enumerate(order)}
    pair_out: dict[tuple[str, str], str | None] = {}
    if any(items[i].reverse_keyed_with for i in order):
        sessions: dict[str, list[dict]] = {}
        for r in responses:
            if by_variant[r["variant_id"]].mechanism == "opinion_trap":
                sessions.setdefault(r["session_id"], []).append(r)
        for sid, rs in sessions.items():
            rs.sort(key=lambda r: pos[r["item_id"]])
            single = {r["item_id"]: TRAP.score(by_variant[r["variant_id"]],
                                                Response(**{k: r[k] for k in Response.model_fields}),
                                                None) for r in rs}
            final = score_reverse_pairs(items, [(r["item_id"], r["answer"]) for r in rs], single)
            pair_out.update({(sid, iid): o for iid, o in final.items()})

    out: list[OutcomeRow] = []
    changes: Counter = Counter()
    for row in base:
        key = (row.session_id, row.variant_id)
        new = row.outcome
        if row.mechanism == "style":
            r = resp[key]
            try:
                new = style.score(by_variant[row.variant_id],
                                  Response(**{k: r[k] for k in Response.model_fields}), None)
            except NoObservation:
                new = None
        elif row.mechanism == "opinion_trap" and pair_out:
            new = pair_out[(row.session_id, row.item_id)]
        if new != row.outcome:
            changes[f"{row.mechanism}: {row.outcome} -> {new}"] += 1
        if new is not None:
            out.append(row.model_copy(update={"outcome": new}))
    return out, changes


def cmd_rescore(a) -> None:
    import yaml

    style = style_mechanism(a.style_profile)
    profile_sha = sha256(a.style_profile)
    for bank in BANKS:
        assessment, items, variants = load(bank)
        for mode in ("cp", "du"):
            rid = run_id(bank, mode)
            run_dir = Path("results/runs") / rid
            current = Path("results/outcomes") / f"{rid}.parquet"
            original = run_dir / "outcomes.harness.parquet"
            if not original.exists():
                original.write_bytes(current.read_bytes())  # the harness's own rows, kept once
            df = read_outcomes(original).astype(object)
            base = [OutcomeRow(**r) for r in df.where(df.notna(), None).to_dict("records")]
            responses = [json.loads(x) for x in
                         (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
            rows, changes = rescore_rows(bank, base, responses, items, variants,
                                         assessment.item_ids, style)
            write_outcomes(rows, current)
            cfg_path = run_dir / "config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cfg["rescore"] = {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "script": "scripts/run_survey_banks.py rescore",
                "from": str(original),
                "style_profile": str(a.style_profile) if "style" in BANKS[bank][0] else None,
                "style_profile_sha256": profile_sha if "style" in BANKS[bank][0] else None,
                "pair_rule": PAIR_RULE if bank == "survey-ipip" else None,
                "n_rows_harness": len(base),
                "n_rows": len(rows),
                "changes": dict(sorted(changes.items())),
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                encoding="utf-8")
            print(f"{rid}: {len(base)} -> {len(rows)} rows; {dict(changes) or 'no changes'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("estimate")
    r = sub.add_parser("run")
    r.add_argument("--bank", choices=list(BANKS), required=True)
    r.add_argument("--mode", choices=["cp", "du"], required=True)
    r.add_argument("--fake", action="store_true")
    rs = sub.add_parser("rescore")
    rs.add_argument("--style-profile", type=Path, default=STYLE_PROFILE)
    a = ap.parse_args()
    {"check": cmd_check, "estimate": cmd_estimate, "run": cmd_run,
     "rescore": cmd_rescore}[a.cmd](a)


if __name__ == "__main__":
    main()

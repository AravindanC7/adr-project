"""Rates for the gpt-mini L0 survey-bank runs (scripts/run_survey_banks.py) vs the pilot humans.

    python scripts/report_survey_banks.py [--out results/survey_banks_v1]

Writes report.md and summary.csv (counts and Wilson 95% CIs; no answer text) and prints the
report. Run `run_survey_banks.py rescore` first, so the IPIP rows carry the pair rule.

Per bank x mode: outcome counts per mechanism (opinion_trap implausible / contradiction, canary
followed_instruction / leaked, style high / mid / low). Human comparison:
  * public human data on the SAME items: the expected implausible rate, i.e. the mean over items
    of the share of respondents choosing an option the trap calls implausible (p <= 0.05);
  * the pilot (results/outcomes/pilot-2026-11-13.parquet), with all sessions and without the 4
    low-effort sessions listed in Archana's handoff README. Its items differ: IPIP likert (4
    items, 2 pairs), OpinionQA MCQ (not GSS), handwritten open-text (not ASURRE).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data/processed/archana-survey-written-v1")
PILOT = Path("results/outcomes/pilot-2026-11-13.parquet")
RUNS = {
    ("survey-ipip", "copy_paste"): "v1-cp-L0-survey-ipip-s5",
    ("survey-ipip", "doc_upload"): "v1-du-L0-survey-ipip-q20",
    ("survey-gss", "copy_paste"): "v1-cp-L0-survey-gss-s5",
    ("survey-gss", "doc_upload"): "v1-du-L0-survey-gss",
    ("assignment-asurre-train", "copy_paste"): "v1-cp-L0-assignment-asurre-train-s5",
    ("assignment-asurre-train", "doc_upload"): "v1-du-L0-assignment-asurre-train",
}
LOW_EFFORT = ("d3ba4f10", "04096570", "01fe7650", "0a87374a")  # session-id prefixes
SHOWN = {"opinion_trap": ("implausible", "contradiction", "plausible"),
         "canary": ("followed_instruction", "leaked", "clean"),
         "style": ("high", "mid", "low")}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def pct(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def counts(frame: pd.DataFrame, who: str, mode: str, summary: list[dict]) -> list[str]:
    lines = []
    for (mech, itype), g in frame[frame.mechanism.notna()].groupby(["mechanism", "item_type"]):
        c, n = Counter(g.outcome), len(g)
        lines.append(f"- {mech} / {itype}: " + ", ".join(f"{o} {pct(c[o], n)}"
                                                        for o in SHOWN[mech]))
        summary += [{"who": who, "mode": mode, "mechanism": mech, "item_type": itype, "outcome": o,
                         "k": c[o], "n": n} for o in SHOWN[mech]]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/survey_banks_v1"))
    out_dir = ap.parse_args().out

    summary: list[dict] = []
    out = ["# gpt-mini L0 on archana-survey-written-v1\n",
           ("Rates are k/n with Wilson 95% CIs. Copy-paste: 5 sessions (samples); document upload: "
           "one session per PDF. IPIP rows carry the shared reverse-keyed pair rule "
           "(`score_reverse_pairs`), so a pair with a neutral answer has no row for its second "
           "item.")]
    for (bank, mode), rid in RUNS.items():
        d = pd.read_parquet(f"results/outcomes/{rid}.parquet")
        if set(d.model) != {"gpt-mini"} or d.heldout_model.any():
            raise SystemExit(f"{rid}: rows from a model other than gpt-mini")
        out.append(f"\n## {bank} x {mode}\n\n`{rid}`, {d.session_id.nunique()} sessions\n")
        out += counts(d, bank, mode, summary)

    out.append("\n## Human comparison\n")
    items = {json.loads(x)["item_id"]: json.loads(x) for x in
             (DATA / "items.jsonl").read_text(encoding="utf-8").splitlines()}
    dists = {json.loads(x)["key"]: json.loads(x) for x in
             (DATA / "human_distributions.jsonl").read_text(encoding="utf-8").splitlines()}
    for bank in ("survey-ipip", "survey-gss"):
        ids = json.loads((DATA / "assessments" / f"{bank}.json").read_text(encoding="utf-8"))[
            "item_ids"]
        rates = [sum(p for p in dists[items[i]["human_dist_key"]]["option_probs"] if p <= 0.05)
                 for i in ids]
        out.append(f"- {bank}, public human data on the same {len(ids)} items: expected "
                   f"implausible rate {100 * sum(rates) / len(rates):.1f}%")
        summary.append({"who": f"{bank} public humans", "mode": "human", "mechanism": "opinion_trap",
                            "item_type": items[ids[0]]["item_type"], "outcome": "implausible (expected)",
                            "k": "", "n": "", "rate": round(sum(rates) / len(rates), 4)})
    pilot = pd.read_parquet(PILOT)
    keep = pilot[~pilot.session_id.str.startswith(LOW_EFFORT)]
    for label, p in (("all sessions", pilot), ("without the 4 low-effort sessions", keep)):
        part = p[p.mechanism.isin(list(SHOWN))]
        out.append(f"\n**Pilot humans, {label} ({p.session_id.nunique()} sessions)**\n")
        out += counts(part, f"pilot ({label})", "human", summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out) + "\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    s = pd.DataFrame(summary)
    has_n = s.n != ""
    s.loc[has_n, "rate"] = [round(k / n, 4) for k, n in zip(s.k[has_n], s.n[has_n])]
    cis = [wilson(k, n) if n != "" else ("", "") for k, n in zip(s.k, s.n)]
    s["ci_low"] = [round(a, 4) if a != "" else "" for a, _ in cis]
    s["ci_high"] = [round(b, 4) if b != "" else "" for _, b in cis]
    s.to_csv(out_dir / "summary.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()

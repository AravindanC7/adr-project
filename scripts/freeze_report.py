"""Freeze reports and post-run checks for the final router runs (scripts/freeze_run.sh, step c).

    python scripts/freeze_report.py --out results/freeze-2026-11-26d/README.md \\
        --unseen-model RUN_DIR [RUN_DIR] --unseen-survey RUN_DIR [RUN_DIR] \\
        [--reference-all results/real-router/rq2-l012-floor750-reopen-all] \\
        [--reference-no-low-effort results/real-router/rq2-l012-floor750-reopen-no-low-effort]

Writes the README of the final run: the FPR rule, then two separate reports, each with TPR at the
A_0.05 and A_0.01 thresholds per attacker level and mode (95% bootstrap CIs over sessions):

  unseen model   claude-haiku composite sessions on the training banks
  unseen survey  gpt-mini and claude-haiku on the held-out OSMI bank

Checks, each failing loudly (exit 1) after writing what it found:
  * every run has exactly the expected AI groups (model:mode:level): each expected group is in
    BOTH n_ai_sessions_by_group and ai_sources_by_group, and each table's count is independently
    exactly 20 (samples 0-19). A group may be short (or absent) only by sessions whose documents
    were all refused in its own attacker runs (revised Nov 26). No groups at all, any other
    missing, short or over-full group, or a group or model outside the matrix fails, naming the
    run, the group, the table and the count.
    Expected groups:
      unseen model   claude-haiku x {L0, L2, L2g} x {copy_paste, doc_upload}  (6 groups)
      unseen survey  claude-haiku and gpt-mini x the same levels and modes   (12 groups)
    gpt-mini has no unseen-model groups by design: --final evaluates held-out rows only (the
    held-out model's, and training models' on the held-out survey), and gpt-mini's training-bank
    rows are train split, so they fit the map and are never evaluated there. A gpt-mini group in
    an unseen-model run is unexpected and fails;
  * every group (gpt-mini's OSMI groups too) has synthetic = fallback = 0 in the per-group source
    table, and unseen-model groups also have missing = 0: every active check was answered from
    the session's own rows, never from other sessions or the failure map (fitted on gpt-mini
    rows). Style is passive, so it is not in the source table;
  * the honest side is the pilot's: each run's thresholds equal #51's run with the same honest
    sessions (all 44, or without the low-effort ones; picked by the run folder's name), because
    the folds, pilot and config are the same.
Pre-registered ablation (configs/data_decisions.json preregistered_ablations): each main run has
a partner run `<run>-style-clear` with --clear-requires-style-human (no model calls). The partners
get the same group-matrix and source checks; each run's config.yaml must have the flag off (main)
or on (ablation). Each report shows the ablation beside the main policy: the primary outcome (L2
wrongly cleared), the costs (honest sessions cleared, checks per honest person) and the guardrail
(honest false pauses). The main policy (#51) stays the headline.
Refusals (decided Nov 26): a document-upload reply that ended normally but answered none of its
questions is recorded as refused, not as a failure (refused.jsonl; no limit, revised twice since). Each report lists the refused questions per attacker run and per group, read from
the run's config.yaml (n_refused_questions) via the router run's inputs.txt.
Reads aggregates only (metrics.json, config.yaml, inputs.txt); no session ids, no answer text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

FPR_RULE = (
    "False-flag rates come only from the 44 pilot honest sessions, with the same folds and "
    "calibrated thresholds as #51; the held-out OSMI bank has no human side, so no FPR is "
    "reported on it."
)
THRESHOLDS = ("A_0.05", "A_0.01")
LEVELS = ("L0", "L2", "L2g")
MODES = ("copy_paste", "doc_upload")
SESSIONS_PER_GROUP = 20  # samples 0-19


def groups(models: tuple[str, ...]) -> frozenset[str]:
    return frozenset(f"{m}:{mode}:{lv}" for m in models for mode in MODES for lv in LEVELS)


UNSEEN_MODEL_MODELS = ("claude-haiku",)
UNSEEN_SURVEY_MODELS = ("gpt-mini", "claude-haiku")
EXPECTED_GROUPS = {"unseen-model": groups(UNSEEN_MODEL_MODELS),
                   "unseen-survey": groups(UNSEEN_SURVEY_MODELS)}


ABLATION_SUFFIX = "-style-clear"
ABLATION_FLAG = "clear_requires_style_human"


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["results"]["router"]


SESSION_TABLES = ("n_ai_sessions_by_group", "ai_sources_by_group")


def _sessions(router: dict, table: str) -> dict[str, int]:
    """AI sessions per group as one table of metrics.json records them."""
    rows = router.get(table) or {}
    if table == "ai_sources_by_group":
        return {g: v.get("sessions") for g, v in rows.items()}
    return dict(rows)


def group_problems(router: dict, expected: frozenset[str], n: int = SESSIONS_PER_GROUP,
                   refused: dict[str, set[int]] | None = None) -> list[str]:
    """Every expected group must be in BOTH session tables, each count independently equal to `n`;
    no group outside the expected matrix (empty list = exactly the expected matrix).

    Revised Nov 26 after seeing the counts (no resampling): a group may have fewer than `n`
    sessions, or none, only if every missing session is a fully refused one, i.e. the shortfall
    equals the number of samples whose documents were all refused in the group's own attacker
    runs (`refused`: group -> sample indexes, from fully_refused_samples). Any other shortfall
    fails."""
    refused = refused or {}
    tables = {t: _sessions(router, t) for t in SESSION_TABLES}
    if not any(tables.values()):
        return [f"no AI groups at all (expected {len(expected)} groups of {n} sessions)"]
    models = {g.split(":")[0] for g in expected}
    bad = []
    for g in sorted(expected):
        k = len(refused.get(g, ()))
        for t, found in tables.items():
            got = found.get(g, 0)
            if got == n or (got < n and n - got == k):
                continue
            if g not in found:
                bad.append(f"missing group {g}: not in {t} (expected {n} sessions, "
                           f"{k} fully refused)")
            elif got < n:
                bad.append(f"group {g} has {got} sessions in {t}, expected exactly {n} "
                           f"({k} fully refused; {n - got - k} missing sessions unexplained)")
            else:
                bad.append(f"group {g} has {got} sessions in {t}, expected exactly {n}")
    for g in sorted(set().union(*tables.values()) - expected):
        counts = ", ".join(f"{t} {found[g]}" for t, found in tables.items() if g in found)
        model = g.split(":")[0]
        what = (f"unexpected model {model}: group {g}" if model not in models
                else f"unexpected group {g}")
        bad.append(f"{what} ({counts} sessions) is not in this report's matrix")
    return bad


def source_problems(router: dict, kind: str,
                    refused: dict[str, set[int]] | None = None) -> list[str]:
    """Every group's active checks came from its own sessions' rows: synthetic = fallback = 0 for
    every group (any model), and missing = 0 for unseen-model groups (claude-haiku ran every
    training bank, so every check has rows) unless every missing check is explained by recorded
    refused or unanswered questions (revised Nov 26): each session with missing checks
    (ai_missing_by_session) must have a document with refused or unanswered questions in one of
    its own group's attacker runs (`refused`: group -> sample indexes, from explained_samples).
    Any unexplained missing check fails. On the OSMI bank only canary has rows, so the
    unseen-survey groups' other checks are missing by design. Style is passive: it is scored on
    every written answer and is not in this source table."""
    bad = []
    refused = refused or {}
    by_session = router.get("ai_missing_by_session") or {}
    for group, src in sorted((router.get("ai_sources_by_group") or {}).items()):
        if src["synthetic"] > 0 or src["fallback"] > 0:
            bad.append(f"{group}: synthetic {src['synthetic']}, fallback {src['fallback']} per "
                       "session (must be 0: every check from the session's own rows)")
        if kind == "unseen-model" and src["missing"] > 0:
            sessions = by_session.get(group) or {}
            unexplained = sorted(s for s in sessions
                                 if _sample(s) not in refused.get(group, set()))
            if not sessions:
                bad.append(f"{group}: missing {src['missing']} checks per session with no "
                           "per-session record (must be 0, or explained by refused or unanswered "
                           "questions)")
            elif unexplained:
                bad.append(f"{group}: missing {src['missing']} checks per session; "
                           f"{len(unexplained)} of {len(sessions)} sessions with missing checks "
                           "have no recorded refused or unanswered question (unexplained missing "
                           "must be 0)")
    return bad


def _sample(session: str) -> int | None:
    """"composite:claude-haiku:doc_upload:L2:16" -> 16."""
    tail = session.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def missing_lines(router: dict, refused: dict[str, set[int]]) -> list[str]:
    """Per unseen-model group: sessions with missing checks, and how many are explained by
    recorded refused or unanswered questions."""
    by_session = router.get("ai_missing_by_session") or {}
    rows = []
    for g, sessions in sorted(by_session.items()):
        explained = [s for s in sessions if _sample(s) in refused.get(g, set())]
        rows.append(f"| {g} | {len(sessions)} | {len(explained)} | "
                    f"{sum(sessions[s] for s in explained) / max(len(sessions), 1):.2f} |")
    if not rows:
        return ["Missing checks: none in any group."]
    return [("Missing checks explained by refused or unanswered questions (sessions with "
             "missing checks; of those, sessions with a document that has refused or unanswered "
             "questions; mean missing checks per session explained):"), "",
            "| group | sessions with missing checks | explained | mean explained |",
            "|---|---|---|---|", *rows]


def threshold_problems(router: dict, reference: dict, tol: float = 1e-9) -> list[str]:
    return [
        f"{t}: {router['thresholds'][t]} vs reference {reference['thresholds'][t]}"
        for t in THRESHOLDS
        if not math.isclose(router["thresholds"][t], reference["thresholds"][t], abs_tol=tol)
    ]


def reference_for(run: Path, ref_all: Path | None, ref_no_low: Path | None) -> Path | None:
    """#51's run with the same honest sessions as this one."""
    return ref_no_low if run.name.endswith("no-low-effort") else ref_all


def _cell(router: dict, t: str, group: str) -> str:
    v = router["results"][t].get(f"tpr[{group}]")
    if v is None:
        return "–"
    ci = router["ci95"][t].get(f"tpr[{group}]")
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def table(router: dict, models: tuple[str, ...],
          refused: dict[str, set[int]] | None = None) -> list[str]:
    """Per group: sessions present, sessions missing because every document was refused, and TPR
    on the present sessions (n beside it). Fully refused sessions are not detections or misses."""
    refused = refused or {}
    n_by = router.get("n_ai_sessions_by_group") or {}
    src = router.get("ai_sources_by_group") or {}
    lines = [
        ("| model | level | mode | AI sessions present | missing: fully refused "
         "| TPR at A_0.05 [95% CI] (n) | TPR at A_0.01 [95% CI] (n) | missing checks per session |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        for level in LEVELS:
            for mode in MODES:
                g = f"{model}:{mode}:{level}"
                k = len(refused.get(g, ()))
                if g not in n_by and not k:
                    continue
                n = n_by.get(g, 0)
                miss = src.get(g, {}).get("missing")
                lines.append(
                    f"| {model} | {level} | {mode.replace('_', '-')} | {n} | {k} | "
                    f"{_cell(router, 'A_0.05', g)} ({n}) | {_cell(router, 'A_0.01', g)} ({n}) | "
                    f"{'–' if miss is None else f'{miss:.1f}'} |"
                )
    return lines


_RUN_ID = re.compile(r"^v1-final-(cp|du)-(L2-generic|L0|L2)-.+-(claude-haiku|gpt-mini)-s\d+$")
_MODE = {"cp": "copy_paste", "du": "doc_upload"}


def _final_runs(run: Path) -> dict[str, dict]:
    """The final attacker runs the router run read (its inputs.txt): run id -> its
    results/runs/<run_id>/config.yaml ({} if absent)."""
    inputs = run / "inputs.txt"
    if not inputs.is_file():
        return {}
    out = {}
    for line in inputs.read_text(encoding="utf-8").splitlines():
        p = Path(line.strip())
        if p.suffix != ".parquet" or not _RUN_ID.match(p.stem):
            continue
        cfg = p.parent.parent / "runs" / p.stem / "config.yaml"
        out[p.stem] = (yaml.safe_load(cfg.read_text(encoding="utf-8")) if cfg.is_file() else {}) or {}
    return out


def refusals(run: Path) -> dict[str, int]:
    """Refused questions per final attacker run (run id -> n_refused_questions, 0 if absent)."""
    return {rid: int(c.get("n_refused_questions") or 0) for rid, c in _final_runs(run).items()}


def unanswered(run: Path) -> dict[str, int]:
    """Unanswered (not refused) questions per final attacker run (run id ->
    n_unanswered_questions, 0 if absent)."""
    return {rid: int(c.get("n_unanswered_questions") or 0) for rid, c in _final_runs(run).items()}


def _samples(run: Path, filename: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for rid in _final_runs(run):
        path = _inputs_dir(run, rid) / filename
        if not path.is_file():
            continue
        g = _group(rid)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.setdefault(g, set()).add(int(json.loads(line)["sample_idx"]))
    return out


def refused_samples(run: Path) -> dict[str, set[int]]:
    """Per group (model:mode:level), the sample indexes with at least one refused question in the
    group's final attacker runs (results/runs/<run_id>/refused.jsonl)."""
    return _samples(run, "refused.jsonl")


def explained_samples(run: Path) -> dict[str, set[int]]:
    """Per group, the sample indexes with at least one refused or unanswered question in the
    group's own final attacker runs (refused.jsonl and unanswered.jsonl)."""
    out = refused_samples(run)
    for g, idx in _samples(run, "unanswered.jsonl").items():
        out.setdefault(g, set()).update(idx)
    return out


def fully_refused_samples(run: Path) -> dict[str, set[int]]:
    """Per group, the sample indexes whose documents were ALL refused in every one of the group's
    final attacker runs: each run is a document-upload run with refused.jsonl entries for every
    one of its documents (n_documents in config.yaml) for that sample, and no answered question
    (responses.jsonl) for it. Such a sample has no outcome rows, so its session is missing from
    the replay; it is reported separately, never counted as a detection or a miss."""
    by_group: dict[str, list[str]] = {}
    for rid in _final_runs(run):
        by_group.setdefault(_group(rid), []).append(rid)
    out: dict[str, set[int]] = {}
    for g, rids in by_group.items():
        samples: set[int] | None = None
        for rid in rids:
            d = _inputs_dir(run, rid)
            conf = _final_runs(run)[rid]
            n_docs = conf.get("n_documents")
            if conf.get("mode") != "doc_upload" or not n_docs or not (d / "refused.jsonl").is_file() \
                    or not (d / "responses.jsonl").is_file():
                samples = set()
                break
            docs: dict[int, set[int]] = {}
            for line in (d / "refused.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    docs.setdefault(int(r["sample_idx"]), set()).add(int(r["doc"]))
            answered = set()
            for line in (d / "responses.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    answered.add(int(json.loads(line)["session_id"].rsplit(":", 2)[-2]))
            full = {s for s, ks in docs.items() if len(ks) == n_docs and s not in answered}
            samples = full if samples is None else samples & full
        if samples:
            out[g] = samples
    return out


def _group(rid: str) -> str:
    mode, level, model = _RUN_ID.match(rid).groups()
    return f"{model}:{_MODE[mode]}:{'L2g' if level == 'L2-generic' else level}"


def _inputs_dir(run: Path, rid: str) -> Path:
    """results/runs/<rid> next to the outcome file the router run read."""
    for line in (run / "inputs.txt").read_text(encoding="utf-8").splitlines():
        p = Path(line.strip())
        if p.stem == rid and p.suffix == ".parquet":
            return p.parent.parent / "runs" / rid
    return Path("results/runs") / rid


def refusal_lines(run: Path) -> list[str]:
    """Refused and unanswered questions per attacker run and per group (model:mode:level)."""
    ref, una = refusals(run), unanswered(run)
    if not ref:
        return ["Refused questions: no final attacker runs listed in inputs.txt."]
    per_group: dict[str, list[int]] = {}
    for rid in ref:
        g = per_group.setdefault(_group(rid), [0, 0])
        g[0] += ref[rid]
        g[1] += una[rid]
    lines = [("Refused and unanswered questions (document upload, replies that ended normally; "
              "recorded, not failures). Refused: the reply matches the refusal pattern, no limit. "
              "Unanswered: it does not, allowed up to 5% of a run's questions. "
              f"{sum(ref.values())} refused and {sum(una.values())} unanswered in all."),
             "", "| group | refused questions | unanswered questions |", "|---|---|---|"]
    lines += [f"| {g} | {r} | {u} |" for g, (r, u) in sorted(per_group.items())]
    lines += ["", "| attacker run | refused questions | unanswered questions |", "|---|---|---|"]
    lines += [f"| {rid} | {ref[rid]} | {una[rid]} |" for rid in sorted(ref)]
    return lines


def honest(router: dict) -> list[str]:
    n = router["n_sessions"]
    out = []
    for t in THRESHOLDS:
        v = router["results"][t]
        k = round(v["honest_pause_rate"] * n)
        out.append(f"{t} = {router['thresholds'][t]:.3f}: {k} of {n} real honest sessions paused "
                   f"(nominal {v.get('nominal_fpr', '–')})")
    return out


def flag_problems(run: Path, want: bool) -> list[str]:
    """The run's config.yaml must record the ablation flag off (main) or on (ablation)."""
    conf = run / "config.yaml"
    if not conf.exists():
        return [f"no config.yaml to check {ABLATION_FLAG}"]
    got = bool((yaml.safe_load(conf.read_text(encoding="utf-8")).get("router_config") or {})
               .get(ABLATION_FLAG, False))
    return [] if got is want else [f"{ABLATION_FLAG} is {got}, expected {want}"]


def pair_problems(runs: list[Path], ablations: list[Path]) -> list[str]:
    want = [r.name + ABLATION_SUFFIX for r in runs]
    got = [r.name for r in ablations]
    return [] if got == want else [f"ablation runs {got} do not pair with {want}"]


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for k of n, by bisection on the binomial CDF."""
    a = (1 - conf) / 2

    def solve(f) -> float:  # f increasing in p on [0, 1]: the p where f(p) = 0
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: (1 - _binom_cdf(k - 1, n, p)) - a)
    upper = 1.0 if k == n else solve(lambda p: a - _binom_cdf(k, n, p))
    return lower, upper


def _pauses(router: dict, t: str) -> str:
    """Honest false pauses, k of n with the exact 95% interval (the pre-registered guardrail)."""
    n = router["n_sessions"]
    k = round(router["results"][t]["honest_pause_rate"] * n)
    lo, hi = clopper_pearson(k, n)
    return f"{k} of {n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def _of(router: dict, t: str, key: str, n: int) -> str:
    v = router["results"][t].get(key)
    return "–" if v is None else f"{round(v * n)} of {n} ({v:.2f})"


def ablation_table(main: dict, abl: dict, models: tuple[str, ...],
                   refused: dict[str, set[int]] | None = None) -> list[str]:
    """The pre-registered measures, main policy (#51) beside the ablation, at both thresholds."""
    lines = ["| measure | threshold | main policy (#51) | ablation (style must look human to clear) |",
             "|---|---|---|---|"]
    for t in THRESHOLDS:
        n_h = main["n_sessions"]
        thr = f"{t} ({main['thresholds'][t]:.3f} / {abl['thresholds'][t]:.3f})"
        for model in models:
            for mode in MODES:
                g = f"{model}:{mode}:L2"
                n = (main.get("n_ai_sessions_by_group") or {}).get(g, 0)
                k = len((refused or {}).get(g, ()))
                if n == 0:
                    cell = f"no sessions ({k} fully refused)"
                    lines.append(f"| **primary:** L2 wrongly cleared, {model} "
                                 f"{mode.replace('_', '-')} | {thr} | {cell} | {cell} |")
                    continue
                note = f"; {k} fully refused, not counted" if k else ""
                lines.append(f"| **primary:** L2 wrongly cleared, {model} {mode.replace('_', '-')} "
                             f"| {thr} | {_of(main, t, f'ai_cleared_rate[{g}]', n)}{note} | "
                             f"{_of(abl, t, f'ai_cleared_rate[{g}]', n)}{note} |")
        lines += [
            (f"| cost: honest sessions cleared | {thr} | "
            f"{_of(main, t, 'honest_cleared_rate', n_h)} | {_of(abl, t, 'honest_cleared_rate', n_h)} |"),
            (f"| cost: checks per honest person | {thr} | "
            f"{main['results'][t]['checks_per_honest_session']:.2f} | "
            f"{abl['results'][t]['checks_per_honest_session']:.2f} |"),
            (f"| guardrail: honest false pauses [exact 95% CI] | {thr} | "
             f"{_pauses(main, t)} | {_pauses(abl, t)} |"),
        ]
    return lines


def secondary_tpr_table(main: dict, abl: dict, models: tuple[str, ...],
                        refused: dict[str, set[int]] | None = None) -> list[str]:
    """Secondary outcome: per-group TPR, main policy beside the ablation, at both thresholds, on the
    present sessions (n), with fully refused sessions beside it."""
    refused = refused or {}
    n_by = main.get("n_ai_sessions_by_group") or {}
    lines = [("| model | level | mode | sessions present | missing: fully refused | "
              "TPR at A_0.05: main / ablation | TPR at A_0.01: main / ablation |"),
             "|---|---|---|---|---|---|---|"]
    for model in models:
        for level in LEVELS:
            for mode in MODES:
                g = f"{model}:{mode}:{level}"
                k = len(refused.get(g, ()))
                if g not in n_by and not k:
                    continue
                lines.append(f"| {model} | {level} | {mode.replace('_', '-')} | "
                             f"{n_by.get(g, 0)} | {k} | "
                             f"{_cell(main, 'A_0.05', g)} / {_cell(abl, 'A_0.05', g)} | "
                             f"{_cell(main, 'A_0.01', g)} / {_cell(abl, 'A_0.01', g)} |")
    return lines


def report(name: str, desc: str, runs: list[Path], models: tuple[str, ...],
           ablations: list[Path] | None = None, ablation_note: str = "") -> list[str]:
    lines = [f"## {name}", "", desc, ""]
    pairs = dict(zip(runs, ablations or [], strict=False))
    for run in runs:
        router = load(run)
        full = fully_refused_samples(run)
        lines += [f"### {run.name} ({router['n_sessions']} honest sessions)", "",
                  ("TPR is on the AI sessions present (n beside it). A session whose documents "
                   "were all refused has no rows: it is listed as fully refused and never "
                   "counted as a detection or a miss."), "",
                  *table(router, models, full), "",
                  "Honest side (the pilot's; see the FPR rule): " + "; ".join(honest(router)), "",
                  *refusal_lines(run), "",
                  *missing_lines(router, explained_samples(run)), ""]
        if run in pairs:
            abl = load(pairs[run])
            lines += [f"#### Pre-registered ablation: {pairs[run].name}", "",
                      ("A session may clear only if its running style total is below 0 when it "
                      "would clear (configs/data_decisions.json preregistered_ablations). The main "
                      "policy (#51) is unchanged and stays the headline. Thresholds are shown as "
                      "main / ablation; each run calibrates them on the same 44-session pilot "
                      "folds."), "",
                      *([ablation_note, ""] if ablation_note else []),
                      *ablation_table(router, abl, models, full), "",
                      "Secondary outcome: per-group TPR [95% CI], main policy / ablation:", "",
                      *secondary_tpr_table(router, abl, models, full), ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unseen-model", type=Path, nargs="+", required=True)
    ap.add_argument("--unseen-survey", type=Path, nargs="+", required=True)
    ap.add_argument("--unseen-model-ablation", type=Path, nargs="+", required=True,
                    help="the --clear-requires-style-human partner of each --unseen-model run")
    ap.add_argument("--unseen-survey-ablation", type=Path, nargs="+", required=True,
                    help="the --clear-requires-style-human partner of each --unseen-survey run")
    ap.add_argument("--reference-all", type=Path, help="#51's rq2-l012-floor750-reopen-all")
    ap.add_argument("--reference-no-low-effort", type=Path,
                    help="#51's rq2-l012-floor750-reopen-no-low-effort")
    a = ap.parse_args()

    lines = ["# Final run (experiment freeze, 2026-11-22)", "", f"**FPR rule.** {FPR_RULE}", ""]
    lines += report(
        "Unseen model",
        "claude-haiku (the held-out model) composite sessions on the training banks (quiz MMLU/ARC,"
        " survey IPIP/GSS, assignment-asurre-train). \"Unseen model\" means an unseen model, not "
        "unseen items: these are the same banks the router's map was fitted on, answered by a "
        "model that never contributed to the map, thresholds or any tuning.",
        a.unseen_model, UNSEEN_MODEL_MODELS, a.unseen_model_ablation,
    )
    lines += report(
        "Unseen survey",
        "gpt-mini and claude-haiku on the held-out OSMI bank (written answers only). This TPR "
        "comes from canary and style only and is not comparable to the headline: only the pilot's "
        "open-text checks (canary, style) have rows here; its other checks have none, so they show "
        "as missing and give no evidence.",
        a.unseen_survey, UNSEEN_SURVEY_MODELS, a.unseen_survey_ablation,
        ablation_note=("On the OSMI bank the evidence is canary and style only, so this "
                       "ablation's result here rests on canary + style evidence only and is not "
                       "comparable to the headline."),
    )

    problems = []
    runs = [("unseen-model", r) for r in a.unseen_model] + [("unseen-survey", r)
                                                           for r in a.unseen_survey]
    problems += pair_problems(a.unseen_model, a.unseen_model_ablation)
    problems += pair_problems(a.unseen_survey, a.unseen_survey_ablation)
    ablation_runs = [("unseen-model", r) for r in a.unseen_model_ablation] + [
        ("unseen-survey", r) for r in a.unseen_survey_ablation]
    for kind, run in ablation_runs:  # same group-matrix and source checks as the main runs
        router = load(run)
        problems += [f"{run.name}: {p}" for p in group_problems(
            router, EXPECTED_GROUPS[kind], refused=fully_refused_samples(run))]
        problems += [f"{run.name}: {p}" for p in source_problems(router, kind, explained_samples(run))]
        problems += [f"{run.name}: {p}" for p in flag_problems(run, True)]
    for kind, run in runs:
        router = load(run)
        problems += [f"{run.name}: {p}" for p in group_problems(
            router, EXPECTED_GROUPS[kind], refused=fully_refused_samples(run))]
        problems += [f"{run.name}: {p}" for p in source_problems(router, kind, explained_samples(run))]
        problems += [f"{run.name}: {p}" for p in flag_problems(run, False)]
        ref = reference_for(run, a.reference_all, a.reference_no_low_effort)
        if ref is None:
            problems.append(f"{run.name}: no #51 reference run to check its thresholds against")
        else:
            problems += [f"{run.name}: threshold differs from #51 ({ref.name}), {p}"
                         for p in threshold_problems(router, load(ref))]
    lines += ["## Post-run checks", "",
              ("Source table: where each group's ACTIVE checks were answered from. Style is "
              "passive (scored on every written answer), so it is not in the source table."), ""]
    lines += [f"- FAILED: {p}" for p in problems] or [
        ("- sources: every group's checks answered from its own sessions' rows (synthetic 0, "
         "fallback 0); unseen-model groups also missing 0, or every missing check in a session "
         "with a recorded refusal."),
        (f"- AI groups: exactly the expected matrix, {SESSIONS_PER_GROUP} sessions in each "
         "(unseen model: claude-haiku, 6 groups; unseen survey: both models, 12 groups), except "
         "sessions whose documents were all refused in the group's own runs (listed per group)."),
        "- thresholds: equal to #51's runs with the same honest sessions.",
        ("- pre-registered ablation: every main run has its -style-clear partner, with the same "
         "group matrix and sources; the flag is off in the main runs and on in the partners."),
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    if problems:
        for p in problems:
            print(f"POST-RUN CHECK FAILED: {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

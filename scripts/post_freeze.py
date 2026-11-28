"""Post-freeze exploratory analyses (after freeze; no API calls; the frozen config is unchanged).

Everything this script writes is labelled "exploratory (after freeze)". It reads the frozen router
runs (results/real-router/<freeze>-unseen-*) and re-creates their replays in-process with the exact
flags the driver used (their config.yaml and scripts/freeze_run.sh), so it adds no model calls.

    # 1. per (run, policy): per-session scores, then a session bootstrap whose draws are shared by
    #    every policy (paired), into results/post-freeze/draws/<run>--<policy>.json
    python scripts/post_freeze.py draws --run freeze-2026-11-26d-unseen-model-all --policy router
    # 2. tables and the plot from the draws and the frozen runs' own files
    python scripts/post_freeze.py report

Paired bootstrap. Draw b resamples the honest sessions (within the pilot, seeded by b only) and the
AI sessions (within their group, seeded by b and the run), exactly as the frozen summary does
(adr/replay/crossfit.summarize), but with the SAME draws for every policy. Each policy's thresholds
are recalibrated per draw from its own honest and resampled-honest scores, as in the frozen
summary. The TPR difference (router minus best_single, router minus all_checks) is taken draw by
draw; the 95% interval is the 2.5th and 97.5th percentiles. Honest draws do not depend on the run,
and the unseen-model and unseen-survey runs with the same honest sessions share their folds, maps
and thresholds, so their AI sessions can be pooled draw by draw.

Evidence per check: for the router on <freeze>-unseen-model-all, the log-likelihood ratio each check
added to each claude-haiku session until it paused at the threshold (or to the end), averaged over
the sessions and fold maps of each group.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

LABEL = "exploratory (after freeze)"
FREEZE = "freeze-2026-11-26d"
RUNS = [f"{FREEZE}-unseen-{k}-{w}" for k in ("model", "survey") for w in ("all", "no-low-effort")]
POLICIES = ("router", "best_single", "all_checks")
THRESHOLDS = ("A_0.05", "A_0.01")
OUT = Path("results/post-freeze")
CALIBRATION = {"all": "rq2-l012-floor750-reopen-all",
               "no-low-effort": "rq2-l012-floor750-reopen-no-low-effort"}
# scripts/freeze_run.sh ROUTER flags (settings only; --jobs and --resume do not change a result)
ROUTER_FLAGS = ["--final", "--bank", "server/pilot_items.json",
                "--cap-variant", "per_mech", "passive_group", "open_text_group", "no_cap",
                "--drop-mechanisms", "canary,decoy,opinion_trap,style"]


class _Captured(Exception):
    pass


def router_argv(run: str, work: Path) -> list[str]:
    """The run's run_real_router.py command line, rebuilt from its config.yaml and the manifest."""
    conf = yaml.safe_load((Path("results/real-router") / run / "config.yaml").read_text())
    man = json.loads((Path("results") / FREEZE / "freeze_manifest.json").read_text())
    argv = [*ROUTER_FLAGS, "--pilot-outcomes", man["pilot"]["path"],
            "--low-effort", man["low_effort_ids"]["path"],
            "--ai-outcomes", *conf["ai_outcomes"]]
    if run.endswith("no-low-effort"):
        argv += ["--exclude-sessions", man["low_effort_ids"]["path"]]
    return [*argv, "--run-id", f"post-freeze-{run}", "--results-dir", str(work)]


def _group(per_fold) -> str:
    return per_fold[0].who.split("|", 1)[0].removeprefix("ai:")


def paired_draws(run: str, policy: str, n_boot: int | None = None) -> dict:
    """Re-create `run`'s replay for `policy` and return its point estimate and per-draw TPR
    numerators and denominators per AI group, plus per-check evidence for the router."""
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    import run_real_router as rr

    from adr.contracts.mechanism import parse_variant_id
    from adr.replay import crossfit as cf

    evidence: list[tuple[str, int, list[str], np.ndarray]] = []
    real_trace = cf.trace

    def trace(result, who, fold, prior):  # also keep which check added each delta (AI only)
        t = real_trace(result, who, fold, prior)
        if who.startswith("ai:"):
            h = result.final.history
            evidence.append((who, fold, [parse_variant_id(v)[1] or "plain" for v, _, _ in h],
                             np.array([d for _, _, d in h], dtype=float)))
        return t

    cf.trace = trace
    out: dict = {}

    def summarize(scored, make, c, assessment, items, human, alphas, boot, n_syn_boot, passive,
                  seed, mins):
        mins = mins or {}
        size: dict[str, int] = defaultdict(int)
        for f in scored.ai.values():
            size[_group(f)] += 1
        ai = {w: f for w, f in sorted(scored.ai.items())
              if size[_group(f)] >= mins.get(cf.group_mode(_group(f)), 0)}
        groups = sorted({_group(f) for f in ai.values()})
        by_group = {g: [w for w, f in ai.items() if _group(f) == g] for g in groups}
        det = {w: [t.score for t in f] for w, f in ai.items()}  # per fold map

        def tally(th: dict, pick: dict[str, list[str]]) -> dict:
            res = {}
            for t in THRESHOLDS:
                a = th[t]
                res[t] = {g: float(sum(np.mean([s >= a for s in det[w]]) for w in ws))
                          for g, ws in pick.items()}
            return res

        real = scored.real
        syn = [t.score for t in scored.synthetic]
        th = cf.thresholds_from([t.score for t in real.values()], syn, alphas)
        out["point"] = {"thresholds": th, "detected": tally(th, by_group),
                        "n": {g: len(ws) for g, ws in by_group.items()}}
        index = cf._index(human)
        pols = [make(fmap, cf.score_config(c)) for fmap in scored.maps]
        honest = sorted(real)
        draws = []
        for b in range(n_boot or boot):
            rh = np.random.default_rng([seed, b])
            picked = [honest[i] for i in rh.integers(0, len(honest), len(honest))]
            ra = np.random.default_rng([seed, b, int.from_bytes(run.encode()[-8:], "big") % 2**31])
            pick = {g: [ws[i] for i in ra.integers(0, len(ws), len(ws))] for g, ws in by_group.items()}
            by_fold: dict[int, list[str]] = defaultdict(list)
            for p in picked:
                by_fold[real[p].fold].append(p)
            syn_b = []
            for k, ss in sorted(by_fold.items()):
                syn_b += [t.score for t in cf._synthetic(pols[k], assessment, items, index, ss, k,
                                                          n_syn_boot, passive, seed,
                                                          scored.prior, tag=f":boot{b}")]
            th_b = cf.thresholds_from([real[p].score for p in picked], syn_b, alphas)
            draws.append({"thresholds": {t: th_b[t] for t in THRESHOLDS},
                          "detected": tally(th_b, pick)})
        out["draws"] = draws
        out["groups"] = groups
        if policy == "router":
            out["evidence"] = evidence_by_check(evidence, th, ai)
        raise _Captured

    def run_tasks(tasks, steps_dir, jobs, resume, seed):
        tasks[f"policy-{policy}"]()

    rr.summarize, rr.run_tasks = summarize, run_tasks
    work = OUT / "_work"
    sys.argv = ["run_real_router.py", *router_argv(run, work)]
    try:
        rr.main()
    except _Captured:
        pass
    return {"label": LABEL, "run": run, "policy": policy, **out}


def evidence_by_check(evidence, th: dict, ai: dict) -> dict:
    """Per group and threshold: mean evidence (sum of log-likelihood ratios) each check added per
    session until the session paused at the threshold (or to the end), and the share of sessions
    where the check added any positive evidence. Averaged over the fold maps."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for who, _fold, mechs, deltas in evidence:
        if who not in ai:
            continue
        g = _group(ai[who])
        running = np.cumsum(deltas)
        for t in THRESHOLDS:
            # pause: the running score (prior + cumulative evidence) first reaches the threshold
            prior = ai[who][0].running[0] - (deltas[0] if len(deltas) else 0)
            hit = np.nonzero(prior + running >= th[t])[0]
            stop = hit[0] + 1 if len(hit) else len(deltas)
            per: dict[str, float] = defaultdict(float)
            for m, d in zip(mechs[:stop], deltas[:stop], strict=True):
                per[m] += d
            for m in {"canary", "decoy", "opinion_trap", "style", "behaviour"} | set(per):
                acc[g][t][m].append(per.get(m, 0.0))
    return {g: {t: {m: {"mean_llr": float(np.mean(v)), "share_positive": float(np.mean(np.array(v) > 0))}
                    for m, v in sorted(ms.items()) if any(x != 0 for x in v)}
                for t, ms in by_t.items()} for g, by_t in sorted(acc.items())}


# ------------------------------------------------------------------ report
def _load(run: str, policy: str) -> dict:
    return json.loads((OUT / "draws" / f"{run}--{policy}.json").read_text())


def _tpr(rec: dict, t: str, groups: list[str], n: dict) -> float:
    d = sum(rec["detected"][t][g] for g in groups)
    return d / sum(n[g] for g in groups)


def diff_rows() -> list[dict]:
    rows = []
    sets = [(r, [r]) for r in RUNS] + [
        (f"pooled-{w} (unseen model + unseen survey)", [f"{FREEZE}-unseen-model-{w}",
                                                        f"{FREEZE}-unseen-survey-{w}"])
        for w in ("all", "no-low-effort")]
    for name, runs in sets:
        recs = {p: [_load(r, p) for r in runs] for p in POLICIES}
        for base in ("best_single", "all_checks"):
            for t in THRESHOLDS:
                def tpr(p, which, recs=recs, t=t):  # pooled over the runs' groups
                    num = den = 0.0
                    for rec in recs[p]:
                        src = rec["point"] if which is None else rec["draws"][which]
                        num += sum(src["detected"][t].values())
                        den += sum(rec["point"]["n"].values())
                    return num / den
                n_draws = min(len(rec["draws"]) for p in POLICIES for rec in recs[p])
                d = [tpr("router", b) - tpr(base, b) for b in range(n_draws)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({"set": name, "comparison": f"router - {base}", "threshold": t,
                             "router_tpr": tpr("router", None), "baseline_tpr": tpr(base, None),
                             "diff": tpr("router", None) - tpr(base, None),
                             "lo": float(lo), "hi": float(hi), "n_boot": n_draws,
                             "share_draws_router_better": float(np.mean(np.array(d) > 0)),
                             "share_draws_router_worse": float(np.mean(np.array(d) < 0))})
    return rows


def check_reproduces() -> list[str]:
    """The re-created replays must give the frozen runs' thresholds and TPR."""
    bad = []
    for r in RUNS:
        frozen = json.loads((Path("results/real-router") / r / "metrics.json").read_text())
        for p in POLICIES:
            rec = _load(r, p)
            fr = frozen["results"][p]
            n = rec["point"]["n"]
            for t in THRESHOLDS:
                a, b = rec["point"]["thresholds"][t], fr["thresholds"][t]
                tp, tf = _tpr(rec["point"], t, list(n), n), fr["results"][t]["tpr"]
                if not (math.isclose(a, b, abs_tol=1e-9) and math.isclose(tp, tf, abs_tol=1e-9)):
                    bad.append(f"{r} {p} {t}: A {a} vs {b}, TPR {tp} vs {tf}")
    return bad


def loo_rows(run: str) -> list[dict]:
    import csv

    path = Path("results/real-router") / run / "leave_one_out.csv"
    frozen = json.loads((Path("results/real-router") / run / "metrics.json").read_text())["results"]
    rows = []
    for r in csv.DictReader(path.open()):
        if r["threshold"] not in THRESHOLDS:
            continue
        full = frozen[r["policy"]]["results"][r["threshold"]]
        rows.append({"dropped": r["dropped"], "policy": r["policy"], "threshold": r["threshold"],
                     "tpr_full": full["tpr"], "tpr_without": float(r["tpr"]),
                     "tpr_without_lo": float(r["tpr_lo"]), "tpr_without_hi": float(r["tpr_hi"]),
                     "checks_per_honest_without": float(r["checks_per_honest_session"]),
                     "honest_pause_rate_without": float(r["honest_pause_rate"])})
    return rows


def cost_rows() -> list[dict]:
    import csv

    rows = []
    panels = [("gpt-mini calibration (#51)", CALIBRATION["all"]),
              ("unseen model", f"{FREEZE}-unseen-model-all"),
              ("unseen survey", f"{FREEZE}-unseen-survey-all")]
    for panel, run in panels:
        for r in csv.DictReader((Path("results/real-router") / run / "comparison.csv").open()):
            if r["threshold"] not in THRESHOLDS:
                continue
            rows.append({"panel": panel, "run": run, "policy": r["policy"],
                         "threshold": r["threshold"],
                         **{k: float(r[k]) for k in ("tpr", "tpr_lo", "tpr_hi",
                                                     "checks_per_honest_session",
                                                     "checks_per_honest_session_lo",
                                                     "checks_per_honest_session_hi",
                                                     "honest_pause_rate")}})
    return rows


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The six compared policies, each a fixed categorical slot (validated palette, light mode)
    # plus its own marker shape, so identity is never colour alone. The router's cap variants
    # are in cost_vs_detection.csv (the table view), not on the plot.
    style = {"router": ("#257de0", "o"), "router_noclear": ("#f47230", "s"),
             "best_single": ("#00b894", "^"), "all_checks": ("#f2a600", "D"),
             "fixed_plan": ("#ee7fac", "v"), "random": ("#008a17", "P")}
    rows = [r for r in rows if r["policy"] in style]
    # Policies with identical numbers in a panel (on the 24-item pilot bank all_checks,
    # fixed_plan and random each serve every check) are drawn once, under one joint label.
    same: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        same[(r["panel"], r["threshold"], r["checks_per_honest_session"], r["tpr"])].append(r)
    rows = [{**rs[0], "label": " = ".join(x["policy"] for x in rs)} for rs in same.values()]
    panels = list(dict.fromkeys(r["panel"] for r in rows))
    fig, axes = plt.subplots(len(THRESHOLDS), len(panels), figsize=(4.2 * len(panels), 7.4),
                             sharey=True, squeeze=False)
    for i, t in enumerate(THRESHOLDS):
        for j, panel in enumerate(panels):
            ax = axes[i][j]
            for r in (x for x in rows if x["panel"] == panel and x["threshold"] == t):
                x, y = r["checks_per_honest_session"], r["tpr"]
                ax.errorbar(x, y,
                            xerr=[[x - r["checks_per_honest_session_lo"]],
                                  [r["checks_per_honest_session_hi"] - x]],
                            yerr=[[y - r["tpr_lo"]], [r["tpr_hi"] - y]],
                            fmt=style[r["policy"]][1], ms=8, color=style[r["policy"]][0],
                            elinewidth=1, capsize=2, markeredgecolor="white",
                            markeredgewidth=1.5, label=r["label"], ecolor="#9c9c9c")
            ax.set_title(f"{panel}, {t}", fontsize=10)
            ax.set_ylim(-0.02, 1.04)
            ax.grid(alpha=0.25, linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == len(THRESHOLDS) - 1:
                ax.set_xlabel("checks per honest person")
            if j == 0:
                ax.set_ylabel("TPR (AI sessions paused)")
    from matplotlib.lines import Line2D

    handles = {}
    for r in rows:
        c, m = style[r["policy"]]
        handles.setdefault(r["label"], Line2D([], [], ls="", marker=m, ms=8, color=c,
                                              markeredgecolor="white", markeredgewidth=1.5))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9)
    fig.suptitle(f"Cost vs detection, all policies, 95% CIs - {LABEL}", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160)


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    with path.open("w", newline="") as f:
        f.write(f"# {LABEL}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_report(_a) -> int:
    bad = check_reproduces()
    if bad:
        for b in bad:
            print(f"NOT REPRODUCED: {b}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    diffs = diff_rows()
    write_csv(diffs, OUT / "paired_tpr_differences.csv")
    loo = loo_rows(f"{FREEZE}-unseen-model-all")
    write_csv(loo, OUT / "leave_one_check_out_unseen_model_all.csv")
    ev = _load(f"{FREEZE}-unseen-model-all", "router")["evidence"]
    (OUT / "evidence_by_check_unseen_model_all.json").write_text(
        json.dumps({"label": LABEL, "evidence": ev}, indent=1))
    cost = cost_rows()
    write_csv(cost, OUT / "cost_vs_detection.csv")
    plot(cost, OUT / "cost_vs_detection.png")
    print(f"{LABEL}: wrote {OUT}/paired_tpr_differences.csv, leave_one_check_out_unseen_model_all"
          ".csv, evidence_by_check_unseen_model_all.json, cost_vs_detection.csv and .png; the "
          "re-created replays reproduce the frozen thresholds and TPR")
    return 0


def cmd_draws(a) -> int:
    rec = paired_draws(a.run, a.policy, a.boot)
    (OUT / "draws").mkdir(parents=True, exist_ok=True)
    (OUT / "draws" / f"{a.run}--{a.policy}.json").write_text(json.dumps(rec))
    print(f"{a.run} {a.policy}: {len(rec['draws'])} draws")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draws")
    d.add_argument("--run", required=True, choices=RUNS)
    d.add_argument("--policy", required=True, choices=POLICIES)
    d.add_argument("--boot", type=int, help="draws (default: the frozen run's, 200)")
    sub.add_parser("report")
    a = ap.parse_args()
    random.seed(0)
    return {"draws": cmd_draws, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

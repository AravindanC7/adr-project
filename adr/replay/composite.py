"""Composite AI sessions for the multi-check router run (RQ2/RQ3). Owner: Arav.

A pilot session mixes quiz MCQs (decoy), survey MCQ/Likert (opinion trap, incl. contradiction)
and open text (canary, style). Each harness run covers ONE bank. `join_ai_sessions` joins the
runs into sessions shaped like a pilot session:

    key = (model, attack_mode, attacker variant, sample index)

All rows with the same key, from every bank, become one session
"composite:<model>:<attack_mode>:<attacker variant>:<sample>". The attacker variant is the
level plus any prompt variant, read from prompt_version (`attacker_variant`): L2 and L2-generic
rows are both stored as attacker_level L2 and differ only in prompt_version ("cp-v1" vs
"cp-v1+l2-generic-v1"), so they become L2 and L2g and never share a session or a group.

Assumption (documented in the PR): harness runs are independent calls, so sample i of the quiz
run and sample i of a survey run are no more alike than any two samples. Pairing them by index
is one arbitrary pairing among equally valid ones; it treats the attacker as one respondent who
took the whole pilot. The sample index is read from the harness session_id:

    copy_paste       "<run_id>:<model>:<sample>"
    doc_upload       "<run_id>:<model>:<sample>:d<k>"   (d<k> = one document chunk of the bank)

so new samples (e.g. document-upload samples 1-19 added by a later run) are picked up without
code changes. A bank is identified by the item sources it covers (e.g. "arc", "ipip"), not by
run_id, so a later run over the same bank adds samples to it. The same (bank, key) coming from two
different runs is an error: two answer sheets for one sample.

A key that is missing in a bank is never dropped silently: the session is kept (its checks on that
bank's item types fall back to the group's pooled rows in replay, adr/replay/donors.py) and the
gap is listed in `CompositeJoin.missing`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

KEY = ("model", "attack_mode", "attacker_variant", "sample")
GROUP = ("model", "attack_mode", "attacker_variant")
# prompt_version suffix ("<base>+<suffix>") -> attacker variant name. None = the suffix is part of
# the level itself (L3's retry prompt), not a variant. Any other suffix gets "<level>+<suffix>",
# so an unknown variant is never merged with its level silently.
PROMPT_VARIANTS: dict[str, str | None] = {"l2-generic-v1": "L2g", "cp-retry-v1": None}


def attacker_variant(level: str, prompt_version: str | None) -> str:
    """"L2" + "cp-v1+l2-generic-v1" -> "L2g"; "L2" + "cp-v1" -> "L2"; "L3" + "cp-v1+cp-retry-v1"
    -> "L3"."""
    if not isinstance(prompt_version, str) or "+" not in prompt_version:
        return level
    suffix = prompt_version.split("+", 1)[1]
    if suffix in PROMPT_VARIANTS:
        return PROMPT_VARIANTS[suffix] or level
    return f"{level}+{suffix}"


def add_attacker_variant(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows with an attacker_variant column (used by the join and by the replay groups)."""
    rows = rows.copy()
    pv = rows["prompt_version"] if "prompt_version" in rows else pd.Series(None, index=rows.index)
    rows["attacker_variant"] = [
        attacker_variant(lv, v) for lv, v in zip(rows["attacker_level"], pv, strict=True)
    ]
    return rows
# Never used for fitting or evaluation before the final run (AGENTS.md section 8 + team decisions).
FORBIDDEN_MODELS = frozenset({"claude-haiku", "gemini-flash"})
FORBIDDEN_SOURCES = frozenset({"asurre_osmi_2017_18"})  # the held-out survey


def sample_index(session_id: str, run_id: str, model: str) -> str:
    """The sample index in a harness session_id (see the module doc)."""
    prefix = f"{run_id}:{model}:"
    if not session_id.startswith(prefix):
        raise ValueError(
            f"AI session_id {session_id!r} is not '<run_id>:<model>:<sample>[:d<k>]' "
            f"for run {run_id!r}, model {model!r}"
        )
    sample = session_id[len(prefix) :].split(":", 1)[0]
    if not sample.isdigit():
        raise ValueError(f"AI session_id {session_id!r}: sample index {sample!r} is not a number")
    return sample


def source_of(item_id: str) -> str:
    return item_id.split(":", 1)[0]


def check_allowed(rows: pd.DataFrame) -> None:
    """Refuse held-out model rows, the excluded models and the held-out (OSMI) survey."""
    bad_model = rows["model"].isin(FORBIDDEN_MODELS)
    if "heldout_model" in rows:
        bad_model |= rows["heldout_model"].fillna(False).astype(bool)
    bad_source = rows["item_id"].map(source_of).isin(FORBIDDEN_SOURCES)
    if "split" in rows:
        bad_source |= rows["split"].eq("heldout_survey")
    if bad_model.any() or bad_source.any():
        raise ValueError(
            f"refusing {int(bad_model.sum())} held-out/excluded-model rows "
            f"({sorted(set(rows.loc[bad_model, 'model']))}) and {int(bad_source.sum())} "
            "held-out-survey rows: never pass these to the router run"
        )


@dataclass(frozen=True)
class CompositeJoin:
    rows: pd.DataFrame  # the AI rows, session_id replaced by the composite session id
    missing: pd.DataFrame  # one row per (key, bank) with no rows: model..sample, bank
    banks: dict[tuple[str, str, str], list[str]]  # per group: the banks seen in that group

    def summary(self) -> str:
        n = self.rows["session_id"].nunique()
        if self.missing.empty:
            return f"{n} composite AI sessions; every key has rows in every bank of its group"
        per = self.missing.groupby(["attack_mode", "attacker_variant", "bank"])["sample"].apply(
            lambda s: ",".join(sorted(s, key=int))
        )
        lines = [f"{n} composite AI sessions; keys MISSING in a bank (kept; see composite.py):"]
        lines += [
            f"  {mode} {variant} / {bank}: samples {samples}"
            for (mode, variant, bank), samples in per.items()
        ]
        return "\n".join(lines)


def _bank_of_run(rows: pd.DataFrame) -> pd.Series:
    """run_id -> bank name: the sorted item sources the run covers, e.g. 'arc' or
    'asurre_nature_phd_2019+asurre_springer_social_2017'."""
    sources = rows.assign(_src=rows["item_id"].map(source_of)).groupby("run_id")["_src"].unique()
    return sources.map(lambda s: "+".join(sorted(s)))


def join_ai_sessions(rows: pd.DataFrame, allow_heldout: bool = False) -> CompositeJoin:
    """Join AI rows across banks into composite sessions (see the module doc). Held-out rows are
    refused unless `allow_heldout` (the final evaluation only)."""
    if rows.empty:
        raise ValueError("no AI rows to join")
    if not rows["respondent_kind"].eq("ai").all():
        raise ValueError("join_ai_sessions takes AI rows only")
    if not allow_heldout:
        check_allowed(rows)
    rows = add_attacker_variant(rows)
    rows["sample"] = [
        sample_index(s, r, m)
        for s, r, m in zip(rows["session_id"], rows["run_id"], rows["model"], strict=True)
    ]
    rows["bank"] = rows["run_id"].map(_bank_of_run(rows))

    # one answer sheet per (bank, key): two runs giving the same sample of a bank is an error
    runs = rows.groupby(["bank", *KEY])["run_id"].nunique()
    dup = runs[runs > 1]
    if not dup.empty:
        raise ValueError(
            "the same sample of a bank comes from more than one run (pass one of them): "
            f"{dup.index[:5].tolist()}"
        )

    banks: dict[tuple[str, str, str], list[str]] = {}
    missing = []
    for group, g in rows.groupby(list(GROUP), sort=True):
        group_banks = sorted(g["bank"].unique())
        banks[tuple(group)] = group_banks
        have = set(zip(g["bank"], g["sample"], strict=True))
        for sample in sorted(g["sample"].unique(), key=int):
            for bank in group_banks:
                if (bank, sample) not in have:
                    missing.append({**dict(zip(GROUP, group, strict=True)), "sample": sample,
                                    "bank": bank})

    rows["session_id"] = [
        f"composite:{m}:{a}:{lv}:{i}"
        for m, a, lv, i in zip(
            rows["model"], rows["attack_mode"], rows["attacker_variant"], rows["sample"],
            strict=True,
        )
    ]
    missing_df = pd.DataFrame(missing, columns=[*KEY, "bank"])
    return CompositeJoin(rows.drop(columns=["sample", "bank"]), missing_df, banks)

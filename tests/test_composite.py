"""adr/replay/composite.py: composite AI sessions joined across banks."""

import pandas as pd
import pytest

from adr.replay.composite import join_ai_sessions, sample_index

# bank tag -> (item source, item_type, mechanism, outcome)
BANKS = {
    "quiz-arc": ("arc", "mcq", "decoy", "picked_decoy"),
    "survey-gss": ("gss", "mcq", "opinion_trap", "plausible"),
    "survey-ipip": ("ipip", "likert", "opinion_trap", "contradiction"),
    "assignment-asurre": ("asurre_nature_phd_2019", "open_text", "canary", "followed_instruction"),
}


def _run(model, mode, bank, samples, run_suffix="", docs=2, level="L0", prompt_version=None):
    tag = {"copy_paste": "cp", "doc_upload": "du"}[mode]
    source, itype, mech, outcome = BANKS[bank]
    run_id = f"v1-{tag}-{level}-{bank}{run_suffix}"
    prompt_version = prompt_version or f"{tag}-v1"
    rows = []
    for sample in samples:
        chunks = [f":d{k}" for k in range(docs)] if mode == "doc_upload" else [""]
        for d in chunks:
            for i in range(2):
                rows.append(
                    {
                        "run_id": run_id,
                        "session_id": f"{run_id}:{model}:{sample}{d}",
                        "respondent_kind": "ai",
                        "model": model,
                        "attack_mode": mode,
                        "attacker_level": level,
                        "prompt_version": prompt_version,
                        "heldout_model": False,
                        "split": "train",
                        "item_type": itype,
                        "item_id": f"{source}:{d}{i}",
                        "mechanism": mech,
                        "outcome": outcome,
                    }
                )
    return rows


def _table(samples=range(3), models=("gpt-mini", "other-model")):
    rows = []
    for model in models:
        for mode in ("copy_paste", "doc_upload"):
            for bank in BANKS:
                rows += _run(model, mode, bank, samples)
    return pd.DataFrame(rows)


def test_sample_index_parses_both_harness_formats():
    assert sample_index("v1-cp-L0-quiz:gpt-mini:3", "v1-cp-L0-quiz", "gpt-mini") == "3"
    assert sample_index("v1-du-L0-quiz:gpt-mini:0:d12", "v1-du-L0-quiz", "gpt-mini") == "0"
    with pytest.raises(ValueError):
        sample_index("other:gpt-mini:0", "v1-cp-L0-quiz", "gpt-mini")


def test_composite_session_holds_every_bank_and_never_mixes():
    raw = _table()
    join = join_ai_sessions(raw)
    comp = join.rows
    assert len(comp) == len(raw) and join.missing.empty
    assert comp["session_id"].nunique() == 2 * 2 * 3  # models x modes x samples
    for sid, g in comp.groupby("session_id"):
        assert g["run_id"].nunique() == len(BANKS), sid  # quiz + survey + open text
        assert set(g["item_type"]) == {"mcq", "likert", "open_text"}
        assert g["model"].nunique() == 1 and g["attack_mode"].nunique() == 1
        assert g["attacker_level"].nunique() == 1
        samples = {
            sample_index(s, r, m)
            for s, r, m in zip(
                raw.loc[g.index, "session_id"], g["run_id"], g["model"], strict=True
            )
        }
        assert len(samples) == 1, sid  # never mixes samples
        assert sid.endswith(":" + samples.pop())
    # all document-upload chunks of one sample stay in that sample's session
    du = comp[comp["attack_mode"].eq("doc_upload") & comp["model"].eq("gpt-mini")]
    assert du["session_id"].nunique() == 3
    assert (du.groupby("session_id")["item_id"].nunique() == 4 * 2 * 2).all()


def test_missing_keys_are_recorded_not_dropped():
    # document upload: the quiz bank has sample 0 only; the survey banks have samples 0-3
    rows = _run("gpt-mini", "doc_upload", "quiz-arc", [0])
    for bank in ("survey-gss", "survey-ipip", "assignment-asurre"):
        rows += _run("gpt-mini", "doc_upload", bank, range(4))
    join = join_ai_sessions(pd.DataFrame(rows))
    assert join.rows["session_id"].nunique() == 4  # samples 1-3 kept, not dropped
    assert sorted(join.missing["sample"]) == ["1", "2", "3"]
    assert set(join.missing["bank"]) == {"arc"}
    assert "doc_upload L0 / arc: samples 1,2,3" in join.summary()


def test_new_samples_from_a_later_run_join_automatically():
    # a later run over the same bank (another run_id) adds samples 1-3
    rows = []
    for bank in BANKS:
        rows += _run("gpt-mini", "doc_upload", bank, [0])
    for bank in ("survey-gss", "survey-ipip"):
        rows += _run("gpt-mini", "doc_upload", bank, range(1, 4), run_suffix="-s1-19")
    join = join_ai_sessions(pd.DataFrame(rows))
    assert join.rows["session_id"].nunique() == 4
    assert join.banks[("gpt-mini", "doc_upload", "L0")] == sorted(
        ["arc", "gss", "ipip", "asurre_nature_phd_2019"]
    )
    # samples 1-3 lack the quiz and open-text banks: recorded per bank
    assert len(join.missing) == 3 * 2


def test_same_sample_from_two_runs_is_an_error():
    rows = _run("gpt-mini", "copy_paste", "quiz-arc", [0, 1])
    rows += _run("gpt-mini", "copy_paste", "quiz-arc", [1], run_suffix="-again")
    with pytest.raises(ValueError, match="more than one run"):
        join_ai_sessions(pd.DataFrame(rows))


@pytest.mark.parametrize(
    "change",
    [
        {"model": "claude-haiku"},
        {"model": "gemini-flash"},
        {"heldout_model": True},
        {"item_id": "asurre_osmi_2017_18:q1"},
        {"split": "heldout_survey"},
    ],
)
def test_held_out_and_excluded_rows_are_refused(change):
    raw = pd.DataFrame(_run("gpt-mini", "copy_paste", "quiz-arc", [0]))
    for col, val in change.items():
        raw.loc[raw.index[0], col] = val
    if "model" in change:  # keep the session_id consistent with the model
        raw.loc[raw.index[0], "session_id"] = f"{raw.loc[raw.index[0], 'run_id']}:{val}:0"
    with pytest.raises(ValueError, match="refusing"):
        join_ai_sessions(raw)


# ------------------------------------------------------------ per-group TPR, small upload groups
def _t(who, score, cleared=False):
    import numpy as np

    from adr.replay.crossfit import Trace

    return Trace(who, 0, score, np.array([score]), np.array([1]), 1, cleared=cleared)


def test_small_upload_group_is_described_not_given_a_tpr():
    from adr.replay import crossfit

    ai = [[_t(f"ai:m:copy_paste:L0|cp{i}", 5.0)] for i in range(5)]
    ai += [[_t("ai:m:doc_upload:L0|du0", 5.0, cleared=False)]]
    mins = crossfit.MIN_GROUP_SESSIONS
    assert mins == {"doc_upload": 6}  # more than 5 upload sessions for a TPR
    size = {"m:copy_paste:L0": 5, "m:doc_upload:L0": 1}
    small = [f for f in ai if size[crossfit.ai_group(f)] < mins.get(
        crossfit.group_mode(crossfit.ai_group(f)), 0)]
    assert [crossfit.ai_group(f) for f in small] == ["m:doc_upload:L0"]  # 5 copy-paste: kept
    rows = crossfit.describe_ai(small, {"A_0.05": 2.0})
    assert rows == [
        {
            "group": "m:doc_upload:L0",
            "session": "du0",
            "score": 5.0,
            "checks": 1.0,
            "paused@A_0.05": 1.0,
            "cleared@A_0.05": 0.0,
        }
    ]


# ------------------------------------------------- --final: dropped rows never reach a session
def test_rows_dropped_by_the_filter_or_final_never_appear_in_a_composite_session():
    from adr.replay.quick_map import final_eval_rows, split_rows

    rows = []
    for bank in BANKS:
        rows += _run("gpt-mini", "copy_paste", bank, range(2))
    haiku = _run("claude-haiku", "copy_paste", "quiz-arc", range(2))
    stale = _run("gemini-flash", "copy_paste", "quiz-arc", range(2))
    for r in haiku + stale:
        r["heldout_model"] = True
    human = dict(rows[0], respondent_kind="human", model=None, session_id="pilot:p1",
                 run_id="pilot")
    table = pd.DataFrame(rows + haiku + stale + [human])

    # run_real_router: the AI-row filter (#30, H1) first, then the join on the filtered rows
    ai = table[table["respondent_kind"].eq("ai")].reset_index(drop=True)
    with pytest.raises(ValueError, match="refusing"):
        join_ai_sessions(ai)  # without --final, held-out rows are refused outright
    comp = join_ai_sessions(ai, allow_heldout=True).rows
    assert "pilot:p1" not in set(comp["session_id"])  # the human row never reaches a session
    assert (comp.groupby("session_id")["model"].nunique() == 1).all()  # never mixes models

    # --final: only the decided held-out model is evaluated; gemini-flash rows are dropped whole
    splits = split_rows(comp)
    kept = final_eval_rows(splits.heldout, "claude-haiku")
    dropped = splits.heldout.loc[splits.heldout.index.difference(kept.index)]
    assert set(dropped["model"]) == {"gemini-flash"} and len(dropped) == len(stale)
    final_sessions = set(kept.loc[kept["respondent_kind"].eq("ai"), "session_id"])
    assert final_sessions == {f"composite:claude-haiku:copy_paste:L0:{i}" for i in range(2)}
    assert not any("gemini-flash" in s for s in final_sessions)
    # and no kept session contains a dropped row
    assert set(dropped.index).isdisjoint(kept.index)


# ------------------------------------------------- attacker variants (L2 vs L2-generic)
def test_attacker_variant_from_prompt_version():
    from adr.replay.composite import attacker_variant

    assert attacker_variant("L0", "cp-v1") == "L0"
    assert attacker_variant("L2", "cp-v1") == "L2"
    assert attacker_variant("L2", "cp-v1+l2-generic-v1") == "L2g"
    assert attacker_variant("L2", "du-v1+l2-generic-v1") == "L2g"
    assert attacker_variant("L3", "cp-v1+cp-retry-v1") == "L3"  # part of L3, not a variant
    assert attacker_variant("L2", "cp-v1+new-v9") == "L2+new-v9"  # unknown: never merged
    assert attacker_variant("L1", None) == "L1"


def test_l2_and_l2_generic_never_share_a_session_or_group():
    from adr.replay.donors import group_of

    rows = []
    for bank in ("quiz-arc", "survey-gss"):
        rows += _run("gpt-mini", "copy_paste", bank, range(2), level="L2")
        rows += _run("gpt-mini", "copy_paste", bank, range(2), level="L2",
                     run_suffix="-generic", prompt_version="cp-v1+l2-generic-v1")
    comp = join_ai_sessions(pd.DataFrame(rows)).rows
    # same model, mode and sample -> two sessions per sample: L2 and L2g
    assert sorted(comp["session_id"].unique()) == [
        "composite:gpt-mini:copy_paste:L2:0", "composite:gpt-mini:copy_paste:L2:1",
        "composite:gpt-mini:copy_paste:L2g:0", "composite:gpt-mini:copy_paste:L2g:1",
    ]
    for sid, g in comp.groupby("session_id"):
        assert g["prompt_version"].nunique() == 1, sid  # never mixed
        assert g["run_id"].nunique() == 2  # quiz + survey, each from its own variant's run
    assert set(comp["attacker_level"]) == {"L2"}  # stored level unchanged (map lookups use it)
    labels = {group_of(r).label for r in comp.to_dict("records")}
    assert labels == {"ai:gpt-mini:copy_paste:L2", "ai:gpt-mini:copy_paste:L2g"}


def test_tpr_is_reported_per_attacker_variant():
    from adr.replay.crossfit import rates

    ai = [
        [_t("ai:m:copy_paste:L0|a", 5.0)],
        [_t("ai:m:copy_paste:L2|b", 1.0)],
        [_t("ai:m:copy_paste:L2g|c", 5.0)],
    ]
    r = rates([_t("h", 0.0)], ai, [0.0], {"A": 2.0})["A"]
    assert (r["tpr[m:copy_paste:L0]"], r["tpr[m:copy_paste:L2]"], r["tpr[m:copy_paste:L2g]"]) == (
        1.0, 0.0, 1.0
    )


def test_a_bank_missing_for_a_whole_group_is_recorded():
    rows = []
    for bank in BANKS:  # copy-paste has every bank
        rows += _run("gpt-mini", "copy_paste", bank, range(2))
    for bank in ("survey-gss", "survey-ipip", "assignment-asurre"):  # upload: no quiz bank yet
        rows += _run("gpt-mini", "doc_upload", bank, range(2))
    join = join_ai_sessions(pd.DataFrame(rows))
    miss = join.missing
    assert set(miss["attack_mode"]) == {"doc_upload"} and set(miss["bank"]) == {"arc"}
    assert sorted(miss["sample"]) == ["0", "1"]
    assert "doc_upload L0 / arc: samples 0,1" in join.summary()

#!/usr/bin/env bash
# Experiment freeze, Sun Nov 22: the ONE final run (AGENTS.md section 8). Do not start it without
# the team's go. It needs #50 (cache-key lock, per-job spend cap) and #51 (calibration) merged and
# the checkout at tag freeze-2026-11-26.
#
#   (a) attacker runs, samples 0-19, run ids containing "final":
#       claude-haiku (held-out model) on every bank the RQ2 composite sessions use plus the
#       held-out OSMI bank; gpt-mini on the held-out OSMI bank only. Levels L0, L2, L2-generic;
#       copy-paste and document upload, both at temperature 0.7 (the training runs' settings).
#       Each provider is its own job: ADR_JOB_ID=freeze-2026-11-26-<model>, with
#       ADR_JOB_SPEND_CAP_USD = $8 for claude-haiku (Nov 26) and $2 for gpt-mini (Nov 25).
#       The caps are fixed here and recorded in the manifest; preflight stops if they differ.
#   (b) scripts/run_real_router.py --final with the frozen config (configs/data_decisions.json
#       router_calibration = the script's defaults), floor750 pilot, all 44 honest sessions and
#       without the 4 low-effort ones, --jobs 8 --resume. Two separate evaluations, each on the
#       training outcome files plus:
#         unseen-model   claude-haiku's runs on the training banks
#         unseen-survey  gpt-mini's and claude-haiku's runs on the held-out OSMI bank
#       Each of the 4 router runs is followed by its pre-registered ablation replay
#       (<run>-style-clear, --clear-requires-style-human; no model calls): 8 router replays.
#   (c) the bootstrap CIs come with (b) (n_boot 200, as in the calibration runs), then
#       scripts/freeze_report.py writes $OUT/README.md (FPR rule, the two reports with TPR at
#       A_0.05 and A_0.01 per level and mode) and FAILS LOUDLY if a group is missing, short,
#       over-full or unexpected (in either session table), any group has synthetic or fallback
#       outcomes > 0 (or missing > 0 in unseen-model), or a threshold differs from #51's.
#
# START after 17:00 Arizona time on Sun Nov 22 (00:00 UTC Mon Nov 23): a new ISO week, so the
# full $25 weekly cap applies. Preflight refuses if the week has less than $10 (both job caps) left.
#
# FPR rule: False-flag rates come only from the 44 pilot honest sessions, with the same folds and
# calibrated thresholds as #51; the held-out OSMI bank has no human side, so no FPR is reported
# on it.
#
# Run from the root of a checkout at the tag, with data/ and cache/ pointing at the shared ones:
#   nohup caffeinate -i bash scripts/freeze_run.sh > results/freeze-2026-11-26/driver.log 2>&1 &
# DRY_RUN=1 prints every command and runs nothing (no preflight, no API calls).
# A restart resumes: finished attacker runs are skipped (their outputs still go to the router),
# router steps resume from steps/*.json.
set -euo pipefail

FREEZE=freeze-2026-11-26
OUT=results/$FREEZE
MANIFEST=$OUT/freeze_manifest.json
LOGS=$OUT/logs
HELDOUT_BANK_DIR=data/processed/osmi-heldout-final
PY=${PY:-python}
REF_DIR=${REF_DIR:-results/real-router}  # #51's runs: rq2-l012-floor750-reopen-{all,no-low-effort}
JOBS=${JOBS:-8}
# USD per job; must equal the manifest's spend_caps_usd. claude-haiku: $12 approved Nov 21,
# lowered to $8 on Nov 26 (week 48 had $13.75 left after the fourth halt; the weekly check needs
# remaining >= the sum of the caps; about $3-4 of claude-haiku work remained).
CAP_CLAUDE_HAIKU=8
# gpt-mini lowered from 10 to 2 on Nov 25 (estimate $0.55): the weekly check needs the week's
# remaining budget to cover the sum of the caps, and after the third halt week 48 had $19.62 left.
CAP_GPT_MINI=2
cap_for() { case "$1" in claude-haiku) echo "$CAP_CLAUDE_HAIKU" ;; gpt-mini) echo "$CAP_GPT_MINI" ;; esac; }
# claude-haiku copy-paste reply limit (decided Nov 21; must equal the manifest's max_tokens): it is
# billed ~190 output tokens per call beyond the reply text, so the training runs' 300 would cut
# replies off. Document upload keeps the training limits; gpt-mini is unchanged. Truncation stays
# fatal (check_run). Raised from 1024 to 4096 on Nov 23: the first launch (tag freeze-2026-11-22)
# halted with 41 of 8,000 replies cut off at 1024.
HAIKU_CP_MAX_TOKENS=4096
# claude-haiku document-upload limit, all banks (decided Nov 24; must equal the manifest's
# max_tokens, both entries): the second launch (tag freeze-2026-11-23) halted with 16 of 400
# document replies cut off at 3000. Quiz and survey banks: the larger of 8192 and twice the largest
# finished reply (2994), rounded up to a power of two, capped at the model's 128K. Written-answer
# banks (asurre-train, held-out OSMI): the same 8192, one limit for all claude-haiku document
# upload (gpt-mini's largest written-answer reply was 564). gpt-mini is unchanged.
HAIKU_DU_MAX_TOKENS=8192
# Refused questions may be at most this fraction of a run's rows (decided Nov 26; see check_run).
REFUSED_MAX_FRAC=0.01
DRY_RUN=${DRY_RUN:-0}
SAMPLES=20
TEMP=0.7

log() { echo "$(date '+%F %T') $*"; }
run() {  # run "$@", or only print it with DRY_RUN=1
  if [ "$DRY_RUN" = 1 ]; then echo "DRY: $*"; else "$@"; fi
}
runlog() {  # runlog LOGFILE CMD...: run CMD appending its output to LOGFILE (DRY_RUN: print it)
  local logfile=$1; shift
  if [ "$DRY_RUN" = 1 ]; then echo "DRY: $* >> $logfile"; else "$@" >> "$logfile" 2>&1; fi
}
fail() { log "STOP: $*"; exit 1; }

# ------------------------------------------------------------------ 0. preflight
# START TIME: after 17:00 Arizona time on Sun Nov 22 (00:00 UTC Mon Nov 23), when a new ISO week
# gives the full $25 weekly cap. The budget check below refuses to start otherwise.
if [ "$DRY_RUN" != 1 ]; then
  [ "$(git describe --exact-match --tags HEAD 2>/dev/null || true)" = "$FREEZE" ] \
    || fail "HEAD is not the $FREEZE tag"
  git fetch -q origin main || fail "git fetch origin main failed"
  git merge-base --is-ancestor HEAD origin/main || fail "tag not on main: HEAD is not in origin/main"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked files have changes"
  grep -q "job_spend_cap_usd" adr/harness/llm_client.py \
    || fail "this checkout has no per-job spend cap (#50): not starting"
  [ -f "$MANIFEST" ] || fail "no $MANIFEST (scripts/freeze_manifest.py prep)"
  caps=$("$PY" -c "import json,sys; c=json.load(open(sys.argv[1]))['spend_caps_usd']; \
print(f\"{c['claude-haiku']:g} {c['gpt-mini']:g}\")" "$MANIFEST")
  [ "$caps" = "$CAP_CLAUDE_HAIKU $CAP_GPT_MINI" ] \
    || fail "spend caps differ: driver $CAP_CLAUDE_HAIKU/$CAP_GPT_MINI, manifest $caps"
  maxtok=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['max_tokens']\
['claude-haiku']['copy_paste'])" "$MANIFEST")
  [ "$maxtok" = "$HAIKU_CP_MAX_TOKENS" ] \
    || fail "claude-haiku copy-paste max_tokens differ: driver $HAIKU_CP_MAX_TOKENS, manifest $maxtok"
  dutok=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['max_tokens']\
['claude-haiku']['doc_upload']['quiz_and_survey'])" "$MANIFEST")
  [ "$dutok" = "$HAIKU_DU_MAX_TOKENS" ] \
    || fail "claude-haiku document-upload max_tokens differ: driver $HAIKU_DU_MAX_TOKENS, manifest $dutok"
  duwtok=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['max_tokens']\
['claude-haiku']['doc_upload']['written_answers'])" "$MANIFEST")
  [ "$duwtok" = "$HAIKU_DU_MAX_TOKENS" ] \
    || fail "claude-haiku written-answer document-upload max_tokens differ: driver $HAIKU_DU_MAX_TOKENS, manifest $duwtok"
  log "preflight: this week's spend against the weekly cap"
  "$PY" scripts/freeze_manifest.py budget \
    || fail "weekly budget below the two jobs' caps (\$$CAP_CLAUDE_HAIKU + \$$CAP_GPT_MINI): start after 17:00 Arizona time Sunday (00:00 UTC Monday)"
  for who in all no-low-effort; do
    [ -f "$REF_DIR/rq2-l012-floor750-reopen-$who/metrics.json" ] \
      || fail "no #51 reference run $REF_DIR/rq2-l012-floor750-reopen-$who (set REF_DIR)"
  done
fi
[ "$DRY_RUN" = 1 ] || mkdir -p "$LOGS"  # a dry run creates no folders
log "preflight: rebuild the held-out bank (#55) and check every input against $MANIFEST"
run "$PY" scripts/build_runs_v1.py --datasets asurre --heldout-asurre-only --output "$HELDOUT_BANK_DIR"
run "$PY" scripts/freeze_manifest.py verify --manifest "$MANIFEST" --heldout-bank "$HELDOUT_BANK_DIR"

# ------------------------------------------------------------------ (a) attacker runs
# level tag -> --level and --system-variant flags
level_flags() {
  case "$1" in
    L0) echo "--level L0" ;;
    L2) echo "--level L2" ;;
    L2g) echo "--level L2 --system-variant L2-generic" ;;
  esac
}

check_run() {  # a finished run with no errors, no truncated replies and no spend-cap stop;
  # refused questions (document upload: a normally finished reply answering none of its
  # questions, refused.jsonl) pass up to REFUSED_MAX_FRAC of the run's rows (decided Nov 26)
  [ "$DRY_RUN" = 1 ] && return 0
  "$PY" scripts/freeze_manifest.py check-run --run-dir "results/runs/$1" \
    --max-refused-frac "$REFUSED_MAX_FRAC"
}

max_tokens_flag() {  # copy-paste only: claude-haiku's limit; gpt-mini keeps the training limits
  [ "$1" = claude-haiku ] && echo "--max-tokens $HAIKU_CP_MAX_TOKENS"
  return 0
}

du_max_tokens() {  # document upload: claude-haiku's limit on every bank (passed explicitly), else the training's
  case "$1:$2" in
    claude-haiku:*) echo "--max-tokens $HAIKU_DU_MAX_TOKENS" ;;
    *:quiz-*) echo "--max-tokens 3000" ;;
  esac
  return 0
}

done_already() {  # restart: skip a run that finished cleanly
  [ "$DRY_RUN" = 1 ] && return 1
  [ -f "results/outcomes/$1.parquet" ] && check_run "$1" >/dev/null 2>&1
}

attack() {  # attack MODEL MODE(cp|du) LEVEL(L0|L2|L2g) BANK; sets RID
  local model=$1 mode=$2 level=$3 bank=$4
  local tag=$level; [ "$level" = L2g ] && tag=L2-generic
  local rid="v1-final-$mode-$tag-$bank-$model-s$SAMPLES"
  RID=$rid
  if done_already "$rid"; then log "skip $rid (finished)"; return; fi
  log "run $rid"
  local flags; flags=$(level_flags "$level")
  case "$bank" in
    quiz-mmlu-200|quiz-arc-200)
      local common=(--final --assessment "data/processed/runs_v1/assessments/$bank.json"
                    --items data/processed/runs_v1/items.jsonl
                    --variants "data/processed/runs_v1/variants/$bank.jsonl"
                    --mechanisms decoy --models "$model" --samples "$SAMPLES"
                    --temperature "$TEMP" --run-id "$rid")
      if [ "$mode" = cp ]; then
        # shellcheck disable=SC2086
        runlog "$LOGS/$rid.log" "$PY" scripts/run_copy_paste.py "${common[@]}" $flags \
          $(max_tokens_flag "$model")
      else
        # shellcheck disable=SC2086
        runlog "$LOGS/$rid.log" "$PY" scripts/run_doc_upload.py "${common[@]}" $flags \
          $(du_max_tokens "$model" "$bank") --questions-per-doc 20
      fi ;;
    *)  # survey and written-answer banks, incl. the held-out OSMI bank; then the shared re-score
      # shellcheck disable=SC2086
      runlog "$LOGS/$rid.log" "$PY" scripts/run_survey_banks.py run --final --model "$model" \
        --bank "$bank" --mode "$mode" $flags --samples "$SAMPLES" --temperature "$TEMP" \
        --run-id "$rid" $([ "$mode" = cp ] && max_tokens_flag "$model") \
        $([ "$mode" = du ] && du_max_tokens "$model" "$bank")
      runlog "$LOGS/$rid.log" "$PY" scripts/run_survey_banks.py rescore --final \
        --runs "$bank=$rid" ;;
  esac
  check_run "$rid" || fail "$rid did not finish cleanly (see $LOGS/$rid.log)"
}

# outputs, also of runs skipped on a restart: unseen-model and unseen-survey inputs
HAIKU_TRAIN_RUNS=()
OSMI_RUNS=()
TRAIN_BANKS="quiz-mmlu-200 quiz-arc-200 survey-ipip survey-gss assignment-asurre-train"
HELDOUT_BANK=assignment-asurre-osmi-heldout

for model in claude-haiku gpt-mini; do
  export ADR_JOB_ID="$FREEZE-$model" ADR_JOB_SPEND_CAP_USD="$(cap_for "$model")"
  banks="$HELDOUT_BANK"
  [ "$model" = claude-haiku ] && banks="$TRAIN_BANKS $HELDOUT_BANK"
  log "job $ADR_JOB_ID (cap \$$ADR_JOB_SPEND_CAP_USD): banks $banks"
  for bank in $banks; do
    for level in L0 L2 L2g; do
      for mode in cp du; do
        attack "$model" "$mode" "$level" "$bank"
        if [ "$bank" = "$HELDOUT_BANK" ]; then OSMI_RUNS+=("results/outcomes/$RID.parquet")
        else HAIKU_TRAIN_RUNS+=("results/outcomes/$RID.parquet"); fi
      done
    done
  done
done
unset ADR_JOB_ID ADR_JOB_SPEND_CAP_USD

# ------------------------------------------------------------------ (b) + (c) the final router runs
FINAL_RUNS=("${HAIKU_TRAIN_RUNS[@]}" "${OSMI_RUNS[@]}")
log "record the ${#FINAL_RUNS[@]} final outcome files in $MANIFEST"
run "$PY" scripts/freeze_manifest.py add-final --manifest "$MANIFEST" --files "${FINAL_RUNS[@]}"
run "$PY" scripts/freeze_manifest.py verify --manifest "$MANIFEST" --heldout-bank "$HELDOUT_BANK_DIR"

# inputs from the manifest: the training outcome files, the floor750 pilot, the low-effort ids
if [ "$DRY_RUN" = 1 ]; then
  TRAINING=(TRAINING_OUTCOMES_FROM_MANIFEST); PILOT=PILOT_FROM_MANIFEST; LOW=LOW_EFFORT_FROM_MANIFEST
else
  TRAINING=()
  while IFS= read -r line; do TRAINING+=("$line"); done < <("$PY" -c \
    "import json,sys; [print(e['path']) for e in json.load(open(sys.argv[1]))['training_outcomes']]" \
    "$MANIFEST")
  PILOT=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['pilot']['path'])" "$MANIFEST")
  LOW=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['low_effort_ids']['path'])" "$MANIFEST")
fi

ROUTER=(--final --pilot-outcomes "$PILOT" --bank server/pilot_items.json --low-effort "$LOW"
        --cap-variant per_mech passive_group open_text_group no_cap
        --drop-mechanisms canary,decoy,opinion_trap,style --jobs "$JOBS" --resume
        --results-dir results)
# Each main run (#51's frozen config) is followed by its pre-registered ablation replay,
# <run>-style-clear, with --clear-requires-style-human: same inputs, no model calls
# (configs/data_decisions.json preregistered_ablations).
UNSEEN_MODEL=(); UNSEEN_SURVEY=(); UNSEEN_MODEL_ABL=(); UNSEEN_SURVEY_ABL=()
for report in unseen-model unseen-survey; do
  if [ "$report" = unseen-model ]; then AI=("${HAIKU_TRAIN_RUNS[@]}"); else AI=("${OSMI_RUNS[@]}"); fi
  for who in all no-low-effort; do
    EX=(); [ "$who" = no-low-effort ] && EX=(--exclude-sessions "$LOW")
    for variant in main style-clear; do
      rid="$FREEZE-$report-$who"; ABL=()
      [ "$variant" = style-clear ] && { rid="$rid-style-clear"; ABL=(--clear-requires-style-human); }
      [ "$DRY_RUN" = 1 ] || mkdir -p "results/real-router/$rid"
      log "router $rid (${#AI[@]} final outcome files)"
      runlog "results/real-router/$rid/run.log" "$PY" -u scripts/run_real_router.py "${ROUTER[@]}" \
        --ai-outcomes "${TRAINING[@]}" "${AI[@]}" ${EX[@]+"${EX[@]}"} ${ABL[@]+"${ABL[@]}"} \
        --run-id "$rid"
      case "$report:$variant" in
        unseen-model:main) UNSEEN_MODEL+=("results/real-router/$rid") ;;
        unseen-survey:main) UNSEEN_SURVEY+=("results/real-router/$rid") ;;
        unseen-model:style-clear) UNSEEN_MODEL_ABL+=("results/real-router/$rid") ;;
        unseen-survey:style-clear) UNSEEN_SURVEY_ABL+=("results/real-router/$rid") ;;
      esac
    done
  done
done

log "reports and post-run checks -> $OUT/README.md"
run "$PY" scripts/freeze_report.py --out "$OUT/README.md" \
  --unseen-model "${UNSEEN_MODEL[@]}" --unseen-survey "${UNSEEN_SURVEY[@]}" \
  --unseen-model-ablation "${UNSEEN_MODEL_ABL[@]}" \
  --unseen-survey-ablation "${UNSEEN_SURVEY_ABL[@]}" \
  --reference-all "$REF_DIR/rq2-l012-floor750-reopen-all" \
  --reference-no-low-effort "$REF_DIR/rq2-l012-floor750-reopen-no-low-effort" \
  || fail "POST-RUN CHECK FAILED (see $OUT/README.md): do not report these numbers"
log "done: attacker runs in results/outcomes, router runs in results/real-router/$FREEZE-unseen-*"

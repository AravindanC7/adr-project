#!/usr/bin/env bash
# Experiment freeze, Sun Nov 22: the ONE final run (AGENTS.md section 8). Do not start it without
# the team's go. It needs #50 (cache-key lock, per-job spend cap) and #51 (calibration) merged and
# the checkout at tag freeze-2026-11-22.
#
#   (a) attacker runs, samples 0-19, run ids containing "final":
#       claude-haiku (held-out model) on every bank the RQ2 composite sessions use plus the
#       held-out OSMI bank; gpt-mini on the held-out OSMI bank only. Levels L0, L2, L2-generic;
#       copy-paste and document upload, both at temperature 0.7 (the training runs' settings).
#       Each provider is its own job: ADR_JOB_ID=freeze-2026-11-22-<model>, with
#       ADR_JOB_SPEND_CAP_USD (default 10).
#   (b) scripts/run_real_router.py --final with the frozen config (configs/data_decisions.json
#       router_calibration = the script's defaults), floor750 pilot, all 44 honest sessions and
#       without the 4 low-effort ones, --jobs 8 --resume.
#   (c) the bootstrap CIs come with (b) (n_boot 200, as in the calibration runs).
#
# Run from the root of a checkout at the tag, with data/ and cache/ pointing at the shared ones:
#   nohup caffeinate -i bash scripts/freeze_run.sh > results/freeze-2026-11-22/driver.log 2>&1 &
# DRY_RUN=1 prints every command and runs nothing (no preflight, no API calls).
# A restart resumes: finished attacker runs are skipped, router steps resume from steps/*.json.
set -euo pipefail

FREEZE=freeze-2026-11-22
OUT=results/$FREEZE
MANIFEST=$OUT/freeze_manifest.json
LOGS=$OUT/logs
HELDOUT_BANK_DIR=data/processed/osmi-heldout-final
PY=${PY:-python}
JOBS=${JOBS:-8}
CAP=${ADR_JOB_SPEND_CAP_USD:-10}
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
if [ "$DRY_RUN" != 1 ]; then
  [ "$(git describe --exact-match --tags HEAD 2>/dev/null || true)" = "$FREEZE" ] \
    || fail "HEAD is not the $FREEZE tag"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked files have changes"
  grep -q "job_spend_cap_usd" adr/harness/llm_client.py \
    || fail "this checkout has no per-job spend cap (#50): not starting"
  [ -f "$MANIFEST" ] || fail "no $MANIFEST (scripts/freeze_manifest.py prep)"
fi
mkdir -p "$LOGS"
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

check_run() {  # a finished run with no errors, no truncated replies and no spend-cap stop
  [ "$DRY_RUN" = 1 ] && return 0
  "$PY" - "$1" <<'PYEOF'
import sys, yaml
from pathlib import Path
rid = sys.argv[1]
cfg = yaml.safe_load((Path("results/runs") / rid / "config.yaml").read_text())
errs = (Path("results/runs") / rid / "errors.jsonl").read_text().splitlines()
bad = {"errors": len(errs), "truncated": cfg.get("n_truncated_replies") or 0,
       "stopped_for_budget": bool(cfg.get("stopped_for_budget"))}
if any(bad.values()) or not cfg.get("finished_at"):
    sys.exit(f"{rid}: {bad}, finished_at={cfg.get('finished_at')}")
print(f"{rid}: ok ({cfg.get('n_rows')} rows)")
PYEOF
}

done_already() {  # restart: skip a run that finished cleanly
  [ "$DRY_RUN" = 1 ] && return 1
  [ -f "results/outcomes/$1.parquet" ] && check_run "$1" >/dev/null 2>&1
}

attack() {  # attack MODEL MODE(cp|du) LEVEL(L0|L2|L2g) BANK
  local model=$1 mode=$2 level=$3 bank=$4
  local tag=$level; [ "$level" = L2g ] && tag=L2-generic
  local rid="v1-final-$mode-$tag-$bank-$model-s$SAMPLES"
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
        runlog "$LOGS/$rid.log" "$PY" scripts/run_copy_paste.py "${common[@]}" $flags
      else
        # shellcheck disable=SC2086
        runlog "$LOGS/$rid.log" "$PY" scripts/run_doc_upload.py "${common[@]}" $flags \
          --max-tokens 3000 --questions-per-doc 20
      fi ;;
    *)  # survey and written-answer banks, incl. the held-out OSMI bank; then the shared re-score
      # shellcheck disable=SC2086
      runlog "$LOGS/$rid.log" "$PY" scripts/run_survey_banks.py run --final --model "$model" \
        --bank "$bank" --mode "$mode" $flags --samples "$SAMPLES" --temperature "$TEMP" \
        --run-id "$rid"
      runlog "$LOGS/$rid.log" "$PY" scripts/run_survey_banks.py rescore --final \
        --runs "$bank=$rid" ;;
  esac
  check_run "$rid" || fail "$rid did not finish cleanly (see $LOGS/$rid.log)"
  FINAL_RUNS+=("results/outcomes/$rid.parquet")
}

FINAL_RUNS=()
TRAIN_BANKS="quiz-mmlu-200 quiz-arc-200 survey-ipip survey-gss assignment-asurre-train"
HELDOUT_BANK=assignment-asurre-osmi-heldout

for model in claude-haiku gpt-mini; do
  export ADR_JOB_ID="$FREEZE-$model" ADR_JOB_SPEND_CAP_USD="$CAP"
  banks="$HELDOUT_BANK"
  [ "$model" = claude-haiku ] && banks="$TRAIN_BANKS $HELDOUT_BANK"
  log "job $ADR_JOB_ID (cap \$$CAP): banks $banks"
  for bank in $banks; do
    for level in L0 L2 L2g; do
      for mode in cp du; do
        attack "$model" "$mode" "$level" "$bank"
      done
    done
  done
done
unset ADR_JOB_ID ADR_JOB_SPEND_CAP_USD

# ------------------------------------------------------------------ (b) + (c) the final router run
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

ROUTER=(--final --ai-outcomes "${TRAINING[@]}" "${FINAL_RUNS[@]}" --pilot-outcomes "$PILOT"
        --bank server/pilot_items.json --low-effort "$LOW"
        --cap-variant per_mech passive_group open_text_group no_cap
        --drop-mechanisms canary,decoy,opinion_trap,style --jobs "$JOBS" --resume
        --results-dir results)
for who in all no-low-effort; do
  rid="$FREEZE-final-router-$who"
  EX=(); [ "$who" = no-low-effort ] && EX=(--exclude-sessions "$LOW")
  mkdir -p "results/real-router/$rid"
  log "router $rid"
  runlog "results/real-router/$rid/run.log" "$PY" -u scripts/run_real_router.py "${ROUTER[@]}" \
    ${EX[@]+"${EX[@]}"} --run-id "$rid"
done
log "done: attacker runs in results/outcomes, router runs in results/real-router/$FREEZE-final-router-*"

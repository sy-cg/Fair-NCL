#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PLAN="${PLAN:-${1:-}}"
PROCESSED_ROOT="${PROCESSED_ROOT:-processed_data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
TABLES_DIR="${TABLES_DIR:-tables}"
RUN_MECHANISM_SUMMARY="${RUN_MECHANISM_SUMMARY:-0}"
DEBUG_MISSING_FIRST="${DEBUG_MISSING_FIRST:-0}"
PHASE_OVERRIDE="${PHASE_OVERRIDE:-}"
FAIR_NCL_TMPDIR="${FAIR_NCL_TMPDIR:-$ROOT_DIR/cache/tmp}"
export FAIR_NCL_TMPDIR
export TMPDIR="$FAIR_NCL_TMPDIR"
export TEMP="$FAIR_NCL_TMPDIR"
export TMP="$FAIR_NCL_TMPDIR"

mkdir -p "$TABLES_DIR"
mkdir -p "$FAIR_NCL_TMPDIR"

if [[ -z "$PLAN" ]]; then
  echo "[ERROR] Missing jobs plan."
  echo "Usage:"
  echo "  bash resume_plan_jobs.sh experiments/taobao_light_parallel.jsonl"
  echo "or"
  echo "  PLAN=experiments/taobao_light_parallel.jsonl bash resume_plan_jobs.sh"
  exit 1
fi

if [[ ! -f "$PLAN" ]]; then
  echo "[ERROR] Plan file not found: $PLAN"
  exit 1
fi

PLAN_BASENAME="$(basename "$PLAN" .jsonl)"

echo "[INFO] Inspecting plan: $PLAN"
TOTAL="$(python -m experiments.plan_status field --plan "$PLAN" --results-root "$RESULTS_ROOT" --field total)"
DONE="$(python -m experiments.plan_status field --plan "$PLAN" --results-root "$RESULTS_ROOT" --field done)"
PHASE="$(python -m experiments.plan_status field --plan "$PLAN" --results-root "$RESULTS_ROOT" --field phase)"

if [[ -n "$PHASE_OVERRIDE" ]]; then
  PHASE="$PHASE_OVERRIDE"
fi

echo "[INFO] Completed jobs: $DONE / $TOTAL"

mapfile -t TODO_INDICES < <(python -m experiments.plan_status todo-indices --plan "$PLAN" --results-root "$RESULTS_ROOT")

if [[ "${#TODO_INDICES[@]}" -eq 0 ]]; then
  echo "[INFO] No remaining jobs. Exporting reports only."
else
  if [[ "$DEBUG_MISSING_FIRST" == "1" ]]; then
    FIRST_INDEX="${TODO_INDICES[0]}"
    echo "[INFO] Running one missing job in debug mode first: index $FIRST_INDEX"
    python run_research_pipeline.py run \
      --jobs "$PLAN" \
      --index "$FIRST_INDEX" \
      --processed-root "$PROCESSED_ROOT" \
      --results-root "$RESULTS_ROOT" \
      --debug \
      --skip-existing
  fi

  echo "[INFO] Resuming remaining jobs..."
  for idx in "${TODO_INDICES[@]}"; do
    JOB_DONE="$(python -m experiments.plan_status job-done --plan "$PLAN" --results-root "$RESULTS_ROOT" --index "$idx")"
    if [[ "$JOB_DONE" == "1" ]]; then
      echo "[INFO] Skip completed job index $idx"
      continue
    fi

    echo "[INFO] Running missing job index $idx"
    python run_research_pipeline.py run \
      --jobs "$PLAN" \
      --index "$idx" \
      --processed-root "$PROCESSED_ROOT" \
      --results-root "$RESULTS_ROOT" \
      --skip-existing
  done
fi

if [[ "$PHASE" == "comparison" || "$PHASE" == "ablation" ]]; then
  OUTPUT_XLSX="$TABLES_DIR/${PLAN_BASENAME}.xlsx"
  echo "[INFO] Exporting $PHASE report to $OUTPUT_XLSX"
  python run_research_pipeline.py summarize \
    --phase "$PHASE" \
    --jobs "$PLAN" \
    --results-root "$RESULTS_ROOT" \
    --output "$OUTPUT_XLSX"
fi

if [[ "$RUN_MECHANISM_SUMMARY" == "1" ]]; then
  OUTPUT_RQ4="$TABLES_DIR/${PLAN_BASENAME}_rq4.xlsx"
  echo "[INFO] Exporting mechanism report to $OUTPUT_RQ4"
  python run_research_pipeline.py summarize-mechanism \
    --jobs "$PLAN" \
    --results-root "$RESULTS_ROOT" \
    --output "$OUTPUT_RQ4"
fi

echo "[INFO] Resume complete for $PLAN"

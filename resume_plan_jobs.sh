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

mkdir -p "$TABLES_DIR"

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
SUMMARY_JSON="$(PLAN="$PLAN" RESULTS_ROOT="$RESULTS_ROOT" python -c 'import json, os; plan=os.environ["PLAN"]; results_root=os.environ["RESULTS_ROOT"]; jobs=[]; 
with open(plan, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        job=json.loads(line); job["_index"]=idx; path=os.path.join(results_root, job["dataset"], job["job_id"], "test_results.json"); job["_done"]=os.path.exists(path); jobs.append(job)
phase=jobs[0]["phase"] if jobs else None
print(json.dumps({"total": len(jobs), "done": sum(1 for j in jobs if j["_done"]), "todo_indices": [j["_index"] for j in jobs if not j["_done"]], "phase": phase}))'
)"

TOTAL="$(SUMMARY_JSON="$SUMMARY_JSON" python -c 'import json, os; print(json.loads(os.environ["SUMMARY_JSON"])["total"])'
)"

DONE="$(SUMMARY_JSON="$SUMMARY_JSON" python -c 'import json, os; print(json.loads(os.environ["SUMMARY_JSON"])["done"])'
)"

PHASE="$(SUMMARY_JSON="$SUMMARY_JSON" python -c 'import json, os; print(json.loads(os.environ["SUMMARY_JSON"])["phase"] or "")'
)"

if [[ -n "$PHASE_OVERRIDE" ]]; then
  PHASE="$PHASE_OVERRIDE"
fi

echo "[INFO] Completed jobs: $DONE / $TOTAL"

mapfile -t TODO_INDICES < <(SUMMARY_JSON="$SUMMARY_JSON" python -c 'import json, os; [print(idx) for idx in json.loads(os.environ["SUMMARY_JSON"])["todo_indices"]]')

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
    JOB_DONE="$(PLAN="$PLAN" RESULTS_ROOT="$RESULTS_ROOT" JOB_INDEX="$idx" python -c 'import json, os; plan=os.environ["PLAN"]; results_root=os.environ["RESULTS_ROOT"]; job_index=int(os.environ["JOB_INDEX"]); 
with open(plan, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == job_index:
            job=json.loads(line); path=os.path.join(results_root, job["dataset"], job["job_id"], "test_results.json"); print("1" if os.path.exists(path) else "0"); break'
)"
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

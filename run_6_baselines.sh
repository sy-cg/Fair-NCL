#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASETS="${DATASETS:-ml-1m lastfm-1k taobao}"
BACKBONES="${BACKBONES:-sasrec bert4rec gru4rec caser}"
SEEDS="${SEEDS:-42}"
PARAMS_FILE="${PARAMS_FILE:-configs/backbone_best.json}"
PROCESSED_ROOT="${PROCESSED_ROOT:-processed_data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
TABLES_DIR="${TABLES_DIR:-tables}"
DEBUG_FIRST="${DEBUG_FIRST:-1}"
RUN_MECHANISM_SUMMARY="${RUN_MECHANISM_SUMMARY:-1}"
PLAN_FULL="${PLAN_FULL:-$EXPERIMENTS_DIR/comparison_full.jsonl}"
PLAN_BASELINES="${PLAN_BASELINES:-$EXPERIMENTS_DIR/comparison_6baselines.jsonl}"

KEEP_METHODS="${KEEP_METHODS:-adv_debias,grl,sm_pcfr,afrl,pfrec,a_fsr}"

mkdir -p "$EXPERIMENTS_DIR" "$RESULTS_ROOT" "$TABLES_DIR"

if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "[ERROR] Params file not found: $PARAMS_FILE"
  echo "        Prepare configs/backbone_best.json first."
  exit 1
fi

for dataset in $DATASETS; do
  if [[ ! -f "$PROCESSED_ROOT/$dataset/processed_data.pkl" ]]; then
    echo "[ERROR] Missing processed dataset: $PROCESSED_ROOT/$dataset/processed_data.pkl"
    echo "        Run preprocessing first."
    exit 1
  fi
done

echo "[INFO] Building full comparison plan..."
python run_research_pipeline.py plan \
  --phase comparison \
  --datasets $DATASETS \
  --backbones $BACKBONES \
  --seeds $SEEDS \
  --params-file "$PARAMS_FILE" \
  --output "$PLAN_FULL"

echo "[INFO] Filtering plan to 6 fairness baselines: $KEEP_METHODS"
KEEP_METHODS="$KEEP_METHODS" PLAN_FULL="$PLAN_FULL" PLAN_BASELINES="$PLAN_BASELINES" python - <<'PY'
import json
import os

keep = {x.strip() for x in os.environ["KEEP_METHODS"].split(",") if x.strip()}
src = os.environ["PLAN_FULL"]
dst = os.environ["PLAN_BASELINES"]

count = 0
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        job = json.loads(line)
        if job.get("method") in keep:
            fout.write(json.dumps(job, ensure_ascii=False) + "\n")
            count += 1

print(f"[INFO] Saved {count} jobs to {dst}")
PY

JOB_COUNT=$(wc -l < "$PLAN_BASELINES")
if [[ "$JOB_COUNT" -le 0 ]]; then
  echo "[ERROR] No jobs were generated in $PLAN_BASELINES"
  exit 1
fi

echo "[INFO] Total baseline jobs: $JOB_COUNT"

if [[ "$DEBUG_FIRST" == "1" ]]; then
  echo "[INFO] Running one debug smoke test before the full batch..."
  python run_research_pipeline.py run \
    --jobs "$PLAN_BASELINES" \
    --index 0 \
    --processed-root "$PROCESSED_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --debug
fi

echo "[INFO] Running all baseline jobs..."
for ((i=0; i<JOB_COUNT; i++)); do
  echo "[INFO] Running job index $i / $((JOB_COUNT - 1))"
  python run_research_pipeline.py run \
    --jobs "$PLAN_BASELINES" \
    --index "$i" \
    --processed-root "$PROCESSED_ROOT" \
    --results-root "$RESULTS_ROOT"
done

echo "[INFO] Exporting comparison table..."
python run_research_pipeline.py summarize \
  --phase comparison \
  --jobs "$PLAN_BASELINES" \
  --results-root "$RESULTS_ROOT" \
  --output "$TABLES_DIR/comparison_6baselines.xlsx"

if [[ "$RUN_MECHANISM_SUMMARY" == "1" ]]; then
  echo "[INFO] Exporting RQ4 mechanism table..."
  python run_research_pipeline.py summarize-mechanism \
    --jobs "$PLAN_BASELINES" \
    --results-root "$RESULTS_ROOT" \
    --output "$TABLES_DIR/rq4_6baselines.xlsx"
fi

echo "[INFO] Done."
echo "[INFO] Jobs file: $PLAN_BASELINES"
echo "[INFO] Main table: $TABLES_DIR/comparison_6baselines.xlsx"
if [[ "$RUN_MECHANISM_SUMMARY" == "1" ]]; then
  echo "[INFO] Mechanism table: $TABLES_DIR/rq4_6baselines.xlsx"
fi

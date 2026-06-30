#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHASE="${PHASE:-augment}"
SEEDS="${SEEDS:-42}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments/tuning_parallel/$PHASE}"

case "$PHASE" in
  augment)
    DEFAULT_PARAMS_FILE="configs/backbone_best.json"
    ;;
  loss)
    DEFAULT_PARAMS_FILE="configs/augment_selected.json"
    ;;
  *)
    echo "[ERROR] PHASE must be augment or loss, got: $PHASE"
    exit 1
    ;;
esac

PARAMS_FILE="${PARAMS_FILE:-$DEFAULT_PARAMS_FILE}"
mkdir -p "$EXPERIMENTS_DIR"

if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "[ERROR] Params file not found: $PARAMS_FILE"
  exit 1
fi

build_lane() {
  local lane_name="$1"
  shift
  local output_path="$EXPERIMENTS_DIR/${PHASE}_${lane_name}.jsonl"

  echo "[INFO] Building $PHASE lane: $lane_name"
  python run_research_pipeline.py plan \
    --phase "$PHASE" \
    --pairs "$@" \
    --seeds $SEEDS \
    --params-file "$PARAMS_FILE" \
    --output "$output_path"
}

# Lanes are sized for one 24 GB RTX3090 plus 90 GB RAM.
# Heavy Taobao / BERT lanes are intentionally isolated; lighter pairs are bundled.
build_lane "lane01_taobao_bert4rec" \
  taobao:bert4rec

build_lane "lane02_taobao_caser" \
  taobao:caser

build_lane "lane03_taobao_sasrec_gru4rec" \
  taobao:sasrec \
  taobao:gru4rec

build_lane "lane04_lastfm_heavy" \
  lastfm-1k:bert4rec \
  lastfm-1k:caser

build_lane "lane05_lastfm_light_ml_bert" \
  lastfm-1k:sasrec \
  lastfm-1k:gru4rec \
  ml-1m:bert4rec

build_lane "lane06_ml_light" \
  ml-1m:sasrec \
  ml-1m:gru4rec \
  ml-1m:caser

echo "[INFO] Parallel tuning plans are ready under: $EXPERIMENTS_DIR"
echo "[INFO] Recommended two-process waves:"
echo "  Wave 1:"
echo "    bash resume_plan_jobs.sh $EXPERIMENTS_DIR/${PHASE}_lane01_taobao_bert4rec.jsonl"
echo "    bash resume_plan_jobs.sh $EXPERIMENTS_DIR/${PHASE}_lane06_ml_light.jsonl"
echo "  Wave 2:"
echo "    bash resume_plan_jobs.sh $EXPERIMENTS_DIR/${PHASE}_lane02_taobao_caser.jsonl"
echo "    bash resume_plan_jobs.sh $EXPERIMENTS_DIR/${PHASE}_lane05_lastfm_light_ml_bert.jsonl"
echo "  Wave 3:"
echo "    bash resume_plan_jobs.sh $EXPERIMENTS_DIR/${PHASE}_lane03_taobao_sasrec_gru4rec.jsonl"
echo "    bash resume_plan_jobs.sh $EXPERIMENTS_DIR/${PHASE}_lane04_lastfm_heavy.jsonl"

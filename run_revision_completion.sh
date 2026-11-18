#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-results_revision_kbs_v2}"
PROCESSED_ROOT="${PROCESSED_ROOT:-processed_data}"
PLAN_DIR="${PLAN_DIR:-experiments/revision_completion_plans}"
OUTPUT_DIR="${OUTPUT_DIR:-tables/supplementary/revision_completion}"

run_plan() {
  local plan="$1"
  PLAN="$plan" RESULTS_ROOT="$RESULTS_ROOT" PROCESSED_ROOT="$PROCESSED_ROOT" \
    bash resume_plan_jobs.sh
}

python -m experiments.revision_completion plan \
  --results-root "$RESULTS_ROOT" --output-dir "$PLAN_DIR"

python -m analysis.audit_revision_completion \
  --results-root "$RESULTS_ROOT" --plan-dir "$PLAN_DIR" \
  --output "$OUTPUT_DIR/coverage_audit.json"

# Validation-only selection. No test result is inspected here.
run_plan "$PLAN_DIR/mce_tune.jsonl"
run_plan "$PLAN_DIR/coserec_tune.jsonl"

python -m experiments.revision_completion finalize \
  --results-root "$RESULTS_ROOT" --output-dir "$PLAN_DIR"

python -m analysis.audit_revision_completion \
  --results-root "$RESULTS_ROOT" --plan-dir "$PLAN_DIR" \
  --output "$OUTPUT_DIR/coverage_audit.json"

for plan in \
  factorial.jsonl \
  random_pool.jsonl \
  sensitivity.jsonl \
  mce_final.jsonl \
  coserec_final.jsonl
do
  run_plan "$PLAN_DIR/$plan"
done

python -m analysis.export_revision_mechanisms \
  --results-root "$RESULTS_ROOT" --processed-root "$PROCESSED_ROOT" \
  --methods fair_ncl baseline

python run_research_pipeline.py summarize-mechanism \
  --jobs experiments/revision_supplementary_final_valid.jsonl \
  --results-root "$RESULTS_ROOT" --phase supplementary \
  --datasets ml-1m taobao --backbones all \
  --output "$OUTPUT_DIR/rq4_mechanism.xlsx"

python tables/make_rq4_mechanism_figures.py \
  --input "$OUTPUT_DIR/rq4_mechanism.xlsx" \
  --sheet rq4_paper_table \
  --output-dir figures/rq4 \
  --seed-count 5

python -m analysis.skew_diagnostics \
  --processed-root "$PROCESSED_ROOT" --datasets ml-1m taobao \
  --output-root tables/supplementary/skew --permutations 20

python -m analysis.report_revision_completion \
  --results-root "$RESULTS_ROOT" \
  --plans \
    "$PLAN_DIR/factorial.jsonl" \
    "$PLAN_DIR/random_pool.jsonl" \
    "$PLAN_DIR/sensitivity.jsonl" \
    "$PLAN_DIR/mce_final.jsonl" \
    "$PLAN_DIR/coserec_final.jsonl" \
  --output-dir "$OUTPUT_DIR"

python -m analysis.audit_revision_completion \
  --results-root "$RESULTS_ROOT" --plan-dir "$PLAN_DIR" \
  --output "$OUTPUT_DIR/coverage_audit.json" \
  --require-complete-plans

echo "Revision experiments and reports completed."

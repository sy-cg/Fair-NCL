# Supplementary experiments for the KBS revision

This workflow implements the agreed revision scope: ML-1M and Taobao only,
five final random seeds (`42`, `2025`, `2026`, `2027`, `2028`), and no new
dataset. Existing tuned parameters are the center of a compact validation-only
audit. Test results are never used to select a configuration.

The revision parameter source is `configs/revision_base_params.json`. It keeps
the submitted settings from `configs/loss_selected.json` for BERT4Rec, GRU4Rec,
and Caser, and uses the post-method-adjustment settings from
`configs/loss_ml_taobao_sasrec.json` for ML-1M/Taobao SASRec. The selected
augmentation settings in `configs/augment_taobao_sasrec.json` are already
contained in the latter loss-stage file and are not overlaid a second time.

## 1. Generate the fixed plans

```bash
python -m experiments.specs \
  --phase supplementary \
  --params-file configs/revision_base_params.json \
  --output experiments/revision_supplementary_old_params.jsonl

python -m experiments.specs \
  --phase validation_audit \
  --params-file configs/revision_base_params.json \
  --output experiments/revision_validation_audit.jsonl
```

The first plan is a 540-job pre-selection template used to verify the final
matrix; do not execute it as the final experiment plan. The audit contains
1,020 jobs: Fair-NCL,
six fairness-aware baselines, and the MCE controls each receive five compact
candidates over three selection seeds. The NCL-SR-style MCE control covers all
four backbones; FairAug+MCE follows its final SASRec/BERT4Rec scope.

For Fair-NCL, candidate `c0` is the submitted dataset--backbone setting. The
other four candidates use the two remaining values from each declared grid:
`epsilon` in `{0.5, 1.0, 2.0}` and `utility_beta` in `{0.2, 0.3, 0.4}`, with
`utility_alpha = 1 - utility_beta`. The augmentation ratio and geometry-loss
weights remain fixed at the submitted center during this compact audit.

## 2. Produce the matched validation reference

The validation selector needs the same three-seed backbone reference. Generate
24 dedicated validation-only jobs:

```bash
python -m experiments.specs \
  --phase validation_reference \
  --params-file configs/backbone_best.json \
  --output experiments/revision_baseline_reference.jsonl
```

Run the reference first, then the compact audit. Both phases are validation
only: they write an explicit completion marker but never evaluate the test
split. The resume script skips a job only when its matching artifact exists.
AFRL and PFRec load the reference backbone with the same dataset, architecture,
and seed before freezing it. Missing, ambiguous, or incompatible reference
checkpoints are hard errors; each repaired run writes
`backbone_initialization.json` with the exact source job.

```bash
export REV_RESULTS=results_revision_kbs_v2

RESULTS_ROOT="$REV_RESULTS" bash resume_plan_jobs.sh experiments/revision_baseline_reference.jsonl
RESULTS_ROOT="$REV_RESULTS" bash resume_plan_jobs.sh experiments/revision_validation_audit.jsonl
```

Use a clean results root. Do not merge jobs generated from the earlier
`loss_selected.json`-centered audit into this directory.

## 3. Select parameters using validation data only

```bash
python -m analysis.select_validation_configs \
  --results-root "$REV_RESULTS" \
  --jobs experiments/revision_validation_audit.jsonl \
  --utility-tolerance 0.15 \
  --output configs/revision_method_overrides.json \
  --summary tables/supplementary/validation_selection.csv \
  --candidate-summary tables/supplementary/validation_candidates.csv
```

The selector fails instead of silently proceeding if any required selection
seed or matched baseline reference is missing. A candidate is utility-feasible
when its paired seed-level NDCG@10 retention satisfies
`mean retention - one standard deviation >= 0.85`, corresponding to the 15%
loss budget. The disadvantaged group for each attribute is fixed from the
matched backbone rather than redefined per candidate. Among feasible candidates,
selection first keeps the validation Pareto frontier over overall NDCG@10,
conservative disadvantaged-group benefit, and worst-attribute normalized-gap
risk. It then maximizes the minimum conservative benefit across gender and age,
followed by minimizing the maximum gap risk. The plan check rejects missing,
unexpected, or duplicate audit jobs before selection. `validation_candidates.csv`
reports all candidates, seed-level dispersion, feasibility diagnostics, and
validation Pareto membership rather than only the winning configuration.

### Repair existing AFRL/PFRec validation runs

Plans generated after the frozen-backbone repair contain
`frozen_backbone_init=validation_reference`, so their job IDs differ from the
invalid runs that froze a random backbone. Before resuming the regenerated
1,020-job plan in an existing results root, move only the old AFRL/PFRec audit
runs into a recoverable backup directory:

```bash
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(os.environ["REV_RESULTS"]).resolve()
backup = root.parent / f"{root.name}_random_frozen_backup"
moved = 0
for job_path in root.glob("*/*/job.json"):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("phase") != "validation_audit" or job.get("method") not in {"afrl", "pfrec"}:
        continue
    run_dir = job_path.parent
    target = backup / job["dataset"] / run_dir.name
    if target.exists():
        raise FileExistsError(f"Backup target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    moved += 1
print(f"Moved {moved} old AFRL/PFRec runs to {backup}")
PY

python -m experiments.specs \
  --phase validation_audit \
  --params-file configs/revision_base_params.json \
  --output experiments/revision_validation_audit.jsonl

RESULTS_ROOT="$REV_RESULTS" bash resume_plan_jobs.sh experiments/revision_validation_audit.jsonl
```

The unchanged 780 audit jobs retain their IDs and are skipped; only the 240
repaired AFRL/PFRec jobs run again. Afterward, rerun the 15% selector from the
previous section.

### Caser-MCE low-weight stability diagnostic

The diagnostic plan contains eight validation-only jobs: ML-1M and Taobao,
Caser, seed 42, and matched uniform/alignment weights in
`{0.01, 0.05, 0.10, 0.25}`. It is a failure-isolation experiment, not a source
of final test results.

```bash
python -m experiments.specs \
  --phase mce_stability \
  --params-file configs/revision_base_params.json \
  --output experiments/revision_caser_mce_stability.jsonl

RESULTS_ROOT="$REV_RESULTS" bash resume_plan_jobs.sh experiments/revision_caser_mce_stability.jsonl
```

Summarize seed-42 utility retention and non-finite optimizer updates:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REV_RESULTS"])
references = {}
diagnostics = []
for job_path in root.glob("*/*/job.json"):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    key = (job.get("dataset"), job.get("backbone"), int(job.get("seed", -1)))
    if job.get("phase") == "validation_reference" and job.get("method") == "baseline":
        result = json.loads(job_path.with_name("validation_results.json").read_text(encoding="utf-8"))
        references[key] = float(result["utility"]["NDCG@10"])
    elif job.get("phase") == "mce_stability":
        diagnostics.append((job, job_path.parent))

print("dataset\tweight\tNDCG@10\tretention\tskipped_nonfinite\tdiagnostic_pass")
for job, run_dir in sorted(diagnostics, key=lambda item: (
    item[0]["dataset"], float(item[0]["params"]["mce_uniform_weight"])
)):
    key = (job["dataset"], job["backbone"], int(job["seed"]))
    result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    ndcg = float(result["utility"]["NDCG@10"])
    retention = ndcg / references[key]
    skipped = sum(int(value) for value in history.get("skipped_nonfinite_updates", []))
    passed = retention >= 0.85 and skipped == 0
    print(job["dataset"], job["params"]["mce_uniform_weight"],
          f"{ndcg:.6f}", f"{retention:.4f}", skipped, passed, sep="\t")
PY
```

If at least one low weight passes on both datasets, repeat this diagnostic with
`--seeds 42 2025 2026` before promoting a weight into the final override. A
single diagnostic seed must not be used directly for final model selection.

## 4. Run the archived final five-seed plan

AFRL/PFRec also require matched references for final seeds 2027 and 2028. Run
the provided 16-job extension before the final plan:

```bash
python -m experiments.specs \
  --phase validation_reference \
  --seeds 2027 2028 \
  --params-file configs/backbone_best.json \
  --output experiments/revision_frozen_backbone_reference_extra.jsonl

RESULTS_ROOT="$REV_RESULTS" bash resume_plan_jobs.sh experiments/revision_frozen_backbone_reference_extra.jsonl
```

The authoritative final plan is
`experiments/revision_supplementary_final_extended_595.jsonl`. It contains
all 595 archived final jobs, including the subsequently completed BERT4Rec
component controls and GRU4Rec/Caser FairAug+MCE controls. It was reconstructed
from the final `job.json` records and checked against the complete five-seed
summary with `python analysis/export_archived_final_runs.py`. The earlier
`revision_supplementary_final_valid.jsonl` contains only 535 of these jobs;
`experiments.specs --phase supplementary` generates a candidate plan, not the
complete archived 595-job plan. Use the supplied frozen plan for reproduction:

```bash
RESULTS_ROOT="$REV_RESULTS" bash resume_plan_jobs.sh experiments/revision_supplementary_final_extended_595.jsonl
```

The final matrix contains:

- all four backbones: backbone, Fair-NCL, VICReg-only, random augmentation,
  similarity augmentation, and the MCE control;
- all four backbones: AdvDebias, GRL, SM-PCFR, AFRL, PFRec, and A-FSR;
- all four backbones: FairAug with the MCE objective;
- SASRec and BERT4Rec: skew-only sampling, no-geometry, no-augmented-CE, and
  submitted raw-skew controls;
- NCL-SR-ID has seven final dataset--backbone groups; ML-1M/Caser did not
  qualify for final evaluation.

All final jobs export per-user results, named gender and age groups,
gender-by-age intersectional groups, augmentation diagnostics, resource-build
time, training/evaluation time, parameter count, and peak CUDA memory.

## 5. Generate skew and statistical reports

```bash
python -m analysis.skew_diagnostics \
  --processed-root processed_data \
  --datasets ml-1m taobao \
  --output-root tables/supplementary/skew

python -m analysis.run_supplementary_analysis \
  --results-root "$REV_RESULTS" \
  --jobs experiments/revision_supplementary_final_extended_595.jsonl \
  --output-dir tables/supplementary/final_5seed_extended \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 42
```

The report contains five-seed mean and standard deviation, group direction,
group counts and per-group changes, paired hierarchical bootstrap intervals
for HR/NDCG at both @10 and @20, exact paired seed-level sign-flip tests, family-wise
Holm adjustments, utility--fairness Pareto membership, augmentation/resource
summaries, and a strict manifest of missing or unexpected jobs. Separate test
Pareto frontiers use marginal-group worst utility and intersectional worst-group
utility, so intersectional evidence is not hidden inside an averaged gap.

With five seeds, the smallest possible two-sided exact sign-flip p-value is
0.0625. Therefore seed-level p-values are reported as robustness evidence, not
used alone to declare significance. The report separately marks bootstrap
support, family-wise Holm support, and their conservative conjunction; table
bolding should use only the last field.

## Reproducibility check

```bash
python -m unittest discover -s tests -v
python -m py_compile \
  config.py fairness/skew.py fairness/augmentation_diagnostics.py \
  methods/augmentations.py methods/mce.py methods/registry.py \
  methods/wrappers.py methods/trainer.py evaluation/full_ranking.py \
  experiments/specs.py experiments/filter_plan.py experiments/runner.py \
  analysis/supplementary_statistics.py \
  analysis/run_supplementary_analysis.py \
  analysis/select_validation_configs.py analysis/skew_diagnostics.py
```

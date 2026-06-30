# Fair-NCL Refactor Guide

This guide defines the target project structure for the three-dataset Fair-NCL
study and records the design decisions that should remain stable across
experiments.

## 1. Dataset Layer

Use `data/unified_preprocessing.py` to convert raw datasets into one common
format:

- `ml-1m`: true timestamp sequence, positive signal is rating >= 4.
- `taobao`: true timestamp sequence, positive signal is click by default.
- `lastfm-1k`: true timestamp listening sequence. By default, artists are used
  as items to reduce sparsity; track-level processing can be enabled for
  sensitivity analysis.

All processed datasets are written to:

```text
processed_data/<dataset>/processed_data.pkl
```

Convenience commands:

```powershell
python run_research_pipeline.py preprocess --datasets ml-1m --max-rows 2000 --output-dir cache/preprocess_smoke
python run_research_pipeline.py preprocess --datasets all --output-dir processed_data
python run_research_pipeline.py preprocess --datasets lastfm-1k --lastfm-item-level artist --output-dir processed_data
```

The processed pickle follows the existing loader contract:

```text
train_data / val_data / test_data:
  user_id, input_seq, target, gender, age_group
```

Item ids are always `1..num_items`; `0` is reserved for padding.

Default preprocessing follows the common sequential recommendation protocol used
in recent top-conference implementations:

- Convert all datasets to implicit positive interactions.
- Apply iterative 5-core user/item filtering.
- Sort each user's interactions chronologically.
- Use leave-one-out splitting: all but the last two interactions for training,
  the penultimate interaction for validation, and the last interaction for test.
- Build item/user id mappings only after filtering, with padding id reserved.
- Estimate fairness-related statistics from training interactions only.

## 2. Experiment Order

The standard experiment order is:

1. Backbone tuning: search `learning_rate`, `hidden_units`, and `dropout_rate` for every `dataset x backbone` combination.
2. Augmentation tuning: search `epsilon`, `augment_ratio`, and `utility_alpha` for every `dataset x backbone` combination, while passing the tuned backbone parameters through `--params-file` or `--params-json`.
3. Loss tuning: search `fair_ncl_align_weight`, `fair_ncl_var_weight`, and `fair_ncl_cov_weight` for every `dataset x backbone` combination, again using the fixed backbone and augmentation parameters from the previous stages.
4. Ablation: run `SASRec` on all three datasets to isolate semantic-aware sampling, skew-aware sampling, controlled augmentation, and non-contrastive losses. Ablation jobs do not reuse loss-stage results because final `fair_ncl` uses the semantic-hybrid framework.
5. Comparison: run all four backbones on all three datasets against backbone baseline, Adv-Debias/GRL, SM/PCFR, AFRL, PFRec, and A-FSR. The `SASRec` `baseline` and `fair_ncl` jobs reuse the matching `ablation` results.

Generate plans with:

```powershell
python run_research_pipeline.py plan --phase backbone --output experiments/backbone.jsonl
python run_research_pipeline.py plan --phase augment --datasets all --backbones all --params-file configs/backbone_best.json --output experiments/augment.jsonl
python run_research_pipeline.py plan --phase loss --datasets all --backbones all --params-file configs/augment_selected.json --output experiments/loss.jsonl
python run_research_pipeline.py plan --phase ablation --params-file configs/loss_selected.json --output experiments/ablation.jsonl
python run_research_pipeline.py plan --phase comparison --params-file configs/loss_selected.json --output experiments/comparison.jsonl
python run_research_pipeline.py plan --phase augment --pairs taobao:bert4rec ml-1m:sasrec --params-file configs/backbone_best.json --output experiments/augment_selected_pairs.jsonl

# Direct module entry is also supported:
python -m experiments.specs --phase backbone --output experiments/backbone.jsonl
python -m experiments.specs --phase augment --datasets all --backbones all --params-file configs/backbone_best.json --output experiments/augment.jsonl
python -m experiments.specs --phase loss --datasets all --backbones all --params-file configs/augment_selected.json --output experiments/loss.jsonl
python -m experiments.specs --phase ablation --params-file configs/loss_selected.json --output experiments/ablation.jsonl
python -m experiments.specs --phase comparison --params-file configs/loss_selected.json --output experiments/comparison.jsonl
python run_research_pipeline.py run --jobs experiments/comparison.jsonl --index 0 --processed-root processed_data --results-root results
```

When tuning on a shared GPU, prefer `--pairs dataset:backbone` over a large
Cartesian matrix. This allows one plan file to contain only the exact
combinations assigned to a parallel lane.

The fixed parameter file should contain the exact JSON object used by the
current stage. A flat JSON object is still supported for SASRec-only phases,
for example:

```json
{
  "learning_rate": 0.001,
  "hidden_units": 128,
  "dropout_rate": 0.2,
  "epsilon": 1.0,
  "augment_ratio": 0.2,
  "utility_alpha": 0.7,
  "utility_beta": 0.3,
  "fair_ncl_align_weight": 1.0,
  "fair_ncl_var_weight": 1.0,
  "fair_ncl_cov_weight": 0.04
}
```

For multi-dataset, multi-backbone tuning and comparison, use a scoped JSON file
so each `dataset x backbone` combination can keep its own tuned parameters:

```json
{
  "__default__": {
    "epsilon": 1.0,
    "augment_ratio": 0.2,
    "utility_alpha": 0.7,
    "utility_beta": 0.3,
    "fair_ncl_align_weight": 1.0,
    "fair_ncl_var_weight": 1.0,
    "fair_ncl_cov_weight": 0.04
  },
  "__dataset_backbones__": {
    "ml-1m": {
      "sasrec": {
        "learning_rate": 0.001,
        "hidden_units": 128,
        "dropout_rate": 0.2
      },
      "bert4rec": {
        "learning_rate": 0.0005,
        "hidden_units": 128,
        "dropout_rate": 0.1
      }
    },
    "lastfm-1k": {
      "sasrec": {
        "learning_rate": 0.0005,
        "hidden_units": 64,
        "dropout_rate": 0.2
      }
    }
  }
}
```

The planner resolves `__default__`, `__datasets__`, `__backbones__`, and
`__dataset_backbones__` into a flat parameter set for each job.
If `--params-file` and `--params-json` are used together, the nested scopes are
merged recursively, so you can override only one backbone or one shared field.
For example, `configs/loss_selected.json` is expected to carry the selected
backbone, augmentation, and loss parameters. Ablation jobs are generated as
fresh jobs; comparison jobs may reuse the matching SASRec ablation outputs for
`baseline` and `fair_ncl`.

## 2.1 Parallel Tuning on One RTX3090

Use the helper script to build resource-aware tuning lanes:

```bash
PHASE=augment bash plan_parallel_tuning_lanes.sh
PHASE=loss bash plan_parallel_tuning_lanes.sh
```

The script creates six plan files under:

```text
experiments/tuning_parallel/<phase>/
```

The default grouping isolates Taobao/BERT4Rec and Taobao/Caser, bundles the
remaining Taobao pairs, and groups lighter LastFM / ML-1M pairs together. For a
24 GB RTX3090 with 90 GB RAM, the recommended starting point is two concurrent
processes:

```bash
bash resume_plan_jobs.sh experiments/tuning_parallel/augment/augment_lane01_taobao_bert4rec.jsonl
bash resume_plan_jobs.sh experiments/tuning_parallel/augment/augment_lane06_ml_light.jsonl
```

After Wave 1 completes, continue with the Wave 2 and Wave 3 commands printed by
the helper script. A third concurrent lane is only recommended after confirming
that system memory and GPU memory still have comfortable headroom.

If the project is continuing from the common bootstrap state where
`backbone` has only finished on `ml-1m x {sasrec, bert4rec, gru4rec, caser}`
and `augment/loss` have only finished on `ml-1m x sasrec`, use:

```bash
PHASE=backbone bash plan_remaining_tuning_after_bootstrap.sh
PHASE=augment bash plan_remaining_tuning_after_bootstrap.sh
PHASE=loss bash plan_remaining_tuning_after_bootstrap.sh
```

Run these phases sequentially. After the remaining backbone plans finish,
append the selected winners into `configs/backbone_best.json`; only then build
the remaining augment plans. After the remaining augment plans finish, append
their selected winners into `configs/augment_selected.json`; only then build
the remaining loss plans.

The planner now validates this dependency chain. If a requested pair lacks the
required upstream selected parameters, plan generation stops immediately instead
of silently falling back to default hyperparameters.

For tuning jobs, the runner automatically disables RQ4 mechanism export unless
`export_mechanism_analysis` is explicitly included in the job parameters. This
removes a costly post-training analysis step without changing model selection or
reported tuning metrics. `eval_interval` is now honored by the shared trainer,
but it remains `1` by default; increasing it is an optional coarse-search speed
knob and changes validation cadence, so it should not be used for final reported
tuning runs unless that protocol is declared in advance.

## 3. Baseline Policy

Backbone baselines:

- SASRec
- BERT4Rec
- GRU4Rec
- Caser

Fairness baselines:

- Adv-Debias / GRL
- SM / PCFR
- AFRL
- PFRec
- A-FSR

Ablation methods:

- `ncl_only`: recommendation plus non-contrastive losses using two dropout views.
- `random_aug`: Fair-NCL objective with random local item replacement.
- `similarity_aug`: Fair-NCL objective with similarity-only replacement.
- `fair_ncl`: semantic-hybrid controlled perturbation using train-only co-occurrence recall, lightweight item-text semantic reranking, and train-only item bias.
- `baseline`: plain tuned backbone baseline used as the ablation reference and the comparison reference.
- `wo_fairness_sampling`: semantic-hybrid replacement without skew-aware sampling.
- `wo_semantic_sampling`: original co-occurrence Fair-NCL sampling with `utility_alpha=0`, removing the semantic-aware sampling term.
- `wo_alignment`: Fair-NCL with `fair_ncl_align_weight=0`.
- `wo_variance`: Fair-NCL with `fair_ncl_var_weight=0`.
- `wo_covariance`: Fair-NCL with `fair_ncl_cov_weight=0`.
- `wo_augmented_ce`: Fair-NCL with `fair_ncl_aug_rec_weight=0`.
- `random_low_skew`: random replacement from a global low-skew item pool.
- `high_skew`: local co-occurrence replacement biased toward higher-skew candidates.

The ablation planner accepts `--ablation-methods` to control the generated
variants. Presets are `legacy`, `requested`/`new`, `main`, `semantic_hybrid`,
`all`/`full`, and `all_with_experimental`.
Use `run --skip-existing` or `resume_plan_jobs.sh` to avoid rerunning completed
jobs.

Preferred implementation policy:

- Keep each baseline behind a method registry or adapter.
- Do not mix method-specific training logic into the base model classes.
- Use the same data split, negative sampling, seeds, early stopping criterion,
  and evaluation candidates for every method.
- Sensitive attributes may be used during training by fairness methods. Inference
  should consume only the user sequence. The active training attributes are
  `gender` and `age_group`.
- The `loss` phase saves the full parameter set together with the result files.
  Later phases first try the deterministic source `job_id`, then fall back to
  matching the saved `job.json` contents when the parameter set is identical.

## 4. Evaluation Protocol

Main recommendation metrics:

- HitRate@10
- NDCG@10

Main fairness metrics:

- HitRate@10 Gap across sensitive groups
- NDCG@10 Gap across sensitive groups

Recommended K values:

```text
K = 5, 10, 20
```

Primary reporting table:

```text
HitRate@10
NDCG@10
Gender HitRate@10 gap / NDCG@10 gap
Age HitRate@10 gap / NDCG@10 gap
```

Do not report logit-mean demographic parity as the primary fairness metric for
this project, because the paper defines user-side utility parity rather than
selection-rate demographic parity.

## 5. Reporting Layer

Two post-processing commands are responsible for paper assets:

```powershell
python run_research_pipeline.py summarize --phase ablation --jobs experiments/ablation.jsonl --results-root results --output tables/ablation.xlsx
python run_research_pipeline.py summarize --phase comparison --jobs experiments/comparison.jsonl --results-root results --output tables/comparison.xlsx
python run_research_pipeline.py summarize-mechanism --jobs experiments/comparison.jsonl --results-root results --output tables/rq4_mechanism.xlsx
python run_research_pipeline.py plot --phase backbone --jobs experiments/backbone.jsonl --results-root results --output-dir figures/backbone
python run_research_pipeline.py plot --phase augment --jobs experiments/augment.jsonl --results-root results --output-dir figures/augment
python run_research_pipeline.py plot --phase loss --jobs experiments/loss.jsonl --results-root results --output-dir figures/loss
```

The Excel workbook exported by `summarize` contains:

- `raw_results`: one row per job with metadata, parameters, and selected metrics.
- `summary_long`: mean/std/count aggregated across seeds for each dataset/backbone/method.
- One paper-style sheet per `dataset/backbone` combination, with formatted `mean +/- std` cells.

For multi-seed results split across several JSONL files, use the dedicated
post-processing script:

```powershell
python analysis/process_multiseed_results.py --phase comparison --jobs experiments/comparison_ml_taobao.jsonl experiments/comparison_ml_taobao_2025_2026.jsonl --results-root results --datasets ml-1m taobao --seeds 42 2025 2026 --output tables/comparison_ml_taobao_3seed.xlsx
python analysis/process_multiseed_results.py --phase ablation --jobs experiments/ablation.jsonl experiments/<ablation_2025_2026>.jsonl --results-root results --datasets ml-1m taobao --backbones sasrec --seeds 42 2025 2026 --output tables/ablation_ml_taobao_3seed.xlsx
```

The multi-seed workbook adds seed-completeness checks, per-seed primary metrics,
method rankings, and deltas against `baseline` and `fair_ncl`.

The parameter-analysis exporter uses a long-form schema:

```text
phase, dataset, backbone, method, seed, param_name, param_value, metric_name, metric_value
```

It then averages over seeds and the remaining grid dimensions to produce the
parameter-sensitivity curves used in the paper.

By default, the plot command writes CSV data and PNG/PDF/SVG figures under:

```text
figures/parameter_analysis/<phase>/
```

Each completed job now also exports RQ4-oriented mechanism-analysis artifacts
under its result directory:

```text
results/<dataset>/<job_id>/mechanism_test_summary.json
results/<dataset>/<job_id>/mechanism_test_repr_sample.npz
```

The mechanism summary includes:

- raw/fair representation geometry (`mean_norm`, `variance_loss`, `covariance_loss`)
- representation shift statistics between encoded and fair representations
- attribute-level linear probe leakage for `gender` and `age_group`
- group-centroid separability statistics before and after the fairness transform

## 6. Fair-NCL Design

Fair-NCL should be implemented as:

```text
x -> controlled_augment(x) = x'
z = encoder(x)
z' = encoder(x')
L = L_rec(x, y)
  + eta * L_rec(x', y)
  + lambda_align * L_align(z, z')
  + lambda_var * L_var(z, z')
  + lambda_cov * L_cov(z, z')
```

Controlled augmentation is exponential-mechanism-inspired, not a formal
end-to-end differential privacy guarantee.

## 7. Migration Notes

The old scripts can remain while the refactor progresses. New code should be
migrated into these layers:

- `data/`: raw-to-unified dataset builders and PyTorch datasets.
- `models/`: backbone architectures only.
- `methods/` or method-specific trainers: Fair-NCL, NCL ablations, Adv-Debias/GRL,
  SM/PCFR, AFRL, PFRec, A-FSR.
- `evaluation/`: ranking and group utility metrics.
- `experiments/`: job planning and phase-specific orchestration.

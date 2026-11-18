# Fair-NCL

Official research code for **Fair-NCL: Fairness-Aware Non-Contrastive Learning
with Controlled Augmentation for Sequential Recommendation**.

The reproducible pipeline uses one entry point, `run_research_pipeline.py`, and
one JSONL job definition per run. Legacy scripts remain in the repository for
historical compatibility but are not used to reproduce the paper or revision.
Start with [the end-to-end reproduction guide](docs/REPRODUCIBILITY.md) for
dataset acquisition, checksums, environment records, the frozen final plan,
table-generation commands, and expected outputs. The
[baseline provenance table](docs/BASELINE_PROVENANCE.md) distinguishes local
adaptations from the cited original systems.


## Environment

For a compatible fresh setup, use Python 3.10 with a CUDA-compatible PyTorch
build, then install:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

PyTorch is intentionally installed separately because its wheel must match the
server CUDA runtime. See [docs/AUTODL_SETUP.md](docs/AUTODL_SETUP.md) for the
server setup and verification commands. Every run writes
`run_provenance.json`, including the exact Python/package versions, CUDA/cuDNN,
GPU model, dataset scale, preprocessing metadata, seed, and Git commit when the
workspace is a valid Git checkout.
The archived final runs used more than one Python/PyTorch/CUDA combination;
their recorded per-run versions are in
`els-cas-templates/source_data/environment/final_run_environment.csv`.
Historical versions of some secondary packages and the code commit were not
recorded, so `requirements.txt` is not a bitwise historical environment lock.

## Data and preprocessing

The revision uses only ML-1M and the public Alimama/Taobao advertising dataset.
Acquire the original files from the sources and follow the checksum and
placement instructions in [the reproduction guide](docs/REPRODUCIBILITY.md).
Place the files under:

- `data/ml-1m/`: `ratings.dat`, `users.dat`, `movies.dat`
- `data/Taobao/`: `raw_sample.csv`, `user_profile.csv`, `ad_feature.csv`

Then run the deterministic unified preprocessing path:

```bash
python run_research_pipeline.py preprocess \
  --datasets ml-1m taobao \
  --output-dir processed_data
```

The processed artifacts include the split metadata and dataset summary. The
pipeline uses chronological leave-two-out validation/test targets and computes
all item-skew resources from training events only.

## KBS revision experiments

The complete fixed workflow, task counts, parameter-selection rule, execution
order, and report-generation commands are in
[docs/SUPPLEMENTARY_EXPERIMENTS.md](docs/SUPPLEMENTARY_EXPERIMENTS.md).
The reviewer-comment coverage and remaining interpretation risks are recorded
in [docs/SUPPLEMENTARY_EXPERIMENT_AUDIT.md](docs/SUPPLEMENTARY_EXPERIMENT_AUDIT.md).

The required order is:

1. run the three-seed validation-only backbone references;
2. run the shared final validation-only parameter audit;
3. select configurations under the predefined 15% paired conservative validation utility constraint;
4. run the final five-seed test experiments;
5. generate skew, group, uncertainty, Pareto, and resource reports.

The frozen plan matching the 595 original archived final runs is
`experiments/revision_supplementary_final_extended_595.jsonl` (595 jobs,
119 complete groups). The older
`experiments/revision_supplementary_final_valid.jsonl` is a 535-job subset.
Use the frozen plan for the final run and strict aggregate report; do not
replace it with a newly generated candidate plan.

The completed CoSeRec, component, sensitivity, and SASRec MCE additions have
separate plans and source mappings in the reproduction guide. Equal candidate
and seed counts in the final validation audit do not imply equal total search
budgets: Fair-NCL also received the disclosed preliminary searches.

`configs/revision_base_params.json` is the fixed parameter provenance for the
revision. It keeps the submitted per-backbone settings from
`configs/loss_selected.json` and replaces ML-1M/Taobao SASRec with the
post-method-adjustment settings from `configs/loss_ml_taobao_sasrec.json`.
The validation reference is generated separately from
`configs/backbone_best.json` so it contains backbone parameters only.

Validation-only jobs deliberately do not evaluate the test set. Their
`test_results.json` contains an explicit `not_run_by_protocol` marker so the
resume script can detect completion without exposing test metrics.

## Main outputs

Each completed final run under
`results_revision_kbs_v2/<dataset>/<job_id>/` contains:

- `job.json`: method, backbone, seed, and explicit parameter overrides;
- `effective_config.json`: all resolved settings in newly executed runs
  (not available for the archived historical runs);
- `run_provenance.json`: environment, hardware, code, data, and split provenance;
- `validation_results.json`: metrics from the HR@10-selected checkpoint;
- `test_results.json`: utility, named group/intersectional metrics, per-user
  ranks, augmentation diagnostics, runtime, and peak CUDA memory;
- `resource_diagnostics.json`: skew/candidate construction time, support, pool,
  and cache diagnostics;
- `history.json` and `best.pt`: optimization history and checkpoint, both directly
  under the job directory.

Final reviewer-facing tables are generated under `tables/supplementary/`.
For the reported comparison, ablation, and robustness table bodies, run
`python tables/export_reproduction_tables.py` after generating the complete
`tables/supplementary/final_5seed_extended/` summary. It writes standalone
TeX and CSV files without editing the manuscript. The updated S1--S10 and
figure generators are listed in the reproduction guide. The source-data
directory under `els-cas-templates/` is a local package. Raw third-party files,
processed user sequences, checkpoints, model caches, credentials, and local
configuration secrets are excluded from the prepared delivery snapshot.

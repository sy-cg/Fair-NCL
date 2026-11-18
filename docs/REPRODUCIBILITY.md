# Reproducing the revision results

Run the commands below from the repository root. Model training requires a
compatible CUDA environment; the result-export commands in Section 5 consume
completed outputs and do not start training. `run_research_pipeline.py` is the
entry point for preprocessing and model runs. `main.py`, `train.py`,
`train_fair_ncl.py`, and the other older training scripts are not the source
of the revised five-seed tables. This document describes the **archived
configurations**; it does not claim that the current code checkout is bitwise
identical to every historical run.

The submission includes the reproduction source code and configuration plans;
it does not include a separate experimental source-data archive. The
`els-cas-templates/source_data/` and `tables/supplementary/` paths below describe
local archived analysis inputs or expected generated outputs. S1--S10 are
internal export identifiers, not journal attachments. Additional derived
records are available from the authors on reasonable request, subject to
the original data terms and privacy constraints. Analysis/export commands
require the corresponding archived inputs or completed outputs from a new
reproduction; they cannot run from the code archive alone.

## 1. Code and environment identity

The prepared local revision is identified by
`docs/release/revision_submission_release.json`. Its `code_snapshot` section
records the code archive, content SHA-256, archive SHA-256, and file count;
`manuscript` identifies the verified clean PDF. The code and LaTeX source
archives and `SHA256SUMS` are supplied under `submission_package/`. No
experimental source-data archive is included. Check the delivered files
from that directory:

```bash
sha256sum -c SHA256SUMS
```

Extract the code archive into a new working directory. The outer package's
`revision_submission_release.json` is intentionally not embedded in that
archive, to avoid self-reference. Create `docs/release/` under the
extracted working directory and copy that manifest there as
`docs/release/revision_submission_release.json` before following the remaining
commands from the extracted root.

This identifies the local deliverable independently of Git. The manifest's
`public_release` status and commit field distinguish a prepared local snapshot
from an externally published revision. A content hash is not a historical
training commit. See [the release checklist](REVISION_RELEASE_CHECKLIST.md)
for the package layout and publication status.

Record `git rev-parse HEAD` and `git status --short` for the checkout used in a
new reproduction. The 595 archived final runs did **not** record a Git commit.
Their available environment fields, one row per run, are in
`els-cas-templates/source_data/environment/final_run_environment.csv`. The
recorded versions span Python 3.10.8 (484 runs) and 3.13.3 (111), PyTorch
2.1.2+cu118 (93), 2.5.1+cu124 (391), and 2.6.0+cu124 (111). GPU models were
RTX 3090 (484) and RTX 2080 Ti (111). The CSV also records NumPy, pandas,
SciPy, CUDA, and cuDNN for each run. Historical versions of tqdm,
scikit-learn, matplotlib, and openpyxl were not captured. The current
`requirements.txt` provides compatible ranges, **not** a historical lockfile.
New runs additionally record these package versions and Git dirty status in
`run_provenance.json`. The manifest's `job_json_sha256` is the digest of the
archived per-run `job.json`, which can be checked when the original result
directory is available.

For a compatible fresh environment, use Python 3.10 and a PyTorch wheel suited
to the installed NVIDIA driver. One of the observed PyTorch profiles can be
installed with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The installation command is a compatibility recipe; the archived version CSV
is the evidence for the actual runs. Exact historical software identity cannot
be established for packages that were not logged. Do not substitute this
recipe for an unrecorded historical environment.

## 2. Acquire and verify the datasets

MovieLens 1M is distributed by [GroupLens](https://grouplens.org/datasets/movielens/1m/).
Download its 1M archive and extract the three input files:

```bash
mkdir -p data/ml-1m data/Taobao
curl -L --fail https://files.grouplens.org/datasets/movielens/ml-1m.zip -o ml-1m.zip
unzip -j ml-1m.zip 'ml-1m/ratings.dat' 'ml-1m/users.dat' 'ml-1m/movies.dat' -d data/ml-1m
```

The Taobao input is the [Alimama/Tianchi display-ad click dataset](https://tianchi.aliyun.com/dataset/56),
also linked through the [Kaggle mirror used in the manuscript](https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom).
Authentication and download terms are controlled by the host. Extract and
place `raw_sample.csv`, `user_profile.csv`, and `ad_feature.csv` directly in
`data/Taobao/`; the behavior-log file is not used. The exact filenames and
SHA-256 digests of the local raw inputs used for the revision are in
`docs/raw_data_sha256.txt`. After obtaining the files, check them with:

```bash
# Optional authenticated Kaggle CLI route; inspect the extracted file layout.
kaggle datasets download pavansanagapati/ad-displayclick-data-on-taobaocom \
  -p data/Taobao --unzip
```

The CLI route requires the user's own Kaggle credentials. If the mirror
delivers nested archives or paths, extract/move only the three named CSV files
into `data/Taobao/` before the checksum check:

```bash
sha256sum -c docs/raw_data_sha256.txt
```

These are checksums of the files used in the archived runs, not claims about
all copies distributed by third-party mirrors. Do not publish the raw data or
processed user sequences as part of this repository without confirming the
source terms.

## 3. Preprocess

```bash
python run_research_pipeline.py preprocess \
  --datasets ml-1m taobao \
  --output-dir processed_data \
  --min-user-interactions 5 \
  --min-item-interactions 5 \
  --max-seq-len 100
```

This path applies the paper's positive-event rules, iterative 5-core filter,
chronological ordering, and leave-two-out validation/test split. Check
`processed_data/{ml-1m,taobao}/summary.json` before training. Expected
`(users, items, interactions)` are `(6034, 3125, 574376)` for ML-1M and
`(31333, 17240, 265275)` for Taobao. A mismatch means the raw files or
preprocessing inputs differ from the archived experiment.

## 4. Configurations, tuning, and model runs

The frozen final configuration for **each of the 595 reported final runs** is
one JSON object in `experiments/revision_supplementary_final_extended_595.jsonl`.
It contains the dataset, backbone, method, seed, explicit parameter overrides,
and job ID; unspecified settings come from `config.py` and its memory-profile
rule. The file was reconstructed from the archived `job.json` files and
checked against the 119 complete five-seed groups in
`tables/supplementary/final_5seed_extended/run_summary.csv`; its 535-job
subset is `experiments/revision_supplementary_final_valid.jsonl`. The
reconstruction/check command is `python analysis/export_archived_final_runs.py`
and requires the original result directory. It is **not** a step in a fresh
run, because the frozen 595-job file is supplied here.

The validation-only reference and five-candidate search specifications are
`experiments/revision_baseline_reference.jsonl` (24 jobs),
`experiments/revision_frozen_backbone_reference_extra.jsonl` (16 jobs), and
`experiments/revision_validation_audit.jsonl` (1,020 jobs). Their generation,
selection rule, and method grids are documented in
`docs/SUPPLEMENTARY_EXPERIMENTS.md`; the selected method overrides are saved
in `configs/revision_method_overrides.json`. For an independent selection audit,
run the reference plans, the audit plan, and then the selector:

```bash
export RESULTS_ROOT=results_revision_kbs_v2
bash resume_plan_jobs.sh experiments/revision_baseline_reference.jsonl
bash resume_plan_jobs.sh experiments/revision_frozen_backbone_reference_extra.jsonl
bash resume_plan_jobs.sh experiments/revision_validation_audit.jsonl
python -m analysis.select_validation_configs \
  --results-root "$RESULTS_ROOT" \
  --jobs experiments/revision_validation_audit.jsonl \
  --utility-tolerance 0.15 \
  --output configs/reproduced_method_overrides.json \
  --summary tables/supplementary/reproduced_validation_selection.csv \
  --candidate-summary tables/supplementary/reproduced_validation_candidates.csv
```

Compare the reproduced overrides and selection tables with the archived
versions before interpreting deviations. For direct reevaluation of the
reported selected settings, use the supplied final plan; AFRL and PFRec first
need the matching frozen-backbone reference checkpoints from the two reference
plans above:

```bash
export RESULTS_ROOT=results_revision_kbs_v2
bash resume_plan_jobs.sh experiments/revision_baseline_reference.jsonl
bash resume_plan_jobs.sh experiments/revision_frozen_backbone_reference_extra.jsonl
bash resume_plan_jobs.sh experiments/revision_supplementary_final_extended_595.jsonl
```

To run one final job explicitly, use
`python run_research_pipeline.py run --jobs experiments/revision_supplementary_final_extended_595.jsonl --index 0 --processed-root processed_data --results-root results_revision_kbs_v2`.
Each new run writes `job.json`, `effective_config.json`,
`run_provenance.json`, `validation_results.json`, `test_results.json`,
`history.json`, and its checkpoint under
`results_revision_kbs_v2/<dataset>/<job_id>/`. The full plan requires
substantial GPU time; a single job verifies wiring but does not reproduce a
five-seed table. `effective_config.json` was added after the archived runs and
is not retroactively available for them.

### CoSeRec open-source code in the shared SASRec pipeline

The separate CoSeRec comparison adds 18 validation-only tuning jobs and 10
five-seed final jobs to the 595-run archive above. Its frozen plans are
`experiments/revision_lean_plans/coserec_tune.jsonl` and
`experiments/revision_lean_plans/coserec_final.jsonl`; the validation-only
selection is recorded in `experiments/revision_lean_plans/selection_manifest.json`.
For each dataset, contrastive weights 0.03, 0.1, and 0.3 are evaluated on seeds
42, 2025, and 2026. The highest mean validation HR@10 selects 0.03 on ML-1M
and 0.3 on Taobao. Tuning `test_results.json` files must have
`status=not_run_by_protocol`; final tests use seeds 42, 2025, 2026, 2027,
and 2028. The final runs reuse the shared SASRec, processed splits, and
full-ranking evaluator. `methods/coserec.py` directly imports the authors'
`Random` augmentation and `NCELoss` classes from `CoSeRec-main/src/`; its data
connector supplies train-only ItemCF-IUF neighbors. The authors' original
end-to-end trainer was not run. The connected path uses a one-negative logistic
recommendation loss instead of Fair-NCL's full-softmax CE, and its utility-only,
three-candidate selection differs from the seven-method fairness-oriented
audit. The comparison uses the open-source CoSeRec modules under shared data
and evaluation, but does not isolate the sampler's effect.

```bash
export RESULTS_ROOT=results_revision_kbs_v2
bash resume_plan_jobs.sh experiments/revision_lean_plans/coserec_tune.jsonl
bash resume_plan_jobs.sh experiments/revision_lean_plans/coserec_final.jsonl
python -m analysis.prepare_coserec_comparison
```

The last command checks all 18 tuning markers, 10 final CoSeRec runs, and their
matched Fair-NCL/plain SASRec seeds; it writes the validation, five-seed,
paired-bootstrap, environment, and RQ1 table-row source data as Supplementary
Data S7 and `els-cas-templates/coserec_main_*.tex`.

## 5. Regenerate reported aggregates and figures

Two routes are distinct. If the archived aggregate CSVs are available locally,
they permit inspection of the reported estimates and figure coordinates.
Re-exporting statistics from raw run outputs requires the corresponding
archived result directories, or outputs from an independent reproduction of
the frozen jobs. The journal package supplies neither those experimental
records nor raw third-party data, processed user-level sequences, per-user
prediction records, or checkpoints. User-level bootstrap intervals cannot
be reconstructed from aggregate means alone.

The commands below regenerate the existing report; they do not add training
experiments, change the selected configurations, or use test results to tune
new weights.

### Current displayed RQ2 tables and summary-based uncertainty bounds

Extract the LaTeX source archive into the same working directory as the code
archive, preserving the `els-cas-templates/` directory. The author-updated
Tables 11, 12, F.1 and F.2 in `cas-dc-sample.tex` are the authoritative
summaries for this analysis. Run:

```bash
python -m analysis.export_displayed_ablation_tables
python -m analysis.summary_ablation_uncertainty
```

The first command exports 88 displayed mean/SD rows, the four exact table
fragments, and eight full-minus-beta-zero point differences at @10/@20 under
`tables/supplementary/rq2_rq5/manuscript_exports/`. Its difference CSV uses
units of 10^-3 and reproduces the point differences in Table F.3. The second
exports 432 uncertainty bounds under
`tables/supplementary/rq2_rq5/summary_uncertainty/`. Both generate the same F.3
interval table and read only the four current manuscript tables.

For five reported seeds, the difference SD is at most the sum of the two
marginal SDs. Under a normal paired-difference model, this gives an envelope
containing all compatible ordinary 95% paired-t intervals. Each displayed
mean and SD has rounding half-unit 0.00005; the interval and p-value upper
bound allow for rounding. Holm adjustment applies to the p-value upper bounds
over 36 comparisons separately per metric/cutoff. These calculations are a
conditional sensitivity analysis: normality is unverified, covariance is not
estimated, and the upper bounds are not p values computed from observed seed
pairs. The intervals are not empirical paired-bootstrap intervals. Failure
of this conservative criterion does not rule out an effect for a particular
pairing. Main Fair-NCL/backbone and VICReg/MCE paired analyses retain their
separate stated scope. Full factorial interactions are not estimated.

### Archived RQ2 and RQ5 analyses

After copying the completed component and sensitivity results into
`results_revision_kbs_v2`, run from the repository root:

```bash
python -m analysis.prepare_rq2_rq5
python figures/gen_fig_parameter_sensitivity.py
```

The first command audits the two `missing_components_beta0_*.jsonl` plans
against saved jobs and test artifacts, adds the matched original references,
and reuses the existing BERT4Rec no-augmented-CE runs. It exports Supplementary
Data S8/S9, archival detailed-ablation TeX tables,
a sensitivity table, and `tables/supplementary/rq2_rq5/analysis_report.md`.
S8 has five seeds per cell; S9 uses three seeds per point, including centers.
Only the three requested SASRec sensitivity axes are read from the broader
completion plan. The second command requires NumPy/matplotlib and writes
PDF, SVG, and PNG plots in the manuscript and figure folders.
Neither command runs training or selects a new configuration from test data.
Archived S8 paired contrasts and its analysis report describe the archived
inputs, not the author-updated displayed tables. Do not transfer their intervals,
p values or dominance counts to the current revision. The F.3 export in this
script now follows the displayed-summary route above. Use that route for the
current Table F.3; paired inference would require the corresponding seed data.

### Representation objective comparison (internal export S10)

The completed plan
`experiments/revision_mce_matched_plans/sasrec_mce_matched.jsonl` contains ten
SASRec MCE runs: five seeds on each dataset. Do not rerun this plan merely to
export its existing results. With those directories and the original final
archive under `results_revision_kbs_v2`, run:

```bash
python -m analysis.prepare_mce_objective_comparison --include-sasrec
```

This exports S10 and the @10/@20 objective-comparison and paired-difference
TeX tables. The numerical comparison contains five settings: SASRec and
GRU4Rec on both datasets, plus BERT4Rec on ML-1M. It has 50 method-run
records, ten method summaries, and 70 metric contrasts. All ten new SASRec
MCE results are reported. Holm adjustment is across the five settings
separately for each metric/cutoff; the minimum adjusted p value is 0.3125.
The separate 50-run MCE audit retains provenance and the ten Caser
zero-update failures. Resource differences remain in the audit rather than
serving as exclusions in the current reporting policy. The MCE numerical
stabilization and precision path are disclosed implementation differences.
No new reference training is needed for this report.

### Primary comparisons, complete group statistics, and Pareto coordinates

After all 595 final jobs complete:

```bash
python -m analysis.run_supplementary_analysis \
  --results-root results_revision_kbs_v2 \
  --jobs experiments/revision_supplementary_final_extended_595.jsonl \
  --output-dir tables/supplementary/final_5seed_extended \
  --bootstrap-iterations 10000 --bootstrap-seed 42
python tables/export_reproduction_tables.py
python -m analysis.prepare_coserec_comparison
python analysis/prepare_reviewer_fairness_data.py
python analysis/prepare_reviewer_cost.py
python analysis/prepare_tuning_budget_audit.py
python analysis/prepare_component_controls.py
python figures/gen_fig_fairness_pareto.py
```

| Analysis item / internal export | Input and generator | Expected output |
| --- | --- | --- |
| Main comparison @10/@20 and SASRec ablation tables | `final_5seed_extended/{run_summary,group_summary}.csv`; `tables/export_reproduction_tables.py` | `tables/supplementary/reproduced_paper_tables/`: four comparison TeX files, `ablation_sasrec_10.tex`, `rq3_robustness_summary.tex`, and the RQ3 CSV |
| Paired intervals, directional groups, Pareto and proxy diagnostics | 595-job plan and final `test_results.json`; `analysis/run_supplementary_analysis.py` | `final_5seed_extended/manifest.json`, `run_summary.csv`, `group_summary.csv`, `paired_comparisons.csv`, `pareto.csv`, `proxy_outcome*.csv` |
| Pareto figure and Supplementary Data S1/S2 | Extended five-seed summary; `analysis/prepare_reviewer_fairness_data.py`, `figures/gen_fig_fairness_pareto.py` | `els-cas-templates/source_data/table_source_data/Supplementary_Data_S1_*.csv`, `Supplementary_Data_S2_*.csv`, and `figures/fairness_pareto_main.pdf` |
| Cost and validation-budget tables, Supplementary Data S3--S5 | Final jobs and validation records; `analysis/prepare_reviewer_cost.py`, `analysis/prepare_tuning_budget_audit.py` | `tables/supplementary/reviewer_protocol_cost/*.csv` and corresponding `source_data/table_source_data/` CSV files |
| Component controls, Supplementary Data S6 | Extended five-seed summary and frozen plan; `analysis/prepare_component_controls.py` | `tables/supplementary/reviewer_component_controls.csv` and `source_data/table_source_data/Supplementary_Data_S6_component_controls.csv` |
| CoSeRec rows in the four RQ1 comparison tables, Supplementary Data S7 | 18 validation-only CoSeRec jobs, 10 final CoSeRec jobs, and matched five-seed SASRec/Fair-NCL results; `analysis/prepare_coserec_comparison.py` | `source_data/table_source_data/Supplementary_Data_S7_*.csv` and `els-cas-templates/coserec_main_*.tex` |
| Current detailed ablations and Table F.3 | Four author-updated LaTeX tables; `analysis/export_displayed_ablation_tables.py`, `analysis/summary_ablation_uncertainty.py` | 88 displayed mean/SD rows, eight full-minus-beta-zero contrasts with conservative t-interval envelopes, and 432 conditional bounds; no seed observations or covariance reconstructed |
| Archived component analysis and internal export S8 | Archived component plans, references and BERT4Rec no-augmented-CE runs; `analysis/prepare_rq2_rq5.py` | Archival run summaries and paired contrasts; these do not supply inferential evidence for the author-updated current tables |
| Parameter sensitivity and Supplementary Data S9 | Completed SASRec sweeps; `analysis/prepare_rq2_rq5.py`, `figures/gen_fig_parameter_sensitivity.py` | 18 three-seed points, 42 unique source jobs, and two dataset-specific PDF/SVG/PNG figures |
| Representation objectives and Supplementary Data S10 | Original MCE archive and the completed ten-job SASRec plan; `analysis/prepare_mce_objective_comparison.py --include-sasrec` | 50 method-run records, ten summaries, 70 paired contrasts, configuration/training audit, and objective-comparison TeX tables |

S1 intentionally has two views of the same Fair-NCL-minus-backbone results.
`Supplementary_Data_S1_paired_comparisons.csv` retains the 96 primary overall
utility and marginal absolute-gap contrasts. The complete
`Supplementary_Data_S1_full_paired_comparisons.csv` contains all 672 contrasts,
including group means, signed and relative gaps, worst-group utility, and
intersectional gaps. The 96 primary rows are a subset of those 672; they are
not additional independent tests. The reported correction families are
preserved rather than recalculated by joining the two files.
The full archival export also retains `gender_age` signed contrasts between
the first and last sorted joint labels. Those are not the manuscript's
four-group disparity measure: the paper uses the maximum-minus-minimum gap
and worst-group utility for intersectional evaluation. Binary signed gaps
refer to gender and age group.

S2 contains 66 points: nine methods in each SASRec block, including CoSeRec,
and eight in each of the other six blocks. The CoSeRec coordinates come from
the completed S7 summary. Its addition does not change the existing methods'
Pareto memberships; Fair-NCL remains on four of eight fronts. The 15% common
test utility constraint is descriptive and does not reselect configurations.

The aggregate report's `manifest.json` must show `expected_jobs: 595`,
`discovered_runs: 595`, no missing or unexpected job IDs, and 119 complete
five-seed groups. `docs/BASELINE_PROVENANCE.md` maps every final-path method
to its local implementation and states where an original system was only
adapted. The table-source CSVs and final-run environment CSV are staged under
`els-cas-templates/source_data/`; their presence locally is not proof of a
public data deposit or a historical code commit.

## 6. Clean manuscript and response locations

The clean manuscript is `els-cas-templates/cas-dc-sample.pdf`; the marked-up
version is `els-cas-templates/cas-dc-marked.pdf`. New and revised passages in
the marked-up manuscript are highlighted in yellow. The regenerated Fair-NCL
response is `point_by_point_response_filled.docx`, accompanied by
`docs/response_work/revised_locations.csv`. Use that table and the response's
section/table/figure references to locate revisions; its page references
follow the clean manuscript. A differently titled paper's response file is
not part of the Fair-NCL submission package.

This revision's verification covers archived-result analysis, generated
tables/figures, document consistency, and the recorded checks. It does not
claim that the entire GPU training archive was rerun during package assembly.

# Reviewer-perspective audit of the supplementary experiments

## Scope and decision

This audit maps the KBS editor/reviewer requests in `Comments.md` to the
supplementary experiment implementation. It evaluates protocol and code only;
no supplementary numerical result is claimed until the GPU jobs finish.

Current protocol status: **ready for validation-only execution, with final test
execution intentionally gated by configuration selection**.

## Critical issues found and repaired

### 1. Revision plans were centered on pre-adjustment SASRec parameters

**Risk.** The first revision plans used `configs/loss_selected.json` directly,
although the submitted ML-1M/Taobao SASRec experiments used the later
`configs/loss_ml_taobao_sasrec.json` settings. The submitted SASRec setting was
therefore absent from the five-candidate validation audit. Multiplicative
epsilon perturbations also produced values outside the search space declared
in the manuscript.

**Repair.** Added `configs/revision_base_params.json`: BERT4Rec, GRU4Rec, and
Caser retain their submitted `loss_selected.json` settings, while both SASRec
settings use the post-method-adjustment values. Candidate `c0` is now the exact
submitted center. The other Fair-NCL candidates use only the remaining values
from the declared epsilon and beta grids. The baseline reference is generated
from `configs/backbone_best.json` and carries no augmentation/loss parameters.

### 2. Test metrics were computed during configuration selection

**Risk.** The original audit jobs and the extracted baseline-reference jobs
would save test metrics even though the selector read only validation files.
This leaves a credible test-peeking criticism.

**Repair.** Added a dedicated `validation_reference` phase and set both
reference and `validation_audit` jobs to `evaluate_test=false`. They save an
explicit `not_run_by_protocol` completion artifact. Only the frozen five-seed
`supplementary` phase evaluates the test split.

### 3. The new MCE controls had no comparable tuning budget

**Risk.** Default MCE loss weights could unfairly weaken the closest
non-contrastive control.

**Repair.** Both `ncl_sr_mce` and `fair_aug_mce` receive five validation
candidates over three seeds. The former covers all four backbones; the latter
matches its SASRec/BERT4Rec final scope. The audit now contains 1,020 jobs and
68 dataset--backbone--method groups, each with exactly five candidates.

### 4. Selection reported variance but did not use it

**Risk.** Selecting the smallest mean validation gap could favor a noisy
configuration, matching Reviewer 2's gap-overfitting concern.

**Repair.** After applying the predefined 15% utility-loss budget through the
paired conservative retention rule `mean(seed-wise NDCG retention) - one
standard deviation >= 0.85`, selection fixes each attribute's disadvantaged
group from the matched backbone. It then uses the feasible three-objective
Pareto frontier and prioritizes the minimum conservative disadvantaged-group
benefit across gender and age, with the maximum normalized-gap risk as the next
criterion. All candidates, their standard deviations, selection status, and
validation Pareto membership are exported. Missing fairness evidence or
incomplete jobs cause a hard failure.

### 5. Statistical reporting did not fully answer the group-benefit question

**Risk.** The first implementation tested only NDCG@10 and did not directly
show each group's change from the matched backbone.

**Repair.** Paired hierarchical bootstrap results now cover HR and NDCG at @10
and @20, overall utility, absolute/relative/signed gaps, worst-group utility,
and every named marginal/intersectional group. Each row contains the baseline
estimate, method estimate, delta, interval, improvement direction, exact
seed-level sign-flip p-value, and family-wise Holm result.

### 6. Result aggregation could silently accept incomplete evidence

**Risk.** A missing method group would not appear in a seed-only manifest, and
paired analyses silently intersected mismatched user sets.

**Repair.** Final reporting checks the declared JSONL plan and rejects missing,
unexpected, or duplicate jobs. Paired inference requires identical seeds,
identical user IDs, and consistent group labels.

## Reviewer requests now covered

| Reviewer concern | Implemented evidence |
|---|---|
| Marginal prevalence confounds raw skew | Shrinkage-stabilized JSD from the interaction prior; raw submitted score retained as a control |
| User vs interaction prevalence choice | Skew diagnostics report both priors, score correlation, and low-skew-set overlap |
| Frequency/popularity confounding | Item-level score/support/popularity tables and rank correlations |
| Temporal stability | Adjacent temporal-quartile score correlations and low-skew overlap |
| Does lower skew affect final fairness? | Per-seed augmentation skew change linked to gap and worst-group changes, with association summaries |
| Who benefits? | Named group values/counts, per-group paired changes, signed/relative gaps, and worst-group utility |
| Intersectional fairness | Gender-by-age values, counts, inference, and a separate intersectional Pareto frontier |
| Statistical robustness | Five final seeds, paired hierarchical bootstrap, exact paired sign-flip tests, and family-wise Holm correction |
| Consistent model selection | Same three seeds, five candidates, 15% paired conservative validation-utility constraint, fixed-group benefit and minimax gap objectives |
| Direct controls | VICReg-only, random+NCL, similarity+NCL, NCL-SR-style MCE, FairAug+MCE, and focused component removals |
| Computational cost | Resource construction time, candidate counts, cache size, training/test time, parameter count, peak CUDA memory, and dataset scale |
| Reproducibility | Root README, consolidated submitted-parameter provenance, fixed JSONL plans, exact job/config artifacts, preprocessing instructions, environment/hardware/code provenance |

## Interpretation constraints after results arrive

- Five seeds are enough for uncertainty estimation but a two-sided exact
  sign-flip test has minimum p-value 0.0625. Do not call a difference
  significant from the seed test alone.
- Use `statistically_supported_improvement` for conservative table emphasis;
  do not bold methods merely because their mean is larger or their gap smaller.
- The NCL-SR comparison is an objective/profile-generation control, not a claim
  of byte-for-byte reproduction of its original implementation.
- The focused ablations identify component contributions and use explicit
  Pareto criteria; they are not a complete 2^4 factorial design.
- Temporal-quartile diagnostics test stability of the skew proxy, not a formal
  guarantee of dynamic fairness.
- Proxy--outcome correlations are empirical associations and must not be
  described as causal or theoretical fairness guarantees.

## Remaining blockers

1. Run the 24 validation-reference jobs and 1,020 validation-audit jobs in the
   configured PyTorch/GPU environment.
2. Freeze `configs/revision_method_overrides.json`, then generate and run the
   540 final five-seed jobs.
3. Inspect the strict result manifest before generating manuscript tables.
4. Revise the manuscript and response letter only after the numerical evidence
   is available; current three-seed/raw-skew statements must not be updated in
   advance of results.
5. Create a valid Git commit before final execution so `run_provenance.json`
   contains a submission commit identifier rather than `null`.

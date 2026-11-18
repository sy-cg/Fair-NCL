# Baseline implementation provenance for the revision

The final five-seed runs instantiate the four ID-based backbones in
`models/sasrec.py`, `models/bert4rec.py`, `models/gru4rec.py`, and
`models/caser.py` through `methods/registry.py`. The same preprocessing,
checkpoint rule, and full-ranking evaluator are used for all methods. The
citations in the manuscript identify the methodological sources; the final
training path uses the **local implementations** below. Do not interpret a
method name as evidence that an unmodified official release was executed.

| Reported name | Job method | Final-path implementation | Relationship to cited method |
| --- | --- | --- | --- |
| Plain backbone | `baseline` | `methods/wrappers.py:BackboneMethod` | Shared local backbone reference. |
| Adv-Debias | `adv_debias` | `methods/wrappers.py:AdvDebiasMethod` | Local adversarial-debiasing adaptation of the cited objective. |
| GRL | `grl` | `methods/wrappers.py:GRLMethod` | Local gradient-reversal adaptation. |
| PCFR-SM | `sm_pcfr` | `methods/wrappers.py:SMPCFRMethod` | Local selective-filter adaptation inspired by PCFR/SM; not an unmodified original implementation. |
| PFRec | `pfrec` | `methods/wrappers.py:PFRecMethod` | Local prompt-based adaptation on the shared ID backbone. |
| AFRL | `afrl` | `methods/wrappers.py:AFRLMethod`, `models/afrl_components.py` | Local alternating alignment/adversarial adaptation on the shared backbone. |
| A-FSR | `a_fsr` | `methods/wrappers.py:AFSRMethod` | Local demographic-agnostic smoothing/robust-loss adaptation. |
| VICReg only | `ncl_only` | `methods/wrappers.py:NCLMethod` | Identity-view regularization control. |
| Random/local augmentation + VICReg | `random_aug`, `similarity_aug` | `methods/wrappers.py:NCLMethod`, `methods/augmentations.py` | Local controls for the view-construction mechanism. |
| NCL-SR-ID | `ncl_sr_mce` | `methods/wrappers.py:NCLMethod`, `methods/mce.py` | Independent ID-based reimplementation of transferable NCL-SR components. It does not reproduce the original text encoder or full text-based system. |
| FairAug + MCE | `fair_aug_mce` | `methods/wrappers.py:NCLMethod`, `methods/mce.py` | Local objective comparison using five complete five-seed settings with matching recorded view parameters and current-rule effective batch sizes: new ML-1M/SASRec and Taobao/SASRec, plus archived ML-1M/BERT4Rec and both GRU4Rec settings. S10 reports 50 method-run records and 70 paired metric contrasts; all ten new SASRec MCE runs have effective training records and zero skipped updates. Resource differences remain in the provenance audit but do not exclude results from this report. Both Caser MCE settings have zero effective training updates and remain failures rather than learned-performance evidence. The MCE path disables autocast and stabilizes the matrix logarithm; configuration agreement does not establish identical historical software or resource contents. |
| CoSeRec | `coserec` | `methods/coserec.py:CoSeRecMethod`, `CoSeRec-main/src/data_augmentation.py`, `CoSeRec-main/src/modules.py` | Directly imports the authors' open-source `Random` augmentation and `NCELoss` classes. The data connector supplies train-only ItemCF-IUF neighbors and uses the shared SASRec backbone, prefix examples, checkpointing, and evaluator. Its one-negative logistic recommendation loss differs from Fair-NCL's full-softmax CE; this is not an execution of the original repository's end-to-end trainer. |
| Fair-NCL | `fair_ncl` | `methods/wrappers.py:NCLMethod`, `methods/augmentations.py` | Proposed method. |

The complete job parameters, including each method's selected center and seed,
are in `experiments/revision_supplementary_final_extended_595.jsonl`; the
shared validation candidates are in `experiments/revision_validation_audit.jsonl`
and `tables/supplementary/validation_candidates.csv`. The separate CoSeRec
search and final SASRec jobs are frozen in `experiments/revision_lean_plans/`
and selected by three-seed validation HR@10. The manuscript's baseline
citations and validation-grid table specify the respective source papers and
search ranges. Source-code provenance for historical local adaptations and
upstream commit identifiers was not captured in the archived run metadata;
therefore no upstream commit is asserted here. The directories under
`cache/external_baselines/` are not the final training entry point.

Supplementary Data S10 now reports 50 method-run records and 70 paired
contrasts for five settings with matching recorded augmentation parameters.
Holm correction is applied across five settings separately for each metric
and cutoff; the minimum adjusted p value is 0.3125. Its source-job file
contains 90 definitions: 50 MCE runs and 40 unique Fair-NCL references.
Separate S10 new-run, new-summary, and new-environment files retain all ten
new SASRec MCE runs, all included in the current comparison. The completed
SASRec plan is not overwritten by the analysis generator. Resource differences
remain in the provenance audit and are not used to exclude results; current
reporting requires no additional reference training. This compares the
implemented objectives without asserting identity of all historical resources
or a causal effect of the symbolic loss alone.

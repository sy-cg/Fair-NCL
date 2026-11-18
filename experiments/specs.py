import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .job_utils import merge_nested_params, merge_params, stable_job_id


DATASETS = ("ml-1m", "lastfm-1k", "taobao")
REVISION_DATASETS = ("ml-1m", "taobao")
FINAL_REVISION_SEEDS = (42, 2025, 2026, 2027, 2028)
SELECTION_SEEDS = (42, 2025, 2026)
FAIR_NCL_AUDIT_EPSILON_GRID = (0.5, 1.0, 2.0)
FAIR_NCL_AUDIT_BETA_GRID = (0.2, 0.3, 0.4)
MCE_STABILITY_WEIGHT_GRID = (0.01, 0.05, 0.1, 0.25)
BACKBONES = ("sasrec", "bert4rec", "gru4rec", "caser")
LEGACY_ABLATION_METHODS = ("baseline", "fair_ncl", "ncl_only", "random_aug", "similarity_aug")
REQUESTED_ABLATION_METHODS = (
    "wo_fairness_sampling",
    "wo_semantic_sampling",
    "wo_alignment",
    "wo_variance",
    "wo_covariance",
    "wo_augmented_ce",
    "random_low_skew",
    "high_skew",
)
ABLATION_METHODS = LEGACY_ABLATION_METHODS + REQUESTED_ABLATION_METHODS
EXPERIMENTAL_ABLATION_METHODS = ("fair_ncl_alpha_tradeoff",)
SEMANTIC_ABLATION_METHODS = ("fair_ncl_semantic_alpha_tradeoff",)
SEMANTIC_HYBRID_ABLATION_METHODS = ("fair_ncl_semantic_hybrid_alpha_tradeoff",)
KNOWN_ABLATION_METHODS = (
    ABLATION_METHODS
    + EXPERIMENTAL_ABLATION_METHODS
    + SEMANTIC_ABLATION_METHODS
    + SEMANTIC_HYBRID_ABLATION_METHODS
)
ABLATION_METHOD_PRESETS = {
    "legacy": LEGACY_ABLATION_METHODS,
    "requested": REQUESTED_ABLATION_METHODS,
    "new": REQUESTED_ABLATION_METHODS,
    "main": ("baseline", "fair_ncl") + REQUESTED_ABLATION_METHODS,
    "alpha_tradeoff": EXPERIMENTAL_ABLATION_METHODS,
    "experimental": EXPERIMENTAL_ABLATION_METHODS,
    "semantic_alpha_tradeoff": SEMANTIC_ABLATION_METHODS,
    "semantic": SEMANTIC_ABLATION_METHODS,
    "semantic_hybrid_alpha_tradeoff": SEMANTIC_HYBRID_ABLATION_METHODS,
    "semantic_hybrid": SEMANTIC_HYBRID_ABLATION_METHODS,
    "hybrid_semantic": SEMANTIC_HYBRID_ABLATION_METHODS,
    "all": ABLATION_METHODS,
    "full": ABLATION_METHODS,
    "all_with_experimental": KNOWN_ABLATION_METHODS,
}
COMPARISON_METHODS = (
    "baseline",
    "fair_ncl",
    "adv_debias",
    "grl",
    "sm_pcfr",
    "afrl",
    "pfrec",
    "a_fsr",
)
SUPPLEMENTARY_DIRECT_METHODS = (
    "baseline",
    "fair_ncl",
    "ncl_only",
    "random_aug",
    "similarity_aug",
    "ncl_sr_mce",
)
SUPPLEMENTARY_FAIRNESS_METHODS = COMPARISON_METHODS[2:]
SUPPLEMENTARY_CLOSEST_METHODS = ("fair_aug_mce",)
SUPPLEMENTARY_FACTORIAL_METHODS = (
    "skew_only_aug",
    "fair_aug_no_geometry",
    "fair_aug_no_augmented_ce",
    "raw_skew_fair_ncl",
)
VALIDATION_AUDIT_METHODS = (
    ("fair_ncl",)
    + SUPPLEMENTARY_FAIRNESS_METHODS
    + ("ncl_sr_mce", "fair_aug_mce")
)
PHASES = (
    "backbone",
    "augment",
    "loss",
    "ablation",
    "comparison",
    "supplementary",
    "validation_reference",
    "validation_audit",
    "mce_stability",
    "all",
)
BACKBONE_TUNING_DATASETS = DATASETS
BACKBONE_TUNING_BACKBONES = BACKBONES
AUGMENT_TUNING_DATASETS = DATASETS
AUGMENT_TUNING_BACKBONES = BACKBONES
LOSS_TUNING_DATASETS = DATASETS
LOSS_TUNING_BACKBONES = BACKBONES
ABLATION_DATASETS = DATASETS
ABLATION_BACKBONES = ("sasrec",)
COMPARISON_DATASETS = DATASETS
COMPARISON_BACKBONES = BACKBONES
SCOPED_DEFAULT_KEY = "__default__"
SCOPED_BACKBONES_KEY = "__backbones__"
SCOPED_DATASETS_KEY = "__datasets__"
SCOPED_DATASET_BACKBONES_KEY = "__dataset_backbones__"
SCOPED_METHODS_KEY = "__methods__"

BASE_PARAMS_REQUIRED_PHASES = {
    "augment",
    "loss",
    "ablation",
    "comparison",
    "supplementary",
    "validation_reference",
    "validation_audit",
    "mce_stability",
    "all",
}
BACKBONE_STAGE_KEYS = ("learning_rate", "hidden_units", "dropout_rate")
AUGMENT_STAGE_KEYS = BACKBONE_STAGE_KEYS + ("epsilon", "augment_ratio", "utility_alpha", "utility_beta")


@dataclass
class ExperimentJob:
    job_id: str
    phase: str
    dataset: str
    method: str
    backbone: str
    seed: int
    params: Dict[str, Any]
    reuse_from_phase: Optional[str] = None


def build_experiment_plan(phase: str,
                          datasets: Optional[Iterable[str]] = None,
                          backbones: Optional[Iterable[str]] = None,
                          seeds: Optional[Iterable[int]] = None,
                          base_params: Optional[Dict[str, Any]] = None,
                          pairs: Optional[Iterable[str]] = None,
                          ablation_methods: Optional[Iterable[str]] = None) -> List[ExperimentJob]:
    phase = phase.lower()
    base_params = dict(base_params or {})
    if seeds is None:
        if phase == "supplementary":
            seeds = FINAL_REVISION_SEEDS
        elif phase in {"validation_reference", "validation_audit"}:
            seeds = SELECTION_SEEDS
        else:
            seeds = (42,)
    seeds = tuple(int(seed) for seed in seeds)

    if phase == "backbone":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        return _backbone_plan(_resolve_pairs(pairs, datasets, backbones), seeds)
    if phase == "augment":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        resolved_pairs = _resolve_pairs(pairs, datasets, backbones)
        _require_pair_params(phase, base_params, resolved_pairs, BACKBONE_STAGE_KEYS)
        return _augment_plan(resolved_pairs, seeds, base_params)
    if phase == "loss":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        resolved_pairs = _resolve_pairs(pairs, datasets, backbones)
        _require_pair_params(phase, base_params, resolved_pairs, AUGMENT_STAGE_KEYS)
        return _loss_plan(resolved_pairs, seeds, base_params)
    if phase == "ablation":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        return _ablation_plan(
            _resolve_pairs(pairs, datasets, backbones),
            seeds,
            base_params,
            ablation_methods=ablation_methods,
        )
    if phase == "comparison":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        return _comparison_plan(_resolve_pairs(pairs, datasets, backbones), seeds, base_params)
    if phase == "supplementary":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        return _supplementary_plan(_resolve_pairs(pairs, datasets, backbones), seeds, base_params)
    if phase == "validation_reference":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        return _validation_reference_plan(_resolve_pairs(pairs, datasets, backbones), seeds, base_params)
    if phase == "validation_audit":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        return _validation_audit_plan(_resolve_pairs(pairs, datasets, backbones), seeds, base_params)
    if phase == "mce_stability":
        datasets, backbones = _resolve_scope(phase, datasets, backbones)
        _require_base_params(phase, base_params)
        return _mce_stability_plan(_resolve_pairs(pairs, datasets, backbones), seeds, base_params)
    if phase == "all":
        _require_base_params(phase, base_params)
        jobs = []
        for item in ("backbone", "augment", "loss", "ablation", "comparison"):
            jobs.extend(build_experiment_plan(
                item,
                datasets,
                backbones,
                seeds,
                base_params,
                pairs,
                ablation_methods=ablation_methods,
            ))
        return jobs
    raise ValueError(f"phase must be one of {', '.join(PHASES)}")


def _backbone_plan(pairs, seeds):
    grid = _grid({
        "learning_rate": [1e-3, 5e-4],
        "hidden_units": [64, 128],
        "dropout_rate": [0.1, 0.2],
    })
    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed, params in itertools.product(pairs, seeds, grid):
        jobs.append(_make_job("backbone", dataset, "baseline", backbone, seed, params))
    return jobs


def _augment_plan(pairs, seeds, base_params):
    grid = []
    for epsilon in [0.5, 1.0, 2.0]:
        for augment_ratio in [0.1, 0.2]:
            for utility_alpha in [0.6, 0.7, 0.8]:
                grid.append({
                    "epsilon": epsilon,
                    "augment_ratio": augment_ratio,
                    "utility_alpha": utility_alpha,
                    "utility_beta": round(1.0 - utility_alpha, 3),
                })

    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed, grid_params in itertools.product(pairs, seeds, grid):
        params = merge_params(_resolve_base_params(base_params, dataset, backbone), grid_params)
        params = _final_fair_ncl_params(params)
        jobs.append(_make_job("augment", dataset, "fair_ncl", backbone, seed, params))
    return jobs


def _loss_plan(pairs, seeds, base_params):
    grid = _grid({
        "fair_ncl_align_weight": [0.5, 1.0],
        "fair_ncl_var_weight": [0.5, 1.0],
        "fair_ncl_cov_weight": [0.02, 0.04, 0.08],
    })
    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed, grid_params in itertools.product(pairs, seeds, grid):
        params = merge_params(_resolve_base_params(base_params, dataset, backbone), grid_params)
        params = _final_fair_ncl_params(params)
        jobs.append(_make_job("loss", dataset, "fair_ncl", backbone, seed, params))
    return jobs


def _ablation_plan(pairs, seeds, base_params, ablation_methods=None):
    selected_methods = _resolve_ablation_methods(ablation_methods)
    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed, method in itertools.product(pairs, seeds, selected_methods):
        resolved_base_params = _resolve_base_params(base_params, dataset, backbone)
        params = _ablation_params(method, resolved_base_params)
        reuse_from_phase = None
        jobs.append(_make_job("ablation", dataset, method, backbone, seed, params, reuse_from_phase))
    return jobs


def _comparison_plan(pairs, seeds, base_params):
    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed, method in itertools.product(pairs, seeds, COMPARISON_METHODS):
        resolved_base_params = _resolve_base_params(base_params, dataset, backbone)
        params = _comparison_params(method, resolved_base_params)
        reuse_from_phase = "ablation" if backbone == "sasrec" and method in {"baseline", "fair_ncl"} else None
        jobs.append(_make_job("comparison", dataset, method, backbone, seed, params, reuse_from_phase))
    return jobs


def _supplementary_plan(pairs, seeds, base_params):
    jobs: List[ExperimentJob] = []
    for dataset, backbone in pairs:
        methods = list(SUPPLEMENTARY_DIRECT_METHODS + SUPPLEMENTARY_FAIRNESS_METHODS)
        if backbone in {"sasrec", "bert4rec"}:
            methods.extend(SUPPLEMENTARY_CLOSEST_METHODS)
        if backbone == "sasrec":
            methods.extend(SUPPLEMENTARY_FACTORIAL_METHODS)

        for seed, method in itertools.product(seeds, methods):
            resolved_base_params = _resolve_method_params(base_params, dataset, backbone, method)
            params = _supplementary_params(method, resolved_base_params)
            jobs.append(_make_job("supplementary", dataset, method, backbone, seed, params))
    return jobs


def _validation_audit_plan(pairs, seeds, base_params):
    jobs: List[ExperimentJob] = []
    for dataset, backbone in pairs:
        methods = [
            method
            for method in VALIDATION_AUDIT_METHODS
            if method != "fair_aug_mce" or backbone in {"sasrec", "bert4rec"}
        ]
        for method in methods:
            resolved_base_params = _resolve_method_params(base_params, dataset, backbone, method)
            for candidate_params in _validation_audit_candidates(method, resolved_base_params):
                for seed in seeds:
                    jobs.append(_make_job(
                        "validation_audit",
                        dataset,
                        method,
                        backbone,
                        seed,
                        candidate_params,
                    ))
    return jobs


def _validation_reference_plan(pairs, seeds, base_params):
    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed in itertools.product(pairs, seeds):
        params = _resolve_base_params(base_params, dataset, backbone)
        params["export_per_user_metrics"] = False
        params["report_intersectional_groups"] = True
        params["evaluate_test"] = False
        params["export_mechanism_analysis"] = False
        jobs.append(_make_job("validation_reference", dataset, "baseline", backbone, seed, params))
    return jobs


def _mce_stability_plan(pairs, seeds, base_params):
    jobs: List[ExperimentJob] = []
    for (dataset, backbone), seed, weight in itertools.product(
        pairs,
        seeds,
        MCE_STABILITY_WEIGHT_GRID,
    ):
        resolved_base_params = _resolve_method_params(
            base_params,
            dataset,
            backbone,
            "ncl_sr_mce",
        )
        params = _supplementary_params("ncl_sr_mce", resolved_base_params)
        params.update({
            "mce_uniform_weight": weight,
            "mce_align_weight": weight,
            "mce_diagnostic": True,
            "evaluate_test": False,
            "export_per_user_metrics": False,
            "export_mechanism_analysis": False,
        })
        jobs.append(_make_job(
            "mce_stability",
            dataset,
            "ncl_sr_mce",
            backbone,
            seed,
            params,
        ))
    return jobs


def _supplementary_params(method: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(base_params)
    params["evaluate_test"] = True
    params["export_per_user_metrics"] = True
    params["report_intersectional_groups"] = True
    if method in SUPPLEMENTARY_FAIRNESS_METHODS:
        return _comparison_params(method, params)

    if method in {"fair_ncl", "fair_aug_mce", "fair_aug_no_geometry", "fair_aug_no_augmented_ce"}:
        params = _final_fair_ncl_params(params)
        params["bias_metric"] = "prior_jsd"
        params["bias_event_source"] = "unique_train_events"
    if method == "fair_aug_mce":
        params["ncl_objective"] = "mce"
    elif method == "fair_aug_no_geometry":
        params["fair_ncl_align_weight"] = 0.0
        params["fair_ncl_var_weight"] = 0.0
        params["fair_ncl_cov_weight"] = 0.0
    elif method == "fair_aug_no_augmented_ce":
        params["fair_ncl_aug_rec_weight"] = 0.0
    elif method == "skew_only_aug":
        params["utility_alpha"] = 0.0
        params["utility_beta"] = 1.0
        params["use_similarity_scores"] = False
        params["similarity_source"] = "cooccurrence"
        params["bias_metric"] = "prior_jsd"
        params["bias_event_source"] = "unique_train_events"
    elif method == "raw_skew_fair_ncl":
        params = _final_fair_ncl_params(params)
        params["bias_metric"] = "raw_range"
        params["bias_event_source"] = "prefix_samples"
        params["bias_min_count"] = 1
    elif method == "ncl_sr_mce":
        params["ncl_objective"] = "mce"
        params["fair_ncl_aug_rec_weight"] = 0.0
        params["utility_alpha"] = 1.0
        params["utility_beta"] = 0.0
        params["use_similarity_scores"] = True
        params["similarity_source"] = "semantic_hybrid"
    elif method == "ncl_only":
        params["fair_ncl_aug_rec_weight"] = 0.0
    return params


def _validation_audit_candidates(method: str, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = _supplementary_params(method, base_params)
    params["export_per_user_metrics"] = False
    params["evaluate_test"] = False
    params["export_mechanism_analysis"] = False
    candidates = [dict(params)]

    if method == "fair_ncl":
        beta = float(params.get("utility_beta", 0.3))
        epsilon = float(params.get("epsilon", 1.0))
        if beta not in FAIR_NCL_AUDIT_BETA_GRID:
            raise ValueError(
                f"Fair-NCL audit center utility_beta={beta} is outside the declared grid "
                f"{FAIR_NCL_AUDIT_BETA_GRID}."
            )
        if epsilon not in FAIR_NCL_AUDIT_EPSILON_GRID:
            raise ValueError(
                f"Fair-NCL audit center epsilon={epsilon} is outside the declared grid "
                f"{FAIR_NCL_AUDIT_EPSILON_GRID}."
            )
        for candidate_beta in FAIR_NCL_AUDIT_BETA_GRID:
            if candidate_beta == beta:
                continue
            candidate = dict(params)
            candidate["utility_beta"] = candidate_beta
            candidate["utility_alpha"] = 1.0 - candidate_beta
            candidates.append(candidate)
        for candidate_epsilon in FAIR_NCL_AUDIT_EPSILON_GRID:
            if candidate_epsilon == epsilon:
                continue
            candidate = dict(params)
            candidate["epsilon"] = candidate_epsilon
            candidates.append(candidate)
    elif method in {"ncl_sr_mce", "fair_aug_mce"}:
        uniform_weight = float(params.get("mce_uniform_weight", 1.0))
        align_weight = float(params.get("mce_align_weight", 1.0))
        for uniform_multiplier, align_multiplier in (
            (0.5, 1.0),
            (2.0, 1.0),
            (1.0, 0.5),
            (1.0, 2.0),
        ):
            candidate = dict(params)
            candidate["mce_uniform_weight"] = uniform_weight * uniform_multiplier
            candidate["mce_align_weight"] = align_weight * align_multiplier
            candidates.append(candidate)
        candidates[0]["mce_uniform_weight"] = uniform_weight
        candidates[0]["mce_align_weight"] = align_weight
    else:
        parameter, default = {
            "adv_debias": ("adv_debias_weight", 0.2),
            "grl": ("grl_weight", 0.2),
            "sm_pcfr": ("sm_lambda", 1.0),
            "afrl": ("afrl_lambda", 1.0),
            "pfrec": ("pfrec_adv_weight", 0.2),
            "a_fsr": ("afsr_uniform_weight", 0.05),
        }[method]
        center = float(params.get(parameter, default))
        for multiplier in (0.25, 0.5, 2.0, 4.0):
            candidate = dict(params)
            candidate[parameter] = center * multiplier
            candidates.append(candidate)

    for index, candidate in enumerate(candidates):
        candidate["audit_candidate"] = f"c{index}"
    return candidates


def _ablation_params(method: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    if method == "fair_ncl":
        return _final_fair_ncl_params(base_params)
    if method == "similarity_aug":
        params = dict(base_params)
        params["utility_alpha"] = 1.0
        params["utility_beta"] = 0.0
        return params
    if method == "wo_fairness_sampling":
        params = _final_fair_ncl_params(base_params)
        params["utility_alpha"] = 1.0
        params["utility_beta"] = 0.0
        return params
    if method == "wo_semantic_sampling":
        params = dict(base_params)
        params["utility_alpha"] = 0.0
        params["use_similarity_scores"] = False
        params["similarity_source"] = "cooccurrence"
        return params
    if method == "wo_alignment":
        params = _final_fair_ncl_params(base_params)
        params["fair_ncl_align_weight"] = 0.0
        return params
    if method == "wo_variance":
        params = _final_fair_ncl_params(base_params)
        params["fair_ncl_var_weight"] = 0.0
        return params
    if method == "wo_covariance":
        params = _final_fair_ncl_params(base_params)
        params["fair_ncl_cov_weight"] = 0.0
        return params
    if method == "wo_augmented_ce":
        params = _final_fair_ncl_params(base_params)
        params["fair_ncl_aug_rec_weight"] = 0.0
        return params
    if method == "fair_ncl_alpha_tradeoff":
        params = dict(base_params)
        params["use_similarity_scores"] = True
        params["similarity_source"] = "cooccurrence"
        return params
    if method == "fair_ncl_semantic_alpha_tradeoff":
        params = dict(base_params)
        params["use_similarity_scores"] = True
        params["similarity_source"] = "semantic"
        return params
    if method == "fair_ncl_semantic_hybrid_alpha_tradeoff":
        return _final_fair_ncl_params(base_params)
    return dict(base_params)


def _comparison_params(method: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = _ablation_params(method, base_params)
    if method in {"afrl", "pfrec"}:
        params["frozen_backbone_init"] = "validation_reference"
    return params


def _final_fair_ncl_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(base_params)
    params["use_similarity_scores"] = True
    params["similarity_source"] = "semantic_hybrid"
    semantic_top_k = int(params.get("semantic_hybrid_top_k") or params.get("k_synonyms", 20))
    semantic_pool_size = int(params.get("semantic_hybrid_pool_size", 50))
    params["semantic_hybrid_top_k"] = semantic_top_k
    params["semantic_hybrid_pool_size"] = max(semantic_pool_size, semantic_top_k)
    params.setdefault("semantic_hashing_dim", 4096)
    return params


def _make_job(phase: str,
              dataset: str,
              method: str,
              backbone: str,
              seed: int,
              params: Dict[str, Any],
              reuse_from_phase: Optional[str] = None) -> ExperimentJob:
    params = dict(params or {})
    job_id = stable_job_id(phase, dataset, method, backbone, seed, params)
    return ExperimentJob(
        job_id=job_id,
        phase=phase,
        dataset=dataset,
        method=method,
        backbone=backbone,
        seed=int(seed),
        params=params,
        reuse_from_phase=reuse_from_phase,
    )


def _require_base_params(phase: str, base_params: Dict[str, Any]) -> None:
    if phase in BASE_PARAMS_REQUIRED_PHASES and not base_params:
        raise ValueError(
            f"Phase '{phase}' requires manual parameters. Provide --params-file or --params-json."
        )


def _require_pair_params(phase: str,
                         base_params: Dict[str, Any],
                         pairs: Sequence[Tuple[str, str]],
                         required_keys: Sequence[str]) -> None:
    missing_messages = []
    for dataset, backbone in pairs:
        resolved = _resolve_base_params(base_params, dataset, backbone)
        missing_keys = [key for key in required_keys if key not in resolved]
        if missing_keys:
            missing_messages.append(f"{dataset}:{backbone} missing {', '.join(missing_keys)}")
    if missing_messages:
        joined = "; ".join(missing_messages)
        raise ValueError(
            f"Phase '{phase}' is missing upstream selected parameters for: {joined}. "
            "Update the scoped params file before generating this plan."
        )


def _resolve_base_params(base_params: Dict[str, Any], dataset: str, backbone: str) -> Dict[str, Any]:
    if not base_params:
        return {}
    if not _is_scoped_params(base_params):
        return dict(base_params)

    resolved: Dict[str, Any] = {}
    default_params = base_params.get(SCOPED_DEFAULT_KEY, {})
    if isinstance(default_params, dict):
        resolved = merge_params(resolved, default_params)

    dataset_scopes = base_params.get(SCOPED_DATASETS_KEY, {})
    if isinstance(dataset_scopes, dict) and isinstance(dataset_scopes.get(dataset), dict):
        dataset_scope = dataset_scopes[dataset]
        resolved = merge_params(resolved, _plain_scope_params(dataset_scope))
        nested_backbones = dataset_scope.get(SCOPED_BACKBONES_KEY, {})
        if isinstance(nested_backbones, dict) and isinstance(nested_backbones.get(backbone), dict):
            resolved = merge_params(resolved, _plain_scope_params(nested_backbones[backbone]))

    backbone_scopes = base_params.get(SCOPED_BACKBONES_KEY, {})
    if isinstance(backbone_scopes, dict) and isinstance(backbone_scopes.get(backbone), dict):
        backbone_scope = backbone_scopes[backbone]
        resolved = merge_params(resolved, _plain_scope_params(backbone_scope))
        nested_datasets = backbone_scope.get(SCOPED_DATASETS_KEY, {})
        if isinstance(nested_datasets, dict) and isinstance(nested_datasets.get(dataset), dict):
            resolved = merge_params(resolved, _plain_scope_params(nested_datasets[dataset]))

    dataset_backbone_scopes = base_params.get(SCOPED_DATASET_BACKBONES_KEY, {})
    if isinstance(dataset_backbone_scopes, dict):
        dataset_scope = dataset_backbone_scopes.get(dataset, {})
        if isinstance(dataset_scope, dict) and isinstance(dataset_scope.get(backbone), dict):
            resolved = merge_params(resolved, _plain_scope_params(dataset_scope[backbone]))

    top_level_dataset = base_params.get(dataset)
    if isinstance(top_level_dataset, dict):
        resolved = merge_params(resolved, _plain_scope_params(top_level_dataset))

    top_level_backbone = base_params.get(backbone)
    if isinstance(top_level_backbone, dict):
        resolved = merge_params(resolved, _plain_scope_params(top_level_backbone))

    return resolved


def _resolve_method_params(base_params: Dict[str, Any],
                           dataset: str,
                           backbone: str,
                           method: str) -> Dict[str, Any]:
    resolved = _resolve_base_params(base_params, dataset, backbone)
    method_scopes = base_params.get(SCOPED_METHODS_KEY, {})
    if not isinstance(method_scopes, dict):
        return resolved
    method_scope = method_scopes.get(method)
    if not isinstance(method_scope, dict):
        return resolved
    return merge_params(resolved, _resolve_base_params(method_scope, dataset, backbone))


def _is_scoped_params(base_params: Dict[str, Any]) -> bool:
    if any(key in base_params for key in (
        SCOPED_DEFAULT_KEY,
        SCOPED_BACKBONES_KEY,
        SCOPED_DATASETS_KEY,
        SCOPED_DATASET_BACKBONES_KEY,
        SCOPED_METHODS_KEY,
    )):
        return True

    for key in BACKBONES + DATASETS:
        if isinstance(base_params.get(key), dict):
            return True
    return False


def _grid(options):
    keys = list(options.keys())
    values = [options[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _resolve_scope(phase: str,
                   datasets: Optional[Iterable[str]],
                   backbones: Optional[Iterable[str]]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    phase = phase.lower()
    if phase == "backbone":
        return _normalize_selection(datasets, BACKBONE_TUNING_DATASETS), _normalize_selection(backbones, BACKBONE_TUNING_BACKBONES)
    if phase == "augment":
        return _normalize_selection(datasets, AUGMENT_TUNING_DATASETS), _normalize_selection(backbones, AUGMENT_TUNING_BACKBONES)
    if phase == "loss":
        return _normalize_selection(datasets, LOSS_TUNING_DATASETS), _normalize_selection(backbones, LOSS_TUNING_BACKBONES)
    if phase == "ablation":
        return _normalize_selection(datasets, ABLATION_DATASETS), _normalize_selection(backbones, ABLATION_BACKBONES)
    if phase == "comparison":
        return _normalize_selection(datasets, COMPARISON_DATASETS), _normalize_selection(backbones, COMPARISON_BACKBONES)
    if phase in {"supplementary", "validation_reference", "validation_audit"}:
        return _normalize_selection(datasets, REVISION_DATASETS), _normalize_selection(backbones, BACKBONES)
    if phase == "mce_stability":
        return _normalize_selection(datasets, REVISION_DATASETS), _normalize_selection(backbones, ("caser",))
    return tuple(datasets or ()), tuple(backbones or ())


def _resolve_pairs(pairs: Optional[Iterable[str]],
                   datasets: Sequence[str],
                   backbones: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    if pairs is None:
        return tuple((dataset, backbone) for dataset, backbone in itertools.product(datasets, backbones))

    resolved: List[Tuple[str, str]] = []
    seen = set()
    for raw_pair in pairs:
        pair = str(raw_pair).strip()
        if ":" not in pair:
            raise ValueError(
                f"Invalid pair '{pair}'. Use the format <dataset>:<backbone>, "
                "for example taobao:bert4rec."
            )
        dataset, backbone = (part.strip().lower() for part in pair.split(":", 1))
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset in pair '{pair}': {dataset}")
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone in pair '{pair}': {backbone}")
        normalized = (dataset, backbone)
        if normalized not in seen:
            resolved.append(normalized)
            seen.add(normalized)
    return tuple(resolved)


def _resolve_ablation_methods(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if values is None:
        return tuple(ABLATION_METHODS)
    if isinstance(values, str):
        values = (values,)

    selected: List[str] = []
    seen = set()
    tokens = []
    for raw_value in values:
        for token in str(raw_value).replace(",", " ").split():
            normalized = token.strip().lower()
            if normalized:
                tokens.append(normalized)

    if not tokens:
        return tuple(ABLATION_METHODS)

    for token in tokens:
        if token in ABLATION_METHOD_PRESETS:
            candidates = ABLATION_METHOD_PRESETS[token]
        else:
            if token not in KNOWN_ABLATION_METHODS:
                valid = ", ".join(KNOWN_ABLATION_METHODS + tuple(ABLATION_METHOD_PRESETS.keys()))
                raise ValueError(f"Unknown ablation method or preset '{token}'. Valid values: {valid}")
            candidates = (token,)
        for method in candidates:
            if method not in seen:
                selected.append(method)
                seen.add(method)

    return tuple(selected)


def _plain_scope_params(scope: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in scope.items()
        if not (isinstance(key, str) and key.startswith("__"))
    }


def _normalize_selection(values: Optional[Iterable[str]], default: Sequence[str]) -> Tuple[str, ...]:
    if values is None:
        return tuple(default)
    normalized = tuple(values)
    if not normalized:
        return tuple(default)
    if len(normalized) == 1 and str(normalized[0]).lower() == "all":
        return tuple(default)
    return normalized


def save_plan(jobs: List[ExperimentJob], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(asdict(job), ensure_ascii=False) + "\n")
    print(f"Saved {len(jobs)} jobs to {output_path}")


def _load_params_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Params file must contain a JSON object: {path}")
    return data


def _load_params_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("params-json must decode to a JSON object.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate Fair-NCL experiment plans.")
    parser.add_argument("--phase", choices=list(PHASES), required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="Omit to use the phase default matrix.")
    parser.add_argument("--backbones", nargs="+", default=None, help="Omit to use the phase default matrix.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional explicit dataset:backbone list. When provided, it overrides the Cartesian dataset/backbone matrix.",
    )
    parser.add_argument(
        "--ablation-methods",
        nargs="+",
        default=None,
        help=(
            "Ablation methods or presets used only with --phase ablation/all. "
            "Presets: legacy, requested/new, main, alpha_tradeoff/experimental, "
            "semantic_alpha_tradeoff/semantic, semantic_hybrid, all/full, all_with_experimental."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Defaults to five final seeds for supplementary, three selection seeds for validation_reference/validation_audit, and 42 otherwise.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="JSON file with fixed parameters. Supports a flat object or scoped keys like __default__/__backbones__/__datasets__/__dataset_backbones__.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Inline JSON object with fixed parameters. Supports a flat object or scoped keys like __default__/__backbones__/__datasets__/__dataset_backbones__.",
    )
    parser.add_argument(
        "--params-overlay-file",
        default=None,
        help="Optional higher-priority JSON overlay, including validation-selected __methods__ scopes.",
    )
    parser.add_argument("--output", default="experiments/jobs.jsonl")
    args = parser.parse_args()

    base_params = merge_nested_params(
        _load_params_file(args.params_file),
        _load_params_file(args.params_overlay_file),
        _load_params_json(args.params_json),
    )
    jobs = build_experiment_plan(
        args.phase,
        args.datasets,
        args.backbones,
        args.seeds,
        base_params=base_params,
        pairs=args.pairs,
        ablation_methods=args.ablation_methods,
    )
    save_plan(jobs, args.output)


if __name__ == "__main__":
    main()

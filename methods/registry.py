from typing import Dict, Optional

from models.base_model import create_model

# Import backbone modules for registry side effects.
import models.bert4rec  # noqa: F401
import models.caser  # noqa: F401
import models.gru4rec  # noqa: F401
import models.sasrec  # noqa: F401

from .augmentations import (
    SequenceAugmenter,
    build_low_skew_item_pool,
    build_cooccurrence_similarity,
    compute_train_item_bias_statistics,
)
from .common import active_sensitive_attributes
from .semantic_similarity import build_semantic_embedding_similarity, build_semantic_hybrid_similarity
from .coserec import CoSeRecMethod, build_itemcf_iuf_similarity
from .wrappers import (
    AFSRMethod,
    AFRLMethod,
    AdvDebiasMethod,
    BackboneMethod,
    GRLMethod,
    NCLMethod,
    PFRecMethod,
    SMPCFRMethod,
)


METHOD_NAMES = (
    "baseline",
    "fair_ncl",
    "fair_ncl_factorial",
    "coserec",
    "ncl_only",
    "random_aug",
    "similarity_aug",
    "fair_ncl_alpha_tradeoff",
    "fair_ncl_semantic_alpha_tradeoff",
    "fair_ncl_semantic_hybrid_alpha_tradeoff",
    "ncl_sr_mce",
    "fair_aug_mce",
    "skew_only_aug",
    "fair_aug_no_geometry",
    "fair_aug_no_augmented_ce",
    "raw_skew_fair_ncl",
    "wo_fairness_sampling",
    "wo_semantic_sampling",
    "wo_alignment",
    "wo_variance",
    "wo_covariance",
    "wo_augmented_ce",
    "random_low_skew",
    "high_skew",
    "adv_debias",
    "grl",
    "sm_pcfr",
    "afrl",
    "pfrec",
    "a_fsr",
)


def create_method(method: str,
                  backbone_name: str,
                  config,
                  train_data: Optional[list] = None,
                  resources: Optional[Dict] = None,
                  item_metadata=None):
    """Create a method wrapper around a sequential backbone."""
    method = method.lower()
    if method not in METHOD_NAMES:
        raise ValueError(f"Unknown method '{method}'. Available: {METHOD_NAMES}")

    backbone = create_model(backbone_name, config)
    resources = resources or build_method_resources(config, train_data or [], item_metadata=item_metadata)

    if method == "coserec":
        if backbone_name != "sasrec":
            raise ValueError("CoSeRec is defined for the SASRec backbone only")
        return CoSeRecMethod(backbone, config, resources["coserec_similarity"])
    if method == "baseline":
        return BackboneMethod(backbone, config, name=method)
    if method == "adv_debias":
        return AdvDebiasMethod(backbone, config, name=method)
    if method == "grl":
        return GRLMethod(backbone, config, name=method)
    if method == "sm_pcfr":
        return SMPCFRMethod(backbone, config, name=method)
    if method == "afrl":
        return AFRLMethod(backbone, config, name=method)
    if method == "pfrec":
        return PFRecMethod(backbone, config, name=method)

    if method == "ncl_only":
        augmenter = _make_augmenter(config, "identity", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method == "random_aug":
        augmenter = _make_augmenter(config, "random", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method in {"similarity_aug", "wo_fairness_sampling"}:
        augmenter = _make_augmenter(config, "similarity", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method == "ncl_sr_mce":
        augmenter = _make_augmenter(config, "fair_ncl_alpha_tradeoff", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method in {"wo_semantic_sampling", "skew_only_aug"}:
        augmenter = _make_augmenter(config, "fair_ncl", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method in {
        "fair_ncl",
        "fair_ncl_factorial",
        "wo_alignment",
        "wo_variance",
        "wo_covariance",
        "wo_augmented_ce",
        "fair_aug_mce",
        "fair_aug_no_geometry",
        "fair_aug_no_augmented_ce",
        "raw_skew_fair_ncl",
    }:
        augmenter = _make_augmenter(config, "fair_ncl_alpha_tradeoff", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method in {
        "fair_ncl_alpha_tradeoff",
        "fair_ncl_semantic_alpha_tradeoff",
        "fair_ncl_semantic_hybrid_alpha_tradeoff",
    }:
        augmenter = _make_augmenter(config, "fair_ncl_alpha_tradeoff", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method == "random_low_skew":
        augmenter = _make_augmenter(config, "random_low_skew", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method == "high_skew":
        augmenter = _make_augmenter(config, "high_skew", resources)
        return NCLMethod(backbone, config, augmenter, name=method)
    if method == "a_fsr":
        return AFSRMethod(backbone, config, resources=resources, name=method)

    raise AssertionError("unreachable")


def build_method_resources(config,
                           train_data: list,
                           item_metadata=None,
                           train_event_sequences=None) -> Dict:
    method = str(getattr(config, "method", "baseline")).lower()
    bias_methods = {
        "fair_ncl",
        "fair_ncl_factorial",
        "fair_ncl_alpha_tradeoff",
        "fair_ncl_semantic_alpha_tradeoff",
        "fair_ncl_semantic_hybrid_alpha_tradeoff",
        "wo_semantic_sampling",
        "wo_alignment",
        "wo_variance",
        "wo_covariance",
        "wo_augmented_ce",
        "random_low_skew",
        "high_skew",
        "skew_only_aug",
        "fair_aug_mce",
        "fair_aug_no_geometry",
        "fair_aug_no_augmented_ce",
        "raw_skew_fair_ncl",
    }
    similarity_methods = {
        "fair_ncl",
        "fair_ncl_factorial",
        "fair_ncl_alpha_tradeoff",
        "fair_ncl_semantic_alpha_tradeoff",
        "fair_ncl_semantic_hybrid_alpha_tradeoff",
        "similarity_aug",
        "wo_fairness_sampling",
        "wo_semantic_sampling",
        "wo_alignment",
        "wo_variance",
        "wo_covariance",
        "wo_augmented_ce",
        "high_skew",
        "a_fsr",
        "ncl_sr_mce",
        "skew_only_aug",
        "fair_aug_mce",
        "fair_aug_no_geometry",
        "fair_aug_no_augmented_ce",
        "raw_skew_fair_ncl",
    }
    if method == "coserec":
        if not train_event_sequences:
            raise ValueError("CoSeRec requires train_event_sequences")
        similarity = build_itemcf_iuf_similarity(
            train_event_sequences, config.cache_dir, config.dataset,
            top_k=int(getattr(config, "coserec_similarity_top_k", 20)),
        )
        return {"coserec_similarity": similarity, "bias_statistics": {"scores": {}, "metric": None}}
    needs_bias = method in bias_methods
    needs_similarity = method in similarity_methods

    bias_statistics = compute_train_item_bias_statistics(
        train_data,
        sensitive_attrs=active_sensitive_attributes(config),
        train_event_sequences=train_event_sequences,
        metric=getattr(config, "bias_metric", "prior_jsd"),
        min_count=getattr(config, "bias_min_count", 5),
        prior_strength=getattr(config, "bias_prior_strength", 5.0),
        smoothing=getattr(config, "bias_smoothing", 1.0),
        prior_source=getattr(config, "bias_prior_source", "interaction"),
        event_source=getattr(config, "bias_event_source", "unique_train_events"),
    ) if train_data and needs_bias else {"scores": {}, "metric": None}
    item_bias_scores = bias_statistics["scores"]
    wants_scored_similarity = needs_similarity and (
        bool(getattr(config, "use_similarity_scores", False))
        or getattr(config, "method", "") in {
            "fair_ncl_alpha_tradeoff",
            "fair_ncl_semantic_alpha_tradeoff",
            "fair_ncl_semantic_hybrid_alpha_tradeoff",
            "fair_ncl",
            "fair_ncl_factorial",
            "wo_alignment",
            "wo_variance",
            "wo_covariance",
            "wo_augmented_ce",
            "ncl_sr_mce",
            "fair_aug_mce",
            "fair_aug_no_geometry",
            "fair_aug_no_augmented_ce",
            "raw_skew_fair_ncl",
        }
    )
    similarity_source = str(getattr(config, "similarity_source", "cooccurrence")).lower()
    if not needs_similarity:
        similarity_candidates = {}
    elif wants_scored_similarity and similarity_source in {"semantic_hybrid", "hybrid_semantic"}:
        if item_metadata is None:
            raise ValueError(
                "Semantic hybrid similarity requires item metadata with item_text. "
                "Pass data['items'] or data['movies'] into build_method_resources."
            )
        semantic_top_k = getattr(config, "semantic_hybrid_top_k", None)
        semantic_top_k = int(semantic_top_k or getattr(config, "k_synonyms", 20))
        semantic_pool_size = int(getattr(config, "semantic_hybrid_pool_size", semantic_top_k))
        semantic_pool_size = max(semantic_pool_size, semantic_top_k)
        candidate_pools = build_cooccurrence_similarity(
            train_data,
            num_items=config.num_items,
            top_k=semantic_pool_size,
            window_size=getattr(config, "similarity_window", 20),
            return_scores=False,
        ) if train_data else {}
        similarity_candidates = build_semantic_hybrid_similarity(
            item_metadata=item_metadata,
            num_items=config.num_items,
            candidate_pools=candidate_pools,
            top_k=semantic_top_k,
            cache_dir=getattr(config, "cache_dir", "cache"),
            hashing_dim=getattr(config, "semantic_hashing_dim", 4096),
        )
    elif wants_scored_similarity and similarity_source == "semantic":
        if item_metadata is None:
            raise ValueError(
                "Semantic similarity requires item metadata with item_text. "
                "Pass data['items'] or data['movies'] into build_method_resources."
            )
        similarity_candidates = build_semantic_embedding_similarity(
            item_metadata=item_metadata,
            num_items=config.num_items,
            top_k=getattr(config, "k_synonyms", 20),
            model_name=getattr(config, "semantic_embedding_model", "BAAI/bge-m3"),
            cache_dir=getattr(config, "cache_dir", "cache"),
            encode_batch_size=getattr(config, "semantic_embedding_batch_size", 128),
            retrieval_batch_size=getattr(config, "semantic_retrieval_batch_size", 512),
            device=getattr(config, "semantic_embedding_device", None),
            search_device=getattr(config, "semantic_search_device", None),
            text_prefix=getattr(config, "semantic_text_prefix", None),
        )
    else:
        similarity_candidates = build_cooccurrence_similarity(
            train_data,
            num_items=config.num_items,
            top_k=getattr(config, "k_synonyms", 20),
            window_size=getattr(config, "similarity_window", 20),
            return_scores=wants_scored_similarity,
            candidate_pool_multiplier=getattr(config, "similarity_candidate_pool_multiplier", 5),
        ) if train_data else {}
    low_skew_items = build_low_skew_item_pool(
        item_bias_scores,
        top_k=getattr(config, "low_skew_pool_size", 500),
    )
    return {
        "similarity_candidates": similarity_candidates,
        "item_bias_scores": item_bias_scores,
        "low_skew_items": low_skew_items,
        "bias_statistics": bias_statistics,
    }


def _make_augmenter(config, mode: str, resources: Dict) -> SequenceAugmenter:
    return SequenceAugmenter(
        num_items=config.num_items,
        max_seq_len=config.max_seq_len,
        mode=mode,
        augment_ratio=getattr(config, "augment_ratio", 0.2),
        similarity_candidates=resources.get("similarity_candidates", {}),
        item_bias_scores=resources.get("item_bias_scores", {}),
        low_skew_items=resources.get("low_skew_items", []),
        utility_alpha=getattr(config, "utility_alpha", 0.7),
        utility_beta=getattr(config, "utility_beta", 0.3),
        epsilon=getattr(config, "epsilon", 1.0),
        unknown_bias_score=getattr(config, "bias_unknown_score", 1.0),
        seed=getattr(config, "seed", 42),
    )






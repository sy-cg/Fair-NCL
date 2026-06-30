from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        key: _tensor_to_device(value, device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _tensor_to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if value.device.type == device.type:
        if device.type != "cuda" or device.index is None or value.device.index == device.index:
            return value
    return value.to(device, non_blocking=True)


def mask_seen_items(full_scores: torch.Tensor,
                    input_seq: torch.Tensor,
                    target: torch.Tensor,
                    pad_idx: int = 0) -> torch.Tensor:
    """Mask history items for full-ranking evaluation while keeping the target rankable."""
    if full_scores.dim() != 2:
        raise ValueError(f"Full-ranking scores must be 2D [B, N], got {tuple(full_scores.shape)}")

    scores = full_scores
    mask_value = torch.finfo(scores.dtype).min

    target = target.long().view(-1)
    target_scores = scores.gather(1, target.unsqueeze(1)).clone()

    seen_mask = torch.zeros_like(scores, dtype=torch.bool)
    history_index = input_seq.long().clamp(min=0, max=scores.size(1) - 1)
    history_valid = input_seq > 0
    seen_mask.scatter_(1, history_index, history_valid)
    seen_mask[:, pad_idx] = True

    scores = scores.masked_fill(seen_mask, mask_value)
    scores.scatter_(1, target.unsqueeze(1), target_scores)
    return scores


def compute_target_ranks(full_scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the 1-based rank of the target item for each user."""
    target = target.long().view(-1)
    target_scores = full_scores.gather(1, target.unsqueeze(1))
    higher = (full_scores > target_scores).sum(dim=1)
    return higher + 1


def evaluate_full_ranking_loader(data_loader,
                                 config,
                                 predict_fn: Callable[[Dict], torch.Tensor],
                                 sensitive_attributes: Optional[Iterable[str]] = None,
                                 desc: str = "evaluate",
                                 legacy_output: bool = False) -> Dict[str, object]:
    """Evaluate a model with full-ranking over the complete item space."""
    topk_list = _normalize_topk(getattr(config, "topk_list", [5, 10, 20]))
    sensitive_attributes = list(sensitive_attributes or getattr(config, "sensitive_attributes", []))

    utility_sums = defaultdict(float)
    group_sums = {
        attr: {
            "HitRate": {k: defaultdict(float) for k in topk_list},
            "NDCG": {k: defaultdict(float) for k in topk_list},
            "count": defaultdict(int),
        }
        for attr in sensitive_attributes
    }
    user_count = 0

    for raw_batch in tqdm(data_loader, desc=desc, leave=False):
        batch = move_batch_to_device(raw_batch, config.device)
        with autocast(enabled=getattr(config, "use_mixed_precision", False)):
            full_scores = predict_fn(batch)

        if full_scores.dim() != 2:
            raise ValueError(f"Predict function must return 2D full scores, got {tuple(full_scores.shape)}")

        target = batch["target"].long().view(-1)
        masked_scores = mask_seen_items(full_scores, batch["input_seq"], target)
        ranks = compute_target_ranks(masked_scores, target)
        batch_metrics = _metrics_from_ranks(ranks, topk_list)

        batch_size = int(target.size(0))
        user_count += batch_size

        for k in topk_list:
            utility_sums[f"HitRate@{k}"] += float(batch_metrics[k]["HitRate"].sum().item())
            utility_sums[f"Precision@{k}"] += float(batch_metrics[k]["Precision"].sum().item())
            utility_sums[f"Recall@{k}"] += float(batch_metrics[k]["Recall"].sum().item())
            utility_sums[f"NDCG@{k}"] += float(batch_metrics[k]["NDCG"].sum().item())
            utility_sums[f"MRR@{k}"] += float(batch_metrics[k]["MRR"].sum().item())

        for attr in sensitive_attributes:
            if attr not in batch:
                continue
            attr_values = batch[attr].detach().cpu().long()
            for group in attr_values.unique().tolist():
                group_mask = attr_values == int(group)
                group_sums[attr]["count"][int(group)] += int(group_mask.sum().item())
                for k in topk_list:
                    group_sums[attr]["HitRate"][k][int(group)] += float(
                        batch_metrics[k]["HitRate"][group_mask.to(batch_metrics[k]["HitRate"].device)].sum().item()
                    )
                    group_sums[attr]["NDCG"][k][int(group)] += float(
                        batch_metrics[k]["NDCG"][group_mask.to(batch_metrics[k]["NDCG"].device)].sum().item()
                    )

    utility = _finalize_utility(utility_sums, user_count, topk_list)
    fairness = _finalize_group_fairness(group_sums, topk_list)
    selected = _selected_metrics(utility, fairness)

    if legacy_output:
        return {
            "performance": _legacy_performance_metrics(utility, topk_list),
            "fairness": _legacy_fairness_metrics(fairness, topk_list),
            "selected": selected,
        }

    return {
        "utility": utility,
        "fairness": fairness,
        "selected": selected,
    }


def _normalize_topk(topk_list: Iterable[int]) -> List[int]:
    values = sorted({int(k) for k in topk_list if int(k) > 0})
    return values or [10]


def _metrics_from_ranks(ranks: torch.Tensor, topk_list: Iterable[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    ranks = ranks.float()
    metrics = {}
    for k in topk_list:
        hits = (ranks <= k).float()
        discount = torch.log2(ranks + 1.0)
        ndcg = torch.where(hits > 0, 1.0 / discount, torch.zeros_like(ranks))
        mrr = torch.where(hits > 0, 1.0 / ranks, torch.zeros_like(ranks))
        metrics[int(k)] = {
            "HitRate": hits,
            "Precision": hits / float(k),
            "Recall": hits,
            "NDCG": ndcg,
            "MRR": mrr,
        }
    return metrics


def _finalize_utility(utility_sums: Dict[str, float], user_count: int, topk_list: Iterable[int]) -> Dict[str, float]:
    if user_count <= 0:
        return {f"{metric}@{k}": 0.0 for k in topk_list for metric in ("HitRate", "Precision", "Recall", "NDCG", "MRR")}

    utility = {}
    for k in topk_list:
        for metric in ("HitRate", "Precision", "Recall", "NDCG", "MRR"):
            key = f"{metric}@{k}"
            utility[key] = float(utility_sums.get(key, 0.0) / user_count)
    return utility


def _finalize_group_fairness(group_sums: Dict[str, Dict], topk_list: Iterable[int]) -> Dict[str, Dict[str, object]]:
    fairness: Dict[str, Dict[str, object]] = {}
    for attr, attr_stats in group_sums.items():
        counts = attr_stats["count"]
        attr_result: Dict[str, object] = {}

        for k in topk_list:
            group_hit = {}
            group_ndcg = {}
            for group, count in counts.items():
                if count <= 0:
                    continue
                group_hit[int(group)] = float(attr_stats["HitRate"][k][group] / count)
                group_ndcg[int(group)] = float(attr_stats["NDCG"][k][group] / count)

            hit_values = list(group_hit.values())
            ndcg_values = list(group_ndcg.values())

            attr_result[f"group_hitrate@{k}"] = group_hit
            attr_result[f"hitrate@{k}_gap"] = float(max(hit_values) - min(hit_values)) if len(hit_values) >= 2 else 0.0
            attr_result[f"hitrate@{k}_worst_group"] = float(min(hit_values)) if hit_values else 0.0
            attr_result[f"hitrate@{k}_std"] = float(np.std(hit_values)) if hit_values else 0.0

            attr_result[f"group_ndcg@{k}"] = group_ndcg
            attr_result[f"ndcg@{k}_gap"] = float(max(ndcg_values) - min(ndcg_values)) if len(ndcg_values) >= 2 else 0.0
            attr_result[f"ndcg@{k}_worst_group"] = float(min(ndcg_values)) if ndcg_values else 0.0
            attr_result[f"ndcg@{k}_std"] = float(np.std(ndcg_values)) if ndcg_values else 0.0

        fairness[attr] = attr_result
    return fairness


def _selected_metrics(utility: Dict[str, float], fairness: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    selected = {
        "HitRate@10": utility.get("HitRate@10", 0.0),
        "NDCG@10": utility.get("NDCG@10", 0.0),
    }
    for attr, attr_metrics in fairness.items():
        selected[f"{attr}_HitRate@10_Gap"] = float(attr_metrics.get("hitrate@10_gap", 0.0))
        selected[f"{attr}_NDCG@10_Gap"] = float(attr_metrics.get("ndcg@10_gap", 0.0))
    return selected


def _legacy_performance_metrics(utility: Dict[str, float], topk_list: Iterable[int]) -> Dict[str, float]:
    performance = dict(utility)
    for k in topk_list:
        performance[f"HR@{k}"] = performance.get(f"HitRate@{k}", 0.0)
    return performance


def _legacy_fairness_metrics(fairness: Dict[str, Dict[str, object]], topk_list: Iterable[int]) -> Dict[str, Dict[str, object]]:
    ref_k = 10 if 10 in topk_list else list(topk_list)[0]
    results: Dict[str, Dict[str, object]] = {}

    for attr, attr_metrics in fairness.items():
        group_rates = attr_metrics.get(f"group_hitrate@{ref_k}", {})
        group_ndcg = attr_metrics.get(f"group_ndcg@{ref_k}", {})
        hit_gap = float(attr_metrics.get(f"hitrate@{ref_k}_gap", 0.0))
        ndcg_gap = float(attr_metrics.get(f"ndcg@{ref_k}_gap", 0.0))

        spd = np.nan
        di = np.nan
        if len(group_rates) >= 2:
            ordered = [group_rates[key] for key in sorted(group_rates.keys())[:2]]
            spd = float(ordered[0] - ordered[1])
            di = float(ordered[0] / (ordered[1] + 1e-8))

        results[attr] = {
            "demographic_parity": float(max(0.0, 1.0 - hit_gap)),
            "equalized_opportunity": float(max(0.0, 1.0 - hit_gap)),
            "group_utility_parity": float(max(0.0, 1.0 - ndcg_gap)),
            "group_utility_gap": ndcg_gap,
            "worst_group_utility": float(attr_metrics.get(f"ndcg@{ref_k}_worst_group", 0.0)),
            "statistical_parity_difference": spd,
            "disparate_impact": di,
            "group_rates": group_rates,
            "group_tpr": group_rates,
            "group_ndcg": group_ndcg,
            "group_hit_rate": group_rates,
        }

    return results

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np


PRIMARY_UTILITY_METRICS = ("HitRate@10", "NDCG@10")
PRIMARY_FAIRNESS_METRICS = ("hitrate@10_gap", "ndcg@10_gap")


def ranking_metrics(scores: np.ndarray,
                    targets: np.ndarray,
                    k_list: Iterable[int] = (5, 10, 20)) -> Dict[str, float]:
    """Compute common top-k metrics for sampled or full ranking evaluation.

    scores: [num_users, num_candidates]
    targets: either [num_users] target candidate indices or one-hot/binary
             matrix with the same shape as scores.
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    result = {}

    for k in k_list:
        k_eff = min(int(k), scores.shape[1])
        topk = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        topk_scores = np.take_along_axis(scores, topk, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        topk = np.take_along_axis(topk, order, axis=1)

        hits, precisions, recalls, ndcgs, mrrs = [], [], [], [], []
        for row, items in enumerate(topk):
            positives = _positive_indices(targets[row], scores.shape[1])
            if len(positives) == 0:
                continue

            hit_count = 0
            dcg = 0.0
            rr = 0.0
            positives_set = set(positives)

            for rank, item in enumerate(items):
                if int(item) in positives_set:
                    hit_count += 1
                    dcg += 1.0 / math.log2(rank + 2)
                    if rr == 0.0:
                        rr = 1.0 / (rank + 1)

            idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(len(positives), k_eff)))
            hits.append(1.0 if hit_count > 0 else 0.0)
            precisions.append(hit_count / k_eff)
            recalls.append(hit_count / len(positives))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            mrrs.append(rr)

        result[f"HitRate@{k}"] = float(np.mean(hits)) if hits else 0.0
        result[f"Precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        result[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        result[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        result[f"MRR@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0

    return result


def group_utility_metrics(scores: np.ndarray,
                          targets: np.ndarray,
                          group_values: np.ndarray,
                          k: int = 10,
                          utility_metric: str = "NDCG") -> Dict[str, object]:
    """Compute user-side group utility fairness metrics."""
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    group_values = np.asarray(group_values)

    group_scores = {}
    for group in sorted(np.unique(group_values).tolist()):
        mask = group_values == group
        if not np.any(mask):
            continue
        group_metric = ranking_metrics(scores[mask], targets[mask], [k])
        group_scores[int(group)] = group_metric.get(f"{utility_metric}@{k}", 0.0)

    values = list(group_scores.values())
    gap = max(values) - min(values) if values else 0.0
    return {
        f"group_{utility_metric.lower()}@{k}": group_scores,
        f"{utility_metric.lower()}@{k}_gap": float(gap),
        f"{utility_metric.lower()}@{k}_worst_group": float(min(values)) if values else 0.0,
        f"{utility_metric.lower()}@{k}_std": float(np.std(values)) if values else 0.0,
    }


def all_group_fairness(scores: np.ndarray,
                       targets: np.ndarray,
                       sensitive_attrs: Dict[str, np.ndarray],
                       k_list: Iterable[int] = (5, 10, 20)) -> Dict[str, Dict[str, object]]:
    fairness = {}
    for attr, values in sensitive_attrs.items():
        fairness[attr] = {}
        for k in k_list:
            fairness[attr].update(group_utility_metrics(scores, targets, values, k=k, utility_metric="NDCG"))
            fairness[attr].update(group_utility_metrics(scores, targets, values, k=k, utility_metric="HitRate"))
    return fairness


def selected_report_metrics(scores: np.ndarray,
                            targets: np.ndarray,
                            sensitive_attrs: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Return the four primary metrics used in the main paper tables."""
    utility = ranking_metrics(scores, targets, [10])
    fairness = all_group_fairness(scores, targets, sensitive_attrs, [10])

    selected = {
        "HitRate@10": utility.get("HitRate@10", 0.0),
        "NDCG@10": utility.get("NDCG@10", 0.0),
    }
    for attr, attr_metrics in fairness.items():
        selected[f"{attr}_HitRate@10_Gap"] = attr_metrics.get("hitrate@10_gap", 0.0)
        selected[f"{attr}_NDCG@10_Gap"] = attr_metrics.get("ndcg@10_gap", 0.0)
    return selected


def _positive_indices(target_row: np.ndarray, num_candidates: int) -> List[int]:
    target_row = np.asarray(target_row)
    if target_row.ndim == 0:
        return [int(target_row)]
    if target_row.shape[0] == num_candidates:
        return np.flatnonzero(target_row > 0).astype(int).tolist()
    return [int(x) for x in target_row.reshape(-1).tolist() if int(x) >= 0]

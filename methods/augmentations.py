import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


class SequenceAugmenter:
    """Sequence augmentation policies used by Fair-NCL and ablations."""

    def __init__(self,
                 num_items: int,
                 max_seq_len: int,
                 mode: str = "none",
                 augment_ratio: float = 0.2,
                 similarity_candidates: Optional[Dict[int, Sequence[Any]]] = None,
                 item_bias_scores: Optional[Dict[int, float]] = None,
                 low_skew_items: Optional[Sequence[int]] = None,
                 utility_alpha: float = 0.7,
                 utility_beta: float = 0.3,
                 epsilon: float = 1.0,
                 seed: int = 42):
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.mode = mode
        self.augment_ratio = float(augment_ratio)
        self.similarity_candidates = similarity_candidates or {}
        self.item_bias_scores = item_bias_scores or {}
        self.low_skew_items = [int(item) for item in (low_skew_items or []) if int(item) > 0]
        self.utility_alpha = float(utility_alpha)
        self.utility_beta = float(utility_beta)
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)

    def augment(self, input_seq: torch.Tensor) -> torch.Tensor:
        if self.mode in {"none", "identity"} or self.augment_ratio <= 0:
            return input_seq.clone()
        if self.mode == "random":
            return self._random_replace(input_seq)
        if self.mode == "random_low_skew":
            return self._random_low_skew_replace(input_seq)
        if self.mode == "similarity":
            return self._similarity_replace(input_seq, sampling_mode="uniform")
        if self.mode == "fair_ncl":
            return self._similarity_replace(input_seq, sampling_mode="low_skew")
        if self.mode == "fair_ncl_alpha_tradeoff":
            return self._similarity_replace(input_seq, sampling_mode="alpha_tradeoff")
        if self.mode == "high_skew":
            return self._similarity_replace(input_seq, sampling_mode="high_skew")
        raise ValueError(f"Unknown augmentation mode: {self.mode}")

    def _random_replace(self, input_seq: torch.Tensor) -> torch.Tensor:
        out = input_seq.clone()
        for row in range(out.size(0)):
            positions = self._sample_positions(out[row])
            for pos in positions:
                out[row, pos] = self.rng.randint(1, self.num_items)
        return out

    def _random_low_skew_replace(self, input_seq: torch.Tensor) -> torch.Tensor:
        out = input_seq.clone()
        fallback_pool = [item for item in self.low_skew_items if item <= self.num_items]
        for row in range(out.size(0)):
            positions = self._sample_positions(out[row])
            for pos in positions:
                item = int(out[row, pos].item())
                candidates = [cand for cand in fallback_pool if cand != item]
                if candidates:
                    out[row, pos] = self.rng.choice(candidates)
                else:
                    out[row, pos] = self.rng.randint(1, self.num_items)
        return out

    def _similarity_replace(self, input_seq: torch.Tensor, sampling_mode: str) -> torch.Tensor:
        out = input_seq.clone()
        for row in range(out.size(0)):
            positions = self._sample_positions(out[row])
            for pos in positions:
                item = int(out[row, pos].item())
                candidates = self.similarity_candidates.get(item)
                if not candidates:
                    candidates = self.similarity_candidates.get(0, [])
                candidates = self._normalize_candidates(item, candidates)
                if not candidates:
                    continue
                out[row, pos] = self._sample_candidate(item, candidates, sampling_mode)
        return out

    def _normalize_candidates(self, item: int, candidates: Sequence[Any]) -> List[Tuple[int, float]]:
        normalized: List[Tuple[int, float]] = []
        for raw_candidate in candidates:
            candidate = raw_candidate
            similarity_score = 1.0
            if isinstance(raw_candidate, dict):
                candidate = raw_candidate.get("candidate", raw_candidate.get("item", raw_candidate.get("id")))
                similarity_score = raw_candidate.get("score", raw_candidate.get("similarity", 1.0))
            elif isinstance(raw_candidate, (tuple, list)) and len(raw_candidate) >= 2:
                candidate = raw_candidate[0]
                similarity_score = raw_candidate[1]

            try:
                candidate_id = int(candidate)
                score = float(similarity_score)
            except (TypeError, ValueError):
                continue

            if candidate_id <= 0 or candidate_id > self.num_items or candidate_id == item:
                continue
            normalized.append((candidate_id, max(0.0, min(1.0, score))))
        return normalized

    def _sample_positions(self, seq: torch.Tensor) -> List[int]:
        valid = torch.nonzero(seq > 0, as_tuple=False).flatten().tolist()
        if not valid:
            return []
        n_replace = max(1, int(round(len(valid) * self.augment_ratio)))
        n_replace = min(n_replace, len(valid))
        return self.rng.sample(valid, n_replace)

    def _sample_candidate(self, item: int, candidates: List[Tuple[int, float]], sampling_mode: str) -> int:
        if sampling_mode == "uniform":
            return self.rng.choice(candidates)[0]

        base_bias = self.item_bias_scores.get(item, 0.0)
        utilities = []
        for cand, similarity_score in candidates:
            cand_bias = self.item_bias_scores.get(cand, 0.0)
            if sampling_mode == "low_skew":
                similarity_term = 1.0
                bias_delta = max(0.0, base_bias - cand_bias)
            elif sampling_mode == "alpha_tradeoff":
                similarity_term = similarity_score
                bias_delta = max(0.0, base_bias - cand_bias)
            elif sampling_mode == "high_skew":
                similarity_term = 1.0
                bias_delta = max(0.0, cand_bias - base_bias)
            else:
                raise ValueError(f"Unknown candidate sampling mode: {sampling_mode}")
            utility = self.utility_alpha * similarity_term + self.utility_beta * bias_delta
            utilities.append(utility)

        max_u = max(utilities)
        weights = [pow(2.718281828, self.epsilon * (u - max_u)) for u in utilities]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(candidates)[0]

        threshold = self.rng.random() * total
        acc = 0.0
        for (cand, _), weight in zip(candidates, weights):
            acc += weight
            if acc >= threshold:
                return cand
        return candidates[-1][0]


def build_cooccurrence_similarity(train_data: Iterable[Dict],
                                  num_items: int,
                                  top_k: int = 20,
                                  window_size: int = 20,
                                  return_scores: bool = False,
                                  candidate_pool_multiplier: int = 5) -> Dict[int, List]:
    """Build sparse item neighbors plus one shared fallback list.

    Returning a sparse dictionary is substantially cheaper than materializing a
    top-k neighbor list for every item id, especially on large Taobao item
    spaces.

    When return_scores=True, neighbors are first expanded to a larger
    co-occurrence pool, then re-ranked by co-occurrence-context cosine and
    returned as (item_id, cosine_score). This gives the alpha/beta sampling
    variant a real per-candidate semantic-preservation term without changing
    the default Fair-NCL behavior.
    """
    counts = defaultdict(lambda: defaultdict(int))
    popularity = defaultdict(int)
    effective_window = max(2, int(window_size))

    for sample in train_data:
        seq = [int(x) for x in sample.get("input_seq", []) if int(x) > 0]
        target = int(sample.get("target", 0))
        window = seq[-effective_window:] + ([target] if target > 0 else [])
        unique = list(dict.fromkeys(window))
        for item in unique:
            popularity[item] += 1
        for i, item_i in enumerate(unique):
            for item_j in unique[i + 1:]:
                counts[item_i][item_j] += 1
                counts[item_j][item_i] += 1

    global_popular = [item for item, _ in sorted(popularity.items(), key=lambda kv: kv[1], reverse=True)]
    fallback = [item for item in global_popular if 0 < item <= num_items][:top_k]

    pool_multiplier = max(1, int(candidate_pool_multiplier))
    candidate_pool_size = top_k * pool_multiplier if return_scores else top_k

    neighbors = {0: [(item, 0.0) for item in fallback] if return_scores else fallback}
    for item, item_counts in counts.items():
        ranked = sorted(item_counts.items(), key=lambda kv: kv[1], reverse=True)
        item_pool = [(cand, count) for cand, count in ranked if cand != item][:candidate_pool_size]
        if not item_pool:
            continue
        if return_scores:
            scored_neighbors = [
                (int(cand), _sparse_cosine(item_counts, counts.get(cand, {})), int(count))
                for cand, count in item_pool
            ]
            scored_neighbors.sort(key=lambda kv: (kv[1], kv[2]), reverse=True)
            neighbors[int(item)] = [
                (cand, score)
                for cand, score, _ in scored_neighbors[:top_k]
            ]
        else:
            neighbors[int(item)] = [cand for cand, _ in item_pool]
    return neighbors


def _sparse_cosine(left: Dict[int, float], right: Dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left

    dot = 0.0
    for key, value in left.items():
        dot += float(value) * float(right.get(key, 0.0))

    if dot <= 0.0:
        return 0.0
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def compute_train_item_bias_scores(train_data: Iterable[Dict],
                                   sensitive_attrs: Sequence[str] = ("gender", "age_group"),
                                   smoothing: float = 1.0) -> Dict[int, float]:
    """Estimate train-only item demographic skew for fairness-aware augmentation."""
    item_group_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    item_total = defaultdict(float)

    for sample in train_data:
        items = [int(x) for x in sample.get("input_seq", []) if int(x) > 0]
        target = int(sample.get("target", 0))
        if target > 0:
            items.append(target)
        for item in items:
            item_total[item] += 1.0
            for attr in sensitive_attrs:
                if attr in sample:
                    item_group_counts[item][attr][int(sample[attr])] += 1.0

    bias_scores = {}
    for item, attr_counts in item_group_counts.items():
        item_bias = 0.0
        for groups in attr_counts.values():
            values = list(groups.values())
            if len(values) < 2:
                values = values + [0.0]
            total = sum(values) + smoothing * len(values)
            probs = [(v + smoothing) / total for v in values]
            item_bias += max(probs) - min(probs)
        bias_scores[int(item)] = item_bias / max(1, len(attr_counts))
    return bias_scores


def build_low_skew_item_pool(item_bias_scores: Dict[int, float],
                             top_k: int = 500) -> List[int]:
    """Return globally low-skew items for the random low-skew ablation."""
    if not item_bias_scores:
        return []
    limit = max(1, int(top_k))
    ranked = sorted(
        ((int(item), float(score)) for item, score in item_bias_scores.items() if int(item) > 0),
        key=lambda kv: (kv[1], kv[0]),
    )
    return [item for item, _ in ranked[:limit]]

import math
from collections import defaultdict

import numpy as np
import torch
from torch.cuda.amp import autocast


class OptimizedDifferentialPrivacyAugmenter:
    """Exponential-mechanism-inspired fairness-aware sequence augmenter.

    This is DP-inspired controlled perturbation rather than a full privacy
    guarantee for the whole recommender pipeline. Sensitive attributes are used
    only during training to estimate directional item bias and sample local,
    preference-preserving replacements.
    """

    def __init__(self, config, fair_synonyms, movie_bias_scores):
        self.config = config
        self.device = config.device
        self.epsilon = getattr(config, 'epsilon', 1.0)
        self.augment_ratio = getattr(config, 'augment_ratio', 0.2)
        self.alpha = getattr(config, 'utility_alpha', 0.7)
        self.beta = getattr(config, 'utility_beta', 0.3)
        self.sensitivity = getattr(config, 'utility_sensitivity', 1.0)
        self.fair_synonyms = fair_synonyms or {}
        self.movie_bias_scores = movie_bias_scores or {}
        self.sensitive_attributes = getattr(config, 'sensitive_attributes', ['gender', 'age_group'])
        self.attribute_dims = getattr(config, 'attribute_dims', {
            'gender': 2,
            'age_group': 2
        })

    def augment_batch_optimized(self, batch):
        with autocast(enabled=getattr(self.config, 'use_mixed_precision', False)):
            return self._augment_batch_gpu(batch)

    def augment_batch(self, batch):
        return self._augment_batch_gpu(batch)

    def _augment_batch_gpu(self, batch):
        if not self.fair_synonyms:
            return batch

        input_seq = batch['input_seq'].clone()
        valid_mask = input_seq > 0
        lengths = valid_mask.sum(dim=1)

        raw_num_aug = (lengths.float() * self.augment_ratio).long()
        num_aug = torch.maximum(torch.ones_like(raw_num_aug), torch.minimum(raw_num_aug, lengths))

        for row in range(input_seq.size(0)):
            positions = torch.nonzero(valid_mask[row], as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                continue

            selected = positions[torch.randperm(positions.numel(), device=input_seq.device)[:num_aug[row]]]
            for pos in selected:
                original_item = int(input_seq[row, pos].item())
                replacement = self._sample_replacement(original_item, batch, row)
                input_seq[row, pos] = replacement

        new_batch = batch.copy()
        new_batch['input_seq'] = input_seq
        return new_batch

    def _sample_replacement(self, original_item, batch, row):
        candidate_ids, similarities = self._candidate_ids_and_similarities(original_item)
        if not candidate_ids:
            return original_item

        original_bias = self._directional_bias(original_item, batch, row)
        utilities = []

        for candidate, similarity in zip(candidate_ids, similarities):
            candidate_bias = self._directional_bias(candidate, batch, row)
            fairness_gain = max(0.0, original_bias - candidate_bias)
            utility = self.alpha * similarity + self.beta * fairness_gain
            utilities.append(max(0.0, min(1.0, utility)))

        utilities = torch.tensor(utilities, dtype=torch.float32, device=self.device)
        scaled = self.epsilon * utilities / (2.0 * max(self.sensitivity, 1e-8))
        probs = torch.softmax(scaled, dim=0)
        sampled_idx = torch.multinomial(probs, 1).item()
        return int(candidate_ids[sampled_idx])

    def _candidate_ids_and_similarities(self, item_id):
        candidates = self.fair_synonyms.get(item_id, [])

        if isinstance(candidates, dict):
            ids = [int(k) for k in candidates.keys()]
            sims = [float(v) for v in candidates.values()]
        else:
            ids = [int(x) for x in candidates]
            sims = [1.0 for _ in ids]

        filtered_ids = []
        filtered_sims = []
        for candidate_id, sim in zip(ids, sims):
            if candidate_id > 0 and candidate_id != item_id:
                filtered_ids.append(candidate_id)
                filtered_sims.append(max(0.0, min(1.0, sim)))

        return filtered_ids, filtered_sims

    def _directional_bias(self, item_id, batch, row):
        info = self.movie_bias_scores.get(int(item_id))
        if not info:
            return 0.0

        group_probs = info.get('group_probs', {})
        values = []

        for attr in self.sensitive_attributes:
            if attr not in batch or attr not in group_probs:
                continue

            label = int(batch[attr][row].item())
            probs = group_probs[attr]
            if label < 0 or label >= len(probs):
                continue

            own_prob = probs[label]
            others = [p for idx, p in enumerate(probs) if idx != label]
            other_prob = float(np.mean(others)) if others else 0.0
            values.append(max(0.0, own_prob - other_prob))

        return float(np.mean(values)) if values else 0.0


def compute_movie_bias_scores_optimized(train_data, users, movies, config):
    """Estimate item directional bias from training interactions only."""
    print("Computing item directional bias scores from training data...")

    sensitive_attributes = getattr(config, 'sensitive_attributes', ['gender', 'age_group'])
    attribute_dims = getattr(config, 'attribute_dims', {
        'gender': 2,
        'age_group': 2
    })

    counts = defaultdict(lambda: {
        attr: np.zeros(attribute_dims[attr], dtype=np.float64)
        for attr in sensitive_attributes
        if attr in attribute_dims
    })
    totals = defaultdict(int)

    for sample in train_data:
        item = int(sample['target'])
        if item <= 0:
            continue
        totals[item] += 1
        for attr in sensitive_attributes:
            if attr not in sample or attr not in attribute_dims:
                continue
            label = int(sample[attr])
            if 0 <= label < attribute_dims[attr]:
                counts[item][attr][label] += 1.0

    min_count = getattr(config, 'bias_min_count', 5)
    smoothing = getattr(config, 'bias_smoothing', 1.0)
    bias_scores = {}

    for item, total in totals.items():
        if total < min_count:
            continue

        group_probs = {}
        attr_bias_values = []
        for attr, attr_counts in counts[item].items():
            smoothed = attr_counts + smoothing
            probs = smoothed / smoothed.sum()
            group_probs[attr] = probs.tolist()

            if len(probs) > 1:
                attr_bias_values.append(float(probs.max() - probs.min()))

        bias_scores[item] = {
            'group_probs': group_probs,
            'total_bias': float(np.mean(attr_bias_values)) if attr_bias_values else 0.0,
            'count': total
        }

    print(f"Computed directional bias scores for {len(bias_scores)} items")
    return bias_scores


def build_genre_similarity_candidates(movies, config):
    """Build lightweight item candidates from MovieLens genre overlap."""
    print("Building genre-based item similarity candidates...")

    top_k = getattr(config, 'k_synonyms', 20)
    threshold = getattr(config, 'similarity_threshold', 0.2)

    item_genres = {}
    for _, row in movies.iterrows():
        item_id = int(row['movie_id'])
        genres = set(str(row.get('genres', '')).split('|'))
        genres.discard('')
        item_genres[item_id] = genres

    candidates = {}
    item_ids = sorted(item_genres.keys())

    for item_id in item_ids:
        sims = []
        genres = item_genres[item_id]
        if not genres:
            continue

        for other_id in item_ids:
            if other_id == item_id:
                continue
            other_genres = item_genres[other_id]
            union = genres | other_genres
            if not union:
                continue
            sim = len(genres & other_genres) / len(union)
            if sim >= threshold:
                sims.append((other_id, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        candidates[item_id] = {other_id: sim for other_id, sim in sims[:top_k]}

    print(f"Built candidates for {len(candidates)} items")
    return candidates

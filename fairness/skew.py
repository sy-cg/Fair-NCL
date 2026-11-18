"""Train-only demographic association statistics for Fair-NCL augmentation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def build_leave_two_out_train_events(sequences: Mapping[int, Sequence[int]]) -> Dict[int, Sequence[int]]:
    """Extract train-only events from full sequences under leave-two-out."""
    return {
        int(user_id): [int(item) for item in sequence[:-2] if int(item) > 0]
        for user_id, sequence in sequences.items()
        if len(sequence) >= 3
    }


def build_train_event_sequences(
        sequences: Mapping[int, Sequence[int]],
        train_data: Sequence[Dict[str, Any]]) -> Dict[int, Sequence[int]]:
    """Return train-only histories, including for legacy processed datasets."""
    train_events = build_leave_two_out_train_events(sequences)
    if train_events:
        return train_events
    return _reconstruct_unique_train_events(train_data)


def prevalence_adjusted_jsd(counts: Sequence[float],
                            prior: Sequence[float],
                            prior_strength: float = 5.0) -> float:
    """Return normalized JSD between an item distribution and the group prior.

    Counts are shrunk toward the population prior before computing the
    divergence. The result is normalized to [0, 1] by the maximum binary JSD
    value, log(2); the same bound also holds for more than two groups.
    """
    if len(counts) != len(prior) or not counts:
        raise ValueError("counts and prior must be non-empty and have equal length")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    if any(float(value) < 0 for value in counts):
        raise ValueError("counts must be non-negative")
    if any(float(value) < 0 for value in prior):
        raise ValueError("prior probabilities must be non-negative")

    prior_total = float(sum(prior))
    if prior_total <= 0:
        raise ValueError("prior probabilities must have positive mass")
    normalized_prior = [float(value) / prior_total for value in prior]

    count_total = float(sum(counts))
    denominator = count_total + float(prior_strength)
    if denominator <= 0:
        posterior = list(normalized_prior)
    else:
        posterior = [
            (float(count) + float(prior_strength) * probability) / denominator
            for count, probability in zip(counts, normalized_prior)
        ]

    midpoint = [(left + right) * 0.5 for left, right in zip(posterior, normalized_prior)]
    divergence = 0.5 * _kl_divergence(posterior, midpoint)
    divergence += 0.5 * _kl_divergence(normalized_prior, midpoint)
    return max(0.0, min(1.0, float(divergence / math.log(2.0))))


def compute_train_item_bias_statistics(
        train_data: Iterable[Dict[str, Any]],
        sensitive_attrs: Sequence[str] = ("gender", "age_group"),
        train_event_sequences: Optional[Mapping[int, Sequence[int]]] = None,
        metric: str = "prior_jsd",
        min_count: int = 5,
        prior_strength: float = 5.0,
        smoothing: float = 1.0,
        prior_source: str = "interaction",
        event_source: str = "unique_train_events") -> Dict[str, Any]:
    """Compute item skew once per unique train event instead of per prefix.

    ``train_event_sequences`` must contain train-only histories. For the
    leave-two-out preprocessing used by this project, callers should pass
    ``data['sequences'][user_id][:-2]``. If it is omitted, the longest observed
    prefix plus its target is used as a backward-compatible reconstruction.
    """
    samples = list(train_data)
    attrs = tuple(str(attr) for attr in sensitive_attrs)
    normalized_metric = str(metric).lower()
    if normalized_metric not in {"prior_jsd", "raw_range"}:
        raise ValueError("metric must be 'prior_jsd' or 'raw_range'")
    normalized_prior_source = str(prior_source).lower()
    if normalized_prior_source not in {"interaction", "user"}:
        raise ValueError("prior_source must be 'interaction' or 'user'")
    normalized_event_source = str(event_source).lower()
    if normalized_event_source not in {"unique_train_events", "prefix_samples"}:
        raise ValueError("event_source must be 'unique_train_events' or 'prefix_samples'")
    if int(min_count) < 1:
        raise ValueError("min_count must be at least 1")

    user_attributes = _collect_user_attributes(samples, attrs)
    if normalized_event_source == "prefix_samples":
        event_records = [
            (
                int(sample["user_id"]),
                [int(item) for item in sample.get("input_seq", []) if int(item) > 0]
                + ([int(sample["target"])] if int(sample.get("target", 0)) > 0 else []),
            )
            for sample in samples
            if "user_id" in sample
        ]
    else:
        event_sequences = _normalize_event_sequences(train_event_sequences)
        if not event_sequences:
            event_sequences = _reconstruct_unique_train_events(samples)
        event_records = list(event_sequences.items())

    group_values = {
        attr: sorted({values[attr] for values in user_attributes.values() if attr in values})
        for attr in attrs
    }
    group_totals = {
        attr: defaultdict(float)
        for attr in attrs
    }
    item_group_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    item_support = defaultdict(int)

    prior_users_seen = set()
    for user_id, sequence in event_records:
        attributes = user_attributes.get(int(user_id), {})
        valid_items = [int(item) for item in sequence if int(item) > 0]
        if not valid_items:
            continue

        for attr in attrs:
            if attr not in attributes:
                continue
            group = int(attributes[attr])
            if normalized_prior_source == "interaction":
                group_totals[attr][group] += len(valid_items)
            elif (int(user_id), attr) not in prior_users_seen:
                group_totals[attr][group] += 1.0
                prior_users_seen.add((int(user_id), attr))

        for item in valid_items:
            item_support[item] += 1
            for attr in attrs:
                if attr in attributes:
                    item_group_counts[item][attr][int(attributes[attr])] += 1.0

    group_priors: Dict[str, Dict[int, float]] = {}
    for attr in attrs:
        groups = group_values[attr]
        total = sum(group_totals[attr][group] for group in groups)
        if total <= 0:
            group_priors[attr] = {}
        else:
            group_priors[attr] = {
                int(group): float(group_totals[attr][group] / total)
                for group in groups
            }

    scores: Dict[int, float] = {}
    per_attribute_scores: Dict[str, Dict[int, float]] = {attr: {} for attr in attrs}
    excluded = 0
    for item in sorted(item_support):
        if item_support[item] < int(min_count):
            excluded += 1
            continue

        attribute_scores = []
        for attr in attrs:
            groups = group_values[attr]
            prior_map = group_priors.get(attr, {})
            if len(groups) < 2 or not prior_map:
                continue
            counts = [item_group_counts[item][attr].get(group, 0.0) for group in groups]
            if normalized_metric == "prior_jsd":
                prior = [prior_map[group] for group in groups]
                attribute_score = prevalence_adjusted_jsd(counts, prior, prior_strength)
            else:
                denominator = sum(counts) + float(smoothing) * len(groups)
                probabilities = [
                    (count + float(smoothing)) / denominator
                    for count in counts
                ]
                attribute_score = max(probabilities) - min(probabilities)
            per_attribute_scores[attr][int(item)] = float(attribute_score)
            attribute_scores.append(attribute_score)

        if attribute_scores:
            scores[int(item)] = float(sum(attribute_scores) / len(attribute_scores))

    return {
        "metric": normalized_metric,
        "prior_source": normalized_prior_source,
        "event_source": normalized_event_source,
        "prior_strength": float(prior_strength),
        "min_count": int(min_count),
        "scores": scores,
        "per_attribute_scores": per_attribute_scores,
        "item_support": {int(item): int(count) for item, count in item_support.items()},
        "group_priors": group_priors,
        "num_unique_train_events": int(sum(item_support.values())),
        "num_scored_items": len(scores),
        "excluded_low_support_items": int(excluded),
    }


def compute_train_item_bias_scores(
        train_data: Iterable[Dict[str, Any]],
        sensitive_attrs: Sequence[str] = ("gender", "age_group"),
        train_event_sequences: Optional[Mapping[int, Sequence[int]]] = None,
        metric: str = "prior_jsd",
        min_count: int = 5,
        prior_strength: float = 5.0,
        smoothing: float = 1.0,
        prior_source: str = "interaction",
        event_source: str = "unique_train_events") -> Dict[int, float]:
    """Compatibility wrapper returning only item scores."""
    return compute_train_item_bias_statistics(
        train_data=train_data,
        sensitive_attrs=sensitive_attrs,
        train_event_sequences=train_event_sequences,
        metric=metric,
        min_count=min_count,
        prior_strength=prior_strength,
        smoothing=smoothing,
        prior_source=prior_source,
        event_source=event_source,
    )["scores"]


def _collect_user_attributes(samples: Sequence[Dict[str, Any]], attrs: Sequence[str]) -> Dict[int, Dict[str, int]]:
    user_attributes: Dict[int, Dict[str, int]] = {}
    for sample in samples:
        if "user_id" not in sample:
            continue
        user_id = int(sample["user_id"])
        values = user_attributes.setdefault(user_id, {})
        for attr in attrs:
            if attr not in sample:
                continue
            value = int(sample[attr])
            previous = values.get(attr)
            if previous is not None and previous != value:
                raise ValueError(f"Conflicting {attr} values for user {user_id}")
            values[attr] = value
    return user_attributes


def _normalize_event_sequences(sequences: Optional[Mapping[int, Sequence[int]]]) -> Dict[int, Sequence[int]]:
    if not sequences:
        return {}
    return {
        int(user_id): [int(item) for item in sequence if int(item) > 0]
        for user_id, sequence in sequences.items()
    }


def _reconstruct_unique_train_events(samples: Sequence[Dict[str, Any]]) -> Dict[int, Sequence[int]]:
    histories: Dict[int, Sequence[int]] = {}
    for sample in samples:
        if "user_id" not in sample:
            continue
        user_id = int(sample["user_id"])
        prefix = [int(item) for item in sample.get("input_seq", []) if int(item) > 0]
        target = int(sample.get("target", 0))
        candidate = prefix + ([target] if target > 0 else [])
        current = list(histories.get(user_id, ()))
        if not current:
            histories[user_id] = candidate
        elif prefix and current[-len(prefix):] == prefix and target > 0:
            histories[user_id] = current + [target]
        elif len(candidate) > len(current):
            histories[user_id] = candidate
    return histories


def _kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    value = 0.0
    for probability, reference in zip(left, right):
        if probability <= 0:
            continue
        if reference <= 0:
            raise ValueError("reference distribution has zero mass where probability is positive")
        value += probability * math.log(probability / reference)
    return float(value)

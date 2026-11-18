"""Statistical utilities for the five-seed Fair-NCL revision experiments."""

from __future__ import annotations

import random
import itertools

import numpy as np
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


def summarize_group_metric(rows: Iterable[Mapping[str, Any]],
                           metric: str,
                           group_attribute: str) -> Dict[str, Any]:
    grouped: Dict[Any, List[float]] = defaultdict(list)
    for row in rows:
        if metric not in row or group_attribute not in row:
            continue
        grouped[row[group_attribute]].append(float(row[metric]))

    group_values = {
        group: float(sum(values) / len(values))
        for group, values in grouped.items()
        if values
    }
    if not group_values:
        return {
            "group_values": {},
            "group_counts": {},
            "signed_gap": 0.0,
            "absolute_gap": 0.0,
            "relative_gap": 0.0,
            "worst_group": 0.0,
            "worst_group_id": None,
            "best_group": 0.0,
            "best_group_id": None,
        }

    ordered_groups = sorted(group_values, key=lambda value: str(value))
    worst_group_id = min(group_values, key=group_values.get)
    best_group_id = max(group_values, key=group_values.get)
    worst_value = group_values[worst_group_id]
    best_value = group_values[best_group_id]
    absolute_gap = best_value - worst_value
    signed_gap = 0.0
    if len(ordered_groups) >= 2:
        signed_gap = group_values[ordered_groups[0]] - group_values[ordered_groups[-1]]

    return {
        "group_values": group_values,
        "group_counts": {group: len(grouped[group]) for group in group_values},
        "signed_gap": float(signed_gap),
        "absolute_gap": float(absolute_gap),
        "relative_gap": float(absolute_gap / best_value) if best_value > 0 else 0.0,
        "worst_group": float(worst_value),
        "worst_group_id": worst_group_id,
        "best_group": float(best_value),
        "best_group_id": best_group_id,
    }


def paired_hierarchical_bootstrap(method_rows: Sequence[Mapping[str, Any]],
                                  baseline_rows: Sequence[Mapping[str, Any]],
                                  metric: str,
                                  group_attribute: Optional[str] = None,
                                  estimand: str = "mean",
                                  group_value: Any = None,
                                  iterations: int = 10000,
                                  confidence: float = 0.95,
                                  random_seed: int = 42) -> Dict[str, Any]:
    """Paired seed-and-user bootstrap with optional within-group stratification."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    method_index = _index_rows(method_rows)
    baseline_index = _index_rows(baseline_rows)
    if set(method_index) != set(baseline_index):
        raise ValueError("method and baseline must contain identical seed sets")
    seeds = sorted(method_index)
    paired_users = {}
    for seed in seeds:
        method_users = set(method_index[seed])
        baseline_users = set(baseline_index[seed])
        if method_users != baseline_users:
            raise ValueError(f"method and baseline must contain identical user sets for seed {seed}")
        users = sorted(method_users)
        if group_attribute:
            mismatched = [
                user_id
                for user_id in users
                if method_index[seed][user_id].get(group_attribute)
                != baseline_index[seed][user_id].get(group_attribute)
            ]
            if mismatched:
                raise ValueError(
                    f"paired rows contain conflicting {group_attribute} labels for seed {seed}"
                )
        paired_users[seed] = users
    if not seeds:
        raise ValueError("method and baseline have no paired seed-user observations")

    method_estimates = []
    baseline_estimates = []
    for seed in seeds:
        method_sample = [method_index[seed][user_id] for user_id in paired_users[seed]]
        baseline_sample = [baseline_index[seed][user_id] for user_id in paired_users[seed]]
        method_estimates.append(_estimate(method_sample, metric, group_attribute, estimand, group_value))
        baseline_estimates.append(_estimate(baseline_sample, metric, group_attribute, estimand, group_value))
    point_deltas = [
        method_value - baseline_value
        for method_value, baseline_value in zip(method_estimates, baseline_estimates)
    ]
    mean_delta = float(sum(point_deltas) / len(point_deltas))

    generator = random.Random(random_seed)
    bootstrap_deltas = []
    for _ in range(iterations):
        sampled_seed_deltas = []
        for seed in generator.choices(seeds, k=len(seeds)):
            sampled_users = _sample_paired_users(
                paired_users[seed],
                method_index[seed],
                group_attribute,
                generator,
            )
            sampled_seed_deltas.append(_paired_seed_delta(
                method_index[seed],
                baseline_index[seed],
                sampled_users,
                metric,
                group_attribute,
                estimand,
                group_value,
            ))
        bootstrap_deltas.append(sum(sampled_seed_deltas) / len(sampled_seed_deltas))

    alpha = 1.0 - confidence
    return {
        "metric": metric,
        "estimand": estimand,
        "group_attribute": group_attribute,
        "group_value": group_value,
        "baseline_estimate": float(sum(baseline_estimates) / len(baseline_estimates)),
        "method_estimate": float(sum(method_estimates) / len(method_estimates)),
        "mean_delta": mean_delta,
        "ci_lower": _percentile(bootstrap_deltas, alpha / 2.0),
        "ci_upper": _percentile(bootstrap_deltas, 1.0 - alpha / 2.0),
        "confidence": float(confidence),
        "iterations": int(iterations),
        "paired_seeds": len(seeds),
        "paired_users": int(sum(len(users) for users in paired_users.values())),
        "seed_deltas": {int(seed): float(delta) for seed, delta in zip(seeds, point_deltas)},
    }


def paired_hierarchical_bootstrap_many(
        method_rows: Sequence[Mapping[str, Any]],
        baseline_rows: Sequence[Mapping[str, Any]],
        specifications: Sequence[Tuple[str, Optional[str], str, Any]],
        iterations: int = 10000,
        confidence: float = 0.95,
        random_seed: int = 42) -> List[Dict[str, Any]]:
    """Compute many paired hierarchical bootstrap estimands from shared draws.

    Specifications are ``(metric, group_attribute, estimand, group_value)``
    tuples. Draws are shared when their group attribute is the same, which
    preserves the bootstrap design while avoiding repeated user resampling.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not specifications:
        return []

    normalized = [tuple(specification) for specification in specifications]
    if any(len(specification) != 4 for specification in normalized):
        raise ValueError("each bootstrap specification must contain four values")

    method_index = _index_rows(method_rows)
    baseline_index = _index_rows(baseline_rows)
    if set(method_index) != set(baseline_index):
        raise ValueError("method and baseline must contain identical seed sets")
    seeds = sorted(method_index)
    if not seeds:
        raise ValueError("method and baseline have no paired seed-user observations")

    paired_users: Dict[int, List[Any]] = {}
    attributes = {specification[1] for specification in normalized if specification[1]}
    metrics = {specification[0] for specification in normalized}
    for seed in seeds:
        method_users = set(method_index[seed])
        baseline_users = set(baseline_index[seed])
        if method_users != baseline_users:
            raise ValueError(f"method and baseline must contain identical user sets for seed {seed}")
        users = sorted(method_users)
        for user_id in users:
            method_row = method_index[seed][user_id]
            baseline_row = baseline_index[seed][user_id]
            for attribute in attributes:
                if method_row.get(attribute) != baseline_row.get(attribute):
                    raise ValueError(
                        f"paired rows contain conflicting {attribute} labels for seed {seed}"
                    )
            for metric in metrics:
                if metric not in method_row or metric not in baseline_row:
                    raise ValueError(
                        f"batched bootstrap requires complete '{metric}' values for paired rows"
                    )
        paired_users[seed] = users

    method_estimates_by_spec: List[List[float]] = []
    baseline_estimates_by_spec: List[List[float]] = []
    for metric, attribute, estimand, group_value in normalized:
        method_estimates = []
        baseline_estimates = []
        for seed in seeds:
            users = paired_users[seed]
            method_sample = [method_index[seed][user_id] for user_id in users]
            baseline_sample = [baseline_index[seed][user_id] for user_id in users]
            method_estimates.append(
                _estimate(method_sample, metric, attribute, estimand, group_value)
            )
            baseline_estimates.append(
                _estimate(baseline_sample, metric, attribute, estimand, group_value)
            )
        method_estimates_by_spec.append(method_estimates)
        baseline_estimates_by_spec.append(baseline_estimates)

    bootstrap_deltas = np.empty((iterations, len(normalized)), dtype=np.float64)
    specifications_by_attribute: Dict[Optional[str], List[int]] = defaultdict(list)
    for index, specification in enumerate(normalized):
        specifications_by_attribute[specification[1]].append(index)
    for attribute, specification_indices in specifications_by_attribute.items():
        attribute_specs = [normalized[index] for index in specification_indices]
        attribute_deltas = _vectorized_bootstrap_attribute(
            method_index,
            baseline_index,
            paired_users,
            seeds,
            attribute_specs,
            iterations,
            random_seed,
        )
        bootstrap_deltas[:, specification_indices] = attribute_deltas

    alpha = 1.0 - confidence
    lower = np.quantile(bootstrap_deltas, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap_deltas, 1.0 - alpha / 2.0, axis=0)
    total_paired_users = int(sum(len(users) for users in paired_users.values()))
    output = []
    for index, (metric, attribute, estimand, group_value) in enumerate(normalized):
        method_estimates = method_estimates_by_spec[index]
        baseline_estimates = baseline_estimates_by_spec[index]
        seed_deltas = [
            method_value - baseline_value
            for method_value, baseline_value in zip(method_estimates, baseline_estimates)
        ]
        output.append({
            "metric": metric,
            "estimand": estimand,
            "group_attribute": attribute,
            "group_value": group_value,
            "baseline_estimate": float(np.mean(baseline_estimates)),
            "method_estimate": float(np.mean(method_estimates)),
            "mean_delta": float(np.mean(seed_deltas)),
            "ci_lower": float(lower[index]),
            "ci_upper": float(upper[index]),
            "confidence": float(confidence),
            "iterations": int(iterations),
            "paired_seeds": len(seeds),
            "paired_users": total_paired_users,
            "seed_deltas": {
                int(seed): float(delta)
                for seed, delta in zip(seeds, seed_deltas)
            },
        })
    return output


def _vectorized_bootstrap_attribute(
        method_index: Mapping[int, Mapping[Any, Mapping[str, Any]]],
        baseline_index: Mapping[int, Mapping[Any, Mapping[str, Any]]],
        paired_users: Mapping[int, Sequence[Any]],
        seeds: Sequence[int],
        specifications: Sequence[Tuple[str, Optional[str], str, Any]],
        iterations: int,
        random_seed: int) -> np.ndarray:
    attribute = specifications[0][1]
    metrics = list(dict.fromkeys(specification[0] for specification in specifications))
    metric_indices = {metric: index for index, metric in enumerate(metrics)}
    seed_data = []
    max_users = 1
    for seed in seeds:
        users = paired_users[seed]
        max_users = max(max_users, len(users))
        method_values = np.asarray([
            [float(method_index[seed][user_id][metric]) for metric in metrics]
            for user_id in users
        ], dtype=np.float64)
        baseline_values = np.asarray([
            [float(baseline_index[seed][user_id][metric]) for metric in metrics]
            for user_id in users
        ], dtype=np.float64)
        if attribute is None:
            groups: List[Any] = []
            group_positions: List[np.ndarray] = []
        else:
            labels = [method_index[seed][user_id].get(attribute) for user_id in users]
            groups = sorted(set(labels), key=str)
            group_positions = [
                np.asarray([index for index, label in enumerate(labels) if label == group])
                for group in groups
            ]
        seed_data.append((method_values, baseline_values, groups, group_positions))

    generator = np.random.default_rng(random_seed)
    result = np.empty((iterations, len(specifications)), dtype=np.float64)
    batch_size = min(256, max(1, 2_000_000 // max_users))
    seed_count = len(seeds)
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        sampled_seed_indices = generator.integers(0, seed_count, size=(size, seed_count))
        delta_sum = np.zeros((size, len(specifications)), dtype=np.float64)
        for slot in range(seed_count):
            selected = sampled_seed_indices[:, slot]
            for seed_index in range(seed_count):
                batch_rows = np.flatnonzero(selected == seed_index)
                if not len(batch_rows):
                    continue
                method_values, baseline_values, groups, group_positions = seed_data[seed_index]
                method_estimates, baseline_estimates = _resample_seed_specifications(
                    method_values,
                    baseline_values,
                    groups,
                    group_positions,
                    specifications,
                    metric_indices,
                    generator,
                    len(batch_rows),
                )
                delta_sum[batch_rows] += method_estimates - baseline_estimates
        result[start:start + size] = delta_sum / seed_count
    return result


def _resample_seed_specifications(
        method_values: np.ndarray,
        baseline_values: np.ndarray,
        groups: Sequence[Any],
        group_positions: Sequence[np.ndarray],
        specifications: Sequence[Tuple[str, Optional[str], str, Any]],
        metric_indices: Mapping[str, int],
        generator: np.random.Generator,
        sample_count: int) -> Tuple[np.ndarray, np.ndarray]:
    method_output = np.zeros((sample_count, len(specifications)), dtype=np.float64)
    baseline_output = np.zeros_like(method_output)
    attribute = specifications[0][1]
    if attribute is None:
        sampled = generator.integers(
            0, len(method_values), size=(sample_count, len(method_values))
        )
        method_means = method_values[sampled].mean(axis=1)
        baseline_means = baseline_values[sampled].mean(axis=1)
        for index, (metric, _, estimand, _) in enumerate(specifications):
            if estimand != "mean":
                raise ValueError(f"estimand '{estimand}' requires group_attribute")
            metric_index = metric_indices[metric]
            method_output[:, index] = method_means[:, metric_index]
            baseline_output[:, index] = baseline_means[:, metric_index]
        return method_output, baseline_output

    method_group_means = np.empty(
        (sample_count, len(groups), len(metric_indices)), dtype=np.float64
    )
    baseline_group_means = np.empty_like(method_group_means)
    for group_index, positions in enumerate(group_positions):
        sampled_within_group = positions[generator.integers(
            0, len(positions), size=(sample_count, len(positions))
        )]
        method_group_means[:, group_index] = method_values[sampled_within_group].mean(axis=1)
        baseline_group_means[:, group_index] = baseline_values[sampled_within_group].mean(axis=1)

    group_lookup = {group: index for index, group in enumerate(groups)}
    for index, (metric, _, estimand, group_value) in enumerate(specifications):
        metric_index = metric_indices[metric]
        method_metric = method_group_means[:, :, metric_index]
        baseline_metric = baseline_group_means[:, :, metric_index]
        if estimand == "group_mean":
            group_index = group_lookup.get(group_value)
            if group_index is not None:
                method_output[:, index] = method_metric[:, group_index]
                baseline_output[:, index] = baseline_metric[:, group_index]
        elif estimand == "gap":
            method_output[:, index] = method_metric.max(axis=1) - method_metric.min(axis=1)
            baseline_output[:, index] = baseline_metric.max(axis=1) - baseline_metric.min(axis=1)
        elif estimand == "signed_gap":
            method_output[:, index] = method_metric[:, 0] - method_metric[:, -1]
            baseline_output[:, index] = baseline_metric[:, 0] - baseline_metric[:, -1]
        elif estimand == "relative_gap":
            method_best = method_metric.max(axis=1)
            baseline_best = baseline_metric.max(axis=1)
            method_gap = method_best - method_metric.min(axis=1)
            baseline_gap = baseline_best - baseline_metric.min(axis=1)
            method_output[:, index] = np.divide(
                method_gap,
                method_best,
                out=np.zeros_like(method_gap),
                where=method_best > 0,
            )
            baseline_output[:, index] = np.divide(
                baseline_gap,
                baseline_best,
                out=np.zeros_like(baseline_gap),
                where=baseline_best > 0,
            )
        elif estimand == "worst_group":
            method_output[:, index] = method_metric.min(axis=1)
            baseline_output[:, index] = baseline_metric.min(axis=1)
        else:
            raise ValueError(f"unknown estimand: {estimand}")
    return method_output, baseline_output


def holm_adjust(p_values: Mapping[Any, float]) -> Dict[Any, float]:
    """Return Holm step-down adjusted p-values."""
    ordered = sorted(p_values.items(), key=lambda item: float(item[1]))
    total = len(ordered)
    adjusted: Dict[Any, float] = {}
    running_max = 0.0
    for index, (key, raw_value) in enumerate(ordered):
        candidate = (total - index) * float(raw_value)
        running_max = max(running_max, candidate)
        adjusted[key] = min(1.0, running_max)
    return adjusted


def exact_paired_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided exact randomization p-value for paired seed differences."""
    nonzero = [abs(float(value)) for value in differences if float(value) != 0.0]
    if not nonzero:
        return 1.0
    if len(nonzero) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 non-zero pairs")
    observed = abs(sum(float(value) for value in differences) / len(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(sum(sign * value for sign, value in zip(signs, nonzero)) / len(differences))
        total += 1
        if statistic + 1e-15 >= observed:
            extreme += 1
    return float(extreme / total)


def pareto_front(rows: Iterable[Mapping[str, Any]],
                 utility_key: str,
                 fairness_key: str,
                 method_key: str = "method",
                 fairness_higher_is_better: bool = True) -> Set[Any]:
    """Return methods not dominated on utility and the selected fairness axis."""
    points = list(rows)
    frontier: Set[Any] = set()
    for candidate in points:
        candidate_utility = float(candidate[utility_key])
        candidate_fairness = float(candidate[fairness_key])
        candidate_fairness_score = candidate_fairness if fairness_higher_is_better else -candidate_fairness
        dominated = False
        for other in points:
            if other is candidate:
                continue
            other_utility = float(other[utility_key])
            other_fairness = float(other[fairness_key])
            other_fairness_score = other_fairness if fairness_higher_is_better else -other_fairness
            no_worse = other_utility >= candidate_utility and other_fairness_score >= candidate_fairness_score
            strictly_better = other_utility > candidate_utility or other_fairness_score > candidate_fairness_score
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(candidate[method_key])
    return frontier


def multi_objective_pareto_front(rows: Iterable[Mapping[str, Any]],
                                 objectives: Mapping[str, bool],
                                 method_key: str = "method") -> Set[Any]:
    """Return rows not dominated across all declared objectives.

    ``objectives`` maps each numeric field to whether larger values are better.
    """
    points = list(rows)
    frontier: Set[Any] = set()
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            comparisons = []
            for key, higher_is_better in objectives.items():
                candidate_value = float(candidate[key])
                other_value = float(other[key])
                if not higher_is_better:
                    candidate_value = -candidate_value
                    other_value = -other_value
                comparisons.append((other_value, candidate_value))
            if (
                all(other_value >= candidate_value for other_value, candidate_value in comparisons)
                and any(other_value > candidate_value for other_value, candidate_value in comparisons)
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(candidate[method_key])
    return frontier


def select_validation_configuration(candidates: Sequence[Mapping[str, Any]],
                                    backbone_ndcg: float,
                                    utility_tolerance: float = 0.15,
                                    utility_key: str = "NDCG@10",
                                    utility_retention_key: str = "utility_retention_conservative",
                                    fairness_benefit_key: str = "fairness_benefit",
                                    gap_risk_key: str = "gap_risk",
                                    worst_group_key: str = "worst_group") -> Dict[str, Any]:
    """Apply conservative utility feasibility, Pareto filtering, and fairness ranking."""
    if not candidates:
        raise ValueError("at least one validation candidate is required")
    if not 0.0 <= float(utility_tolerance) < 1.0:
        raise ValueError("utility_tolerance must be in [0, 1)")
    threshold = float(backbone_ndcg) * (1.0 - float(utility_tolerance))
    retention_floor = 1.0 - float(utility_tolerance)
    scored_candidates = []
    for candidate in candidates:
        scored = dict(candidate)
        scored["fairness_risk_score"] = float(candidate[gap_risk_key])
        scored["fairness_improving"] = float(candidate[fairness_benefit_key]) >= 0.0
        scored_candidates.append(scored)
    qualified = [
        candidate
        for candidate in scored_candidates
        if float(candidate[utility_retention_key]) >= retention_floor
    ]
    if qualified:
        frontier = multi_objective_pareto_front(
            qualified,
            objectives={
                utility_key: True,
                fairness_benefit_key: True,
                gap_risk_key: False,
            },
            method_key="candidate",
        )
        efficient = [candidate for candidate in qualified if candidate["candidate"] in frontier]
        selected = min(
            efficient,
            key=lambda candidate: (
                -float(candidate[fairness_benefit_key]),
                float(candidate[gap_risk_key]),
                -float(candidate.get(worst_group_key, 0.0)),
                -float(candidate[utility_key]),
                str(candidate.get("candidate", "")),
            ),
        )
        status = "qualified"
    else:
        selected = max(scored_candidates, key=lambda candidate: float(candidate[utility_key]))
        status = "no_qualified_candidate"
    result = dict(selected)
    result["selection_status"] = status
    result["utility_threshold"] = threshold
    result["utility_tolerance"] = float(utility_tolerance)
    result["utility_retention_floor"] = retention_floor
    return result


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[Any, Mapping[str, Any]]]:
    indexed: Dict[int, Dict[Any, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        seed = int(row["seed"])
        user_id = row["user_id"]
        if user_id in indexed[seed]:
            raise ValueError(f"duplicate paired observation for seed {seed}, user {user_id}")
        indexed[seed][user_id] = row
    return indexed


def _sample_paired_users(user_ids: Sequence[Any],
                         method_rows: Mapping[Any, Mapping[str, Any]],
                         group_attribute: Optional[str],
                         generator: random.Random) -> List[Any]:
    if not group_attribute:
        return generator.choices(list(user_ids), k=len(user_ids))
    strata: Dict[Any, List[Any]] = defaultdict(list)
    for user_id in user_ids:
        strata[method_rows[user_id].get(group_attribute)].append(user_id)
    sampled: List[Any] = []
    for users in strata.values():
        sampled.extend(generator.choices(users, k=len(users)))
    return sampled


def _paired_seed_delta(method_rows: Mapping[Any, Mapping[str, Any]],
                       baseline_rows: Mapping[Any, Mapping[str, Any]],
                       user_ids: Sequence[Any],
                       metric: str,
                       group_attribute: Optional[str],
                       estimand: str,
                       group_value: Any) -> float:
    method_sample = [method_rows[user_id] for user_id in user_ids]
    baseline_sample = [baseline_rows[user_id] for user_id in user_ids]
    return _estimate(method_sample, metric, group_attribute, estimand, group_value) - _estimate(
        baseline_sample,
        metric,
        group_attribute,
        estimand,
        group_value,
    )


def _estimate(rows: Sequence[Mapping[str, Any]],
              metric: str,
              group_attribute: Optional[str],
              estimand: str,
              group_value: Any) -> float:
    if estimand in {"mean", "group_mean"}:
        if estimand == "group_mean" and (group_attribute is None or group_value is None):
            raise ValueError("group_mean requires group_attribute and group_value")
        values = [
            float(row[metric])
            for row in rows
            if metric in row and (group_attribute is None or group_value is None or row.get(group_attribute) == group_value)
        ]
        return float(sum(values) / len(values)) if values else 0.0
    if group_attribute is None:
        raise ValueError(f"estimand '{estimand}' requires group_attribute")
    summary = summarize_group_metric(rows, metric, group_attribute)
    key = {
        "gap": "absolute_gap",
        "signed_gap": "signed_gap",
        "relative_gap": "relative_gap",
        "worst_group": "worst_group",
    }.get(estimand)
    if key is None:
        raise ValueError(f"unknown estimand: {estimand}")
    return float(summary[key])


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)

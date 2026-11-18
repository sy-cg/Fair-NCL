"""Offline calibration diagnostics for raw and prevalence-adjusted item skew."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from fairness.skew import (
    build_leave_two_out_train_events,
    compute_train_item_bias_statistics,
)


def build_skew_diagnostics(data: Mapping[str, Any],
                           sensitive_attributes: Sequence[str] = ("gender", "age_group"),
                           min_count: int = 5,
                           prior_strength: float = 5.0,
                           permutations: int = 20,
                           random_seed: int = 42) -> Dict[str, Any]:
    train_data = data["train_data"]
    train_events = build_leave_two_out_train_events(data.get("sequences", {}))
    adjusted = compute_train_item_bias_statistics(
        train_data,
        sensitive_attrs=sensitive_attributes,
        train_event_sequences=train_events or None,
        metric="prior_jsd",
        min_count=min_count,
        prior_strength=prior_strength,
    )
    adjusted_user_prior = compute_train_item_bias_statistics(
        train_data,
        sensitive_attrs=sensitive_attributes,
        train_event_sequences=train_events or None,
        metric="prior_jsd",
        min_count=min_count,
        prior_strength=prior_strength,
        prior_source="user",
    )
    raw_unique = compute_train_item_bias_statistics(
        train_data,
        sensitive_attrs=sensitive_attributes,
        train_event_sequences=train_events or None,
        metric="raw_range",
        min_count=min_count,
    )
    raw_submitted = compute_train_item_bias_statistics(
        train_data,
        sensitive_attrs=sensitive_attributes,
        train_event_sequences=None,
        metric="raw_range",
        min_count=1,
        event_source="prefix_samples",
    )

    items = sorted(
        set(adjusted["item_support"])
        | set(adjusted_user_prior["item_support"])
        | set(raw_submitted["item_support"])
    )
    item_rows = []
    for item in items:
        row = {
            "item_id": int(item),
            "train_event_support": int(adjusted["item_support"].get(item, 0)),
            "train_popularity": int(adjusted["item_support"].get(item, 0)),
            "submitted_prefix_support": int(raw_submitted["item_support"].get(item, 0)),
            "adjusted_skew": adjusted["scores"].get(item),
            "adjusted_user_prior_skew": adjusted_user_prior["scores"].get(item),
            "raw_unique_skew": raw_unique["scores"].get(item),
            "raw_submitted_skew": raw_submitted["scores"].get(item),
        }
        for attr in sensitive_attributes:
            row[f"adjusted_{attr}"] = adjusted["per_attribute_scores"].get(attr, {}).get(item)
            row[f"adjusted_user_prior_{attr}"] = adjusted_user_prior["per_attribute_scores"].get(attr, {}).get(item)
            row[f"raw_unique_{attr}"] = raw_unique["per_attribute_scores"].get(attr, {}).get(item)
            row[f"raw_submitted_{attr}"] = raw_submitted["per_attribute_scores"].get(attr, {}).get(item)
        item_rows.append(row)

    correlations = {
        "adjusted_vs_log_support": _spearman_from_rows(item_rows, "adjusted_skew", "train_event_support", log_right=True),
        "adjusted_vs_log_popularity": _spearman_from_rows(
            item_rows, "adjusted_skew", "train_popularity", log_right=True
        ),
        "raw_unique_vs_log_support": _spearman_from_rows(item_rows, "raw_unique_skew", "train_event_support", log_right=True),
        "raw_submitted_vs_log_support": _spearman_from_rows(item_rows, "raw_submitted_skew", "submitted_prefix_support", log_right=True),
        "adjusted_vs_raw_unique": _spearman_from_rows(item_rows, "adjusted_skew", "raw_unique_skew"),
        "adjusted_vs_raw_submitted": _spearman_from_rows(item_rows, "adjusted_skew", "raw_submitted_skew"),
        "interaction_vs_user_adjusted": _spearman_from_rows(
            item_rows,
            "adjusted_skew",
            "adjusted_user_prior_skew",
        ),
    }
    for attr in sensitive_attributes:
        correlations[f"adjusted_vs_raw_unique_{attr}"] = _spearman_from_rows(
            item_rows,
            f"adjusted_{attr}",
            f"raw_unique_{attr}",
        )

    return {
        "summary": {
            "dataset": data.get("dataset"),
            "sensitive_attributes": list(sensitive_attributes),
            "group_priors": adjusted["group_priors"],
            "prior_sensitivity": {
                "interaction_prior": adjusted["group_priors"],
                "user_prior": adjusted_user_prior["group_priors"],
            },
            "min_count": int(min_count),
            "prior_strength": float(prior_strength),
            "unique_train_events": adjusted["num_unique_train_events"],
            "submitted_prefix_events": raw_submitted["num_unique_train_events"],
            "scored_items": adjusted["num_scored_items"],
            "correlations": correlations,
            "low_skew_decile_overlap": _low_score_overlap(adjusted["scores"], raw_unique["scores"], 0.10),
            "interaction_vs_user_low_skew_decile_overlap": _low_score_overlap(
                adjusted["scores"],
                adjusted_user_prior["scores"],
                0.10,
            ),
            "permutation_null": _permutation_null(
                train_data,
                train_events,
                sensitive_attributes,
                min_count,
                prior_strength,
                permutations,
                random_seed,
            ),
            "temporal_quartiles": _temporal_quartile_diagnostics(
                train_data,
                train_events,
                sensitive_attributes,
                min_count,
                prior_strength,
            ),
        },
        "items": item_rows,
    }


def write_skew_diagnostics(diagnostics: Mapping[str, Any], output_dir: Path | str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "skew_diagnostics.json").write_text(
        json.dumps(diagnostics["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = diagnostics["items"]
    if rows:
        with (output_path / "item_skew_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _permutation_null(train_data,
                      train_events,
                      sensitive_attributes,
                      min_count,
                      prior_strength,
                      permutations,
                      random_seed):
    if permutations <= 0:
        return {"permutations": 0}
    user_attrs = {}
    for sample in train_data:
        user_attrs.setdefault(int(sample["user_id"]), {
            attr: int(sample[attr]) for attr in sensitive_attributes if attr in sample
        })
    user_ids = sorted(user_attrs)
    generator = random.Random(random_seed)
    means = []
    for _ in range(int(permutations)):
        shuffled_attrs = {attr: [user_attrs[user].get(attr) for user in user_ids] for attr in sensitive_attributes}
        for values in shuffled_attrs.values():
            generator.shuffle(values)
        permuted_samples = [
            {
                "user_id": user,
                **{attr: shuffled_attrs[attr][index] for attr in sensitive_attributes},
            }
            for index, user in enumerate(user_ids)
        ]
        statistics = compute_train_item_bias_statistics(
            permuted_samples,
            sensitive_attrs=sensitive_attributes,
            train_event_sequences=train_events,
            metric="prior_jsd",
            min_count=min_count,
            prior_strength=prior_strength,
        )
        values = list(statistics["scores"].values())
        means.append(sum(values) / len(values) if values else 0.0)
    return {
        "permutations": int(permutations),
        "mean_adjusted_skew": sum(means) / len(means),
        "min_mean_adjusted_skew": min(means),
        "max_mean_adjusted_skew": max(means),
    }


def _temporal_quartile_diagnostics(train_data,
                                   train_events,
                                   sensitive_attributes,
                                   min_count,
                                   prior_strength):
    quartile_scores = []
    for quartile in range(4):
        segment_sequences = {}
        for user_id, sequence in train_events.items():
            start = math.floor(len(sequence) * quartile / 4)
            end = math.floor(len(sequence) * (quartile + 1) / 4)
            if end > start:
                segment_sequences[user_id] = sequence[start:end]
        statistics = compute_train_item_bias_statistics(
            train_data,
            sensitive_attrs=sensitive_attributes,
            train_event_sequences=segment_sequences,
            metric="prior_jsd",
            min_count=min_count,
            prior_strength=prior_strength,
        )
        quartile_scores.append(statistics["scores"])
    adjacent = []
    for index in range(3):
        adjacent.append({
            "from_quartile": index + 1,
            "to_quartile": index + 2,
            "rank_correlation": _spearman_maps(quartile_scores[index], quartile_scores[index + 1]),
            "low_skew_decile_overlap": _low_score_overlap(quartile_scores[index], quartile_scores[index + 1], 0.10),
        })
    return adjacent


def _spearman_from_rows(rows, left_key, right_key, log_right=False):
    pairs = [
        (float(row[left_key]), float(row[right_key]))
        for row in rows
        if row.get(left_key) is not None and row.get(right_key) is not None
    ]
    if log_right:
        pairs = [(left, math.log1p(right)) for left, right in pairs]
    return _spearman([left for left, _ in pairs], [right for _, right in pairs])


def _spearman_maps(left: Mapping[int, float], right: Mapping[int, float]):
    shared = sorted(set(left) & set(right))
    return _spearman([left[item] for item in shared], [right[item] for item in shared])


def _spearman(left, right):
    if len(left) < 2 or len(right) != len(left):
        return None

    def ranks(values):
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        output = [0.0] * len(values)
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for position in range(start, end):
                output[ordered[position]] = rank
            start = end
        return output

    x, y = ranks(left), ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None

def _low_score_overlap(left: Mapping[int, float], right: Mapping[int, float], fraction: float):
    shared = set(left) & set(right)
    if not shared:
        return None
    limit = max(1, int(len(shared) * fraction))
    left_low = set(sorted(shared, key=lambda item: (left[item], item))[:limit])
    right_low = set(sorted(shared, key=lambda item: (right[item], item))[:limit])
    return float(len(left_low & right_low) / limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prevalence-adjusted skew diagnostics.")
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--datasets", nargs="+", default=["ml-1m", "taobao"])
    parser.add_argument("--output-root", default="tables/supplementary/skew")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--prior-strength", type=float, default=5.0)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()

    for dataset in args.datasets:
        processed_path = Path(args.processed_root) / dataset / "processed_data.pkl"
        with processed_path.open("rb") as handle:
            data = pickle.load(handle)
        diagnostics = build_skew_diagnostics(
            data,
            min_count=args.min_count,
            prior_strength=args.prior_strength,
            permutations=args.permutations,
        )
        write_skew_diagnostics(diagnostics, Path(args.output_root) / dataset)
        print(f"Wrote skew diagnostics for {dataset}.")


if __name__ == "__main__":
    main()


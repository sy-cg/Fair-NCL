"""Aggregate five-seed supplementary runs into reviewer-facing statistics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from experiments.specs import FINAL_REVISION_SEEDS

from .supplementary_statistics import (
    exact_paired_sign_flip_pvalue,
    holm_adjust,
    paired_hierarchical_bootstrap_many,
    pareto_front,
    summarize_group_metric,
)


PRIMARY_METRICS = ("HitRate@10", "NDCG@10", "HitRate@20", "NDCG@20")
GROUP_ATTRIBUTES = ("gender", "age_group", "gender_age")


def build_supplementary_report(results_root: Path | str,
                               phase: str = "supplementary",
                               bootstrap_iterations: int = 10000,
                               bootstrap_seed: int = 42,
                               expected_jobs_path: Path | str | None = None,
                               require_complete: bool = False,
                               progress: bool = False) -> Dict[str, Any]:
    runs = _discover_runs(Path(results_root), phase)
    plan_audit = _audit_expected_plan(runs, phase, expected_jobs_path)
    if require_complete:
        if expected_jobs_path is None:
            raise ValueError("require_complete needs an expected_jobs_path")
        missing_count = len(plan_audit["missing_job_ids"])
        unexpected_count = len(plan_audit["unexpected_job_ids"])
        duplicate_count = len(plan_audit["duplicate_result_job_ids"])
        if missing_count or unexpected_count or duplicate_count:
            raise ValueError(
                f"Supplementary result set is incomplete: missing {missing_count} expected jobs, "
                f"found {unexpected_count} unexpected jobs and {duplicate_count} duplicate job ids."
            )
    run_summary = _aggregate_runs(runs)
    group_summary = _aggregate_groups(runs)
    paired_comparisons = _paired_comparisons(
        runs,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        progress=progress,
    )
    pareto_rows = _pareto_rows(run_summary, group_summary)
    proxy_outcome = _proxy_outcome_rows(runs)
    proxy_outcome_summary = _proxy_outcome_summary(proxy_outcome)
    incomplete_groups = []
    expected_seeds = set(FINAL_REVISION_SEEDS)
    for row in run_summary:
        observed = set(row.pop("_seed_values"))
        missing = sorted(expected_seeds - observed)
        unexpected = sorted(observed - expected_seeds)
        if missing or unexpected:
            incomplete_groups.append({
                "dataset": row["dataset"],
                "backbone": row["backbone"],
                "method": row["method"],
                "missing_seeds": missing,
                "unexpected_seeds": unexpected,
            })

    return {
        "manifest": {
            "phase": phase,
            "expected_seeds": list(FINAL_REVISION_SEEDS),
            "discovered_runs": len(runs),
            "incomplete_groups": incomplete_groups,
            **plan_audit,
        },
        "run_summary": run_summary,
        "group_summary": group_summary,
        "paired_comparisons": paired_comparisons,
        "pareto": pareto_rows,
        "proxy_outcome": proxy_outcome,
        "proxy_outcome_summary": proxy_outcome_summary,
    }


def write_supplementary_report(report: Mapping[str, Any], output_dir: Path | str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(
        json.dumps(report["manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_path / "run_summary.csv", report["run_summary"])
    _write_csv(output_path / "group_summary.csv", report["group_summary"])
    _write_csv(output_path / "paired_comparisons.csv", report["paired_comparisons"])
    _write_csv(output_path / "pareto.csv", report["pareto"])
    _write_csv(output_path / "proxy_outcome.csv", report["proxy_outcome"])
    _write_csv(output_path / "proxy_outcome_summary.csv", report["proxy_outcome_summary"])


def _discover_runs(results_root: Path, phase: str) -> List[Dict[str, Any]]:
    runs = []
    for job_path in results_root.glob("*/*/job.json"):
        result_path = job_path.with_name("test_results.json")
        if not result_path.exists():
            continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if str(job.get("phase", "")) != phase:
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        resource_path = job_path.with_name("resource_diagnostics.json")
        resource = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.exists() else {}
        provenance_path = job_path.with_name("run_provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
        runs.append({
            "job": job,
            "result": result,
            "resource": resource,
            "provenance": provenance,
            "path": str(job_path.parent),
        })
    return runs


def _audit_expected_plan(runs: Sequence[Mapping[str, Any]],
                         phase: str,
                         expected_jobs_path: Path | str | None) -> Dict[str, Any]:
    observed_ids = [str(run["job"].get("job_id", "")) for run in runs]
    duplicate_ids = sorted({job_id for job_id in observed_ids if job_id and observed_ids.count(job_id) > 1})
    if expected_jobs_path is None:
        return {
            "expected_plan": None,
            "expected_jobs": None,
            "missing_job_ids": [],
            "unexpected_job_ids": [],
            "duplicate_result_job_ids": duplicate_ids,
            "plan_complete": None,
        }

    plan_path = Path(expected_jobs_path)
    expected_jobs = [
        json.loads(line)
        for line in plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_jobs = [job for job in expected_jobs if str(job.get("phase", "")) == phase]
    expected_ids = {str(job["job_id"]) for job in expected_jobs}
    observed_id_set = {job_id for job_id in observed_ids if job_id}
    missing = sorted(expected_ids - observed_id_set)
    unexpected = sorted(observed_id_set - expected_ids)
    return {
        "expected_plan": str(plan_path),
        "expected_jobs": len(expected_ids),
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "duplicate_result_job_ids": duplicate_ids,
        "plan_complete": not (missing or unexpected or duplicate_ids),
    }


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_runs(runs)
    output = []
    for (dataset, backbone, method), method_runs in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "seed_count": len(method_runs),
            "seeds": "|".join(str(run["job"]["seed"]) for run in sorted(method_runs, key=lambda item: int(item["job"]["seed"]))),
            "_seed_values": [int(run["job"]["seed"]) for run in method_runs],
        }
        dataset_provenance = [run.get("provenance", {}).get("dataset", {}) for run in method_runs]
        for key in ("num_users", "num_items", "num_train_samples"):
            values = {
                int(provenance[key])
                for provenance in dataset_provenance
                if provenance.get(key) is not None
            }
            row[key] = next(iter(values)) if len(values) == 1 else None
        environments = [run.get("provenance", {}).get("environment", {}) for run in method_runs]
        row["gpu_names"] = "|".join(sorted({
            str(environment["gpu_name"])
            for environment in environments
            if environment.get("gpu_name")
        }))
        row["torch_versions"] = "|".join(sorted({
            str(environment["torch_version"])
            for environment in environments
            if environment.get("torch_version")
        }))
        row["git_commits"] = "|".join(sorted({
            str(run.get("provenance", {}).get("code", {}).get("git_commit"))
            for run in method_runs
            if run.get("provenance", {}).get("code", {}).get("git_commit")
        }))
        for metric in PRIMARY_METRICS:
            values = [float(run["result"].get("utility", {}).get(metric, 0.0)) for run in method_runs]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _sample_std(values)
        runtime_keys = ("training_seconds", "test_evaluation_seconds", "parameter_count", "peak_cuda_memory_bytes")
        for key in runtime_keys:
            values = [
                float(run["result"].get("runtime", {}).get(key, 0.0))
                for run in method_runs
                if key in run["result"].get("runtime", {})
            ]
            row[f"{key}_mean"] = _mean(values) if values else None
            row[f"{key}_std"] = _sample_std(values) if values else None
        for key in (
            "attempted_positions",
            "replaced_positions",
            "replacement_rate",
            "lower_skew_fraction",
            "mean_skew_change",
            "mean_similarity",
        ):
            values = [
                float(run["result"].get("augmentation_diagnostics", {}).get(key))
                for run in method_runs
                if run["result"].get("augmentation_diagnostics", {}).get(key) is not None
            ]
            row[f"{key}_mean"] = _mean(values) if values else None
            row[f"{key}_std"] = _sample_std(values) if values else None
        for key in ("resource_build_seconds", "cache_bytes", "similarity_items", "low_skew_pool_items"):
            values = [
                float(run.get("resource", {}).get(key, 0.0))
                for run in method_runs
                if key in run.get("resource", {})
            ]
            row[f"{key}_mean"] = _mean(values) if values else None
            row[f"{key}_std"] = _sample_std(values) if values else None
        output.append(row)
    return output


def _aggregate_groups(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_runs(runs)
    output = []
    for (dataset, backbone, method), method_runs in sorted(grouped.items()):
        for attribute in GROUP_ATTRIBUTES:
            for metric in PRIMARY_METRICS:
                seed_summaries = []
                for run in method_runs:
                    users = run["result"].get("per_user", [])
                    if users and all(attribute in user and metric in user for user in users):
                        seed_summaries.append(summarize_group_metric(users, metric, attribute))
                if not seed_summaries:
                    continue
                row = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "attribute": attribute,
                    "metric": metric,
                    "seed_count": len(seed_summaries),
                }
                for key in ("signed_gap", "absolute_gap", "relative_gap", "worst_group", "best_group"):
                    values = [float(summary[key]) for summary in seed_summaries]
                    row[f"{key}_mean"] = _mean(values)
                    row[f"{key}_std"] = _sample_std(values)
                group_ids = sorted(
                    {group for summary in seed_summaries for group in summary["group_values"]},
                    key=str,
                )
                row["group_values"] = json.dumps({
                    str(group): _mean([
                        summary["group_values"][group]
                        for summary in seed_summaries
                        if group in summary["group_values"]
                    ])
                    for group in group_ids
                }, ensure_ascii=False, sort_keys=True)
                row["group_counts"] = json.dumps({
                    str(group): _mean([
                        summary["group_counts"][group]
                        for summary in seed_summaries
                        if group in summary["group_counts"]
                    ])
                    for group in group_ids
                }, ensure_ascii=False, sort_keys=True)
                row["worst_group_ids"] = "|".join(
                    str(summary["worst_group_id"])
                    for summary in seed_summaries
                )
                row["best_group_ids"] = "|".join(
                    str(summary["best_group_id"])
                    for summary in seed_summaries
                )
                output.append(row)
    return output


def _paired_comparisons(runs: Sequence[Mapping[str, Any]],
                        bootstrap_iterations: int,
                        bootstrap_seed: int,
                        progress: bool = False) -> List[Dict[str, Any]]:
    grouped = _group_runs(runs)
    settings = sorted({(dataset, backbone) for dataset, backbone, _ in grouped})
    total_methods = sum(
        len({
            method
            for candidate_dataset, candidate_backbone, method in grouped
            if candidate_dataset == dataset
            and candidate_backbone == backbone
            and method != "baseline"
        })
        for dataset, backbone in settings
    )
    method_index = 0
    comparisons = []
    raw_p_values_by_family = defaultdict(dict)
    for dataset, backbone in settings:
        baseline_runs = grouped.get((dataset, backbone, "baseline"), [])
        baseline_users = _combined_user_rows(baseline_runs)
        if not baseline_users:
            continue
        methods = sorted(
            method
            for candidate_dataset, candidate_backbone, method in grouped
            if candidate_dataset == dataset and candidate_backbone == backbone and method != "baseline"
        )
        for method in methods:
            method_users = _combined_user_rows(grouped[(dataset, backbone, method)])
            if not method_users:
                continue
            method_index += 1
            specifications = []
            for metric in PRIMARY_METRICS:
                if not all(metric in row for row in method_users + baseline_users):
                    continue
                specifications.append((metric, None, "mean", None))
                for attribute in GROUP_ATTRIBUTES:
                    specifications.extend([
                        (metric, attribute, "gap", None),
                        (metric, attribute, "signed_gap", None),
                        (metric, attribute, "relative_gap", None),
                        (metric, attribute, "worst_group", None),
                    ])
                    shared_groups = sorted(
                        {row.get(attribute) for row in method_users if attribute in row}
                        & {row.get(attribute) for row in baseline_users if attribute in row},
                        key=str,
                    )
                    specifications.extend(
                        (metric, attribute, "group_mean", group)
                        for group in shared_groups
                    )
            specifications = [
                specification
                for specification in specifications
                if not specification[1]
                or all(specification[1] in row for row in method_users + baseline_users)
            ]
            if progress:
                print(
                    f"[bootstrap {method_index}/{total_methods}] {dataset}/{backbone}/{method}: "
                    f"{len(specifications)} estimands, {bootstrap_iterations} shared iterations",
                    flush=True,
                )
            bootstrap_results = paired_hierarchical_bootstrap_many(
                method_users,
                baseline_users,
                specifications=specifications,
                iterations=bootstrap_iterations,
                random_seed=bootstrap_seed,
            )
            for (metric, attribute, estimand, group_value), bootstrap in zip(
                    specifications, bootstrap_results):
                comparison_id = (
                    f"{dataset}|{backbone}|{method}|{metric}|{attribute or 'overall'}|"
                    f"{estimand}|{group_value if group_value is not None else 'all'}"
                )
                family_id = (
                    f"{method}|{metric}|{attribute or 'overall'}|{estimand}|"
                    f"{group_value if group_value is not None else 'all'}"
                )
                raw_p = exact_paired_sign_flip_pvalue(list(bootstrap["seed_deltas"].values()))
                row = {
                    "comparison_id": comparison_id,
                    "comparison_family": family_id,
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    **bootstrap,
                    "raw_p_value": raw_p,
                }
                comparisons.append(row)
                raw_p_values_by_family[family_id][comparison_id] = raw_p

    adjusted = {}
    for family_values in raw_p_values_by_family.values():
        adjusted.update(holm_adjust(family_values))
    for row in comparisons:
        row["holm_p_value"] = adjusted[row["comparison_id"]]
        row["ci_excludes_zero"] = bool(row["ci_lower"] > 0.0 or row["ci_upper"] < 0.0)
        row["holm_significant"] = bool(row["holm_p_value"] < 0.05)
        direction = _improvement_direction(row["estimand"])
        row["improvement_direction"] = direction
        row["bootstrap_supported_improvement"] = _bootstrap_supports_improvement(row, direction)
        row["statistically_supported_improvement"] = bool(
            row["bootstrap_supported_improvement"] and row["holm_significant"]
        )
    return comparisons


def _improvement_direction(estimand: str) -> str:
    if estimand in {"gap", "relative_gap"}:
        return "lower"
    if estimand == "signed_gap":
        return "toward_zero"
    return "higher"


def _bootstrap_supports_improvement(row: Mapping[str, Any], direction: str) -> bool:
    if direction == "higher":
        return bool(float(row["ci_lower"]) > 0.0)
    if direction == "lower":
        return bool(float(row["ci_upper"]) < 0.0)
    point_improves = abs(float(row["method_estimate"])) < abs(float(row["baseline_estimate"]))
    return bool(point_improves and row["ci_excludes_zero"])


def _pareto_rows(run_summary: Sequence[Mapping[str, Any]],
                 group_summary: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    worst_group_lookup = {
        (row["dataset"], row["backbone"], row["method"], row["attribute"]): float(row["worst_group_mean"])
        for row in group_summary
        if row["metric"] == "NDCG@10" and row["attribute"] in {"gender", "age_group", "gender_age"}
    }
    settings: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for summary in run_summary:
        marginal_worst_values = [
            worst_group_lookup[key]
            for attribute in ("gender", "age_group")
            for key in [(summary["dataset"], summary["backbone"], summary["method"], attribute)]
            if key in worst_group_lookup
        ]
        if marginal_worst_values:
            settings[(summary["dataset"], summary["backbone"], "marginal_joint_worst")].append({
                "dataset": summary["dataset"],
                "backbone": summary["backbone"],
                "method": summary["method"],
                "fairness_axis": "marginal_joint_worst",
                "NDCG@10": float(summary["NDCG@10_mean"]),
                "fairness_value": min(marginal_worst_values),
            })
        intersection_key = (
            summary["dataset"], summary["backbone"], summary["method"], "gender_age"
        )
        if intersection_key in worst_group_lookup:
            settings[(summary["dataset"], summary["backbone"], "intersectional_worst")].append({
                "dataset": summary["dataset"],
                "backbone": summary["backbone"],
                "method": summary["method"],
                "fairness_axis": "intersectional_worst",
                "NDCG@10": float(summary["NDCG@10_mean"]),
                "fairness_value": worst_group_lookup[intersection_key],
            })

    output = []
    for points in settings.values():
        frontier = pareto_front(
            points,
            utility_key="NDCG@10",
            fairness_key="fairness_value",
        )
        for point in points:
            output.append({**point, "is_pareto": point["method"] in frontier})
    return sorted(
        output,
        key=lambda row: (row["dataset"], row["backbone"], row["fairness_axis"], row["method"]),
    )


def _proxy_outcome_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_runs(runs)
    output = []
    settings = sorted({(dataset, backbone) for dataset, backbone, _ in grouped})
    for dataset, backbone in settings:
        baseline_by_seed = {
            int(run["job"]["seed"]): run
            for run in grouped.get((dataset, backbone, "baseline"), [])
        }
        for (run_dataset, run_backbone, method), method_runs in grouped.items():
            if (run_dataset, run_backbone) != (dataset, backbone) or method == "baseline":
                continue
            for run in method_runs:
                diagnostics = run["result"].get("augmentation_diagnostics", {})
                mean_skew_change = diagnostics.get("mean_skew_change")
                seed = int(run["job"]["seed"])
                baseline = baseline_by_seed.get(seed)
                if mean_skew_change is None or baseline is None:
                    continue
                method_users = run["result"].get("per_user", [])
                baseline_users = baseline["result"].get("per_user", [])
                if not method_users or not baseline_users:
                    continue
                for attribute in GROUP_ATTRIBUTES:
                    if not all(attribute in row for row in method_users + baseline_users):
                        continue
                    for metric in PRIMARY_METRICS:
                        if not all(metric in row for row in method_users + baseline_users):
                            continue
                        method_summary = summarize_group_metric(method_users, metric, attribute)
                        baseline_summary = summarize_group_metric(baseline_users, metric, attribute)
                        output.append({
                            "dataset": dataset,
                            "backbone": backbone,
                            "method": method,
                            "seed": seed,
                            "attribute": attribute,
                            "metric": metric,
                            "bias_metric": run["job"].get("params", {}).get("bias_metric"),
                            "mean_skew_change": float(mean_skew_change),
                            "lower_skew_fraction": diagnostics.get("lower_skew_fraction"),
                            "replacement_rate": diagnostics.get("replacement_rate"),
                            "baseline_gap": float(baseline_summary["absolute_gap"]),
                            "method_gap": float(method_summary["absolute_gap"]),
                            "gap_delta": float(
                                method_summary["absolute_gap"] - baseline_summary["absolute_gap"]
                            ),
                            "baseline_worst_group": float(baseline_summary["worst_group"]),
                            "method_worst_group": float(method_summary["worst_group"]),
                            "worst_group_delta": float(
                                method_summary["worst_group"] - baseline_summary["worst_group"]
                            ),
                        })
    return sorted(
        output,
        key=lambda row: (
            row["dataset"], row["backbone"], row["method"], row["seed"], row["attribute"], row["metric"]
        ),
    )


def _proxy_outcome_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["attribute"], row["metric"])].append(row)
    output = []
    for (method, attribute, metric), values in sorted(grouped.items()):
        skew = [float(row["mean_skew_change"]) for row in values]
        output.append({
            "method": method,
            "attribute": attribute,
            "metric": metric,
            "observations": len(values),
            "skew_change_vs_gap_delta_spearman": _spearman(
                skew,
                [float(row["gap_delta"]) for row in values],
            ),
            "skew_change_vs_worst_group_delta_spearman": _spearman(
                skew,
                [float(row["worst_group_delta"]) for row in values],
            ),
        })
    return output


def _spearman(left: Sequence[float], right: Sequence[float]):
    if len(left) < 3 or len(left) != len(right):
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = _mean(left_ranks)
    right_mean = _mean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left_ranks) ** 0.5
    right_scale = sum((value - right_mean) ** 2 for value in right_ranks) ** 0.5
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return float(numerator / (left_scale * right_scale))


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _combined_user_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        seed = int(run["job"]["seed"])
        for user in run["result"].get("per_user", []):
            rows.append({**user, "seed": seed})
    return rows


def _group_runs(runs: Sequence[Mapping[str, Any]]):
    grouped = defaultdict(list)
    for run in runs:
        job = run["job"]
        grouped[(job["dataset"], job["backbone"], job["method"])].append(run)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze five-seed Fair-NCL supplementary runs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="tables/supplementary")
    parser.add_argument("--phase", default="supplementary")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--jobs", default=None, help="Final JSONL plan used to reject missing or unexpected runs.")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    report = build_supplementary_report(
        args.results_root,
        phase=args.phase,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        expected_jobs_path=args.jobs,
        require_complete=bool(args.jobs) and not args.allow_incomplete,
        progress=True,
    )
    write_supplementary_report(report, args.output_dir)
    print(json.dumps(report["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

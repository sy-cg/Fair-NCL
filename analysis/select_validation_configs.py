"""Select compact-audit configurations using validation artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .supplementary_statistics import multi_objective_pareto_front, select_validation_configuration


SELECTION_SEEDS = (42, 2025, 2026)


def build_validation_selection(results_root: Path | str,
                               utility_tolerance: float = 0.15,
                               audit_jobs_path: Path | str | None = None) -> List[Dict[str, Any]]:
    audit_runs, baseline_runs = _discover_validation_runs(Path(results_root))
    if not audit_runs:
        raise ValueError("No completed validation_audit runs were found.")
    _validate_audit_plan_completion(audit_runs, audit_jobs_path)
    expected_seeds = set(SELECTION_SEEDS)
    for setting, runs in baseline_runs.items():
        observed = {int(run["job"]["seed"]) for run in runs}
        if observed != expected_seeds or len(runs) != len(expected_seeds):
            raise ValueError(
                f"Incomplete baseline validation reference for {setting[0]}:{setting[1]}: "
                f"expected seeds {sorted(expected_seeds)}, found {sorted(observed)}."
            )
    baseline_by_seed = {
        setting: {int(run["job"]["seed"]): run for run in runs}
        for setting, runs in baseline_runs.items()
    }
    baseline_ndcg = {}
    baseline_disadvantaged_groups = {}
    for setting, runs in baseline_runs.items():
        utilities = [float(run["validation"]["utility"]["NDCG@10"]) for run in runs]
        if any(not math.isfinite(value) or value <= 0.0 for value in utilities):
            raise ValueError(f"Invalid baseline validation utility for {setting[0]}:{setting[1]}.")
        baseline_ndcg[setting] = _mean(utilities)
        baseline_disadvantaged_groups[setting] = _fixed_disadvantaged_groups(runs)
    candidate_groups = defaultdict(list)
    for run in audit_runs:
        job = run["job"]
        key = (
            job["dataset"],
            job["backbone"],
            job["method"],
            job.get("params", {}).get("audit_candidate"),
        )
        candidate_groups[key].append(run)

    method_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (dataset, backbone, method, candidate), runs in candidate_groups.items():
        observed = {int(run["job"]["seed"]) for run in runs}
        if observed != expected_seeds or len(runs) != len(expected_seeds):
            raise ValueError(
                f"Incomplete validation candidate {dataset}:{backbone}:{method}:{candidate}: "
                f"expected seeds {sorted(expected_seeds)}, found {sorted(observed)}."
            )
        setting = (dataset, backbone)
        if setting not in baseline_by_seed:
            raise ValueError(
                f"Missing baseline validation reference for {dataset}:{backbone}. "
                "Run the fixed backbone with validation_results.json before selecting audit configurations."
            )
        runs_by_seed = {int(run["job"]["seed"]): run for run in runs}
        ordered_runs = [runs_by_seed[seed] for seed in SELECTION_SEEDS]
        utilities = [float(run["validation"]["utility"]["NDCG@10"]) for run in ordered_runs]
        utility_retentions = []
        normalized_gaps = []
        worst_groups = []
        attribute_normalized_gaps = {attribute: [] for attribute in ("gender", "age_group")}
        attribute_disadvantaged_gains = {attribute: [] for attribute in ("gender", "age_group")}
        for seed, run, utility in zip(SELECTION_SEEDS, ordered_runs, utilities):
            if not math.isfinite(utility):
                raise ValueError(
                    f"Non-finite validation utility in {run['job'].get('job_id', '<unknown>')}"
                )
            baseline_run = baseline_by_seed[setting][seed]
            baseline_utility = float(baseline_run["validation"]["utility"]["NDCG@10"])
            utility_retentions.append(utility / baseline_utility)
            evidence = _required_fairness_evidence(run)
            baseline_evidence = _required_fairness_evidence(baseline_run)
            per_seed_gaps = []
            per_seed_worst = []
            for attribute in ("gender", "age_group"):
                normalized_gap = (
                    evidence[attribute]["gap"] / utility if utility > 0 else float("inf")
                )
                attribute_normalized_gaps[attribute].append(normalized_gap)
                per_seed_gaps.append(normalized_gap)
                per_seed_worst.append(evidence[attribute]["worst_group"])
                disadvantaged_group = baseline_disadvantaged_groups[setting][attribute]
                if disadvantaged_group not in evidence[attribute]["group_values"]:
                    raise ValueError(
                        f"Run {run['job'].get('job_id', '<unknown>')} is missing baseline "
                        f"disadvantaged group {disadvantaged_group!r} for {attribute}."
                    )
                attribute_disadvantaged_gains[attribute].append(
                    evidence[attribute]["group_values"][disadvantaged_group]
                    - baseline_evidence[attribute]["group_values"][disadvantaged_group]
                )
            normalized_gaps.append(_mean(per_seed_gaps))
            worst_groups.append(min(per_seed_worst))

        attribute_fairness_benefit = {
            attribute: _mean(values) - _sample_std(values)
            for attribute, values in attribute_disadvantaged_gains.items()
        }
        attribute_gap_risk = {
            attribute: _mean(values) + _sample_std(values)
            for attribute, values in attribute_normalized_gaps.items()
        }
        utility_retention_mean = _mean(utility_retentions)
        utility_retention_std = _sample_std(utility_retentions)
        fairness_benefit = min(attribute_fairness_benefit.values())
        gap_risk = max(attribute_gap_risk.values())

        params = dict(ordered_runs[0]["job"].get("params", {}))
        method_groups[(dataset, backbone, method)].append({
            "candidate": candidate,
            "NDCG@10": _mean(utilities),
            "NDCG@10_std": _sample_std(utilities),
            "utility_retention_mean": utility_retention_mean,
            "utility_retention_std": utility_retention_std,
            "utility_retention_conservative": utility_retention_mean - utility_retention_std,
            "normalized_gap": _mean(normalized_gaps),
            "normalized_gap_std": _sample_std(normalized_gaps),
            "attribute_gap_risk": attribute_gap_risk,
            "gap_risk": gap_risk,
            "attribute_fairness_benefit": attribute_fairness_benefit,
            "fairness_benefit": fairness_benefit,
            "fairness_improving": fairness_benefit >= 0.0,
            "worst_group": _mean(worst_groups),
            "worst_group_std": _sample_std(worst_groups),
            "seed_count": len(runs),
            "params": params,
        })

    selections = []
    for (dataset, backbone, method), candidates in sorted(method_groups.items()):
        if len(candidates) != 5:
            raise ValueError(
                f"Incomplete validation audit for {dataset}:{backbone}:{method}: "
                f"expected 5 candidates, found {len(candidates)}."
            )
        setting = (dataset, backbone)
        if setting not in baseline_ndcg:
            raise ValueError(
                f"Missing baseline validation reference for {dataset}:{backbone}. "
                "Run the fixed backbone with validation_results.json before selecting audit configurations."
            )
        utility_retention_floor = 1.0 - float(utility_tolerance)
        qualified_candidates = [
            candidate
            for candidate in candidates
            if float(candidate["utility_retention_conservative"]) >= utility_retention_floor
        ]
        frontier = multi_objective_pareto_front(
            qualified_candidates,
            objectives={"NDCG@10": True, "fairness_benefit": True, "gap_risk": False},
            method_key="candidate",
        )
        annotated_candidates = [
            {
                **candidate,
                "fairness_risk_score": float(candidate["gap_risk"]),
                "is_validation_pareto": candidate["candidate"] in frontier,
            }
            for candidate in candidates
        ]
        selected = select_validation_configuration(
            annotated_candidates,
            backbone_ndcg=baseline_ndcg[setting],
            utility_tolerance=utility_tolerance,
        )
        selections.append({
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "backbone_NDCG@10": baseline_ndcg[setting],
            "baseline_disadvantaged_groups": baseline_disadvantaged_groups[setting],
            "all_candidates": annotated_candidates,
            **selected,
        })
    return selections


def build_method_override_config(selections: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    methods: Dict[str, Any] = {}
    for selection in selections:
        params = {
            key: value
            for key, value in selection["params"].items()
            if key not in {"audit_candidate", "export_per_user_metrics"}
        }
        method_scope = methods.setdefault(selection["method"], {"__dataset_backbones__": {}})
        dataset_scope = method_scope["__dataset_backbones__"].setdefault(selection["dataset"], {})
        dataset_scope[selection["backbone"]] = params
    return {"__methods__": methods}


def _discover_validation_runs(results_root: Path):
    audit_runs = []
    baseline_runs = defaultdict(list)
    for validation_path in results_root.glob("*/*/validation_results.json"):
        job_path = validation_path.with_name("job.json")
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        run = {"job": job, "validation": validation}
        if job.get("phase") == "validation_audit":
            audit_runs.append(run)
        elif (
            job.get("phase") == "validation_reference"
            and job.get("method") == "baseline"
            and int(job.get("seed", -1)) in SELECTION_SEEDS
        ):
            baseline_runs[(job["dataset"], job["backbone"])].append(run)
    return audit_runs, baseline_runs


def _validate_audit_plan_completion(audit_runs, audit_jobs_path: Path | str | None) -> None:
    if audit_jobs_path is None:
        return
    expected_jobs = [
        json.loads(line)
        for line in Path(audit_jobs_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_ids = {
        str(job["job_id"])
        for job in expected_jobs
        if job.get("phase") == "validation_audit"
    }
    observed_ids = [str(run["job"].get("job_id", "")) for run in audit_runs]
    duplicate_ids = {job_id for job_id in observed_ids if job_id and observed_ids.count(job_id) > 1}
    observed_id_set = {job_id for job_id in observed_ids if job_id}
    missing = expected_ids - observed_id_set
    unexpected = observed_id_set - expected_ids
    if missing or unexpected or duplicate_ids:
        raise ValueError(
            f"Validation audit is incomplete: missing {len(missing)} expected jobs, "
            f"found {len(unexpected)} unexpected jobs and {len(duplicate_ids)} duplicate job ids."
        )


def _required_fairness_evidence(run):
    job_id = run["job"].get("job_id", "<unknown>")
    fairness = run["validation"].get("fairness", {})
    evidence = {}
    for attribute in ("gender", "age_group"):
        metrics = fairness.get(attribute)
        required = ("ndcg@10_gap", "ndcg@10_worst_group", "group_ndcg@10")
        if not isinstance(metrics, dict) or any(key not in metrics for key in required):
            raise ValueError(f"Run {job_id} is missing fairness evidence for {attribute}.")
        group_values = metrics["group_ndcg@10"]
        if not isinstance(group_values, dict) or len(group_values) < 2:
            raise ValueError(f"Run {job_id} is missing fairness evidence for {attribute}.")
        group_values = {str(group): float(value) for group, value in group_values.items()}
        gap = float(metrics["ndcg@10_gap"])
        worst = float(metrics["ndcg@10_worst_group"])
        if (
            not math.isfinite(gap)
            or not math.isfinite(worst)
            or any(not math.isfinite(value) for value in group_values.values())
        ):
            raise ValueError(f"Run {job_id} has non-finite fairness evidence for {attribute}.")
        evidence[attribute] = {
            "gap": gap,
            "worst_group": worst,
            "group_values": group_values,
        }
    return evidence


def _fixed_disadvantaged_groups(runs) -> Dict[str, str]:
    evidence_by_run = [_required_fairness_evidence(run) for run in runs]
    disadvantaged = {}
    for attribute in ("gender", "age_group"):
        group_sets = [set(evidence[attribute]["group_values"]) for evidence in evidence_by_run]
        if any(groups != group_sets[0] for groups in group_sets[1:]):
            setting = runs[0]["job"]
            raise ValueError(
                f"Baseline group identities differ across seeds for "
                f"{setting['dataset']}:{setting['backbone']}:{attribute}."
            )
        group_means = {
            group: _mean([
                evidence[attribute]["group_values"][group]
                for evidence in evidence_by_run
            ])
            for group in group_sets[0]
        }
        disadvantaged[attribute] = min(group_means, key=lambda group: (group_means[group], group))
    return disadvantaged


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _write_selection_csv(path: Path, selections: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for selection in selections:
        rows.append({
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list)) else value
            for key, value in selection.items()
        })
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_candidate_rows(selections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for selection in selections:
        for candidate in selection.get("all_candidates", []):
            rows.append({
                "dataset": selection["dataset"],
                "backbone": selection["backbone"],
                "method": selection["method"],
                "backbone_NDCG@10": selection["backbone_NDCG@10"],
                "selected": candidate["candidate"] == selection["candidate"],
                **candidate,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select validation-audit configurations under a utility constraint.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--utility-tolerance", type=float, default=0.15)
    parser.add_argument("--output", default="configs/revision_method_overrides.json")
    parser.add_argument("--summary", default="tables/supplementary/validation_selection.csv")
    parser.add_argument(
        "--candidate-summary",
        default="tables/supplementary/validation_candidates.csv",
    )
    parser.add_argument("--jobs", default=None, help="Validation-audit JSONL plan used to reject incomplete results.")
    args = parser.parse_args()

    selections = build_validation_selection(
        args.results_root,
        args.utility_tolerance,
        audit_jobs_path=args.jobs,
    )
    overrides = build_method_override_config(selections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_selection_csv(summary_path, selections)
    candidate_path = Path(args.candidate_summary)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    _write_selection_csv(candidate_path, _flatten_candidate_rows(selections))
    print(f"Selected {len(selections)} dataset-backbone-method configurations.")


if __name__ == "__main__":
    main()

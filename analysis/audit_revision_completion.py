"""Audit existing and planned coverage for the reviewer-driven experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from experiments.revision_completion import SEEDS
from experiments.job_utils import result_artifact_is_complete


EXISTING_METHODS = ("baseline", "fair_ncl", "ncl_sr_mce", "a_fsr")
DIAGNOSTIC_PHASES = {"validation_audit", "mce_stability"}


def _existing_groups(results_root):
    groups = defaultdict(set)
    checkpoints = defaultdict(int)
    incomplete = []
    for job_path in Path(results_root).glob("*/*/job.json"):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        method = job.get("method")
        seed = int(job.get("seed", -1))
        if job.get("phase") != "supplementary" or method not in EXISTING_METHODS or seed not in SEEDS:
            continue
        if result_artifact_is_complete(job, results_root):
            groups[(job["dataset"], job["backbone"], method)].add(seed)
        else:
            incomplete.append({
                "dataset": job["dataset"],
                "backbone": job["backbone"],
                "method": method,
                "seed": seed,
                "job_id": job["job_id"],
            })
        if job_path.with_name("best.pt").exists():
            checkpoints[method] += 1
    return groups, checkpoints, incomplete


def _diagnostic_training_failures(results_root):
    grouped = defaultdict(list)
    for job_path in Path(results_root).glob("*/*/job.json"):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("method") != "ncl_sr_mce" or job.get("phase") not in DIAGNOSTIC_PHASES:
            continue
        history_path = job_path.with_name("history.json")
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        skipped = [int(value) for value in history.get("skipped_nonfinite_updates", [])]
        train_metrics = [
            values for key, values in history.items()
            if key.startswith("train_") and isinstance(values, list) and values
        ]
        failed = bool(skipped) and sum(skipped) > 0 and not train_metrics
        grouped[(job["dataset"], job["backbone"])].append({
            "phase": job["phase"],
            "seed": int(job.get("seed", -1)),
            "job_id": job["job_id"],
            "skipped_nonfinite_updates": sum(skipped),
            "zero_finite_training_updates": failed,
        })

    failures = []
    for (dataset, backbone), runs in sorted(grouped.items()):
        failed_runs = [run for run in runs if run["zero_finite_training_updates"]]
        if not failed_runs:
            continue
        failures.append({
            "dataset": dataset,
            "backbone": backbone,
            "method": "ncl_sr_mce",
            "diagnostic_runs": len(runs),
            "zero_finite_update_runs": len(failed_runs),
            "all_diagnostic_runs_failed": len(failed_runs) == len(runs),
            "seeds": sorted({run["seed"] for run in failed_runs}),
            "phases": sorted({run["phase"] for run in failed_runs}),
            "total_skipped_nonfinite_updates": sum(
                run["skipped_nonfinite_updates"] for run in failed_runs
            ),
            "evidence_job_ids": [run["job_id"] for run in failed_runs],
        })
    return failures


def _planned(plan_dir, results_root):
    output = {}
    for path in sorted(Path(plan_dir).glob("*.jsonl")):
        jobs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        incomplete_job_ids = [
            job["job_id"] for job in jobs
            if not result_artifact_is_complete(job, results_root)
        ]
        output[path.stem] = {
            "jobs": len(jobs),
            "completed_jobs": len(jobs) - len(incomplete_job_ids),
            "incomplete_jobs": len(incomplete_job_ids),
            "complete": not incomplete_job_ids,
            "incomplete_job_ids": incomplete_job_ids,
            "methods": sorted({job["method"] for job in jobs}),
            "datasets": sorted({job["dataset"] for job in jobs}),
            "backbones": sorted({job["backbone"] for job in jobs}),
            "test_evaluation": any(job["params"].get("evaluate_test", True) for job in jobs),
        }
    return output


def build_audit(results_root, plan_dir):
    groups, checkpoints, incomplete = _existing_groups(results_root)
    diagnostic_failures = _diagnostic_training_failures(results_root)
    failure_pairs = {
        (row["dataset"], row["backbone"]): row
        for row in diagnostic_failures
        if set(row["seeds"]) >= {42, 2025, 2026}
    }
    matrix = []
    for dataset in ("ml-1m", "taobao"):
        for backbone in ("sasrec", "bert4rec", "gru4rec", "caser"):
            for method in EXISTING_METHODS:
                seeds = groups.get((dataset, backbone, method), set())
                matrix.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "completed_seeds": sorted(seeds),
                    "seed_count": len(seeds),
                    "missing_seeds": sorted(set(SEEDS) - seeds),
                })
    missing_ncl = []
    for dataset in ("ml-1m", "taobao"):
        for backbone in ("sasrec", "bert4rec", "gru4rec", "caser"):
            if len(groups.get((dataset, backbone, "ncl_sr_mce"), set())) >= len(SEEDS):
                continue
            failure = failure_pairs.get((dataset, backbone))
            missing_ncl.append({
                "dataset": dataset,
                "backbone": backbone,
                "status": "documented_training_failure" if failure else "missing",
                "diagnostic_evidence": failure,
            })
    return {
        "seeds": list(SEEDS),
        "existing_matrix": matrix,
        "incomplete_formal_runs": incomplete,
        "checkpoint_counts": dict(checkpoints),
        "planned": _planned(plan_dir, results_root),
        "constraints": {
            "rerun_ncl_sr_mce": False,
            "rerun_a_fsr": False,
            "run_cost_profile": False,
        },
        "diagnostic_training_failures": diagnostic_failures,
        "known_unfilled_existing_cells": {
            "ncl_sr_mce": missing_ncl,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results_revision_kbs_v2")
    parser.add_argument("--plan-dir", default="experiments/revision_completion_plans")
    parser.add_argument("--output", default="tables/supplementary/revision_completion/coverage_audit.json")
    parser.add_argument(
        "--require-complete-plans",
        action="store_true",
        help="Exit nonzero when any plan in --plan-dir still has incomplete jobs.",
    )
    args = parser.parse_args()
    audit = build_audit(args.results_root, args.plan_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.require_complete_plans:
        incomplete = [
            name for name, state in audit["planned"].items()
            if not state["complete"]
        ]
        if incomplete:
            raise SystemExit(f"Incomplete experiment plans: {', '.join(incomplete)}")


if __name__ == "__main__":
    main()

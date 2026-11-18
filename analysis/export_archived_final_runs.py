"""Freeze the reported five-seed jobs and their recorded environments."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_revision_kbs_v2"
SUMMARY = ROOT / "tables/supplementary/final_5seed_extended/run_summary.csv"
PLAN = ROOT / "experiments/revision_supplementary_final_extended_595.jsonl"
PACKAGE_PLAN = ROOT / "els-cas-templates/source_data/configs_and_experiment_specs/revision_supplementary_final_extended_595.jsonl"
ENVIRONMENTS = ROOT / "els-cas-templates/source_data/environment/final_run_environment.csv"
PARTIAL_PLAN = ROOT / "experiments/revision_supplementary_final_valid.jsonl"
SEEDS = {42, 2025, 2026, 2027, 2028}


def main():
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    expected = {
        (row["dataset"], row["backbone"], row["method"]): row
        for row in summaries
    }
    if len(expected) != len(summaries):
        raise ValueError("Duplicate groups in the reported run summary")

    jobs = []
    environments = []
    observed = defaultdict(set)
    for job_path in sorted(RESULTS.glob("*/*/job.json")):
        raw_job = job_path.read_bytes()
        job = json.loads(raw_job)
        if job.get("phase") != "supplementary":
            continue
        key = (job["dataset"], job["backbone"], job["method"])
        if key not in expected:
            raise ValueError(f"Unreported supplementary group: {key}")
        seed = int(job["seed"])
        if seed not in SEEDS or seed in observed[key]:
            raise ValueError(f"Unexpected or duplicate seed: {key}, {seed}")
        if job_path.parent.name != job["job_id"] or job_path.parent.parent.name != job["dataset"]:
            raise ValueError(f"Job path does not match its identifier: {job_path}")
        test_path = job_path.with_name("test_results.json")
        provenance_path = job_path.with_name("run_provenance.json")
        if not test_path.is_file() or not provenance_path.is_file():
            raise ValueError(f"Missing final-run artifact beside {job_path}")
        if "selected" not in json.loads(test_path.read_text(encoding="utf-8")):
            raise ValueError(f"No selected test result in {test_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        env = provenance.get("environment", {})
        environments.append({
            "dataset": job["dataset"],
            "backbone": job["backbone"],
            "method": job["method"],
            "seed": seed,
            "job_id": job["job_id"],
            "job_json_sha256": hashlib.sha256(raw_job).hexdigest(),
            "platform": env.get("platform", ""),
            "python_version": env.get("python_version", ""),
            "torch_version": env.get("torch_version", ""),
            "numpy_version": env.get("numpy_version", ""),
            "pandas_version": env.get("pandas_version", ""),
            "scipy_version": env.get("scipy_version", ""),
            "cuda_version": env.get("cuda_version", ""),
            "cudnn_version": env.get("cudnn_version", ""),
            "gpu_name": env.get("gpu_name", ""),
            "recorded_git_commit": provenance.get("code", {}).get("git_commit") or "",
        })
        jobs.append(job)
        observed[key].add(seed)

    for key, row in expected.items():
        reported_seeds = {int(seed) for seed in row["seeds"].split("|")}
        if observed[key] != reported_seeds or len(observed[key]) != int(row["seed_count"]):
            raise ValueError(f"Run summary and archived jobs disagree: {key}")
    if len(jobs) != 595 or any(seeds != SEEDS for seeds in observed.values()):
        raise ValueError(f"Expected 119 complete five-seed groups; found {len(jobs)} jobs")

    partial_jobs = {
        job["job_id"]: job
        for line in PARTIAL_PLAN.read_text(encoding="utf-8").splitlines() if line.strip()
        for job in [json.loads(line)]
    }
    archived_jobs = {job["job_id"]: job for job in jobs}
    if any(archived_jobs.get(job_id) != job for job_id, job in partial_jobs.items()):
        raise ValueError("Archived jobs disagree with the existing 535-job plan")

    jobs.sort(key=lambda job: (job["dataset"], job["backbone"], job["method"], int(job["seed"])))
    environments.sort(key=lambda row: (row["dataset"], row["backbone"], row["method"], row["seed"]))
    plan_text = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    PLAN.write_text(plan_text, encoding="utf-8")
    PACKAGE_PLAN.parent.mkdir(parents=True, exist_ok=True)
    PACKAGE_PLAN.write_text(plan_text, encoding="utf-8")
    ENVIRONMENTS.parent.mkdir(parents=True, exist_ok=True)
    with ENVIRONMENTS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(environments[0]))
        writer.writeheader()
        writer.writerows(environments)
    print(f"Wrote {len(jobs)} jobs to {PLAN.relative_to(ROOT)}")
    print(f"Wrote {len(environments)} environment records to {ENVIRONMENTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Summarize validation-gap variability for selected Fair-NCL configurations."""

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_revision_kbs_v2"
SELECTION = ROOT / "tables/supplementary/validation_selection.csv"
OUTPUT = ROOT / "tables/supplementary/validation_gap_stability.csv"
SELECTION_SEEDS = {42, 2025, 2026}
FINAL_SEEDS = {42, 2025, 2026, 2027, 2028}
TUNED_KEYS = (
    "learning_rate", "hidden_units", "dropout_rate", "epsilon", "augment_ratio",
    "utility_alpha", "utility_beta", "fair_ncl_align_weight",
    "fair_ncl_var_weight", "fair_ncl_cov_weight",
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _gaps(result):
    fairness = result["fairness"]
    gender = float(fairness["gender"]["ndcg@10_gap"])
    age = float(fairness["age_group"]["ndcg@10_gap"])
    return gender, age, (gender + age) / 2


def _summary(values, prefix):
    return {f"{prefix}_mean": mean(values), f"{prefix}_sd": stdev(values)}


def _jobs(dataset, backbone, phase):
    pattern = f"{phase}_{dataset}_fair_ncl_{backbone}_s*/job.json"
    return [_read(path) for path in (RESULTS / dataset).glob(pattern)]


def _job_dir(job):
    return RESULTS / job["dataset"] / job["job_id"]


def build_rows():
    with SELECTION.open(encoding="utf-8-sig", newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["method"] == "fair_ncl"]
    if len(selected) != 8:
        raise ValueError(f"Expected 8 selected Fair-NCL configurations, found {len(selected)}")

    rows = []
    for choice in selected:
        dataset, backbone, candidate = (
            choice["dataset"], choice["backbone"], choice["candidate"]
        )
        audit = [
            job for job in _jobs(dataset, backbone, "validation_audit")
            if job["params"].get("audit_candidate") == candidate
        ]
        final = _jobs(dataset, backbone, "supplementary")
        if {int(job["seed"]) for job in audit} != SELECTION_SEEDS or len(audit) != 3:
            raise ValueError(f"Incomplete selection runs: {dataset}/{backbone}/{candidate}")
        if {int(job["seed"]) for job in final} != FINAL_SEEDS or len(final) != 5:
            raise ValueError(f"Incomplete final runs: {dataset}/{backbone}")

        tuned = json.loads(choice["params"])
        for job in final:
            for key in TUNED_KEYS:
                if job["params"].get(key) != tuned.get(key):
                    raise ValueError(f"Final run differs from selected {key}: {job['job_id']}")

        audit_gaps = [_gaps(_read(_job_dir(job) / "validation_results.json")) for job in audit]
        final_gaps = [
            (
                _gaps(_read(_job_dir(job) / "validation_results.json")),
                _gaps(_read(_job_dir(job) / "test_results.json")),
            )
            for job in final
        ]
        val_means = [pair[0][2] for pair in final_gaps]
        test_means = [pair[1][2] for pair in final_gaps]
        deltas = [test - val for val, test in zip(val_means, test_means)]
        row = {"dataset": dataset, "backbone": backbone, "candidate": candidate,
               "selection_seed_count": 3, "final_seed_count": 5}
        for index, label in enumerate(("gender", "age", "average")):
            row.update(_summary([gaps[index] for gaps in audit_gaps], f"selection_{label}_gap"))
        row.update(_summary(val_means, "final_validation_average_gap"))
        row.update(_summary(test_means, "final_test_average_gap"))
        row.update(_summary(deltas, "test_minus_validation_average_gap"))
        row["test_gap_higher_seed_count"] = sum(delta > 0 for delta in deltas)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["dataset"], row["backbone"]))


if __name__ == "__main__":
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} verified setting rows to {OUTPUT}")

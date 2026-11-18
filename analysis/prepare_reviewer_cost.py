"""Summarize saved final runs for the reviewer protocol and cost analysis."""

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_revision_efficiency import benchmark_manifest  # noqa: E402


METHODS = {"baseline", "fair_ncl", "adv_debias", "grl", "sm_pcfr", "a_fsr", "afrl", "pfrec"}
OUT = ROOT / "tables" / "supplementary" / "reviewer_protocol_cost"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def main():
    plan = ROOT / "experiments" / "revision_supplementary_final_extended_595.jsonl"
    jobs = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = [job for job in jobs if job["method"] in METHODS]
    assert len(jobs) == 320
    manifest = OUT / "final_main_run_manifest.csv"
    write_csv(manifest, [
        {"dataset": job["dataset"], "backbone": job["backbone"],
         "method": job["method"], "seed": job["seed"], "job_id": job["job_id"],
         "result_dir": str(Path("results_revision_kbs_v2") / job["dataset"] / job["job_id"])}
        for job in jobs
    ], ["dataset", "backbone", "method", "seed", "job_id", "result_dir"])

    raw_path = OUT / "final_main_run_costs.csv"
    audit = benchmark_manifest(manifest, raw_path, OUT / "cost_audit.json")
    assert audit["processed_rows"] == 320 and not audit["missing_rows"] and not audit["error_rows"]
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames) + ["trained_epochs"]
    for row in rows:
        history = json.loads((ROOT / row["result_dir"] / "history.json").read_text(encoding="utf-8"))
        row["trained_epochs"] = len(history["train_loss"])
        assert row["trained_epochs"] > 0
        assert row["training_seconds"] and row["peak_cuda_memory_gb"] and row["gpu_name"]
    write_csv(raw_path, rows, fields)

    grouped = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["method"])
        grouped.setdefault(key, []).append(row)
    assert len(grouped) == 64 and all(len(group) == 5 for group in grouped.values())

    paired = []
    for dataset in ("ml-1m", "taobao"):
        for backbone in ("sasrec", "bert4rec", "gru4rec", "caser"):
            baseline = {int(row["seed"]): row for row in grouped[(dataset, backbone, "baseline")]}
            fair = {int(row["seed"]): row for row in grouped[(dataset, backbone, "fair_ncl")]}
            assert baseline.keys() == fair.keys() == {42, 2025, 2026, 2027, 2028}
            assert all(baseline[seed]["gpu_name"] == fair[seed]["gpu_name"] for seed in baseline)
            b_rows, f_rows = list(baseline.values()), list(fair.values())
            paired.append({
                "dataset": dataset, "backbone": backbone, "paired_seeds": 5,
                "gpu_models": "; ".join(sorted({row["gpu_name"] for row in b_rows})),
                "baseline_train_h_median": median(b_rows, "training_seconds") / 3600,
                "fair_ncl_train_h_median": median(f_rows, "training_seconds") / 3600,
                "train_ratio_median": statistics.median(
                    float(fair[seed]["training_seconds"]) / float(baseline[seed]["training_seconds"])
                    for seed in baseline),
                "baseline_epochs_median": median(b_rows, "trained_epochs"),
                "fair_ncl_epochs_median": median(f_rows, "trained_epochs"),
                "per_epoch_ratio_median": statistics.median(
                    (float(fair[seed]["training_seconds"]) / float(fair[seed]["trained_epochs"])) /
                    (float(baseline[seed]["training_seconds"]) / float(baseline[seed]["trained_epochs"]))
                    for seed in baseline),
                "baseline_peak_cuda_gb_median": median(b_rows, "peak_cuda_memory_gb"),
                "fair_ncl_peak_cuda_gb_median": median(f_rows, "peak_cuda_memory_gb"),
                "baseline_resource_s_median": median(b_rows, "resource_build_seconds"),
                "fair_ncl_resource_s_median": median(f_rows, "resource_build_seconds"),
                "fair_ncl_similarity_items_median": median(f_rows, "similarity_items"),
            })
    write_csv(OUT / "paired_fair_ncl_backbone_costs.csv", paired, list(paired[0]))

    selections = list(csv.DictReader((ROOT / "tables" / "supplementary" / "validation_selection.csv").open(
        encoding="utf-8-sig", newline="")))
    audit_rows = []
    for method in sorted(METHODS - {"baseline"}):
        method_rows = [row for row in selections if row["method"] == method]
        statuses = Counter(row["selection_status"] for row in method_rows)
        assert len(method_rows) == 8 and sum(statuses.values()) == 8
        audit_rows.append({"method": method, "dataset_backbone_blocks": 8,
                           "candidates_per_block": 5, "validation_seeds_per_candidate": 3,
                           "validation_runs": 120, "qualified_selections": statuses["qualified"],
                           "utility_fallbacks": statuses["no_qualified_candidate"]})
    write_csv(OUT / "shared_validation_audit_summary.csv", audit_rows, list(audit_rows[0]))
    print(json.dumps({"main_run_rows": len(rows), "paired_blocks": len(paired),
                      "audit_methods": len(audit_rows), "audit_fallbacks": sum(r["utility_fallbacks"] for r in audit_rows)},
                     indent=2))


if __name__ == "__main__":
    main()

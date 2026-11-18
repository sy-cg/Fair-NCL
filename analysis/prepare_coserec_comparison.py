"""Audit CoSeRec tuning and prepare its paired SASRec comparison."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from analysis.supplementary_statistics import (
    exact_paired_sign_flip_pvalue,
    holm_adjust,
    paired_hierarchical_bootstrap_many,
)


ROOT = Path("results_revision_kbs_v2")
PLANS = Path("experiments/revision_lean_plans")
MAIN_PLAN = Path("experiments/revision_supplementary_final_extended_595.jsonl")
OUT = Path("tables/supplementary/reviewer_coserec")
SOURCE = Path("els-cas-templates/source_data/table_source_data")
SEEDS = (42, 2025, 2026, 2027, 2028)
METRICS = ("HitRate@10", "NDCG@10", "gender_HitRate@10_Gap",
           "gender_NDCG@10_Gap", "age_group_HitRate@10_Gap",
           "age_group_NDCG@10_Gap", "gender_NDCG@10_WorstGroup",
           "age_group_NDCG@10_WorstGroup", "HitRate@20", "NDCG@20",
           "gender_HitRate@20_Gap", "gender_NDCG@20_Gap",
           "age_group_HitRate@20_Gap", "age_group_NDCG@20_Gap",
           "gender_NDCG@20_WorstGroup", "age_group_NDCG@20_WorstGroup")


def read_plan(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def result_path(job, name):
    return ROOT / job["dataset"] / job["job_id"] / name


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_tuning():
    grouped = defaultdict(list)
    rows = []
    for job in read_plan(PLANS / "coserec_tune.jsonl"):
        saved = json.loads(result_path(job, "job.json").read_text(encoding="utf-8"))
        assert saved == job, job["job_id"]
        marker = json.loads(result_path(job, "test_results.json").read_text(encoding="utf-8"))
        assert marker.get("status") == "not_run_by_protocol" and "selected" not in marker, job["job_id"]
        validation = json.loads(result_path(job, "validation_results.json").read_text(encoding="utf-8"))
        score = float(validation["utility"]["HitRate@10"])
        key = (job["dataset"], float(job["params"]["coserec_cf_weight"]))
        grouped[key].append((int(job["seed"]), score))
        rows.append({"dataset": key[0], "backbone": "sasrec", "cf_weight": key[1],
                     "seed": job["seed"], "validation_HR10": score,
                     "job_id": job["job_id"], "test_status": marker["status"]})
    assert len(rows) == 18
    summary = []
    for (dataset, weight), values in sorted(grouped.items()):
        assert {seed for seed, _ in values} == set(SEEDS[:3])
        scores = [score for _, score in values]
        summary.append({"dataset": dataset, "backbone": "sasrec", "cf_weight": weight,
                        "validation_HR10_mean": mean(scores),
                        "validation_HR10_sd": stdev(scores), "n": 3})
    selected = {dataset: max((row for row in summary if row["dataset"] == dataset),
                             key=lambda row: (row["validation_HR10_mean"],
                                              f"cf_{row['cf_weight']:g}"))["cf_weight"]
                for dataset in ("ml-1m", "taobao")}
    manifest = json.loads((PLANS / "selection_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["validation_selection"]["coserec"]:
        assert selected[row["dataset"]] == row["selected_params"]["coserec_cf_weight"]
    return rows, summary, selected


def collect_final(selected):
    final_plan = read_plan(PLANS / "coserec_final.jsonl")
    reference = [job for job in read_plan(MAIN_PLAN)
                 if job["backbone"] == "sasrec" and job["method"] in ("baseline", "fair_ncl")]
    jobs = final_plan + reference
    grouped = defaultdict(dict)
    per_user = {}
    run_rows = []
    environments = []
    for job in jobs:
        if job["method"] == "coserec":
            assert job["phase"] == "revision_coserec_final"
            assert job["params"]["coserec_cf_weight"] == selected[job["dataset"]]
            assert json.loads(result_path(job, "job.json").read_text(encoding="utf-8")) == job
            provenance = json.loads(result_path(job, "run_provenance.json").read_text(encoding="utf-8"))
            environment = provenance["environment"]
            environments.append({"dataset": job["dataset"], "backbone": "sasrec",
                                 "seed": job["seed"], "job_id": job["job_id"],
                                 "python_version": environment["python_version"],
                                 "torch_version": environment["torch_version"],
                                 "numpy_version": environment["numpy_version"],
                                 "cuda_version": environment["cuda_version"],
                                 "cudnn_version": environment["cudnn_version"],
                                 "gpu_name": environment["gpu_name"],
                                 "git_commit": provenance["code"].get("git_commit")})
        result = json.loads(result_path(job, "test_results.json").read_text(encoding="utf-8"))
        data = dict(result["selected"])
        for metric in ("HitRate@20", "NDCG@20"):
            data[metric] = result["utility"][metric]
        for attribute in ("gender", "age_group"):
            fairness = result["fairness"][attribute]
            data[f"{attribute}_HitRate@20_Gap"] = fairness["hitrate@20_gap"]
            data[f"{attribute}_NDCG@20_Gap"] = fairness["ndcg@20_gap"]
            data[f"{attribute}_NDCG@20_WorstGroup"] = fairness["ndcg@20_worst_group"]
        key = (job["dataset"], job["method"])
        seed = int(job["seed"])
        assert seed not in grouped[key]
        grouped[key][seed] = data
        per_user[job["dataset"], job["method"], seed] = result["per_user"]
        run_rows.append({"dataset": job["dataset"], "backbone": "sasrec",
                         "method": job["method"], "seed": seed, "job_id": job["job_id"],
                         **{metric: data[metric] for metric in METRICS}})
    assert len(final_plan) == 10 and len(run_rows) == 30
    for seeds in grouped.values():
        assert set(seeds) == set(SEEDS)
    return grouped, per_user, run_rows, environments


def summarize(grouped, per_user):
    summary = []
    paired = []
    specifications = (("NDCG@10", None, "mean", None),
                      ("NDCG@10", "gender", "gap", None),
                      ("NDCG@10", "age_group", "gap", None))
    for dataset in ("ml-1m", "taobao"):
        fair = grouped[dataset, "fair_ncl"]
        for method in ("baseline", "fair_ncl", "coserec"):
            values = grouped[dataset, method]
            row = {"dataset": dataset, "backbone": "sasrec", "method": method,
                   "n": len(values)}
            for metric in METRICS:
                sequence = [float(values[seed][metric]) for seed in SEEDS]
                row[f"{metric}_mean"] = mean(sequence)
                row[f"{metric}_sd"] = stdev(sequence)
            summary.append(row)
        method_rows = [dict(seed=seed, **user) for seed in SEEDS
                       for user in per_user[dataset, "coserec", seed]]
        fair_rows = [dict(seed=seed, **user) for seed in SEEDS
                     for user in per_user[dataset, "fair_ncl", seed]]
        intervals = paired_hierarchical_bootstrap_many(
            method_rows, fair_rows, specifications, iterations=10000, random_seed=42)
        for metric, interval in zip(("NDCG@10", "gender_NDCG@10_Gap",
                                     "age_group_NDCG@10_Gap"), intervals):
            deltas = [float(grouped[dataset, "coserec"][seed][metric])
                      - float(fair[seed][metric]) for seed in SEEDS]
            paired.append({"dataset": dataset, "backbone": "sasrec",
                           "contrast": "coserec_minus_fair_ncl", "metric": metric,
                           "mean_delta": mean(deltas), "ci_low": interval["ci_lower"],
                           "ci_high": interval["ci_upper"],
                           "paired_p_exact": exact_paired_sign_flip_pvalue(deltas),
                           "n_seeds": len(SEEDS), "bootstrap_iterations": 10000})
    adjusted = holm_adjust({(row["dataset"], row["metric"]): row["paired_p_exact"]
                            for row in paired})
    for row in paired:
        row["holm_p"] = adjusted[row["dataset"], row["metric"]]
    return summary, paired


def paper_table(summary):
    lookup = {(row["dataset"], row["method"]): row for row in summary}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\caption{CoSeRec open-source augmentation and contrastive modules connected to the shared SASRec pipeline at @10. Values are five-seed mean $\pm$ sample standard deviation. G/A gap is the absolute gender/age-group NDCG gap; lower is better. CoSeRec uses a single-negative logistic recommendation loss. Paired intervals and tests are in Supplementary Data~S7.}",
             r"\label{tab:coserec_direct}",
             r"\begin{tabular}{llcccc}",
             r"\toprule", r"Dataset & Method & HR@10 & NDCG@10 & G gap & A gap \\",
             r"\midrule"]
    for dataset, title in (("ml-1m", "ML-1M"), ("taobao", "Taobao")):
        for method, name in (("baseline", "SASRec"), ("fair_ncl", "Fair-NCL"),
                             ("coserec", "CoSeRec")):
            row = lookup[dataset, method]
            cells = [f"{row[f'{metric}_mean']:.4f}$\\pm${row[f'{metric}_sd']:.4f}"
                     for metric in ("HitRate@10", "NDCG@10", "gender_NDCG@10_Gap",
                                    "age_group_NDCG@10_Gap")]
            label = title if method == "baseline" else ""
            lines.append(f"{label} & {name} & " + " & ".join(cells) + r" \\")
        if dataset == "ml-1m":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    content = "\n".join(lines) + "\n"
    (Path("els-cas-templates") / "coserec_direct_table.tex").write_text(content, encoding="utf-8")
    (Path("els-cas-templates/source_data/reproduced_paper_tables") /
     "coserec_direct_table.tex").write_text(content, encoding="utf-8")


def paper_main_rows(summary):
    lookup = {(row["dataset"], row["method"]): row for row in summary}
    for dataset, label in (("ml-1m", "ml1m"), ("taobao", "taobao")):
        row = lookup[dataset, "coserec"]
        for cutoff in (10, 20):
            metrics = (f"HitRate@{cutoff}", f"NDCG@{cutoff}",
                       f"gender_HitRate@{cutoff}_Gap", f"gender_NDCG@{cutoff}_Gap",
                       f"age_group_HitRate@{cutoff}_Gap", f"age_group_NDCG@{cutoff}_Gap")
            cells = [f"{row[f'{metric}_mean']:.4f}$\\pm${row[f'{metric}_sd']:.4f}"
                     for metric in metrics]
            content = " & CoSeRec & " + " & ".join(cells) + r" \\" + "\n"
            name = f"coserec_main_{label}_{cutoff}.tex"
            for directory in (Path("els-cas-templates"),
                              Path("els-cas-templates/source_data/reproduced_paper_tables")):
                (directory / name).write_text(content, encoding="utf-8")


def main():
    tuning, tuning_summary, selected = audit_tuning()
    grouped, per_user, runs, environments = collect_final(selected)
    summary, paired = summarize(grouped, per_user)
    for directory in (OUT, SOURCE):
        write_csv(directory / "Supplementary_Data_S7_tuning.csv", tuning)
        write_csv(directory / "Supplementary_Data_S7_tuning_summary.csv", tuning_summary)
        write_csv(directory / "Supplementary_Data_S7_five_seed_runs.csv", runs)
        write_csv(directory / "Supplementary_Data_S7_summary.csv", summary)
        write_csv(directory / "Supplementary_Data_S7_paired.csv", paired)
        write_csv(directory / "Supplementary_Data_S7_environment.csv", environments)
    paper_table(summary)
    paper_main_rows(summary)
    print("Audited 18 validation-only tuning and 10 CoSeRec final runs; wrote S7 and paper table.")


if __name__ == "__main__":
    main()

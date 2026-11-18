"""Prepare reviewer-facing inference and Pareto source data.

Inputs are the complete five-seed summaries under
``tables/supplementary/final_5seed_extended``. The script creates paired-comparison
and Pareto source data without recomputing or changing any model result.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "tables" / "supplementary" / "final_5seed_extended"
OUTPUT_DIR = ROOT / "tables" / "supplementary" / "reviewer_fairness"
SOURCE_DATA_DIR = ROOT / "els-cas-templates" / "source_data" / "table_source_data"
FIGURE_DATA_DIR = ROOT / "els-cas-templates" / "source_data" / "figure_source_data"
COSEREC_SUMMARY = ROOT / "tables" / "supplementary" / "reviewer_coserec" / "Supplementary_Data_S7_summary.csv"

MAIN_METHODS = (
    "baseline",
    "adv_debias",
    "grl",
    "sm_pcfr",
    "a_fsr",
    "afrl",
    "pfrec",
    "fair_ncl",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows prepared for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_paired_comparisons() -> list[dict[str, object]]:
    rows = read_csv(INPUT_DIR / "paired_comparisons.csv")
    selected = [
        row
        for row in rows
        if row["method"] == "fair_ncl"
        and row["metric"] in {"HitRate@10", "NDCG@10", "HitRate@20", "NDCG@20"}
        and (
            row["estimand"] == "mean"
            or (
                row["estimand"] == "gap"
                and row["group_attribute"] in {"gender", "age_group"}
            )
        )
    ]
    selected.sort(key=lambda row: (
        row["dataset"], row["backbone"], row["metric"],
        row["estimand"], row["group_attribute"]
    ))
    if len(selected) != 96:
        raise ValueError(f"Expected 96 primary paired comparisons, found {len(selected)}")
    return selected


def prepare_full_paired_comparisons() -> list[dict[str, object]]:
    rows = read_csv(INPUT_DIR / "paired_comparisons.csv")
    selected = [row for row in rows if row["method"] == "fair_ncl"]
    if len(selected) != 672 or len({row["comparison_id"] for row in selected}) != 672:
        raise ValueError("Expected 672 unique full Fair-NCL paired comparisons")
    return selected


def prepare_pareto() -> list[dict[str, object]]:
    runs = read_csv(INPUT_DIR / "run_summary.csv")
    groups = read_csv(INPUT_DIR / "group_summary.csv")
    coserec = [row for row in read_csv(COSEREC_SUMMARY) if row["method"] == "coserec"]
    gaps = {
        (row["dataset"], row["backbone"], row["method"], row["attribute"]):
            float(row["absolute_gap_mean"])
        for row in groups
        if row["metric"] == "NDCG@10"
        and row["attribute"] in {"gender", "age_group"}
    }

    output: list[dict[str, object]] = []
    settings = sorted({(row["dataset"], row["backbone"]) for row in runs})
    for dataset, backbone in settings:
        block: list[dict[str, object]] = []
        for row in runs:
            if (row["dataset"], row["backbone"]) != (dataset, backbone):
                continue
            if row["method"] not in MAIN_METHODS:
                continue
            prefix = (dataset, backbone, row["method"])
            gender_gap = gaps[prefix + ("gender",)]
            age_gap = gaps[prefix + ("age_group",)]
            block.append({
                "dataset": dataset,
                "backbone": backbone,
                "method": row["method"],
                "ndcg_10": float(row["NDCG@10_mean"]),
                "gender_ndcg_gap_10": gender_gap,
                "age_ndcg_gap_10": age_gap,
                "mean_ndcg_gap_10": (gender_gap + age_gap) / 2.0,
            })
        for row in coserec:
            if (row["dataset"], row["backbone"]) != (dataset, backbone):
                continue
            gender_gap = float(row["gender_NDCG@10_Gap_mean"])
            age_gap = float(row["age_group_NDCG@10_Gap_mean"])
            block.append({
                "dataset": dataset,
                "backbone": backbone,
                "method": "coserec",
                "ndcg_10": float(row["NDCG@10_mean"]),
                "gender_ndcg_gap_10": gender_gap,
                "age_ndcg_gap_10": age_gap,
                "mean_ndcg_gap_10": (gender_gap + age_gap) / 2.0,
            })
        expected = len(MAIN_METHODS) + (backbone == "sasrec")
        if len(block) != expected:
            raise ValueError(
                f"Expected {expected} main methods for "
                f"{dataset}/{backbone}, found {len(block)}"
            )
        for point in block:
            dominators = [
                other["method"]
                for other in block
                if other is not point
                and float(other["ndcg_10"]) >= float(point["ndcg_10"])
                and float(other["mean_ndcg_gap_10"]) <= float(point["mean_ndcg_gap_10"])
                and (
                    float(other["ndcg_10"]) > float(point["ndcg_10"])
                    or float(other["mean_ndcg_gap_10"]) < float(point["mean_ndcg_gap_10"])
                )
            ]
            point["is_pareto"] = not dominators
            point["dominated_by"] = "|".join(str(method) for method in dominators)
        output.extend(block)
    if len(output) != 66:
        raise ValueError(f"Expected 66 main-table Pareto rows, found {len(output)}")
    return output


def prepare_common_utility(pareto: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detailed = []
    for dataset, backbone in sorted({(row["dataset"], row["backbone"]) for row in pareto}):
        block = [row for row in pareto if (row["dataset"], row["backbone"]) == (dataset, backbone)]
        baseline = next(row for row in block if row["method"] == "baseline")
        baseline_utility = float(baseline["ndcg_10"])
        baseline_gap = float(baseline["mean_ndcg_gap_10"])
        feasible = [row for row in block if float(row["ndcg_10"]) / baseline_utility >= 0.85]
        for point in block:
            utility = float(point["ndcg_10"])
            gap = float(point["mean_ndcg_gap_10"])
            passed = point in feasible
            dominated = any(
                other is not point
                and float(other["ndcg_10"]) >= utility
                and float(other["mean_ndcg_gap_10"]) <= gap
                and (float(other["ndcg_10"]) > utility or float(other["mean_ndcg_gap_10"]) < gap)
                for other in feasible
            )
            detailed.append({
                "dataset": dataset, "backbone": backbone, "method": point["method"],
                "ndcg_10": utility, "mean_ndcg_gap_10": gap,
                "backbone_ndcg_10": baseline_utility, "backbone_mean_ndcg_gap_10": baseline_gap,
                "test_utility_retention": utility / baseline_utility,
                "test_feasible_15pct": passed,
                "lower_gap_than_backbone": gap < baseline_gap,
                "is_test_pareto_15pct": passed and not dominated,
            })
    if len(detailed) != 66:
        raise ValueError(f"Expected 66 constrained-test rows, found {len(detailed)}")
    summary = []
    for method in MAIN_METHODS + ("coserec",):
        rows = [row for row in detailed if row["method"] == method]
        summary.append({
            "method": method,
            "evaluated_blocks": len(rows),
            "median_test_utility_retention": median(float(row["test_utility_retention"]) for row in rows),
            "test_feasible_blocks": sum(bool(row["test_feasible_15pct"]) for row in rows),
            "feasible_lower_gap_blocks": sum(
                bool(row["test_feasible_15pct"]) and bool(row["lower_gap_than_backbone"])
                for row in rows
            ),
            "constrained_test_pareto_blocks": sum(bool(row["is_test_pareto_15pct"]) for row in rows),
        })
    return detailed, summary


def main() -> None:
    paired = prepare_paired_comparisons()
    full_paired = prepare_full_paired_comparisons()
    pareto = prepare_pareto()
    constrained, constrained_summary = prepare_common_utility(pareto)

    paired_path = OUTPUT_DIR / "paired_fair_ncl_vs_backbone.csv"
    pareto_path = OUTPUT_DIR / "pareto_main_methods_ndcg10.csv"
    write_csv(paired, paired_path)
    write_csv(full_paired, OUTPUT_DIR / "full_paired_fair_ncl_vs_backbone.csv")
    write_csv(pareto, pareto_path)

    write_csv(paired, SOURCE_DATA_DIR / "Supplementary_Data_S1_paired_comparisons.csv")
    write_csv(full_paired, SOURCE_DATA_DIR / "Supplementary_Data_S1_full_paired_comparisons.csv")
    write_csv(pareto, SOURCE_DATA_DIR / "Supplementary_Data_S2_pareto.csv")
    write_csv(pareto, FIGURE_DATA_DIR / "fairness_pareto_main.csv")
    write_csv(constrained, OUTPUT_DIR / "common_utility_15pct.csv")
    write_csv(constrained_summary, OUTPUT_DIR / "common_utility_15pct_summary.csv")
    write_csv(constrained, SOURCE_DATA_DIR / "Supplementary_Data_S2_common_utility_15pct.csv")
    write_csv(constrained_summary, SOURCE_DATA_DIR / "Supplementary_Data_S2_common_utility_15pct_summary.csv")

    fair_rows = [row for row in pareto if row["method"] == "fair_ncl"]
    fair_fronts = sum(bool(row["is_pareto"]) for row in fair_rows)
    gap_ci_exclusions = sum(
        row["estimand"] == "gap" and row["ci_excludes_zero"] == "True"
        for row in paired
    )
    holm_significant = sum(row["holm_significant"] == "True" for row in paired)
    print(f"Prepared {len(paired)} paired comparisons")
    print(f"Prepared {len(full_paired)} full paired comparisons, including all group estimands")
    print(f"Prepared {len(pareto)} Pareto rows; Fair-NCL on {fair_fronts}/8 fronts")
    print(f"Gap intervals excluding zero: {gap_ci_exclusions}; Holm significant: {holm_significant}")


if __name__ == "__main__":
    main()

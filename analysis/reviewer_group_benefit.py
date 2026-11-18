"""Extract test changes for validation-defined disadvantaged groups."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("tables/supplementary")
OUTPUT = ROOT / "final_5seed" / "validation_fixed_group_benefit.csv"
METRICS = ("NDCG@10", "NDCG@20")
ATTRIBUTES = ("gender", "age_group")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    selections = [
        row for row in read_csv(ROOT / "validation_selection.csv")
        if row["method"] == "fair_ncl"
    ]
    comparisons = {
        (
            row["dataset"], row["backbone"], row["metric"],
            row["group_attribute"], row["group_value"],
        ): row
        for row in read_csv(ROOT / "final_5seed" / "paired_comparisons.csv")
        if row["method"] == "fair_ncl" and row["estimand"] == "group_mean"
    }
    rows = []
    for selection in selections:
        fixed_groups = json.loads(selection["baseline_disadvantaged_groups"])
        for attribute in ATTRIBUTES:
            disadvantaged = str(fixed_groups[attribute])
            other = str(1 - int(disadvantaged))
            for metric in METRICS:
                key = (selection["dataset"], selection["backbone"], metric, attribute)
                low = comparisons[(*key, disadvantaged)]
                high = comparisons[(*key, other)]
                rows.append({
                    "dataset": selection["dataset"],
                    "backbone": selection["backbone"],
                    "attribute": attribute,
                    "metric": metric,
                    "disadvantaged_group": disadvantaged,
                    "other_group": other,
                    "baseline_disadvantaged": low["baseline_estimate"],
                    "fair_ncl_disadvantaged": low["method_estimate"],
                    "disadvantaged_delta": low["mean_delta"],
                    "disadvantaged_ci_lower": low["ci_lower"],
                    "disadvantaged_ci_upper": low["ci_upper"],
                    "disadvantaged_holm_p": low["holm_p_value"],
                    "other_delta": high["mean_delta"],
                    "other_ci_lower": high["ci_lower"],
                    "other_ci_upper": high["ci_upper"],
                })
    rows.sort(key=lambda row: (
        row["dataset"], row["backbone"], row["attribute"], row["metric"]
    ))
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

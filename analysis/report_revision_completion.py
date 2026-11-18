"""Summarize completed revision plans with paired five-seed contrasts."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

from experiments.revision_completion import _fair_jobs

METRICS = (
    "HitRate@10", "NDCG@10",
    "gender_NDCG@10_Gap", "age_group_NDCG@10_Gap",
    "gender_NDCG@10_WorstGroup", "age_group_NDCG@10_WorstGroup",
    "gender_age_NDCG@10_Gap", "gender_age_NDCG@10_WorstGroup",
)
DIRECT_METHODS = ("baseline", "ncl_sr_mce", "a_fsr", "fair_ncl", "coserec")


def _expected_seed_count(stage):
    return 3 if stage == "sensitivity" else 5


def _plan_backbones(plan_paths):
    backbones = set()
    for plan_path in plan_paths:
        for line in Path(plan_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                backbones.add(json.loads(line)["backbone"])
    return tuple(sorted(backbones))


def _read_result(results_root, job):
    path = Path(results_root) / job["dataset"] / job["job_id"] / "test_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Run all plan jobs first: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if "selected" not in result:
        raise ValueError(f"Test evaluation is missing: {path}")
    return result["selected"]


def _paired_pvalue(deltas):
    observed = abs(mean(deltas))
    null = [
        abs(mean(value * sign for value, sign in zip(deltas, signs)))
        for signs in itertools.product((-1, 1), repeat=len(deltas))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def _row(key, metric, values, deltas):
    return {
        "dataset": key[0], "backbone": key[1], "stage": key[2],
        "condition": key[3], "metric": metric,
        "seeds": "|".join(str(seed) for seed in sorted(values)),
        "n": len(values),
        "mean": mean(values.values()),
        "std": stdev(values.values()) if len(values) > 1 else 0.0,
        "paired_delta_vs_fair": mean(deltas.values()),
        "paired_p_exact": _paired_pvalue(list(deltas.values())),
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _holm_adjust(rows, p_key="paired_p_exact", output_key="holm_p"):
    """Adjust each dataset/backbone/stage/metric comparison family."""
    families = {}
    for index, row in enumerate(rows):
        family = (row["dataset"], row["backbone"], row.get("stage", "factorial_effects"), row["metric"])
        families.setdefault(family, []).append((index, float(row[p_key])))
    for members in families.values():
        ordered = sorted(members, key=lambda item: item[1])
        running = 0.0
        adjusted = {}
        count = len(ordered)
        for rank, (index, value) in enumerate(ordered):
            running = max(running, min(1.0, (count - rank) * value))
            adjusted[index] = running
        for index, value in adjusted.items():
            rows[index][output_key] = value


def _direct_method_rows(results_root, plan_paths, fair_values):
    """Create one paper-table source across existing and new direct methods."""
    jobs = []
    for path in Path(results_root).glob("*/*/job.json"):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("phase") == "supplementary" and job.get("method") in DIRECT_METHODS[:-1]:
            jobs.append(job)
    for plan_path in plan_paths:
        for line in Path(plan_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                job = json.loads(line)
                if job.get("method") == "coserec":
                    jobs.append(job)

    grouped = {}
    for job in jobs:
        seed = int(job["seed"])
        if seed not in (42, 2025, 2026, 2027, 2028):
            continue
        try:
            selected = _read_result(results_root, job)
        except FileNotFoundError:
            continue
        key = (job["dataset"], job["backbone"], job["method"])
        grouped.setdefault(key, {})[seed] = selected

    rows = []
    for (dataset, backbone, method), seeds in sorted(grouped.items()):
        missing = sorted({42, 2025, 2026, 2027, 2028} - set(seeds))
        for metric in METRICS:
            values = {seed: float(result[metric]) for seed, result in seeds.items() if metric in result}
            shared = {
                seed: value - float(fair_values[(dataset, backbone, seed)][metric])
                for seed, value in values.items()
                if (dataset, backbone, seed) in fair_values
            }
            row = {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "metric": metric,
                "seed_count": len(values),
                "missing_seeds": "|".join(map(str, missing)),
                "mean": mean(values.values()),
                "std": stdev(values.values()) if len(values) > 1 else 0.0,
                "paired_delta_vs_fair": mean(shared.values()) if shared else None,
                "paired_p_exact": _paired_pvalue(list(shared.values())) if len(shared) == 5 else None,
            }
            rows.append(row)
    comparable = [row for row in rows if row["paired_p_exact"] is not None]
    _holm_adjust(comparable)
    adjusted = {
        (row["dataset"], row["backbone"], row["method"], row["metric"]): row["holm_p"]
        for row in comparable
    }
    for row in rows:
        row["holm_p"] = adjusted.get((row["dataset"], row["backbone"], row["method"], row["metric"]))
    return rows


def summarize(results_root, plan_paths, output_dir):
    fair = _fair_jobs(results_root, required_backbones=_plan_backbones(plan_paths))
    fair_values = {
        key: _read_result(results_root, job)
        for key, job in fair.items()
    }
    direct_rows = _direct_method_rows(results_root, plan_paths, fair_values)
    _write_csv(Path(output_dir) / "direct_method_comparison.csv", direct_rows)
    grouped = {}
    factorial = {}
    for plan_path in plan_paths:
        stage = Path(plan_path).stem
        for line in Path(plan_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            job = json.loads(line)
            key = (job["dataset"], job["backbone"], int(job["seed"]))
            result = _read_result(results_root, job)
            params = job["params"]
            condition = (
                params.get("factorial_cell")
                or (f"{params['sensitivity_axis']}={params[params['sensitivity_axis']]:g}"
                    if "sensitivity_axis" in params else
                    f"cf_weight={params['coserec_cf_weight']:g}"
                    if "coserec_cf_weight" in params else job["method"])
            )
            for metric in METRICS:
                if metric not in result or metric not in fair_values[key]:
                    raise KeyError(f"Missing metric {metric} for {job['job_id']}")
                bucket = grouped.setdefault((key[0], key[1], stage, condition, metric), {})
                bucket[key[2]] = (float(result[metric]), float(result[metric]) - float(fair_values[key][metric]))
                if stage == "factorial":
                    factorial[(key[0], key[1], key[2], condition, metric)] = float(result[metric])

    rows = []
    for group_key, seed_values in sorted(grouped.items()):
        values = {seed: pair[0] for seed, pair in seed_values.items()}
        deltas = {seed: pair[1] for seed, pair in seed_values.items()}
        expected = _expected_seed_count(group_key[2])
        if len(values) != expected:
            raise ValueError(
                f"Expected {expected} seeds for {group_key}, got {sorted(values)}"
            )
        rows.append(_row(group_key, group_key[-1], values, deltas))
    _holm_adjust(rows)
    _write_csv(Path(output_dir) / "paired_summary.csv", rows)

    effects = []
    pairs = sorted({(dataset, backbone) for dataset, backbone, _, _, _ in factorial})
    for dataset, backbone in pairs:
        for metric in METRICS:
            cells_by_seed = {}
            for seed in (42, 2025, 2026, 2027, 2028):
                cells = {
                    cell: factorial[(dataset, backbone, seed, cell, metric)]
                    for cell in (f"{bits:04b}"[::-1] for bits in range(15))
                }
                cells["1111"] = float(fair_values[(dataset, backbone, seed)][metric])
                cells_by_seed[seed] = cells
            for factor in range(4):
                deltas = []
                for cells in cells_by_seed.values():
                    on = mean(value for cell, value in cells.items() if cell[factor] == "1")
                    off = mean(value for cell, value in cells.items() if cell[factor] == "0")
                    deltas.append(on - off)
                effects.append({
                    "dataset": dataset, "backbone": backbone, "metric": metric,
                    "effect": f"main_{('similarity', 'skew', 'aug_ce', 'geometry')[factor]}",
                    "mean": mean(deltas), "std": stdev(deltas),
                    "paired_p_exact": _paired_pvalue(deltas),
                })
            for left, right in itertools.combinations(range(4), 2):
                deltas = []
                for cells in cells_by_seed.values():
                    averages = {
                        (a, b): mean(value for cell, value in cells.items()
                                     if cell[left] == a and cell[right] == b)
                        for a, b in itertools.product("01", repeat=2)
                    }
                    deltas.append(averages["1", "1"] - averages["1", "0"]
                                  - averages["0", "1"] + averages["0", "0"])
                effects.append({
                    "dataset": dataset, "backbone": backbone, "metric": metric,
                    "effect": f"interaction_{left}_{right}",
                    "mean": mean(deltas), "std": stdev(deltas),
                    "paired_p_exact": _paired_pvalue(deltas),
                })
    _holm_adjust(effects)
    _write_csv(Path(output_dir) / "factorial_effects.csv", effects)
    print(f"Wrote {len(rows)} paired summaries and {len(effects)} factorial effects.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results_revision_kbs_v2")
    parser.add_argument("--plans", nargs="+", required=True)
    parser.add_argument("--output-dir", default="tables/supplementary/revision_completion")
    args = parser.parse_args()
    summarize(args.results_root, args.plans, args.output_dir)


if __name__ == "__main__":
    main()

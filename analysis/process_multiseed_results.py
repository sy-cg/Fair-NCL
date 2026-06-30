from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.reporting import (
    METHOD_LABELS,
    aggregate_numeric_table,
    build_paper_tables,
    filter_records,
    load_job_records,
    _attach_display_columns,
    _discover_metric_cols,
    _metric_display_label,
    _ordered_raw_columns,
    _ordered_summary_columns,
    _sheet_name,
    _sort_summary_rows,
)


DEFAULT_PRIMARY_METRICS = (
    "HitRate@10",
    "NDCG@10",
    "gender_HitRate@10_Gap",
    "gender_NDCG@10_Gap",
    "age_group_HitRate@10_Gap",
    "age_group_NDCG@10_Gap",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multi-seed Fair-NCL job results and export paper-ready tables."
    )
    parser.add_argument("--jobs", nargs="+", required=True, help="One or more JSONL job files.")
    parser.add_argument("--phase", choices=["ablation", "comparison"], required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--backbones", nargs="+", default=["all"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_PRIMARY_METRICS),
        help="Metric names without selected__ prefix. Use 'all' to keep every metric.",
    )
    parser.add_argument("--baseline-method", default="baseline")
    parser.add_argument("--ours-method", default="fair_ncl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-dir", default=None, help="Optional directory for CSV copies.")
    args = parser.parse_args()

    args.jobs = expand_job_paths(args.jobs)
    records, duplicates = load_combined_records(args.jobs, args.results_root)
    records = filter_records(records, phase=args.phase, datasets=args.datasets, backbones=args.backbones)
    records = _filter_methods(records, args.methods)
    records = _filter_seeds(records, args.seeds)
    if records.empty:
        raise ValueError("No matching jobs were found after filtering.")

    metric_cols = _discover_metric_cols(records)
    if not metric_cols:
        raise ValueError("No completed result metrics were found. Check test_results.json files.")

    primary_metric_cols = _resolve_metric_cols(metric_cols, args.metrics)
    completed = records[records["status"] == "ok"].copy()
    if completed.empty:
        raise ValueError("No completed jobs were found after filtering.")

    summary_all = aggregate_numeric_table(
        completed,
        ["dataset", "backbone", "method"],
        metric_cols,
    )
    summary_all = _sort_summary_rows(_attach_display_columns(summary_all), args.phase)

    summary_primary = aggregate_numeric_table(
        completed,
        ["dataset", "backbone", "method"],
        primary_metric_cols,
    )
    summary_primary = _sort_summary_rows(_attach_display_columns(summary_primary), args.phase)

    seed_status = build_seed_status(records, expected_seeds=args.seeds)
    per_seed_primary = build_per_seed_table(records, primary_metric_cols)
    delta_baseline = build_delta_table(summary_primary, primary_metric_cols, args.baseline_method)
    delta_ours = build_delta_table(summary_primary, primary_metric_cols, args.ours_method)
    ranking_primary = build_ranking_table(summary_primary, primary_metric_cols)
    paper_tables = build_paper_tables(summary_primary, args.phase, primary_metric_cols)
    readme = build_readme(args, records, metric_cols, primary_metric_cols)

    tables: Dict[str, pd.DataFrame] = {
        "readme": readme,
        "raw_results": records,
        "duplicate_jobs": duplicates,
        "missing_jobs": records[records["status"] != "ok"].copy(),
        "seed_status": seed_status,
        "summary_all_metrics": summary_all,
        "summary_primary": summary_primary,
        "per_seed_primary": per_seed_primary,
        f"delta_vs_{args.baseline_method}": delta_baseline,
        f"delta_vs_{args.ours_method}": delta_ours,
        "ranking_primary": ranking_primary,
    }
    for name, table in paper_tables.items():
        tables[f"paper_{name}"] = table

    export_tables(tables, args.output, metric_cols, primary_metric_cols)
    if args.csv_dir:
        export_csv_tables(tables, args.csv_dir)

    print(f"Saved multi-seed result workbook to {args.output}")
    if args.csv_dir:
        print(f"Saved CSV copies to {args.csv_dir}")


def load_combined_records(job_paths: Sequence[str], results_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in job_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Job file not found: {path}")
        frame = load_job_records(path, results_root)
        frame["plan_path"] = path
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if combined.empty or "job_id" not in combined.columns:
        return combined, pd.DataFrame()

    duplicate_mask = combined.duplicated("job_id", keep=False)
    duplicates = combined[duplicate_mask].copy()
    combined = combined.drop_duplicates("job_id", keep="first").reset_index(drop=True)
    return combined, duplicates.reset_index(drop=True)


def expand_job_paths(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)

    deduplicated: List[str] = []
    seen = set()
    for path in paths:
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        deduplicated.append(path)
        seen.add(normalized)
    return deduplicated


def _filter_methods(df: pd.DataFrame, methods: Optional[Sequence[str]]) -> pd.DataFrame:
    if not methods:
        return df.reset_index(drop=True)
    selected = {str(method).lower() for method in methods}
    return df[df["method"].astype(str).str.lower().isin(selected)].reset_index(drop=True)


def _filter_seeds(df: pd.DataFrame, seeds: Optional[Sequence[int]]) -> pd.DataFrame:
    if not seeds:
        return df.reset_index(drop=True)
    selected = {int(seed) for seed in seeds}
    return df[df["seed"].astype(int).isin(selected)].reset_index(drop=True)


def _resolve_metric_cols(metric_cols: Sequence[str], requested_metrics: Sequence[str]) -> List[str]:
    if len(requested_metrics) == 1 and str(requested_metrics[0]).lower() == "all":
        return list(metric_cols)

    available = set(metric_cols)
    resolved = []
    missing = []
    for metric in requested_metrics:
        metric_name = str(metric)
        col = metric_name if metric_name.startswith("selected__") else f"selected__{metric_name}"
        if col in available:
            resolved.append(col)
        else:
            missing.append(metric_name)

    if missing:
        print(f"Warning: missing metrics skipped: {', '.join(missing)}")
    if not resolved:
        raise ValueError("None of the requested metrics were available.")
    return resolved


def build_seed_status(df: pd.DataFrame, expected_seeds: Optional[Sequence[int]] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if expected_seeds:
        expected = sorted({int(seed) for seed in expected_seeds})
    else:
        expected = sorted({int(seed) for seed in df["seed"].dropna().tolist()})

    rows = []
    group_cols = ["phase", "dataset", "backbone", "method"]
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        completed = group[group["status"] == "ok"]
        planned_seeds = sorted({int(seed) for seed in group["seed"].dropna().tolist()})
        completed_seeds = sorted({int(seed) for seed in completed["seed"].dropna().tolist()})
        missing_seeds = [seed for seed in expected if seed not in completed_seeds]
        status_counts = group["status"].value_counts(dropna=False).to_dict()
        row = dict(zip(group_cols, keys))
        row.update({
            "expected_seeds": ",".join(str(seed) for seed in expected),
            "planned_seeds": ",".join(str(seed) for seed in planned_seeds),
            "completed_seeds": ",".join(str(seed) for seed in completed_seeds),
            "missing_seeds": ",".join(str(seed) for seed in missing_seeds),
            "expected_n": len(expected),
            "completed_n": len(completed_seeds),
            "missing_n": len(missing_seeds),
            "is_complete": len(missing_seeds) == 0,
            "ok_jobs": int(status_counts.get("ok", 0)),
            "missing_jobs": int(status_counts.get("missing", 0)),
            "error_jobs": int(status_counts.get("error", 0)),
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _attach_display_columns(result)
    return result


def build_per_seed_table(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    base = [
        "status",
        "phase",
        "dataset",
        "dataset_display",
        "backbone",
        "backbone_display",
        "method",
        "method_display",
        "seed",
        "job_id",
        "plan_path",
        "result_path",
    ]
    cols = [col for col in base + list(metric_cols) if col in df.columns]
    result = df[cols].copy()
    return result.sort_values(
        [col for col in ["dataset", "backbone", "method", "seed"] if col in result.columns]
    ).reset_index(drop=True)


def build_delta_table(summary: pd.DataFrame,
                      metric_cols: Sequence[str],
                      reference_method: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["dataset", "backbone"]
    for (dataset, backbone), group in summary.groupby(group_cols, dropna=False, sort=False):
        ref_rows = group[group["method"] == reference_method]
        if ref_rows.empty:
            continue
        ref = ref_rows.iloc[0]
        for _, row in group.iterrows():
            for metric_col in metric_cols:
                metric_name = metric_col[len("selected__"):] if metric_col.startswith("selected__") else metric_col
                mean_col = f"{metric_col}__mean"
                std_col = f"{metric_col}__std"
                count_col = f"{metric_col}__count"
                value = row.get(mean_col, np.nan)
                ref_value = ref.get(mean_col, np.nan)
                if pd.isna(value) or pd.isna(ref_value):
                    continue
                delta = float(value) - float(ref_value)
                pct = np.nan if abs(float(ref_value)) < 1e-12 else delta / abs(float(ref_value))
                direction = metric_direction(metric_name)
                preferred_delta = -delta if direction == "lower" else delta
                rows.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": row.get("method"),
                    "method_display": row.get("method_display", row.get("method")),
                    "reference_method": reference_method,
                    "reference_display": METHOD_LABELS.get(reference_method, reference_method),
                    "metric_name": metric_name,
                    "metric_display": _metric_display_label(metric_name),
                    "direction": direction,
                    "mean": value,
                    "std": row.get(std_col, np.nan),
                    "count": row.get(count_col, np.nan),
                    "reference_mean": ref_value,
                    "delta": delta,
                    "pct_delta": pct,
                    "preferred_delta": preferred_delta,
                    "is_reference": row.get("method") == reference_method,
                })

    return pd.DataFrame(rows)


def build_ranking_table(summary: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for (dataset, backbone), group in summary.groupby(["dataset", "backbone"], dropna=False, sort=False):
        for metric_col in metric_cols:
            metric_name = metric_col[len("selected__"):] if metric_col.startswith("selected__") else metric_col
            mean_col = f"{metric_col}__mean"
            work = group[["method", "method_display", mean_col]].copy()
            work = work.dropna(subset=[mean_col])
            if work.empty:
                continue
            ascending = metric_direction(metric_name) == "lower"
            work = work.sort_values(mean_col, ascending=ascending).reset_index(drop=True)
            for idx, row in work.iterrows():
                rows.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "metric_name": metric_name,
                    "metric_display": _metric_display_label(metric_name),
                    "direction": "lower" if ascending else "higher",
                    "rank": idx + 1,
                    "method": row["method"],
                    "method_display": row["method_display"],
                    "mean": row[mean_col],
                })
    return pd.DataFrame(rows)


def metric_direction(metric_name: str) -> str:
    lowered = metric_name.lower()
    if "gap" in lowered or lowered.endswith("_std") or lowered.endswith(" std"):
        return "lower"
    return "higher"


def build_readme(args, records: pd.DataFrame,
                 metric_cols: Sequence[str],
                 primary_metric_cols: Sequence[str]) -> pd.DataFrame:
    payload = {
        "jobs": list(args.jobs),
        "phase": args.phase,
        "results_root": args.results_root,
        "datasets": args.datasets,
        "backbones": args.backbones,
        "methods": args.methods,
        "seeds": args.seeds,
        "total_planned_rows": int(len(records)),
        "completed_rows": int((records["status"] == "ok").sum()) if "status" in records else 0,
        "available_metric_count": len(metric_cols),
        "primary_metrics": [
            col[len("selected__"):] if col.startswith("selected__") else col
            for col in primary_metric_cols
        ],
        "notes": [
            "summary_* sheets aggregate over seeds by dataset/backbone/method.",
            "For HitRate/NDCG/Recall/Precision/MRR, higher is better.",
            "For Gap and Std fairness metrics, lower is better.",
            "delta_vs_* uses preferred_delta so larger is always better.",
        ],
    }
    return pd.DataFrame([
        {"field": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value}
        for key, value in payload.items()
    ])


def export_tables(tables: Dict[str, pd.DataFrame],
                  output_path: str,
                  metric_cols: Sequence[str],
                  primary_metric_cols: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for raw_name, table in tables.items():
            if table is None or table.empty:
                continue
            sheet_name = _sheet_name(raw_name)
            export_df = table.copy()
            if raw_name == "raw_results":
                cols = _ordered_raw_columns(export_df, metric_cols)
                export_df = export_df[cols]
            elif raw_name in {"summary_all_metrics", "summary_primary"}:
                cols = _ordered_summary_columns(
                    export_df,
                    metric_cols if raw_name == "summary_all_metrics" else primary_metric_cols,
                )
                export_df = export_df[cols]
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)


def export_csv_tables(tables: Dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for raw_name, table in tables.items():
        if table is None or table.empty:
            continue
        path = os.path.join(output_dir, f"{_sheet_name(raw_name)}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

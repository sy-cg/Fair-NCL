from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .reporting import (
    BACKBONE_LABELS,
    DATASET_LABELS,
    METHOD_LABELS,
    PRIMARY_REPORT_METRICS,
    _attach_display_columns,
    _format_mean_std,
    _infer_phase,
    _is_all_selection,
    _method_order,
    _ordered_raw_columns,
    _safe_count,
    _safe_std,
    _sheet_name,
    _sort_summary_rows,
)


MECHANISM_METRICS = (
    "mechanism__representation__transform__cosine_mean",
    "mechanism__representation__transform__delta_l2_mean",
    "mechanism__gender__probe_raw__balanced_accuracy",
    "mechanism__gender__probe_fair__balanced_accuracy",
    "mechanism__gender__probe_balanced_accuracy_drop",
    "mechanism__gender__separability_raw__between_within_ratio",
    "mechanism__gender__separability_fair__between_within_ratio",
    "mechanism__gender__separability_between_within_ratio_drop",
    "mechanism__age_group__probe_raw__balanced_accuracy",
    "mechanism__age_group__probe_fair__balanced_accuracy",
    "mechanism__age_group__probe_balanced_accuracy_drop",
    "mechanism__age_group__separability_raw__between_within_ratio",
    "mechanism__age_group__separability_fair__between_within_ratio",
    "mechanism__age_group__separability_between_within_ratio_drop",
)

MECHANISM_METRIC_LABELS = {
    "mechanism__representation__transform__cosine_mean": "Raw->Fair Cosine",
    "mechanism__representation__transform__delta_l2_mean": "Raw->Fair L2",
    "mechanism__gender__probe_raw__balanced_accuracy": "Gender Probe BA Raw",
    "mechanism__gender__probe_fair__balanced_accuracy": "Gender Probe BA Fair",
    "mechanism__gender__probe_balanced_accuracy_drop": "Gender Probe BA Drop",
    "mechanism__gender__separability_raw__between_within_ratio": "Gender Sep Raw",
    "mechanism__gender__separability_fair__between_within_ratio": "Gender Sep Fair",
    "mechanism__gender__separability_between_within_ratio_drop": "Gender Sep Drop",
    "mechanism__age_group__probe_raw__balanced_accuracy": "Age Probe BA Raw",
    "mechanism__age_group__probe_fair__balanced_accuracy": "Age Probe BA Fair",
    "mechanism__age_group__probe_balanced_accuracy_drop": "Age Probe BA Drop",
    "mechanism__age_group__separability_raw__between_within_ratio": "Age Sep Raw",
    "mechanism__age_group__separability_fair__between_within_ratio": "Age Sep Fair",
    "mechanism__age_group__separability_between_within_ratio_drop": "Age Sep Drop",
}

RQ4_PAPER_METRICS = (
    "mechanism__representation__transform__cosine_mean",
    "mechanism__gender__probe_raw__balanced_accuracy",
    "mechanism__gender__probe_fair__balanced_accuracy",
    "mechanism__gender__probe_balanced_accuracy_drop",
    "mechanism__age_group__probe_raw__balanced_accuracy",
    "mechanism__age_group__probe_fair__balanced_accuracy",
    "mechanism__age_group__probe_balanced_accuracy_drop",
    "mechanism__gender__separability_raw__between_within_ratio",
    "mechanism__gender__separability_fair__between_within_ratio",
    "mechanism__age_group__separability_raw__between_within_ratio",
    "mechanism__age_group__separability_fair__between_within_ratio",
)


def export_mechanism_report(jobs_path: str,
                            results_root: str,
                            output_path: str,
                            phase: Optional[str] = None,
                            datasets: Optional[Sequence[str]] = None,
                            backbones: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    df = load_mechanism_records(jobs_path, results_root)
    df = filter_mechanism_records(df, phase=phase, datasets=datasets, backbones=backbones)
    if df.empty:
        raise ValueError("No matching jobs were found for the mechanism report.")

    metric_cols = [metric for metric in MECHANISM_METRICS if metric in df.columns]
    if not metric_cols:
        raise ValueError("No mechanism-analysis metrics were found in the experiment results.")

    completed = df[df["mechanism_status"] == "ok"].copy()
    if completed.empty:
        raise ValueError("No completed mechanism-analysis results were found.")

    report_phase = _infer_phase(completed, phase)
    summary = aggregate_mechanism_table(completed, ["dataset", "backbone", "method"], metric_cols)
    summary = _attach_display_columns(summary)
    summary = _sort_summary_rows(summary, report_phase)
    paper = build_mechanism_paper_table(summary, report_phase)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_results", index=False, columns=_mechanism_raw_columns(df, metric_cols))
        summary.to_excel(writer, sheet_name="summary_long", index=False, columns=_mechanism_summary_columns(summary, metric_cols))
        if not paper.empty:
            paper.to_excel(writer, sheet_name="rq4_paper_table", index=False)

        failed = df[df["mechanism_status"] != "ok"].copy()
        if not failed.empty:
            failed.to_excel(writer, sheet_name="missing_or_failed", index=False, columns=_mechanism_raw_columns(df, metric_cols))

    return {
        "raw_results": df,
        "summary_long": summary,
        "paper_table": paper,
    }


def load_mechanism_records(jobs_path: str, results_root: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(jobs_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            records.append(_build_mechanism_record(job, results_root))
    return pd.DataFrame.from_records(records)


def filter_mechanism_records(df: pd.DataFrame,
                             phase: Optional[str] = None,
                             datasets: Optional[Sequence[str]] = None,
                             backbones: Optional[Sequence[str]] = None) -> pd.DataFrame:
    result = df.copy()
    if phase and phase.lower() != "all" and "phase" in result.columns:
        result = result[result["phase"] == phase]
    if not _is_all_selection(datasets) and "dataset" in result.columns:
        result = result[result["dataset"].isin(list(datasets))]
    if not _is_all_selection(backbones) and "backbone" in result.columns:
        result = result[result["backbone"].isin(list(backbones))]
    return result.reset_index(drop=True)


def aggregate_mechanism_table(df: pd.DataFrame,
                              group_cols: Sequence[str],
                              metric_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(group_cols))

    agg_spec = {}
    for metric_col in metric_cols:
        agg_spec[f"{metric_col}__mean"] = (metric_col, "mean")
        agg_spec[f"{metric_col}__std"] = (metric_col, _safe_std)
        agg_spec[f"{metric_col}__count"] = (metric_col, _safe_count)

    return df.groupby(list(group_cols), dropna=False, sort=False).agg(**agg_spec).reset_index()


def build_mechanism_paper_table(summary_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    method_order = _method_order(phase, summary_df["method"].dropna().tolist())
    method_rank = {method: idx for idx, method in enumerate(method_order)}

    rows: List[Dict] = []
    work_df = summary_df.copy()
    work_df["_method_rank"] = work_df["method"].map(method_rank).fillna(len(method_rank)).astype(int)
    work_df = work_df.sort_values(["dataset", "backbone", "_method_rank", "method"]).reset_index(drop=True)

    for _, row in work_df.iterrows():
        paper_row = {
            "Dataset": DATASET_LABELS.get(row["dataset"], row["dataset"]),
            "Backbone": BACKBONE_LABELS.get(row["backbone"], row["backbone"]),
            "Method": METHOD_LABELS.get(row["method"], row["method"]),
        }
        for metric in RQ4_PAPER_METRICS:
            mean_col = f"{metric}__mean"
            std_col = f"{metric}__std"
            if mean_col not in row.index:
                continue
            paper_row[MECHANISM_METRIC_LABELS.get(metric, metric)] = _format_mean_std(
                row.get(mean_col, np.nan),
                row.get(std_col, np.nan),
            )
        rows.append(paper_row)

    return pd.DataFrame(rows)


def _build_mechanism_record(job: Dict, results_root: str) -> Dict:
    dataset = job["dataset"]
    job_id = job["job_id"]
    output_dir = os.path.join(results_root, dataset, job_id)
    test_results_path = os.path.join(output_dir, "test_results.json")
    mechanism_path = os.path.join(output_dir, "mechanism_test_summary.json")

    record = {
        "job_id": job_id,
        "phase": job.get("phase", ""),
        "dataset": dataset,
        "dataset_display": DATASET_LABELS.get(dataset, dataset),
        "method": job["method"],
        "method_display": METHOD_LABELS.get(job["method"], job["method"]),
        "backbone": job["backbone"],
        "backbone_display": BACKBONE_LABELS.get(job["backbone"], job["backbone"]),
        "seed": int(job.get("seed", 42)),
        "output_dir": output_dir,
        "reuse_from_phase": job.get("reuse_from_phase"),
        "reuse_from_job_id": job.get("reuse_from_job_id"),
        "params_json": json.dumps(job.get("params", {}), ensure_ascii=False, sort_keys=True),
        "mechanism_status": "missing",
        "mechanism_error": "",
    }

    params = job.get("params", {})
    for key in sorted(params.keys()):
        record[f"param__{key}"] = params[key]

    for metric in PRIMARY_REPORT_METRICS:
        record[f"selected__{metric}"] = np.nan
    for metric in MECHANISM_METRICS:
        record[metric] = np.nan

    test_results = _safe_read_json(test_results_path)
    if test_results:
        for metric in PRIMARY_REPORT_METRICS:
            record[f"selected__{metric}"] = _nested_get(test_results, ["selected", metric], default=np.nan)
        mechanism_meta = test_results.get("mechanism", {})
        if isinstance(mechanism_meta, dict) and mechanism_meta.get("status") == "failed":
            record["mechanism_status"] = "failed"
            record["mechanism_error"] = str(mechanism_meta.get("error", ""))

    mechanism_summary = _safe_read_json(mechanism_path)
    if mechanism_summary:
        record["mechanism_status"] = "ok"
        _fill_mechanism_metrics(record, mechanism_summary)

    return record


def _fill_mechanism_metrics(record: Dict, mechanism_summary: Dict) -> None:
    for metric in (
        "mean_norm",
        "std_norm",
        "variance_loss",
        "covariance_loss",
    ):
        record[f"mechanism__representation__raw__{metric}"] = _nested_get(mechanism_summary, ["representation", "raw", metric], np.nan)
        record[f"mechanism__representation__fair__{metric}"] = _nested_get(mechanism_summary, ["representation", "fair", metric], np.nan)

    for metric in ("delta_l2_mean", "delta_l2_std", "cosine_mean", "cosine_std"):
        record[f"mechanism__representation__transform__{metric}"] = _nested_get(mechanism_summary, ["representation", "transform", metric], np.nan)

    for attr in ("gender", "age_group"):
        for split in ("raw", "fair"):
            prefix = f"mechanism__{attr}__probe_{split}"
            source = _nested_get(mechanism_summary, ["attributes", attr, f"probe_{split}"], {})
            if isinstance(source, dict):
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "num_samples_total", "num_samples_used", "folds"):
                    record[f"{prefix}__{metric}"] = source.get(metric, np.nan)

            sep_prefix = f"mechanism__{attr}__separability_{split}"
            sep_source = _nested_get(mechanism_summary, ["attributes", attr, f"separability_{split}"], {})
            if isinstance(sep_source, dict):
                for metric in ("centroid_distance_mean", "centroid_distance_min", "centroid_distance_max",
                               "within_group_dispersion_mean", "between_within_ratio", "num_groups"):
                    record[f"{sep_prefix}__{metric}"] = sep_source.get(metric, np.nan)

        raw_probe = record.get(f"mechanism__{attr}__probe_raw__balanced_accuracy", np.nan)
        fair_probe = record.get(f"mechanism__{attr}__probe_fair__balanced_accuracy", np.nan)
        record[f"mechanism__{attr}__probe_balanced_accuracy_drop"] = _safe_diff(raw_probe, fair_probe)

        raw_sep = record.get(f"mechanism__{attr}__separability_raw__between_within_ratio", np.nan)
        fair_sep = record.get(f"mechanism__{attr}__separability_fair__between_within_ratio", np.nan)
        record[f"mechanism__{attr}__separability_between_within_ratio_drop"] = _safe_diff(raw_sep, fair_sep)


def _nested_get(data: Dict, path: Sequence[str], default=np.nan):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_diff(raw_value, fair_value):
    if pd.isna(raw_value) or pd.isna(fair_value):
        return np.nan
    return float(raw_value) - float(fair_value)


def _safe_read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _mechanism_raw_columns(df: pd.DataFrame, metric_cols: Sequence[str]) -> List[str]:
    extra_metric_cols = [f"selected__{metric}" for metric in PRIMARY_REPORT_METRICS if f"selected__{metric}" in df.columns]
    base = [
        "mechanism_status", "mechanism_error", "phase",
        "dataset", "dataset_display",
        "backbone", "backbone_display",
        "method", "method_display",
        "seed", "job_id", "reuse_from_phase", "reuse_from_job_id", "output_dir", "params_json",
    ]
    param_cols = sorted([col for col in df.columns if col.startswith("param__")])
    return [col for col in base + param_cols + extra_metric_cols + list(metric_cols) if col in df.columns]


def _mechanism_summary_columns(df: pd.DataFrame, metric_cols: Sequence[str]) -> List[str]:
    base = [
        "dataset", "dataset_display",
        "backbone", "backbone_display",
        "method", "method_display",
    ]
    metric_summary_cols = []
    for metric_col in metric_cols:
        metric_summary_cols.extend([
            f"{metric_col}__mean",
            f"{metric_col}__std",
            f"{metric_col}__count",
        ])
    return [col for col in base + metric_summary_cols if col in df.columns]

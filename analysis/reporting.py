from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.specs import ABLATION_METHODS, COMPARISON_METHODS


# -----------------------------------------------------------------------------
# Metric configuration
# -----------------------------------------------------------------------------
# Your test_results.json already stores full metrics in:
#   - utility: HitRate/Precision/Recall/NDCG/MRR at @5/@10/@20
#   - fairness: gender/age_group group metrics, gap, worst_group, std at @5/@10/@20
# The old reporting.py only read test_results["selected"], so it only exported @10.
# This version flattens utility + fairness dynamically and keeps the old selected__
# column prefix for backward compatibility with plotting/aggregation code.

DEFAULT_TOP_KS = (5, 10, 20)
UTILITY_METRIC_NAMES = ("HitRate", "Precision", "Recall", "NDCG", "MRR")
FAIRNESS_ATTRS = ("gender", "age_group")
FAIRNESS_BASE_METRICS = ("HitRate", "NDCG")
FAIRNESS_STATS = ("Gap", "WorstGroup", "Std")

PRIMARY_UTILITY_METRICS = tuple(
    f"{metric}@{k}"
    for k in DEFAULT_TOP_KS
    for metric in UTILITY_METRIC_NAMES
)

PRIMARY_FAIRNESS_METRICS = tuple(
    f"{attr}_{metric}@{k}_{stat}"
    for attr in FAIRNESS_ATTRS
    for k in DEFAULT_TOP_KS
    for metric in FAIRNESS_BASE_METRICS
    for stat in FAIRNESS_STATS
)

PRIMARY_REPORT_METRICS = PRIMARY_UTILITY_METRICS + PRIMARY_FAIRNESS_METRICS

PHASE_PARAMETER_MAP = {
    "backbone": ("learning_rate", "hidden_units", "dropout_rate"),
    "augment": ("epsilon", "augment_ratio", "utility_alpha"),
    "loss": ("fair_ncl_align_weight", "fair_ncl_var_weight", "fair_ncl_cov_weight"),
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "fair_ncl": "Fair-NCL",
    "fair_ncl_alpha_tradeoff": "Fair-NCL (alpha trade-off)",
    "fair_ncl_semantic_alpha_tradeoff": "Fair-NCL (semantic alpha trade-off)",
    "fair_ncl_semantic_hybrid_alpha_tradeoff": "Fair-NCL (semantic hybrid alpha trade-off)",
    "ncl_only": "NCL-only",
    "random_aug": "RandomAug",
    "similarity_aug": "Similarity-only Aug",
    "wo_fairness_sampling": "w/o Skew-aware Sampling",
    "wo_semantic_sampling": "w/o Semantic-aware Sampling",
    "wo_alignment": "w/o Alignment",
    "wo_variance": "w/o Variance",
    "wo_covariance": "w/o Covariance",
    "wo_augmented_ce": "w/o Augmented CE",
    "random_low_skew": "Random Low-skew Replacement",
    "high_skew": "High-skew Replacement",
    "adv_debias": "Adv-Debias",
    "grl": "GRL",
    "sm_pcfr": "SM / PCFR",
    "afrl": "AFRL",
    "pfrec": "PFRec",
    "a_fsr": "A-FSR",
}

DATASET_LABELS = {
    "ml-1m": "ML-1M",
    "lastfm-1k": "LastFM-1K",
    "taobao": "Taobao",
}
DATASET_ORDER = {name: idx for idx, name in enumerate(DATASET_LABELS.keys())}

BACKBONE_LABELS = {
    "sasrec": "SASRec",
    "bert4rec": "BERT4Rec",
    "gru4rec": "GRU4Rec",
    "caser": "Caser",
}
BACKBONE_ORDER = {name: idx for idx, name in enumerate(BACKBONE_LABELS.keys())}

ATTR_DISPLAY_LABELS = {
    "gender": "Gender",
    "age_group": "Age Group",
}

METRIC_LABELS = {
    metric: metric for metric in PRIMARY_UTILITY_METRICS
}
METRIC_LABELS.update({
    f"{attr}_{metric}@{k}_{stat}": f"{ATTR_DISPLAY_LABELS.get(attr, attr.replace('_', ' ').title())} {metric}@{k} {stat}"
    for attr in FAIRNESS_ATTRS
    for k in DEFAULT_TOP_KS
    for metric in FAIRNESS_BASE_METRICS
    for stat in FAIRNESS_STATS
})

PHASE_METHOD_ORDER = {
    "ablation": list(ABLATION_METHODS),
    "comparison": list(COMPARISON_METHODS),
}


# -----------------------------------------------------------------------------
# Public loading/filtering/aggregation APIs
# -----------------------------------------------------------------------------

def load_job_records(jobs_path: str, results_root: str) -> pd.DataFrame:
    """Load job definitions and the corresponding test results into one frame."""
    records: List[Dict] = []
    with open(jobs_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            records.append(_build_record(job, results_root))
    return pd.DataFrame.from_records(records)


def filter_records(df: pd.DataFrame,
                   phase: Optional[str] = None,
                   datasets: Optional[Sequence[str]] = None,
                   backbones: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Apply common report filters."""
    result = df.copy()
    if phase and phase.lower() != "all" and "phase" in result.columns:
        result = result[result["phase"] == phase]
    if not _is_all_selection(datasets) and "dataset" in result.columns:
        result = result[result["dataset"].isin(list(datasets))]
    if not _is_all_selection(backbones) and "backbone" in result.columns:
        result = result[result["backbone"].isin(list(backbones))]
    return result.reset_index(drop=True)


def aggregate_numeric_table(df: pd.DataFrame,
                            group_cols: Sequence[str],
                            metric_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate metrics with mean/std/count for each group."""
    if df.empty:
        return pd.DataFrame(columns=list(group_cols))

    available_metric_cols = [col for col in metric_cols if col in df.columns]
    if not available_metric_cols:
        return pd.DataFrame(columns=list(group_cols))

    work_df = df.copy()
    for metric_col in available_metric_cols:
        work_df[metric_col] = pd.to_numeric(work_df[metric_col], errors="coerce")

    agg_spec = {}
    for metric_col in available_metric_cols:
        agg_spec[f"{metric_col}__mean"] = (metric_col, "mean")
        agg_spec[f"{metric_col}__std"] = (metric_col, _safe_std)
        agg_spec[f"{metric_col}__count"] = (metric_col, _safe_count)

    summary = work_df.groupby(list(group_cols), dropna=False, sort=False).agg(**agg_spec).reset_index()
    return summary


def build_parameter_curve_frame(df: pd.DataFrame,
                                phase: str,
                                metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert job-level rows into a long frame for parameter sensitivity plots."""
    if metrics is None:
        metrics = PRIMARY_REPORT_METRICS

    metric_cols = [f"selected__{metric}" for metric in metrics if f"selected__{metric}" in df.columns]
    param_cols = [col for col in df.columns if col.startswith("param__")]
    if df.empty or not metric_cols or not param_cols:
        return pd.DataFrame(columns=[
            "phase", "dataset", "backbone", "method", "seed",
            "param_name", "param_value", "metric_name", "metric_value",
        ])

    rows: List[Dict] = []
    ok_rows = df[df["status"] == "ok"] if "status" in df.columns else df
    for _, row in ok_rows.iterrows():
        for param_col in param_cols:
            param_value = row.get(param_col)
            if pd.isna(param_value):
                continue
            param_name = param_col[len("param__"):]
            for metric in metrics:
                metric_col = f"selected__{metric}"
                if metric_col not in df.columns:
                    continue
                metric_value = row.get(metric_col)
                if pd.isna(metric_value):
                    continue
                rows.append({
                    "phase": row.get("phase", phase),
                    "dataset": row.get("dataset"),
                    "backbone": row.get("backbone"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "param_name": param_name,
                    "param_value": _to_python(param_value),
                    "metric_name": metric,
                    "metric_value": _to_python(metric_value),
                })
    return pd.DataFrame.from_records(rows)


def aggregate_parameter_curve_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long parameter frame across seeds and other settings."""
    if long_df.empty:
        return pd.DataFrame(columns=[
            "phase", "dataset", "backbone", "method",
            "param_name", "param_value", "metric_name",
            "mean", "std", "count",
        ])

    summary = long_df.groupby(
        ["phase", "dataset", "backbone", "method", "param_name", "param_value", "metric_name"],
        dropna=False,
        sort=False,
    )["metric_value"].agg(
        mean="mean",
        std=_safe_std,
        count=_safe_count,
    ).reset_index()
    return summary


def export_experiment_report(jobs_path: str,
                             results_root: str,
                             output_path: str,
                             phase: Optional[str] = None,
                             datasets: Optional[Sequence[str]] = None,
                             backbones: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """Export ablation/comparison results as an Excel workbook."""
    df = load_job_records(jobs_path, results_root)
    df = filter_records(df, phase=phase, datasets=datasets, backbones=backbones)
    if df.empty:
        raise ValueError("No matching jobs were found for the report.")

    metric_cols = _discover_metric_cols(df)
    if not metric_cols:
        raise ValueError(
            "No report metrics were found in the experiment results. "
            "Check whether test_results.json contains utility/fairness/selected metrics."
        )

    completed = df[df["status"] == "ok"].copy()
    if completed.empty:
        raise ValueError("No completed experiment results were found.")

    report_phase = _infer_phase(completed, phase)
    summary = aggregate_numeric_table(completed, ["dataset", "backbone", "method"], metric_cols)
    summary = _attach_display_columns(summary)
    summary = _sort_summary_rows(summary, report_phase)
    paper_tables = build_paper_tables(summary, report_phase, metric_cols)

    raw_columns = _ordered_raw_columns(df, metric_cols)
    summary_columns = _ordered_summary_columns(summary, metric_cols)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_results", index=False, columns=raw_columns)
        summary.to_excel(writer, sheet_name="summary_long", index=False, columns=summary_columns)

        missing = df[df["status"] != "ok"].copy() if "status" in df.columns else pd.DataFrame()
        if not missing.empty:
            missing.to_excel(writer, sheet_name="missing_jobs", index=False, columns=_ordered_raw_columns(df, metric_cols))

        for sheet_name, table in paper_tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)

    return {
        "raw_results": df,
        "summary_long": summary,
        "paper_tables": paper_tables,
    }


def build_paper_tables(summary_df: pd.DataFrame,
                       phase: str,
                       metric_cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Build paper-friendly wide tables with formatted mean/std cells."""
    if summary_df.empty:
        return {}

    method_order = _method_order(phase, summary_df["method"].dropna().tolist())
    method_rank = {method: idx for idx, method in enumerate(method_order)}
    metric_names = [col[len("selected__"):] if col.startswith("selected__") else col for col in metric_cols]
    display_metrics = [_metric_display_label(metric) for metric in metric_names]

    paper_tables: Dict[str, pd.DataFrame] = {}
    for (dataset, backbone), combo_df in summary_df.groupby(["dataset", "backbone"], dropna=False, sort=False):
        combo_df = combo_df.copy()
        combo_df["_method_rank"] = combo_df["method"].map(method_rank).fillna(len(method_rank)).astype(int)
        combo_df = combo_df.sort_values(["_method_rank", "method"]).reset_index(drop=True)

        rows: List[Dict] = []
        for _, row in combo_df.iterrows():
            paper_row = {
                "Method": METHOD_LABELS.get(row["method"], row["method"]),
            }
            for metric_col, metric_name, display_name in zip(metric_cols, metric_names, display_metrics):
                mean_value = row.get(f"{metric_col}__mean", np.nan)
                std_value = row.get(f"{metric_col}__std", np.nan)
                paper_row[display_name] = _format_mean_std(mean_value, std_value)
            rows.append(paper_row)

        sheet_name = _sheet_name(f"{dataset}_{backbone}")
        paper_tables[sheet_name] = pd.DataFrame(rows)

    return paper_tables


# -----------------------------------------------------------------------------
# Record building and result flattening
# -----------------------------------------------------------------------------

def _build_record(job: Dict, results_root: str) -> Dict:
    dataset = job["dataset"]
    job_id = job["job_id"]
    method = job["method"]
    backbone = job["backbone"]
    seed = int(job.get("seed", 42))
    phase = job.get("phase", "")
    output_dir = os.path.join(results_root, dataset, job_id)
    result_path = os.path.join(output_dir, "test_results.json")

    record = {
        "job_id": job_id,
        "phase": phase,
        "phase_display": phase.replace("_", " ").title(),
        "dataset": dataset,
        "dataset_display": DATASET_LABELS.get(dataset, dataset),
        "method": method,
        "method_display": METHOD_LABELS.get(method, method),
        "backbone": backbone,
        "backbone_display": BACKBONE_LABELS.get(backbone, backbone),
        "seed": seed,
        "reuse_from_phase": job.get("reuse_from_phase"),
        "reuse_from_job_id": job.get("reuse_from_job_id"),
        "output_dir": output_dir,
        "result_path": result_path,
        "status": "ok" if os.path.exists(result_path) else "missing",
        "params_json": json.dumps(job.get("params", {}), ensure_ascii=False, sort_keys=True),
    }

    params = job.get("params", {})
    for key in sorted(params.keys()):
        record[f"param__{key}"] = _to_python(params[key])

    if record["status"] == "ok":
        try:
            test_results = _read_json(result_path)
            flattened_metrics = _flatten_result_metrics(
                test_results,
                include_group_values=True,
                include_selected=True,
            )
            for metric_name, metric_value in flattened_metrics.items():
                record[f"selected__{metric_name}"] = _to_python(metric_value)
        except Exception as exc:  # keep reporting robust when one JSON is broken
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def _flatten_result_metrics(result: Dict,
                            include_group_values: bool = True,
                            include_selected: bool = True) -> Dict[str, float]:
    """Flatten test_results.json into scalar metric columns.

    Output examples:
        HitRate@5
        NDCG@20
        gender_HitRate@10_Gap
        gender_HitRate@10_WorstGroup
        gender_HitRate@10_Std
        gender_Group_HitRate@10_0
        age_group_Group_NDCG@20_1

    Notes:
        - utility/fairness are treated as the source of truth.
        - selected is appended only for keys that were not already exported.
    """
    flat: Dict[str, float] = {}

    # 1) Utility metrics: HitRate@5, Precision@5, Recall@5, NDCG@5, MRR@5, ...
    utility = result.get("utility", {})
    if isinstance(utility, dict):
        for metric_name, value in utility.items():
            if _is_number(value):
                flat[str(metric_name)] = _to_python(value)

    # 2) Fairness metrics by sensitive attribute.
    fairness = result.get("fairness", {})
    if isinstance(fairness, dict):
        for attr_name, attr_metrics in fairness.items():
            if not isinstance(attr_metrics, dict):
                continue

            for raw_metric_name, value in attr_metrics.items():
                metric_name = _normalize_fair_metric_name(str(raw_metric_name))

                # Scalar metrics: hitrate@10_gap, ndcg@20_worst_group, ...
                if _is_number(value):
                    flat[f"{attr_name}_{metric_name}"] = _to_python(value)

                # Group-wise values: group_hitrate@10: {"0": ..., "1": ...}
                elif include_group_values and isinstance(value, dict):
                    for group_id, group_value in value.items():
                        if _is_number(group_value):
                            flat[f"{attr_name}_{metric_name}_{group_id}"] = _to_python(group_value)

    # 3) Backward compatibility: selected metrics.
    # Usually selected duplicates @10 metrics. Use setdefault so utility/fairness wins.
    if include_selected:
        selected = result.get("selected", {})
        if isinstance(selected, dict):
            for metric_name, value in selected.items():
                if _is_number(value):
                    flat.setdefault(str(metric_name), _to_python(value))

    return flat


def _normalize_fair_metric_name(name: str) -> str:
    """Normalize fairness metric names from result JSON.

    Examples:
        hitrate@10_gap       -> HitRate@10_Gap
        ndcg@20_worst_group  -> NDCG@20_WorstGroup
        ndcg@5_std           -> NDCG@5_Std
        group_hitrate@10     -> Group_HitRate@10
        group_ndcg@20        -> Group_NDCG@20
    """
    normalized = name
    normalized = normalized.replace("group_hitrate", "Group_HitRate")
    normalized = normalized.replace("group_ndcg", "Group_NDCG")
    normalized = normalized.replace("hitrate", "HitRate")
    normalized = normalized.replace("ndcg", "NDCG")
    normalized = normalized.replace("_gap", "_Gap")
    normalized = normalized.replace("_worst_group", "_WorstGroup")
    normalized = normalized.replace("_std", "_Std")
    return normalized


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _safe_count(values: pd.Series) -> int:
    return int(pd.to_numeric(values, errors="coerce").notna().sum())


def _safe_std(values: pd.Series) -> float:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    if len(cleaned) <= 1:
        return 0.0 if len(cleaned) == 1 else np.nan
    return float(cleaned.std(ddof=1))


def _format_mean_std(mean_value, std_value) -> str:
    if pd.isna(mean_value):
        return ""
    if pd.isna(std_value):
        return f"{float(mean_value):.4f}"
    return f"{float(mean_value):.4f} \u00b1 {float(std_value):.4f}"


def _discover_metric_cols(df: pd.DataFrame) -> List[str]:
    """Discover all flattened metric columns and return them in a stable order."""
    metric_cols = [col for col in df.columns if col.startswith("selected__")]
    numeric_metric_cols = []
    for col in metric_cols:
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        if numeric_values.notna().any():
            numeric_metric_cols.append(col)
    return sorted(numeric_metric_cols, key=_metric_col_sort_key)


def _metric_col_sort_key(col: str) -> Tuple:
    metric = col[len("selected__"):] if col.startswith("selected__") else col
    return _metric_sort_key(metric)


def _metric_sort_key(metric: str) -> Tuple:
    """Sort metrics as utility first, fairness scalar second, group values third."""
    utility_match = re.fullmatch(r"([A-Za-z]+)@(\d+)", metric)
    if utility_match:
        metric_name, k = utility_match.group(1), int(utility_match.group(2))
        return (
            0,
            _topk_rank(k),
            _utility_metric_rank(metric_name),
            metric,
        )

    fairness_scalar_match = re.fullmatch(
        r"(.+)_(HitRate|NDCG)@(\d+)_(Gap|WorstGroup|Std)",
        metric,
    )
    if fairness_scalar_match:
        attr, metric_name, k, stat = fairness_scalar_match.groups()
        return (
            1,
            _attr_rank(attr),
            _topk_rank(int(k)),
            _fairness_metric_rank(metric_name),
            _fairness_stat_rank(stat),
            metric,
        )

    fairness_group_match = re.fullmatch(
        r"(.+)_Group_(HitRate|NDCG)@(\d+)_(.+)",
        metric,
    )
    if fairness_group_match:
        attr, metric_name, k, group_id = fairness_group_match.groups()
        return (
            2,
            _attr_rank(attr),
            _topk_rank(int(k)),
            _fairness_metric_rank(metric_name),
            str(group_id),
            metric,
        )

    return (9, metric)


def _topk_rank(k: int) -> int:
    if k in DEFAULT_TOP_KS:
        return DEFAULT_TOP_KS.index(k)
    return len(DEFAULT_TOP_KS) + k


def _utility_metric_rank(metric: str) -> int:
    return UTILITY_METRIC_NAMES.index(metric) if metric in UTILITY_METRIC_NAMES else len(UTILITY_METRIC_NAMES)


def _fairness_metric_rank(metric: str) -> int:
    return FAIRNESS_BASE_METRICS.index(metric) if metric in FAIRNESS_BASE_METRICS else len(FAIRNESS_BASE_METRICS)


def _fairness_stat_rank(stat: str) -> int:
    return FAIRNESS_STATS.index(stat) if stat in FAIRNESS_STATS else len(FAIRNESS_STATS)


def _attr_rank(attr: str) -> int:
    return FAIRNESS_ATTRS.index(attr) if attr in FAIRNESS_ATTRS else len(FAIRNESS_ATTRS)


def _ordered_raw_columns(df: pd.DataFrame, metric_cols: Sequence[str]) -> List[str]:
    base = [
        "status", "error", "phase", "phase_display",
        "dataset", "dataset_display",
        "backbone", "backbone_display",
        "method", "method_display",
        "seed", "job_id", "reuse_from_phase", "reuse_from_job_id",
        "output_dir", "result_path", "params_json",
    ]
    param_cols = sorted([col for col in df.columns if col.startswith("param__")])
    selected_cols = [col for col in metric_cols if col in df.columns]
    rest = [col for col in df.columns if col not in base + param_cols + selected_cols]
    return [col for col in base + param_cols + selected_cols + rest if col in df.columns]


def _ordered_summary_columns(df: pd.DataFrame, metric_cols: Sequence[str]) -> List[str]:
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
    rest = [col for col in df.columns if col not in base + metric_summary_cols]
    return [col for col in base + metric_summary_cols + rest if col in df.columns]


def _attach_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["dataset_display"] = result["dataset"].map(DATASET_LABELS).fillna(result["dataset"])
    result["backbone_display"] = result["backbone"].map(BACKBONE_LABELS).fillna(result["backbone"])
    result["method_display"] = result["method"].map(METHOD_LABELS).fillna(result["method"])
    return result


def _sort_summary_rows(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    method_order = _method_order(phase, result["method"].dropna().tolist())
    method_rank = {method: idx for idx, method in enumerate(method_order)}
    result["_dataset_rank"] = result["dataset"].map(DATASET_ORDER).fillna(len(DATASET_ORDER)).astype(int)
    result["_backbone_rank"] = result["backbone"].map(BACKBONE_ORDER).fillna(len(BACKBONE_ORDER)).astype(int)
    result["_method_rank"] = result["method"].map(method_rank).fillna(len(method_rank)).astype(int)
    result = result.sort_values(["_dataset_rank", "_backbone_rank", "_method_rank", "method"]).reset_index(drop=True)
    return result.drop(columns=["_dataset_rank", "_backbone_rank", "_method_rank"])


def _method_order(phase: str, observed_methods: Sequence[str]) -> List[str]:
    if phase in PHASE_METHOD_ORDER:
        preferred = list(PHASE_METHOD_ORDER[phase])
        remaining = [method for method in observed_methods if method not in preferred]
        return preferred + sorted(set(remaining))
    return sorted(set(observed_methods))


def _infer_phase(df: pd.DataFrame, fallback: Optional[str] = None) -> str:
    if fallback and fallback.lower() != "all":
        return fallback
    if "phase" not in df.columns or df.empty:
        return fallback or "unknown"
    phases = [phase for phase in df["phase"].dropna().unique().tolist() if phase]
    if len(phases) == 1:
        return phases[0]
    return "mixed"


def _sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned[:31] if len(cleaned) > 31 else cleaned


def _is_all_selection(values: Optional[Sequence[str]]) -> bool:
    if values is None:
        return True
    if len(values) == 0:
        return True
    return len(values) == 1 and str(values[0]).lower() == "all"


def _metric_display_label(metric: str) -> str:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]

    group_match = re.fullmatch(r"(.+)_Group_(HitRate|NDCG)@(\d+)_(.+)", metric)
    if group_match:
        attr, metric_name, k, group_id = group_match.groups()
        return f"{_attr_display(attr)} Group {group_id} {metric_name}@{k}"

    scalar_match = re.fullmatch(r"(.+)_(HitRate|NDCG)@(\d+)_(Gap|WorstGroup|Std)", metric)
    if scalar_match:
        attr, metric_name, k, stat = scalar_match.groups()
        return f"{_attr_display(attr)} {metric_name}@{k} {stat}"

    return metric


def _attr_display(attr: str) -> str:
    return ATTR_DISPLAY_LABELS.get(attr, attr.replace("_", " ").title())

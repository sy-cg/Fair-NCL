from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import (
    PRIMARY_REPORT_METRICS,
    PHASE_PARAMETER_MAP,
    aggregate_parameter_curve_frame,
    build_parameter_curve_frame,
    filter_records,
    load_job_records,
    _sheet_name,
)


# -----------------------------------------------------------------------------
# Paper-facing plot configuration
# -----------------------------------------------------------------------------

MAIN_METRICS = [
    "NDCG@10",
    "gender_NDCG@10_Gap",
    "age_group_NDCG@10_Gap",
]

APPENDIX_METRICS = [
    "HitRate@10",
    "NDCG@10",
    "gender_HitRate@10_Gap",
    "gender_NDCG@10_Gap",
    "age_group_HitRate@10_Gap",
    "age_group_NDCG@10_Gap",
]

MAIN_PHASE_PARAMETERS = {
    "augment": ["epsilon", "augment_ratio", "utility_beta"],
    "loss": ["fair_ncl_align_weight", "fair_ncl_var_weight", "fair_ncl_cov_weight"],
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def export_parameter_plots(jobs_path: str,
                           results_root: str,
                           output_dir: str,
                           phase: str,
                           datasets: Optional[Sequence[str]] = None,
                           backbones: Optional[Sequence[str]] = None,
                           metrics: Optional[Sequence[str]] = None,
                           file_format: str = "png") -> Dict[str, pd.DataFrame]:
    """Export parameter-sensitivity plots.

    Output design:
        1. Main-paper figures for augment/loss:
           - One figure per backbone.
           - 3 rows = NDCG@10, Gender NDCG@10 Gap, Age Group NDCG@10 Gap.
           - 3 columns = phase parameters.
           - Curves in each subplot = different datasets.

        2. Appendix figures:
           - One figure per backbone.
           - Rows = 6 metrics:
               HitRate@10, NDCG@10,
               gender_HitRate@10_Gap, gender_NDCG@10_Gap,
               age_group_HitRate@10_Gap, age_group_NDCG@10_Gap.
           - Columns = parameters of the current phase.
           - Curves in each subplot = different datasets.

    Notes:
        - This function keeps the original run_research_pipeline.py interface unchanged.
        - For phase='augment', the main figure corresponds to Figure 1.
        - For phase='loss', the main figure corresponds to Figure 2.
    """
    if phase not in PHASE_PARAMETER_MAP:
        raise ValueError(f"Unsupported plotting phase '{phase}'.")

    requested_metrics = list(metrics) if metrics is not None else list(PRIMARY_REPORT_METRICS)
    metrics_needed = _deduplicate_preserve_order(
        requested_metrics + MAIN_METRICS + APPENDIX_METRICS
    )

    records = load_job_records(jobs_path, results_root)
    records = filter_records(records, phase=phase, datasets=datasets, backbones=backbones)
    if records.empty:
        raise ValueError("No matching jobs were found for parameter plotting.")

    if "status" in records.columns:
        records = records[records["status"] == "ok"].copy()
    if records.empty:
        raise ValueError("No completed jobs were found for parameter plotting.")

    long_df = build_parameter_curve_frame(records, phase=phase, metrics=metrics_needed)
    if long_df.empty:
        raise ValueError("No usable parameter curve data were found.")

    summary_df = aggregate_parameter_curve_frame(long_df)
    if summary_df.empty:
        raise ValueError("No usable parameter summary data were found.")

    phase_dir = os.path.join(output_dir, phase)
    csv_dir = os.path.join(phase_dir, "csv")
    main_dir = os.path.join(phase_dir, "main")
    appendix_dir = os.path.join(phase_dir, "appendix")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(appendix_dir, exist_ok=True)

    long_csv = os.path.join(csv_dir, f"{phase}_parameter_curve_raw.csv")
    summary_csv = os.path.join(csv_dir, f"{phase}_parameter_curve_summary.csv")
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    figure_paths: List[str] = []

    # Main-paper figures: only for augment and loss.
    if phase in MAIN_PHASE_PARAMETERS:
        figure_paths.extend(
            _export_main_paper_figures(
                summary_df=summary_df,
                phase=phase,
                output_dir=main_dir,
                file_format=file_format,
            )
        )

    # Appendix figures: complete 6 metrics x phase parameters.
    figure_paths.extend(
        _export_appendix_figures(
            summary_df=summary_df,
            phase=phase,
            output_dir=appendix_dir,
            file_format=file_format,
        )
    )

    return {
        "raw": long_df,
        "summary": summary_df,
        "figure_paths": pd.DataFrame({"figure_path": figure_paths}),
    }


# -----------------------------------------------------------------------------
# Figure exporters
# -----------------------------------------------------------------------------

def _export_main_paper_figures(summary_df: pd.DataFrame,
                               phase: str,
                               output_dir: str,
                               file_format: str) -> List[str]:
    """Export 3x3 main-paper figures for augment/loss."""
    figure_paths: List[str] = []
    parameter_names = MAIN_PHASE_PARAMETERS[phase]
    metric_names = MAIN_METRICS

    backbones_in_data = _sorted_values(summary_df["backbone"].dropna().unique().tolist(), kind="backbone")

    for backbone in backbones_in_data:
        backbone_df = summary_df[summary_df["backbone"] == backbone].copy()
        if backbone_df.empty:
            continue

        fig_path = _plot_metric_parameter_grid(
            df=backbone_df,
            phase=phase,
            backbone=backbone,
            metric_names=metric_names,
            parameter_names=parameter_names,
            output_dir=output_dir,
            file_format=file_format,
            prefix="main",
            title=f"{_display(backbone, 'backbone')} Parameter Sensitivity ({_display_phase(phase)})",
            figsize=(12.0, 8.4),
            legend_mode="top",
        )
        if fig_path:
            figure_paths.append(fig_path)

    return figure_paths


def _export_appendix_figures(summary_df: pd.DataFrame,
                             phase: str,
                             output_dir: str,
                             file_format: str) -> List[str]:
    """Export appendix figures with 6 metrics x phase parameters."""
    figure_paths: List[str] = []
    parameter_names = list(PHASE_PARAMETER_MAP[phase])
    metric_names = APPENDIX_METRICS

    backbones_in_data = _sorted_values(summary_df["backbone"].dropna().unique().tolist(), kind="backbone")

    for backbone in backbones_in_data:
        backbone_df = summary_df[summary_df["backbone"] == backbone].copy()
        if backbone_df.empty:
            continue

        width = max(4.0 * len(parameter_names), 8.0)
        height = max(2.6 * len(metric_names), 10.0)

        fig_path = _plot_metric_parameter_grid(
            df=backbone_df,
            phase=phase,
            backbone=backbone,
            metric_names=metric_names,
            parameter_names=parameter_names,
            output_dir=output_dir,
            file_format=file_format,
            prefix="appendix",
            title=f"{_display(backbone, 'backbone')} Full Parameter Sensitivity ({_display_phase(phase)})",
            figsize=(width, height),
            legend_mode="top",
        )
        if fig_path:
            figure_paths.append(fig_path)

    return figure_paths


def _plot_metric_parameter_grid(df: pd.DataFrame,
                                phase: str,
                                backbone: str,
                                metric_names: Sequence[str],
                                parameter_names: Sequence[str],
                                output_dir: str,
                                file_format: str,
                                prefix: str,
                                title: str,
                                figsize: Tuple[float, float],
                                legend_mode: str = "top") -> Optional[str]:
    """Plot rows=metrics, columns=parameters; curves=datasets."""
    n_rows = len(metric_names)
    n_cols = len(parameter_names)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=False,
    )

    has_any_subplot = False
    legend_handles = None
    legend_labels = None

    for row_idx, metric_name in enumerate(metric_names):
        for col_idx, param_name in enumerate(parameter_names):
            ax = axes[row_idx][col_idx]
            param_df = df[
                (df["metric_name"] == metric_name) &
                (df["param_name"] == param_name)
            ].copy()

            if param_df.empty:
                ax.set_axis_off()
                continue

            _plot_parameter_curve_multi_dataset(
                ax=ax,
                param_df=param_df,
                param_name=param_name,
                metric_name=metric_name,
                show_legend=False,
            )
            has_any_subplot = True

            if row_idx == 0:
                ax.set_title(_display(param_name, "param"), fontsize=11)
            else:
                ax.set_title("")

            if col_idx == 0:
                ax.set_ylabel(_display_metric(metric_name), fontsize=10)
            else:
                ax.set_ylabel("")

            if row_idx == n_rows - 1:
                ax.set_xlabel(_display(param_name, "param"), fontsize=10)
            else:
                ax.set_xlabel("")

            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    legend_handles = handles
                    legend_labels = labels

    if not has_any_subplot:
        plt.close(fig)
        return None

    fig.suptitle(title, fontsize=13, y=0.995)

    if legend_mode == "top" and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
    else:
        fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{prefix}_{phase}_{_safe_name(backbone)}_multi_dataset.{file_format}"
    full_path = os.path.join(output_dir, file_name)
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return full_path


# -----------------------------------------------------------------------------
# Plot primitives
# -----------------------------------------------------------------------------

def _plot_parameter_curve_multi_dataset(ax: plt.Axes,
                                        param_df: pd.DataFrame,
                                        param_name: str,
                                        metric_name: str,
                                        show_legend: bool = True) -> None:
    """Plot one parameter subplot with multiple dataset curves."""
    if param_df.empty:
        ax.set_axis_off()
        return

    dataset_values = _sorted_values(param_df["dataset"].dropna().unique().tolist(), kind="dataset")

    for dataset in dataset_values:
        dataset_df = param_df[param_df["dataset"] == dataset].copy()
        if dataset_df.empty:
            continue

        dataset_df = _collapse_duplicate_param_values(dataset_df)
        dataset_df = _sort_curve_frame(dataset_df)

        if dataset_df.empty:
            continue

        x_values = dataset_df["param_value"].tolist()
        y_values = pd.to_numeric(dataset_df["mean"], errors="coerce").tolist()

        x_numeric = _try_numeric_list(x_values)
        if x_numeric is not None:
            x_plot = x_numeric
            ax.plot(
                x_plot,
                y_values,
                marker=_dataset_marker(dataset),
                linewidth=1.7,
                markersize=5,
                label=_display(dataset, "dataset"),
            )
            if param_name == "learning_rate":
                ax.set_xscale("log")
        else:
            x_plot = list(range(len(x_values)))
            ax.plot(
                x_plot,
                y_values,
                marker=_dataset_marker(dataset),
                linewidth=1.7,
                markersize=5,
                label=_display(dataset, "dataset"),
            )
            ax.set_xticks(x_plot)
            ax.set_xticklabels([str(x) for x in x_values])

    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.7)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    if show_legend:
        ax.legend(frameon=False, fontsize=8)

    if "Gap" in metric_name:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(bottom=max(0.0, ymin))


def _collapse_duplicate_param_values(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate dataset-param points from multiple seeds/settings.

    The summary frame is usually grouped by:
        phase, dataset, backbone, method, param_name, param_value, metric_name

    In parameter sensitivity plots, we average over non-focal parameters and seeds.
    This gives a clean marginal trend for each parameter.
    """
    if df.empty:
        return df

    work = df.copy()
    work["mean"] = pd.to_numeric(work["mean"], errors="coerce")
    work["std"] = pd.to_numeric(work["std"], errors="coerce")
    if "count" not in work.columns:
        work["count"] = 1
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(1)

    grouped = work.groupby(
        ["dataset", "backbone", "param_name", "param_value", "metric_name"],
        dropna=False,
        sort=False,
    ).agg(
        mean=("mean", "mean"),
        std=("std", "mean"),
        count=("count", "sum"),
    ).reset_index()

    return grouped


# -----------------------------------------------------------------------------
# Sorting and display helpers
# -----------------------------------------------------------------------------

def _sort_curve_frame(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    if ordered.empty:
        return ordered

    ordered = ordered.sort_values("param_value", key=lambda s: s.map(_sort_key))
    return ordered.reset_index(drop=True)


def _sort_key(value):
    if pd.isna(value):
        return (2, "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    text = str(value)
    try:
        return (0, float(text))
    except (TypeError, ValueError):
        return (1, text)


def _try_numeric_list(values: Sequence) -> Optional[List[float]]:
    result: List[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            return None
    return result


def _sorted_values(values: Sequence[str], kind: str = "generic") -> List[str]:
    if kind == "dataset":
        order = {
            "ml-1m": 0,
            "lastfm-1k": 1,
            "taobao": 2,
        }
    elif kind == "backbone":
        order = {
            "sasrec": 0,
            "bert4rec": 1,
            "gru4rec": 2,
            "caser": 3,
        }
    else:
        order = {}

    return sorted(values, key=lambda x: (order.get(x, len(order)), str(x)))

def _dataset_marker(dataset: str) -> str:
    return {
        "ml-1m": "^",
        "taobao": "s",
        "lastfm-1k": "D",
    }.get(dataset, "o")


def _display(value: str, kind: str = "generic") -> str:
    if kind == "dataset":
        return {
            "ml-1m": "ML-1M",
            "lastfm-1k": "LastFM-1K",
            "taobao": "Taobao",
        }.get(value, value)

    if kind == "backbone":
        return {
            "sasrec": "SASRec",
            "bert4rec": "BERT4Rec",
            "gru4rec": "GRU4Rec",
            "caser": "Caser",
        }.get(value, value)

    if kind == "param":
        return {
            "learning_rate": "Learning Rate",
            "hidden_units": "Hidden Units",
            "dropout_rate": "Dropout Rate",
            "epsilon": r"$\epsilon$",
            "augment_ratio": "Augment Ratio",
            "utility_alpha": r"$\alpha$",
            "utility_beta": r"$\beta$",
            "fair_ncl_align_weight": r"$\lambda_{align}$",
            "fair_ncl_var_weight": r"$\lambda_{var}$",
            "fair_ncl_cov_weight": r"$\lambda_{cov}$",
        }.get(value, value.replace("_", " ").title())

    return value


def _display_phase(phase: str) -> str:
    return {
        "backbone": "Backbone",
        "augment": "Augmentation",
        "loss": "Fair-NCL Loss",
    }.get(phase, phase.replace("_", " ").title())


def _display_metric(metric_name: str) -> str:
    return {
        "HitRate@5": "HitRate@5",
        "NDCG@5": "NDCG@5",
        "HitRate@10": "HitRate@10",
        "NDCG@10": "NDCG@10",
        "HitRate@20": "HitRate@20",
        "NDCG@20": "NDCG@20",

        "gender_HitRate@5_Gap": "Gender HR@5 Gap",
        "gender_NDCG@5_Gap": "Gender NDCG@5 Gap",
        "age_group_HitRate@5_Gap": "Age HR@5 Gap",
        "age_group_NDCG@5_Gap": "Age NDCG@5 Gap",

        "gender_HitRate@10_Gap": "Gender HR@10 Gap",
        "gender_NDCG@10_Gap": "Gender NDCG@10 Gap",
        "age_group_HitRate@10_Gap": "Age HR@10 Gap",
        "age_group_NDCG@10_Gap": "Age NDCG@10 Gap",

        "gender_HitRate@20_Gap": "Gender HR@20 Gap",
        "gender_NDCG@20_Gap": "Gender NDCG@20 Gap",
        "age_group_HitRate@20_Gap": "Age HR@20 Gap",
        "age_group_NDCG@20_Gap": "Age NDCG@20 Gap",
    }.get(metric_name, metric_name)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return cleaned.strip("_") or "plot"


def _deduplicate_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

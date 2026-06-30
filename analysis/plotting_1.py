from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def export_parameter_plots(jobs_path: str,
                           results_root: str,
                           output_dir: str,
                           phase: str,
                           datasets: Optional[Sequence[str]] = None,
                           backbones: Optional[Sequence[str]] = None,
                           metrics: Optional[Sequence[str]] = None,
                           file_format: str = "png") -> Dict[str, pd.DataFrame]:
    """Export parameter-sensitivity plots and the underlying curve data."""
    if phase not in PHASE_PARAMETER_MAP:
        raise ValueError(f"Unsupported plotting phase '{phase}'.")

    if metrics is None:
        metrics = PRIMARY_REPORT_METRICS

    records = load_job_records(jobs_path, results_root)
    records = filter_records(records, phase=phase, datasets=datasets, backbones=backbones)
    if records.empty:
        raise ValueError("No matching jobs were found for parameter plotting.")

    long_df = build_parameter_curve_frame(records, phase=phase, metrics=metrics)
    if long_df.empty:
        raise ValueError("No usable parameter curve data were found.")

    summary_df = aggregate_parameter_curve_frame(long_df)

    phase_dir = os.path.join(output_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)
    long_csv = os.path.join(phase_dir, f"{phase}_parameter_curve_raw.csv")
    summary_csv = os.path.join(phase_dir, f"{phase}_parameter_curve_summary.csv")
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    figure_paths: List[str] = []
    parameter_names = list(PHASE_PARAMETER_MAP[phase])
    combos = summary_df[["dataset", "backbone"]].drop_duplicates().itertuples(index=False, name=None)

    for dataset, backbone in combos:
        combo_df = summary_df[
            (summary_df["dataset"] == dataset) &
            (summary_df["backbone"] == backbone)
        ].copy()
        if combo_df.empty:
            continue

        for metric in metrics:
            metric_df = combo_df[combo_df["metric_name"] == metric].copy()
            if metric_df.empty:
                continue

            fig, axes = plt.subplots(
                1,
                len(parameter_names),
                figsize=(5.0 * len(parameter_names), 4.2),
                sharey=True,
                constrained_layout=True,
            )
            if len(parameter_names) == 1:
                axes = [axes]

            for ax, param_name in zip(axes, parameter_names):
                param_df = metric_df[metric_df["param_name"] == param_name].copy()
                if param_df.empty:
                    ax.set_axis_off()
                    continue
                _plot_parameter_curve(ax, param_df, param_name, metric)

            fig.suptitle(
                f"{_display(dataset, 'dataset')} / {_display(backbone, 'backbone')} / {_display_metric(metric)}",
                fontsize=12,
            )
            fig_path = os.path.join(
                phase_dir,
                _sheet_name(dataset),
                _sheet_name(backbone),
            )
            os.makedirs(fig_path, exist_ok=True)
            file_name = f"{phase}_{_safe_name(dataset)}_{_safe_name(backbone)}_{_safe_name(metric)}.{file_format}"
            full_path = os.path.join(fig_path, file_name)
            fig.savefig(full_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            figure_paths.append(full_path)

    return {
        "raw": long_df,
        "summary": summary_df,
        "figure_paths": pd.DataFrame({"figure_path": figure_paths}),
    }


def _plot_parameter_curve(ax: plt.Axes,
                          param_df: pd.DataFrame,
                          param_name: str,
                          metric_name: str) -> None:
    ordered = _sort_curve_frame(param_df)
    if ordered.empty:
        ax.set_axis_off()
        return

    x_values = ordered["param_value"].tolist()
    y_values = ordered["mean"].tolist()
    y_errors = ordered["std"].fillna(0.0).tolist()

    ax.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#1f77b4",
    )
    if param_name == "learning_rate":
        ax.set_xscale("log")
    ax.set_title(_display(param_name, "param"), fontsize=11)
    ax.set_xlabel(_display(param_name, "param"))
    ax.set_ylabel(_display_metric(metric_name))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)


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
            "epsilon": "Epsilon",
            "augment_ratio": "Augment Ratio",
            "utility_alpha": "Utility Alpha",
            "fair_ncl_align_weight": "Align Weight",
            "fair_ncl_var_weight": "Var Weight",
            "fair_ncl_cov_weight": "Cov Weight",
        }.get(value, value.replace("_", " ").title())
    return value


def _display_metric(metric_name: str) -> str:
    return {
        "HitRate@10": "HitRate@10",
        "NDCG@10": "NDCG@10",
        "gender_HitRate@10_Gap": "Gender HitRate@10 Gap",
        "gender_NDCG@10_Gap": "Gender NDCG@10 Gap",
        "age_group_HitRate@10_Gap": "Age Group HitRate@10 Gap",
        "age_group_NDCG@10_Gap": "Age Group NDCG@10 Gap",
    }.get(metric_name, metric_name)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return cleaned.strip("_") or "plot"

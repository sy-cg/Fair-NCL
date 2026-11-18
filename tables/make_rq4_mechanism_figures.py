from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


# -----------------------------------------------------------------------------
# Nature-style visual configuration
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.transparent": False,
})


DATASET_ORDER = ["ML-1M", "Taobao"]#, "LastFM-1K"
BACKBONE_ORDER = ["SASRec", "BERT4Rec", "GRU4Rec", "Caser"]

DATASET_MARKERS = {
    "ML-1M": "^",
    "Taobao": "s",
    #"LastFM-1K": "D",
}

BACKBONE_COLORS = {
    "SASRec": "#0072B2",
    "BERT4Rec": "#D55E00",
    "GRU4Rec": "#009E73",
    "Caser": "#CC79A7",
}

REQUIRED_COLUMNS = [
    "Dataset",
    "Backbone",
    "Method",
    "Raw->Fair Cosine",
    "Gender Probe BA Raw",
    "Gender Probe BA Fair",
    "Gender Probe BA Drop",
    "Age Probe BA Raw",
    "Age Probe BA Fair",
    "Age Probe BA Drop",
    "Gender Sep Raw",
    "Gender Sep Fair",
    "Age Sep Raw",
    "Age Sep Fair",
]

NUMERIC_MECHANISM_COLUMNS = [
    "Raw->Fair Cosine",
    "Gender Probe BA Raw",
    "Gender Probe BA Fair",
    "Gender Probe BA Drop",
    "Age Probe BA Raw",
    "Age Probe BA Fair",
    "Age Probe BA Drop",
    "Gender Sep Raw",
    "Gender Sep Fair",
    "Age Sep Raw",
    "Age Sep Fair",
]


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Nature-style RQ4 mechanism figures and tables."
    )
    parser.add_argument(
        "--input",
        default="results/mechanism_comparison_ml_taobao_3seed.xlsx",
        help="Input CSV/TSV/XLSX mechanism table.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name. Defaults to rq4_paper_table for XLSX inputs.",
    )
    parser.add_argument("--output-dir", default="figures/rq4", help="Output directory.")
    parser.add_argument("--baseline-name", default="Baseline")
    parser.add_argument("--fair-name", default="Fair-NCL")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Seed count stated in generated table captions (for example, 5).",
    )

    parser.add_argument(
        "--summary-vlim",
        type=float,
        default=None,
        help="Symmetric color limit for main heatmaps. If omitted, inferred from data.",
    )
    parser.add_argument(
        "--attr-vlim",
        type=float,
        default=None,
        help="Symmetric color limit for attribute-specific heatmaps. If omitted, inferred from data.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input_table(args.input, sheet_name=args.sheet)
    validate_columns(df)

    numeric_df = add_numeric_columns(df)
    full_df = build_full_baseline_fair_table(
        numeric_df,
        baseline_name=args.baseline_name,
        fair_name=args.fair_name,
    )
    summary_df = build_rq4_summary(
        numeric_df,
        baseline_name=args.baseline_name,
        fair_name=args.fair_name,
    )

    full_csv = output_dir / "rq4_full_mechanism_baseline_fair.csv"
    full_tex = output_dir / "rq4_full_mechanism_baseline_fair.tex"
    summary_csv = output_dir / "rq4_mechanism_summary.csv"
    summary_tex = output_dir / "rq4_mechanism_summary.tex"

    full_df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    write_full_mechanism_latex(full_df, full_tex, seed_count=args.seed_count)
    write_summary_latex(summary_df, summary_tex, seed_count=args.seed_count)

    main_fig = output_dir / f"rq4_main_mechanism_figure.{args.format}"
    attr_fig = output_dir / f"rq4_attribute_specific_heatmaps.{args.format}"
    transform_fig = output_dir / f"rq4_transform_diagnostics.{args.format}"

    plot_main_mechanism_figure(
        summary_df=summary_df,
        output_path=main_fig,
        dpi=args.dpi,
        symmetric_vlim=args.summary_vlim,
    )
    plot_attribute_specific_heatmaps(
        summary_df=summary_df,
        output_path=attr_fig,
        dpi=args.dpi,
        symmetric_vlim=args.attr_vlim,
    )
    plot_transform_diagnostics(
        full_df=full_df,
        output_path=transform_fig,
        baseline_name=args.baseline_name,
        fair_name=args.fair_name,
        dpi=args.dpi,
    )

    print(f"Saved full baseline/Fair-NCL CSV to: {full_csv}")
    print(f"Saved full baseline/Fair-NCL LaTeX table to: {full_tex}")
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved summary LaTeX table to: {summary_tex}")
    print(f"Saved main RQ4 figure to: {main_fig}")
    print(f"Saved attribute-specific figure to: {attr_fig}")
    print(f"Saved transform diagnostic figure to: {transform_fig}")


# -----------------------------------------------------------------------------
# Data loading and parsing
# -----------------------------------------------------------------------------

def read_input_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        target_sheet = sheet_name or "rq4_paper_table"
        return pd.read_excel(path_obj, sheet_name=target_sheet)

    return pd.read_csv(path_obj, sep=None, engine="python")


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input table is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in NUMERIC_MECHANISM_COLUMNS:
        result[f"{col}__mean"] = result[col].apply(parse_mean_value)
        result[f"{col}__std"] = result[col].apply(parse_std_value)

    return result


def parse_mean_value(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = strip_latex_highlights(str(value).strip())
    for token in ("±", "+/-"):
        text = text.replace(token, r"$\pm$")
    split_token = find_pm_token(text)
    if split_token:
        text = text.split(split_token, 1)[0].strip()
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    elif "+/-" in text:
        text = text.split("+/-", 1)[0].strip()

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else np.nan


def parse_std_value(value) -> float:
    if pd.isna(value):
        return np.nan
    cleaned_text = strip_latex_highlights(str(value).strip())
    split_token = find_pm_token(cleaned_text)
    if split_token:
        right = cleaned_text.split(split_token, 1)[1].strip()
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
        return float(match.group(0)) if match else np.nan
    text = strip_latex_highlights(str(value).strip())

    if "±" in text:
        right = text.split("±", 1)[1].strip()
    elif "+/-" in text:
        right = text.split("+/-", 1)[1].strip()
    else:
        return np.nan

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
    return float(match.group(0)) if match else np.nan


def find_pm_token(text: str) -> Optional[str]:
    for token in (r"$\pm$", r"\pm", "±", "¡À", "+/-"):
        if token in text:
            return token
    return None


def strip_latex_highlights(text: str) -> str:
    previous = None
    result = text
    while previous != result:
        previous = result
        result = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", result)
        result = re.sub(r"\\underline\{([^{}]*)\}", r"\1", result)
    return result


# -----------------------------------------------------------------------------
# Summary construction
# -----------------------------------------------------------------------------

def build_full_baseline_fair_table(
    df: pd.DataFrame,
    baseline_name: str,
    fair_name: str,
) -> pd.DataFrame:
    keep = df[df["Method"].isin([baseline_name, fair_name])].copy()
    keep = apply_display_order(keep)
    return keep[REQUIRED_COLUMNS].reset_index(drop=True)


def build_rq4_summary(
    df: pd.DataFrame,
    baseline_name: str = "Baseline",
    fair_name: str = "Fair-NCL",
) -> pd.DataFrame:
    rows: List[Dict] = []

    for (dataset, backbone), group in df.groupby(["Dataset", "Backbone"], sort=False):
        base_rows = group[group["Method"] == baseline_name]
        fair_rows = group[group["Method"] == fair_name]

        if base_rows.empty or fair_rows.empty:
            print(f"[WARN] Skip {dataset} / {backbone}: missing {baseline_name} or {fair_name}.")
            continue

        base = base_rows.iloc[0]
        fair = fair_rows.iloc[0]

        base_gender_probe = base["Gender Probe BA Fair__mean"]
        base_age_probe = base["Age Probe BA Fair__mean"]
        fair_gender_probe = fair["Gender Probe BA Fair__mean"]
        fair_age_probe = fair["Age Probe BA Fair__mean"]

        base_gender_sep = base["Gender Sep Fair__mean"]
        base_age_sep = base["Age Sep Fair__mean"]
        fair_gender_sep = fair["Gender Sep Fair__mean"]
        fair_age_sep = fair["Age Sep Fair__mean"]

        base_avg_probe = np.nanmean([base_gender_probe, base_age_probe])
        fair_avg_probe = np.nanmean([fair_gender_probe, fair_age_probe])
        base_avg_sep = np.nanmean([base_gender_sep, base_age_sep])
        fair_avg_sep = np.nanmean([fair_gender_sep, fair_age_sep])

        row = {
            "dataset": dataset,
            "backbone": backbone,

            "baseline_avg_probe_ba": base_avg_probe,
            "fair_avg_probe_ba": fair_avg_probe,
            "avg_probe_reduction_pct": reduction_pct(base_avg_probe, fair_avg_probe),

            "baseline_gender_probe_ba": base_gender_probe,
            "fair_gender_probe_ba": fair_gender_probe,
            "gender_probe_reduction_pct": reduction_pct(base_gender_probe, fair_gender_probe),

            "baseline_age_probe_ba": base_age_probe,
            "fair_age_probe_ba": fair_age_probe,
            "age_probe_reduction_pct": reduction_pct(base_age_probe, fair_age_probe),

            "baseline_avg_sep": base_avg_sep,
            "fair_avg_sep": fair_avg_sep,
            "avg_sep_reduction_pct": reduction_pct(base_avg_sep, fair_avg_sep),

            "baseline_gender_sep": base_gender_sep,
            "fair_gender_sep": fair_gender_sep,
            "gender_sep_reduction_pct": reduction_pct(base_gender_sep, fair_gender_sep),

            "baseline_age_sep": base_age_sep,
            "fair_age_sep": fair_age_sep,
            "age_sep_reduction_pct": reduction_pct(base_age_sep, fair_age_sep),

            "fair_raw_to_fair_cosine": fair["Raw->Fair Cosine__mean"],
            "fair_gender_probe_ba_drop": fair["Gender Probe BA Drop__mean"],
            "fair_age_probe_ba_drop": fair["Age Probe BA Drop__mean"],
            "fair_avg_probe_ba_drop": np.nanmean([
                fair["Gender Probe BA Drop__mean"],
                fair["Age Probe BA Drop__mean"],
            ]),
        }

        improvement_flags = {
            "gender_probe": fair_gender_probe < base_gender_probe,
            "age_probe": fair_age_probe < base_age_probe,
            "gender_sep": fair_gender_sep < base_gender_sep,
            "age_sep": fair_age_sep < base_age_sep,
        }
        row["improved_probe_dims"] = int(improvement_flags["gender_probe"]) + int(improvement_flags["age_probe"])
        row["improved_sep_dims"] = int(improvement_flags["gender_sep"]) + int(improvement_flags["age_sep"])
        row["improved_probe_label"] = f"{row['improved_probe_dims']}/2"
        row["improved_sep_label"] = f"{row['improved_sep_dims']}/2"

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = apply_summary_order(summary)
    return summary


def reduction_pct(baseline_value: float, fair_value: float) -> float:
    if pd.isna(baseline_value) or pd.isna(fair_value) or baseline_value == 0:
        return np.nan
    return (baseline_value - fair_value) / baseline_value * 100.0


def apply_summary_order(summary: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    backbone_order = {name: idx for idx, name in enumerate(BACKBONE_ORDER)}

    result = summary.copy()
    result["_dataset_rank"] = result["dataset"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    result["_backbone_rank"] = result["backbone"].map(backbone_order).fillna(len(backbone_order)).astype(int)
    result = result.sort_values(["_dataset_rank", "_backbone_rank", "dataset", "backbone"])
    result = result.drop(columns=["_dataset_rank", "_backbone_rank"]).reset_index(drop=True)
    return result


def apply_display_order(df: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    backbone_order = {name: idx for idx, name in enumerate(BACKBONE_ORDER)}
    method_order = {"Baseline": 0, "Fair-NCL": 1}

    result = df.copy()
    result["_dataset_rank"] = result["Dataset"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    result["_backbone_rank"] = result["Backbone"].map(backbone_order).fillna(len(backbone_order)).astype(int)
    result["_method_rank"] = result["Method"].map(method_order).fillna(len(method_order)).astype(int)
    result = result.sort_values(["_dataset_rank", "_backbone_rank", "_method_rank", "Dataset", "Backbone", "Method"])
    return result.drop(columns=["_dataset_rank", "_backbone_rank", "_method_rank"])


# -----------------------------------------------------------------------------
# LaTeX tables
# -----------------------------------------------------------------------------

def write_summary_latex(summary: pd.DataFrame,
                        output_path: Path,
                        seed_count: Optional[int] = None) -> None:
    seed_label = f"{seed_count}-seed" if seed_count is not None else "multi-seed"
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Representation-level mechanism summary for Fair-NCL. "
        r"ProbeRed and SepRed denote the relative reduction of average probe balanced accuracy "
        r"and average sensitive-group separability compared with the corresponding backbone baseline. "
        f"Values are computed from {seed_label} mean mechanism diagnostics.}}"
    )
    lines.append(r"\label{tab:rq4_mechanism_summary}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Backbone & ProbeRed & SepRed & Probe Dims & Sep Dims \\")
    lines.append(r"\midrule")

    current_dataset = None
    for _, row in summary.iterrows():
        if current_dataset is not None and row["dataset"] != current_dataset:
            lines.append(r"\midrule")
        current_dataset = row["dataset"]

        lines.append(
            f"{latex_escape(str(row['dataset']))} & "
            f"{latex_escape(str(row['backbone']))} & "
            f"{format_pct(row['avg_probe_reduction_pct'])} & "
            f"{format_pct(row['avg_sep_reduction_pct'])} & "
            f"{row['improved_probe_label']} & "
            f"{row['improved_sep_label']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_full_mechanism_latex(full_df: pd.DataFrame,
                               output_path: Path,
                               seed_count: Optional[int] = None) -> None:
    seed_label = f"{seed_count}-seed" if seed_count is not None else "multi-seed"
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Full mechanism metrics for Baseline and Fair-NCL. "
        f"Only the corresponding backbone baseline and Fair-NCL are retained, and cells report {seed_label} mean $\\pm$ std values.}}"
    )
    lines.append(r"\label{tab:rq4_full_mechanism_baseline_fair}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllrrrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Backbone & Method & Cosine & G-Probe Raw & G-Probe Fair & G-Probe Drop "
        r"& A-Probe Raw & A-Probe Fair & A-Probe Drop & G-Sep Raw & G-Sep Fair & A-Sep Raw & A-Sep Fair \\"
    )
    lines.append(r"\midrule")

    current_dataset = None
    for _, row in full_df.iterrows():
        if current_dataset is not None and row["Dataset"] != current_dataset:
            lines.append(r"\midrule")
        current_dataset = row["Dataset"]

        values = [
            latex_escape(str(row["Dataset"])),
            latex_escape(str(row["Backbone"])),
            latex_escape(str(row["Method"])),
            fmt_mechanism_cell(row["Raw->Fair Cosine"]),
            fmt_mechanism_cell(row["Gender Probe BA Raw"]),
            fmt_mechanism_cell(row["Gender Probe BA Fair"]),
            fmt_mechanism_cell(row["Gender Probe BA Drop"]),
            fmt_mechanism_cell(row["Age Probe BA Raw"]),
            fmt_mechanism_cell(row["Age Probe BA Fair"]),
            fmt_mechanism_cell(row["Age Probe BA Drop"]),
            fmt_mechanism_cell(row["Gender Sep Raw"]),
            fmt_mechanism_cell(row["Gender Sep Fair"]),
            fmt_mechanism_cell(row["Age Sep Raw"]),
            fmt_mechanism_cell(row["Age Sep Fair"]),
        ]
        lines.append(" & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def fmt_mechanism_cell(value) -> str:
    if pd.isna(value):
        return "--"
    text = strip_latex_highlights(str(value).strip())
    text = text.replace("+/-", r"$\pm$")
    text = text.replace(chr(177), r"$\pm$")
    text = text.replace(chr(161) + chr(192), r"$\pm$")
    text = text.replace("\xa1\xc0", r"$\pm$")
    text = text.replace("±", r"$\pm$")
    return latex_escape_preserve_math(text)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.2f}\\%"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def latex_escape_preserve_math(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# -----------------------------------------------------------------------------
# Main RQ4 figure
# -----------------------------------------------------------------------------

def plot_main_mechanism_figure(
    summary_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
    symmetric_vlim: Optional[float] = None,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 3.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]},
    )

    plot_heatmap_on_axis(
        ax=axes[0],
        summary=summary_df,
        value_col="avg_sep_reduction_pct",
        title="a  Group separability reduction",
        cbar_label="SepRed (%)",
        symmetric_vlim=symmetric_vlim,
    )

    plot_heatmap_on_axis(
        ax=axes[1],
        summary=summary_df,
        value_col="avg_probe_reduction_pct",
        title="b  Probe leakage reduction",
        cbar_label="ProbeRed (%)",
        symmetric_vlim=symmetric_vlim,
    )

    plot_improvement_count_panel(
        ax=axes[2],
        summary=summary_df,
        title="c  Reduction consistency",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_attribute_specific_heatmaps(
    summary_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
    symmetric_vlim: Optional[float] = None,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.8, 6.2),
        constrained_layout=True,
    )

    panels = [
        ("gender_sep_reduction_pct", "a  Gender separability reduction", "SepRed (%)"),
        ("age_sep_reduction_pct", "b  Age separability reduction", "SepRed (%)"),
        ("gender_probe_reduction_pct", "c  Gender probe reduction", "ProbeRed (%)"),
        ("age_probe_reduction_pct", "d  Age probe reduction", "ProbeRed (%)"),
    ]

    for ax, (value_col, title, cbar_label) in zip(axes.ravel(), panels):
        plot_heatmap_on_axis(
            ax=ax,
            summary=summary_df,
            value_col=value_col,
            title=title,
            cbar_label=cbar_label,
            symmetric_vlim=symmetric_vlim,
        )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_transform_diagnostics(
    full_df: pd.DataFrame,
    output_path: Path,
    baseline_name: str,
    fair_name: str,
    dpi: int = 300,
) -> None:
    df = add_numeric_columns(full_df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.4),
        constrained_layout=True,
    )

    plot_method_strip(
        ax=axes[0],
        df=df,
        value_col="Raw->Fair Cosine__mean",
        title="a  Raw-to-fair representation cosine",
        y_label="Cosine",
        baseline_name=baseline_name,
        fair_name=fair_name,
        y_ref=1.0,
    )

    df["Avg Probe BA Drop__mean"] = df[[
        "Gender Probe BA Drop__mean",
        "Age Probe BA Drop__mean",
    ]].mean(axis=1)

    plot_method_strip(
        ax=axes[1],
        df=df,
        value_col="Avg Probe BA Drop__mean",
        title="b  Raw-to-fair probe BA drop",
        y_label="Avg. probe BA drop",
        baseline_name=baseline_name,
        fair_name=fair_name,
        y_ref=0.0,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot primitives
# -----------------------------------------------------------------------------

def plot_heatmap_on_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    symmetric_vlim: Optional[float] = None,
    annotate_fmt: str = "{:.1f}",
) -> None:
    datasets = ordered_unique(summary["dataset"].tolist(), DATASET_ORDER)
    backbones = ordered_unique(summary["backbone"].tolist(), BACKBONE_ORDER)

    pivot = summary.pivot(index="dataset", columns="backbone", values=value_col)
    pivot = pivot.reindex(index=datasets, columns=backbones)
    values = pivot.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    if symmetric_vlim is None:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            vmax_abs = 1.0
        else:
            vmax_abs = max(abs(np.nanmin(finite_values)), abs(np.nanmax(finite_values)))
            vmax_abs = max(vmax_abs, 1e-6)
    else:
        vmax_abs = float(symmetric_vlim)

    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f2f2f2")

    im = ax.imshow(
        masked_values,
        norm=norm,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_title(title, loc="left")
    ax.set_xticks(np.arange(len(backbones)))
    ax.set_xticklabels(backbones, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    annotate_fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.5)


def plot_improvement_count_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
) -> None:
    total_settings = len(summary)

    counts = {
        "Gender\nSep": int((summary["gender_sep_reduction_pct"] > 0).sum()),
        "Age\nSep": int((summary["age_sep_reduction_pct"] > 0).sum()),
        "Gender\nProbe": int((summary["gender_probe_reduction_pct"] > 0).sum()),
        "Age\nProbe": int((summary["age_probe_reduction_pct"] > 0).sum()),
    }

    labels = list(counts.keys())
    values = list(counts.values())

    x = np.arange(len(labels))
    colors = ["#9ecae1", "#9ecae1", "#fdd0a2", "#fdd0a2"]

    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title(title, loc="left")
    ax.set_ylabel(f"Reduced settings out of {total_settings}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(total_settings, 1))

    for i, value in enumerate(values):
        ax.text(i, value + 0.15, f"{value}/{total_settings}", ha="center", va="bottom", fontsize=7)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(False)


def plot_method_strip(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    baseline_name: str,
    fair_name: str,
    y_ref: Optional[float] = None,
) -> None:
    methods = [baseline_name, fair_name]
    method_to_x = {method: idx for idx, method in enumerate(methods)}

    for _, row in df.iterrows():
        method = row["Method"]
        if method not in method_to_x:
            continue

        dataset = row["Dataset"]
        backbone = row["Backbone"]
        x = method_to_x[method]
        jitter = backbone_jitter(backbone)

        ax.scatter(
            x + jitter,
            row[value_col],
            marker=DATASET_MARKERS.get(dataset, "o"),
            s=42,
            color=BACKBONE_COLORS.get(backbone, "0.35"),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    if y_ref is not None:
        ax.axhline(y_ref, linestyle="--", linewidth=0.7, color="0.45", zorder=1)

    ax.set_title(title, loc="left")
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    datasets = ordered_unique(df["Dataset"].tolist(), DATASET_ORDER)
    backbones = ordered_unique(df["Backbone"].tolist(), BACKBONE_ORDER)

    dataset_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.25",
            marker=DATASET_MARKERS.get(dataset, "o"),
            linestyle="None",
            markersize=6,
            label=dataset,
        )
        for dataset in datasets
    ]

    backbone_handles = [
        mlines.Line2D(
            [],
            [],
            color=BACKBONE_COLORS.get(backbone, "0.35"),
            marker="o",
            linestyle="None",
            markersize=6,
            label=backbone,
        )
        for backbone in backbones
    ]

    legend1 = ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="best",
        frameon=False,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=backbone_handles,
        title="Backbone",
        loc="lower right",
        frameon=False,
    )


def backbone_jitter(backbone: str) -> float:
    mapping = {
        "SASRec": -0.18,
        "BERT4Rec": -0.06,
        "GRU4Rec": 0.06,
        "Caser": 0.18,
    }
    return mapping.get(backbone, 0.0)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ordered_unique(values: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    observed = list(dict.fromkeys(values))
    order = {name: idx for idx, name in enumerate(preferred_order)}
    return sorted(observed, key=lambda x: (order.get(x, len(order)), str(x)))


if __name__ == "__main__":
    main()

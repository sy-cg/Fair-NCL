from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

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

# Okabe-Ito inspired, colorblind-friendly palette.
BACKBONE_COLORS = {
    "SASRec": "#0072B2",
    "BERT4Rec": "#D55E00",
    "GRU4Rec": "#009E73",
    "Caser": "#CC79A7",
}

REQUIRED_COLUMNS = [
    "dataset_display",
    "backbone_display",
    "method_display",
    "selected__NDCG@10",
    "selected__NDCG@20",
    "selected__gender_NDCG@10_Gap",
    "selected__age_group_NDCG@10_Gap",
    "selected__gender_NDCG@20_Gap",
    "selected__age_group_NDCG@20_Gap",
]


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Nature-style RQ3 robustness figures and summary table."
    )
    parser.add_argument(
        "--input",
        default="results/comparison_ml_taobao_3seed.xlsx",
        help="Input CSV/TSV/XLSX file.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name. Defaults to summary_all_metrics for XLSX inputs.",
    )
    parser.add_argument("--output-dir", default="figures/rq3", help="Output directory.")
    parser.add_argument("--baseline-name", default="Baseline", help="Name of baseline method.")
    parser.add_argument("--fair-name", default="Fair-NCL", help="Name of proposed method.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    # Optional fixed symmetric limits. If omitted, limits are inferred per metric family.
    parser.add_argument("--utility-vlim", type=float, default=None,
                        help="Symmetric color limit for Delta NDCG heatmaps, e.g., 6.")
    parser.add_argument("--fairness-vlim", type=float, default=None,
                        help="Symmetric color limit for GapRed heatmaps, e.g., 130.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input_table(args.input, sheet_name=args.sheet)
    df = normalize_multiseed_columns(df)
    validate_columns(df)

    summary = build_rq3_summary(
        df=df,
        baseline_name=args.baseline_name,
        fair_name=args.fair_name,
    )

    summary_csv = output_dir / "rq3_robustness_summary.csv"
    summary_tex = output_dir / "rq3_robustness_summary.tex"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_latex_summary_table(summary, summary_tex)

    main_fig = output_dir / f"rq3_main_figure.{args.format}"
    extended_fig = output_dir / f"rq3_extended_20_heatmaps.{args.format}"

    plot_main_rq3_figure(
        summary=summary,
        output_path=main_fig,
        dpi=args.dpi,
        utility_vlim=args.utility_vlim,
        fairness_vlim=args.fairness_vlim,
    )

    plot_extended_20_heatmaps(
        summary=summary,
        output_path=extended_fig,
        dpi=args.dpi,
        utility_vlim=args.utility_vlim,
        fairness_vlim=args.fairness_vlim,
    )

    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved LaTeX table to: {summary_tex}")
    print(f"Saved main RQ3 figure to: {main_fig}")
    print(f"Saved extended @20 figure to: {extended_fig}")


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------

def read_input_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        target_sheet = sheet_name or "summary_all_metrics"
        return pd.read_excel(path_obj, sheet_name=target_sheet)

    # sep=None infers comma/tab separators.
    return pd.read_csv(path_obj, sep=None, engine="python")


def normalize_multiseed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Use three-seed mean columns when a process_multiseed_results workbook is provided."""
    result = df.copy()
    for col in REQUIRED_COLUMNS:
        mean_col = f"{col}__mean"
        if col.startswith("selected__") and col not in result.columns and mean_col in result.columns:
            result[col] = result[mean_col]
    return result


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input table is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


# -----------------------------------------------------------------------------
# RQ3 summary construction
# -----------------------------------------------------------------------------

def build_rq3_summary(
    df: pd.DataFrame,
    baseline_name: str = "Baseline",
    fair_name: str = "Fair-NCL",
) -> pd.DataFrame:
    work = df.copy()

    metric_cols = [col for col in work.columns if col.startswith("selected__")]
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rows: List[Dict] = []

    group_cols = ["dataset_display", "backbone_display"]
    for (dataset, backbone), group in work.groupby(group_cols, sort=False):
        baseline_rows = group[group["method_display"] == baseline_name]
        fair_rows = group[group["method_display"] == fair_name]

        if baseline_rows.empty or fair_rows.empty:
            print(
                f"[WARN] Skip {dataset} / {backbone}: "
                f"missing {baseline_name} or {fair_name}."
            )
            continue

        base = baseline_rows.iloc[0]
        fair = fair_rows.iloc[0]

        row: Dict = {
            "dataset": dataset,
            "backbone": backbone,
        }

        for k in [10, 20]:
            base_ndcg = base[f"selected__NDCG@{k}"]
            fair_ndcg = fair[f"selected__NDCG@{k}"]

            row[f"baseline_ndcg@{k}"] = base_ndcg
            row[f"fair_ndcg@{k}"] = fair_ndcg
            row[f"delta_ndcg@{k}_pct"] = relative_change_pct(fair_ndcg, base_ndcg)

            base_gender_gap = base[f"selected__gender_NDCG@{k}_Gap"]
            base_age_gap = base[f"selected__age_group_NDCG@{k}_Gap"]
            fair_gender_gap = fair[f"selected__gender_NDCG@{k}_Gap"]
            fair_age_gap = fair[f"selected__age_group_NDCG@{k}_Gap"]

            base_avg_gap = np.nanmean([base_gender_gap, base_age_gap])
            fair_avg_gap = np.nanmean([fair_gender_gap, fair_age_gap])

            row[f"baseline_gender_ndcg_gap@{k}"] = base_gender_gap
            row[f"fair_gender_ndcg_gap@{k}"] = fair_gender_gap
            row[f"baseline_age_ndcg_gap@{k}"] = base_age_gap
            row[f"fair_age_ndcg_gap@{k}"] = fair_age_gap
            row[f"baseline_avg_ndcg_gap@{k}"] = base_avg_gap
            row[f"fair_avg_ndcg_gap@{k}"] = fair_avg_gap
            row[f"avg_ndcg_gap_reduction@{k}_pct"] = gap_reduction_pct(
                fair_gap=fair_avg_gap,
                baseline_gap=base_avg_gap,
            )

        improved_gap_count = 0
        improved_gap_total = 0
        for col in [
            "selected__gender_NDCG@10_Gap",
            "selected__age_group_NDCG@10_Gap",
            "selected__gender_NDCG@20_Gap",
            "selected__age_group_NDCG@20_Gap",
        ]:
            base_val = base[col]
            fair_val = fair[col]
            if pd.notna(base_val) and pd.notna(fair_val):
                improved_gap_total += 1
                if fair_val < base_val:
                    improved_gap_count += 1

        row["improved_ndcg_gaps"] = improved_gap_count
        row["total_ndcg_gaps"] = improved_gap_total
        row["improved_ndcg_gaps_label"] = f"{improved_gap_count}/{improved_gap_total}"

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = apply_display_order(summary)
    return summary


def relative_change_pct(new_value: float, baseline_value: float) -> float:
    if pd.isna(new_value) or pd.isna(baseline_value) or baseline_value == 0:
        return np.nan
    return (new_value - baseline_value) / baseline_value * 100.0


def gap_reduction_pct(fair_gap: float, baseline_gap: float) -> float:
    if pd.isna(fair_gap) or pd.isna(baseline_gap) or baseline_gap == 0:
        return np.nan
    return (baseline_gap - fair_gap) / baseline_gap * 100.0


def apply_display_order(summary: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    backbone_order = {name: idx for idx, name in enumerate(BACKBONE_ORDER)}

    result = summary.copy()
    result["_dataset_rank"] = result["dataset"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    result["_backbone_rank"] = result["backbone"].map(backbone_order).fillna(len(backbone_order)).astype(int)
    result = result.sort_values(["_dataset_rank", "_backbone_rank", "dataset", "backbone"])
    result = result.drop(columns=["_dataset_rank", "_backbone_rank"]).reset_index(drop=True)
    return result


# -----------------------------------------------------------------------------
# LaTeX summary table
# -----------------------------------------------------------------------------

def write_latex_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    lines: List[str] = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Robustness summary of Fair-NCL across datasets and backbones. "
        r"Utility change is measured by relative NDCG change over the corresponding backbone baseline, computed from three-seed mean values. "
        r"Fairness improvement is measured by the relative reduction of the average NDCG-based group gap.}"
    )
    lines.append(r"\label{tab:rq3_robustness_summary}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Backbone & $\Delta$NDCG@10 & $\Delta$NDCG@20 "
        r"& GapRed@10 & GapRed@20 & Improved Gaps \\"
    )
    lines.append(r"\midrule")

    current_dataset = None
    for _, row in summary.iterrows():
        if current_dataset is not None and row["dataset"] != current_dataset:
            lines.append(r"\midrule")
        current_dataset = row["dataset"]

        dataset = latex_escape(str(row["dataset"]))
        backbone = latex_escape(str(row["backbone"]))

        lines.append(
            f"{dataset} & {backbone} & "
            f"{format_pct(row['delta_ndcg@10_pct'])} & "
            f"{format_pct(row['delta_ndcg@20_pct'])} & "
            f"{format_pct(row['avg_ndcg_gap_reduction@10_pct'])} & "
            f"{format_pct(row['avg_ndcg_gap_reduction@20_pct'])} & "
            f"{row['improved_ndcg_gaps_label']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


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


# -----------------------------------------------------------------------------
# Main and extended figures
# -----------------------------------------------------------------------------

def plot_main_rq3_figure(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
    utility_vlim: float | None = None,
    fairness_vlim: float | None = None,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 3.6),
        constrained_layout=True,
    )

    plot_tradeoff_scatter(
        ax=axes[0],
        summary=summary,
        x_col="delta_ndcg@10_pct",
        y_col="avg_ndcg_gap_reduction@10_pct",
        title="a  Utility--fairness trade-off",
        x_label=r"$\Delta$NDCG@10 (%)",
        y_label="Avg. NDCG gap reduction@10 (%)",
    )

    plot_heatmap_on_axis(
        ax=axes[1],
        summary=summary,
        value_col="avg_ndcg_gap_reduction@10_pct",
        title="b  Fairness robustness",
        cbar_label="Gap reduction (%)",
        annotate_fmt="{:.1f}",
        symmetric_vlim=fairness_vlim,
    )

    plot_heatmap_on_axis(
        ax=axes[2],
        summary=summary,
        value_col="delta_ndcg@10_pct",
        title="c  Utility change",
        cbar_label=r"$\Delta$NDCG (%)",
        annotate_fmt="{:.1f}",
        symmetric_vlim=utility_vlim,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_extended_20_heatmaps(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
    utility_vlim: float | None = None,
    fairness_vlim: float | None = None,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.6),
        constrained_layout=True,
    )

    plot_heatmap_on_axis(
        ax=axes[0],
        summary=summary,
        value_col="avg_ndcg_gap_reduction@20_pct",
        title="a  Avg. NDCG gap reduction@20",
        cbar_label="Gap reduction (%)",
        annotate_fmt="{:.1f}",
        symmetric_vlim=fairness_vlim,
    )

    plot_heatmap_on_axis(
        ax=axes[1],
        summary=summary,
        value_col="delta_ndcg@20_pct",
        title=r"b  $\Delta$NDCG@20",
        cbar_label=r"$\Delta$NDCG (%)",
        annotate_fmt="{:.1f}",
        symmetric_vlim=utility_vlim,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel A: trade-off scatter
# -----------------------------------------------------------------------------

def plot_tradeoff_scatter(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    datasets = ordered_unique(summary["dataset"].tolist(), DATASET_ORDER)
    backbones = ordered_unique(summary["backbone"].tolist(), BACKBONE_ORDER)

    backbone_to_color = {
        backbone: BACKBONE_COLORS.get(backbone, "0.35")
        for backbone in backbones
    }

    for _, row in summary.iterrows():
        dataset = row["dataset"]
        backbone = row["backbone"]

        ax.scatter(
            row[x_col],
            row[y_col],
            marker=DATASET_MARKERS.get(dataset, "o"),
            s=58,
            color=backbone_to_color.get(backbone, "0.35"),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.axhline(0, linestyle="--", linewidth=0.7, color="0.35", zorder=1)
    ax.axvline(0, linestyle="--", linewidth=0.7, color="0.35", zorder=1)
    ax.grid(False)

    ax.set_title(title, loc="left")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

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
            color=backbone_to_color.get(backbone, "0.35"),
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
        loc="upper right",
        frameon=False,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=backbone_handles,
        title="Backbone",
        loc="lower right",
        frameon=False,
    )


# -----------------------------------------------------------------------------
# Panels B/C: heatmaps
# -----------------------------------------------------------------------------

def plot_heatmap_on_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    annotate_fmt: str = "{:.1f}",
    symmetric_vlim: float | None = None,
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ordered_unique(values: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    observed = list(dict.fromkeys(values))
    order = {name: idx for idx, name in enumerate(preferred_order)}
    return sorted(observed, key=lambda x: (order.get(x, len(order)), str(x)))


if __name__ == "__main__":
    main()

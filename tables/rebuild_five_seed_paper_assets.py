"""Rebuild manuscript result tables and RQ3 figures from the final five-seed CSVs."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from make_rq3_rq4_multiseed_figures import (
    write_rq3_extended_figure,
    write_rq3_main_figure,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tables" / "supplementary" / "final_5seed_extended"
PAPER = ROOT / "els-cas-templates" / "cas-dc-sample.tex"
FIGURES = ROOT / "figures" / "rq3"
METHODS = ("baseline", "adv_debias", "grl", "sm_pcfr", "a_fsr", "afrl", "pfrec", "fair_ncl")
ABLATIONS = (
    ("baseline", "Baseline"),
    ("similarity_aug", "Similarity-only replacement"),
    ("skew_only_aug", "Skew-only replacement"),
    ("random_aug", "Random replacement"),
    ("ncl_only", "Geometry without augmentation"),
    ("fair_aug_no_geometry", "w/o geometric losses"),
    ("fair_aug_no_augmented_ce", "w/o augmented-view CE"),
    ("fair_ncl", r"\textbf{Fair-NCL}"),
)
DATASETS = ("ml-1m", "taobao")
BACKBONES = ("sasrec", "bert4rec", "gru4rec", "caser")
DISPLAY = {"sasrec": "SASRec", "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec", "caser": "Caser"}
METHOD_DISPLAY = {
    "baseline": "", "adv_debias": "Adv-Debias+", "grl": "GRL+", "sm_pcfr": "PCFR-SM+",
    "a_fsr": "A-FSR+", "afrl": "AFRL+", "pfrec": "PFRec+", "fair_ncl": "Fair-NCL+",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


RUN = {(r["dataset"], r["backbone"], r["method"]): r for r in read_csv("run_summary.csv")}
GAP = {(r["dataset"], r["backbone"], r["method"], r["attribute"], r["metric"]): r
       for r in read_csv("group_summary.csv")}


def cell(dataset: str, backbone: str, method: str, metric: str) -> tuple[float, float]:
    if metric.startswith("G-") or metric.startswith("A-"):
        attribute = "gender" if metric[0] == "G" else "age_group"
        kind = "HitRate" if "HR" in metric else "NDCG"
        cutoff = metric.rsplit("@", 1)[1]
        row = GAP[(dataset, backbone, method, attribute, f"{kind}@{cutoff}")]
        return float(row["absolute_gap_mean"]), float(row["absolute_gap_std"])
    kind = "HitRate" if metric.startswith("HR") else "NDCG"
    cutoff = metric.rsplit("@", 1)[1]
    row = RUN[(dataset, backbone, method)]
    return float(row[f"{kind}@{cutoff}_mean"]), float(row[f"{kind}@{cutoff}_std"])


def format_cell(value: tuple[float, float]) -> str:
    return f"{value[0]:.4f}$\\pm${value[1]:.4f}"


def table(label: str, caption: str, body: list[str]) -> str:
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{" + caption + "}", r"\label{" + label + "}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Dataset & Method & HR & NDCG & G-HR Gap & G-NDCG Gap & A-HR Gap & A-NDCG Gap \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}",
    ])


def main_table(cutoff: int, dataset: str) -> str:
    metrics = tuple(f"{name}@{cutoff}" for name in ("HR", "NDCG", "G-HR", "G-NDCG", "A-HR", "A-NDCG"))
    body: list[str] = []
    for b_index, backbone in enumerate(BACKBONES):
        if b_index:
            body.append(r"\cmidrule(lr){2-8}")
        for m_index, method in enumerate(METHODS):
            if b_index == m_index == 0:
                prefix = rf"\multirow{{{len(BACKBONES) * len(METHODS)}}}{{*}}{{{'ML-1M' if dataset == 'ml-1m' else 'Taobao'}}}"
            else:
                prefix = ""
            method_name = METHOD_DISPLAY[method] + DISPLAY[backbone]
            if method == "fair_ncl":
                method_name = r"\textbf{" + method_name + "}"
            values = [format_cell(cell(dataset, backbone, method, metric)) for metric in metrics]
            body.append(" & ".join([prefix, method_name, *values]) + r" \\")
    label = f"tab:comparison_{cutoff}" + ("_taobao" if dataset == "taobao" else "")
    return table(label,
                 f"Main comparison results for {'ML-1M' if dataset == 'ml-1m' else 'Taobao'} at @{cutoff}. "
                 "Values are mean $\\pm$ std over five random seeds. "
                 "HR/NDCG are higher-is-better and absolute group gaps are lower-is-better. "
                 "Mean ranks are not highlighted as statistically supported differences.", body)


def ablation_table() -> str:
    metrics = ("HR@10", "NDCG@10", "G-HR@10", "G-NDCG@10", "A-HR@10", "A-NDCG@10")
    body = []
    for dataset in DATASETS:
        if body:
            body.append(r"\midrule")
        for index, (method, label) in enumerate(ABLATIONS):
            prefix = rf"\multirow{{{len(ABLATIONS)}}}{{*}}{{{'ML-1M' if dataset == 'ml-1m' else 'Taobao'}}}" if index == 0 else ""
            values = [format_cell(cell(dataset, "sasrec", method, metric)) for metric in metrics]
            body.append(" & ".join([prefix, label, *values]) + r" \\")
    return table("tab:ablation_sasrec_10",
                 "Five-seed SASRec component controls at @10 under the prevalence-adjusted skew definition. "
                 "All cells report mean $\\pm$ std; smaller absolute group gaps indicate better fairness.", body)


def replace_table(tex: str, label: str, replacement: str) -> str:
    marker = r"\label{" + label + "}"
    loc = tex.index(marker)
    start = tex.rfind(r"\begin{table*}[t]", 0, loc)
    end = tex.index(r"\end{table*}", loc) + len(r"\end{table*}")
    assert start >= 0
    if label.startswith("tab:comparison_") and (marker.replace("}", "_taobao}")) in tex[end:]:
        second_loc = tex.index(marker.replace("}", "_taobao}"), end)
        end = tex.index(r"\end{table*}", second_loc) + len(r"\end{table*}")
    return tex[:start] + replacement + tex[end:]


def rq3_summary() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            row: dict[str, object] = {"dataset": "ML-1M" if dataset == "ml-1m" else "Taobao",
                                      "backbone": DISPLAY[backbone]}
            improved = 0
            for cutoff in (10, 20):
                base = cell(dataset, backbone, "baseline", f"NDCG@{cutoff}")[0]
                fair = cell(dataset, backbone, "fair_ncl", f"NDCG@{cutoff}")[0]
                row[f"baseline_ndcg@{cutoff}"] = base
                row[f"fair_ndcg@{cutoff}"] = fair
                row[f"delta_ndcg@{cutoff}_pct"] = 100 * (fair - base) / base
                means = []
                for method in ("baseline", "fair_ncl"):
                    gender = cell(dataset, backbone, method, f"G-NDCG@{cutoff}")[0]
                    age = cell(dataset, backbone, method, f"A-NDCG@{cutoff}")[0]
                    row[f"{'baseline' if method == 'baseline' else 'fair'}_gender_ndcg_gap@{cutoff}"] = gender
                    row[f"{'baseline' if method == 'baseline' else 'fair'}_age_ndcg_gap@{cutoff}"] = age
                    means.append((gender + age) / 2)
                row[f"baseline_avg_ndcg_gap@{cutoff}"] = means[0]
                row[f"fair_avg_ndcg_gap@{cutoff}"] = means[1]
                row[f"avg_ndcg_gap_reduction@{cutoff}_pct"] = 100 * (means[0] - means[1]) / means[0]
                improved += sum(
                    cell(dataset, backbone, "fair_ncl", f"{attribute}-NDCG@{cutoff}")[0] <
                    cell(dataset, backbone, "baseline", f"{attribute}-NDCG@{cutoff}")[0]
                    for attribute in ("G", "A")
                )
            row["improved_ndcg_gaps"] = improved
            row["total_ndcg_gaps"] = 4
            row["improved_ndcg_gaps_label"] = f"{improved}/4"
            result.append(row)
    return result


def rq3_table(summary: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Five-seed robustness summary. Utility change is relative NDCG change over the matched backbone; GapRed is the relative reduction in the average absolute gender and age NDCG gaps. Improved Gaps counts lower attribute-specific NDCG gaps at @10 and @20. Percentages use unrounded five-seed means.}",
        r"\label{tab:rq3_robustness_summary}", r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Dataset & Backbone & $\Delta$NDCG@10 & $\Delta$NDCG@20 & GapRed@10 & GapRed@20 & Improved Gaps \\",
        r"\midrule",
    ]
    for index, row in enumerate(summary):
        if index == 4:
            lines.append(r"\midrule")
        dataset = r"\multirow{4}{*}{ML-1M}" if index == 0 else (r"\multirow{4}{*}{Taobao}" if index == 4 else "")
        values = [f"{float(row[key]):.2f}\\%" for key in (
            "delta_ndcg@10_pct", "delta_ndcg@20_pct",
            "avg_ndcg_gap_reduction@10_pct", "avg_ndcg_gap_reduction@20_pct")]
        lines.append(" & ".join([dataset, str(row["backbone"]), *values,
                                  str(row["improved_ndcg_gaps_label"])]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    assert all(int(row["seed_count"]) == 5 for row in RUN.values())
    assert all(int(row["seed_count"]) == 5 for row in GAP.values())
    tex = PAPER.read_text(encoding="utf-8")
    for cutoff in (10, 20):
        two_tables = main_table(cutoff, "ml-1m") + "\n\n" + main_table(cutoff, "taobao")
        tex = replace_table(tex, f"tab:comparison_{cutoff}", two_tables)
    tex = replace_table(tex, "tab:ablation_sasrec_10", ablation_table())
    summary = rq3_summary()
    marker = r"\label{tab:rq3_robustness_summary}"
    position = tex.index(marker)
    start = tex.rfind(r"\begin{table}[t]", 0, position)
    end = tex.index(r"\end{table}", position) + len(r"\end{table}")
    tex = tex[:start] + rq3_table(summary) + tex[end:]
    PAPER.write_text(tex, encoding="utf-8")

    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "rq3_robustness_summary.tex").write_text(rq3_table(summary) + "\n", encoding="utf-8")
    with (FIGURES / "rq3_robustness_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for name, writer in (("rq3_main_figure", write_rq3_main_figure),
                         ("rq3_extended_20_heatmaps", write_rq3_extended_figure)):
        tex_path = FIGURES / f"{name}.tex"
        writer(summary, tex_path)
        subprocess.run([r"F:\texlive\2024\bin\windows\pdflatex.exe", "-interaction=nonstopmode",
                        "-halt-on-error", tex_path.name], cwd=FIGURES, check=True,
                       stdout=subprocess.DEVNULL)
        shutil.copy2(FIGURES / f"{name}.pdf", ROOT / "els-cas-templates" / f"{name}.pdf")
        for suffix in (".aux", ".log"):
            (FIGURES / f"{name}{suffix}").unlink(missing_ok=True)
    print("Rebuilt RQ1/RQ2 tables and RQ3 figures from complete five-seed summaries.")
    for row in summary:
        print(row["dataset"], row["backbone"],
              f"dNDCG10={row['delta_ndcg@10_pct']:.2f}%",
              f"gapred10={row['avg_ndcg_gap_reduction@10_pct']:.2f}%",
              f"dNDCG20={row['delta_ndcg@20_pct']:.2f}%",
              f"gapred20={row['avg_ndcg_gap_reduction@20_pct']:.2f}%",
              row["improved_ndcg_gaps_label"])


if __name__ == "__main__":
    main()

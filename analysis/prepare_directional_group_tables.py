"""Prepare reviewer-facing group utilities and LaTeX tables from five-seed results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tables" / "supplementary" / "final_5seed"
TEX = ROOT / "els-cas-templates"
DATASETS = ("ml-1m", "taobao")
BACKBONES = ("sasrec", "bert4rec", "gru4rec", "caser")
METRICS = ("HitRate@10", "NDCG@10", "HitRate@20", "NDCG@20")
ATTRIBUTES = ("gender", "age_group", "gender_age")
GROUPS = {"gender": ("0", "1"), "age_group": ("0", "1"),
          "gender_age": ("g0_a0", "g0_a1", "g1_a0", "g1_a1")}
NAMES = {"sasrec": "SASRec", "bert4rec": "BERT4Rec",
         "gru4rec": "GRU4Rec", "caser": "Caser"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scaled(value: float, signed: bool = False) -> str:
    number = value * 1000
    precision = 3 if number != 0 and abs(number) < 0.005 else 2
    return f"{number:+.{precision}f}" if signed else f"{number:.{precision}f}"


def pair(before: float, after: float, signed: bool = False) -> str:
    return f"{scaled(before, signed)}/{scaled(after, signed)}"


def latex_table(label: str, caption: str, columns: str, header: str,
                rows: list[str]) -> str:
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *[row + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])


def main() -> None:
    summaries = {
        (r["dataset"], r["backbone"], r["method"], r["attribute"], r["metric"]): r
        for r in read_csv(SOURCE / "group_summary.csv")
        if r["method"] in {"baseline", "fair_ncl"}
    }
    comparisons = {
        (r["dataset"], r["backbone"], r["group_attribute"], r["metric"], r["group_value"]): r
        for r in read_csv(SOURCE / "paired_comparisons.csv")
        if r["method"] == "fair_ncl" and r["estimand"] == "group_mean"
    }
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for attribute in ATTRIBUTES:
                for metric in METRICS:
                    before = summaries[(dataset, backbone, "baseline", attribute, metric)]
                    after = summaries[(dataset, backbone, "fair_ncl", attribute, metric)]
                    before_values = json.loads(before["group_values"])
                    after_values = json.loads(after["group_values"])
                    counts = json.loads(before["group_counts"])
                    assert set(before_values) == set(after_values) == set(counts) == set(GROUPS[attribute])
                    assert counts == json.loads(after["group_counts"])
                    for group in GROUPS[attribute]:
                        comparison = comparisons[(dataset, backbone, attribute, metric, group)]
                        b, f = float(before_values[group]), float(after_values[group])
                        assert abs(b - float(comparison["baseline_estimate"])) < 1e-10
                        assert abs(f - float(comparison["method_estimate"])) < 1e-10
                        assert abs((f - b) - float(comparison["mean_delta"])) < 1e-10
                        rows.append({
                            "dataset": dataset, "backbone": backbone, "attribute": attribute,
                            "metric": metric, "group": group, "users": int(float(counts[group])),
                            "baseline": b, "fair_ncl": f, "delta": float(comparison["mean_delta"]),
                            "ci_lower": float(comparison["ci_lower"]),
                            "ci_upper": float(comparison["ci_upper"]),
                            "holm_p": float(comparison["holm_p_value"]),
                        })
    output = SOURCE / "directional_group_reporting.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    indexed = {(r["dataset"], r["backbone"], r["attribute"], r["metric"], r["group"]): r
               for r in rows}

    main_rows = []
    for backbone in BACKBONES:
        for metric in METRICS[:2]:
            group0 = indexed[("taobao", backbone, "gender", metric, "0")]
            group1 = indexed[("taobao", backbone, "gender", metric, "1")]
            before = summaries[("taobao", backbone, "baseline", "gender", metric)]
            after = summaries[("taobao", backbone, "fair_ncl", "gender", metric)]
            ci = lambda r: (f"{scaled(r['delta'], True)} "
                            f"[{scaled(r['ci_lower'], True)},{scaled(r['ci_upper'], True)}]")
            main_rows.append(" & ".join([
                NAMES[backbone],
                "HR" if metric.startswith("HitRate") else "NDCG",
                pair(group0["baseline"], group0["fair_ncl"]), f"${ci(group0)}$",
                pair(group1["baseline"], group1["fair_ncl"]), f"${ci(group1)}$",
                pair(float(before["signed_gap_mean"]), float(after["signed_gap_mean"]), True),
                pair(float(before["absolute_gap_mean"]), float(after["absolute_gap_mean"])),
            ]))
    main_caption = (r"Taobao gender-group HR@10 and NDCG@10 before (B, plain backbone) and after "
                    r"(F, Fair-NCL), in units of $10^{-3}$. Group 0 has 4,578 users (minority); "
                    r"group 1 has 26,755. Brackets are 95\% paired hierarchical-bootstrap intervals "
                    r"for F$-$B. $D=M_0-M_1$; $|D|$ averages the seed-level absolute gap. "
                    r"None of the group-mean changes survives the five-seed Holm-adjusted exact test.")
    (TEX / "group_directional_main.tex").write_text(latex_table(
        "tab:directional_taobao_gender", main_caption, "llcccccc",
        r"Backbone & Metric & $M_0$: B/F & $\Delta M_0$ [95\% CI] & $M_1$: B/F & $\Delta M_1$ [95\% CI] & $D$: B/F & $|D|$: B/F",
        main_rows), encoding="utf-8")

    appendix = []
    for metric in METRICS:
        metric_rows = []
        for dataset in DATASETS:
            for backbone in BACKBONES:
                for attribute in ATTRIBUTES[:2]:
                    g0 = indexed[(dataset, backbone, attribute, metric, "0")]
                    g1 = indexed[(dataset, backbone, attribute, metric, "1")]
                    before = summaries[(dataset, backbone, "baseline", attribute, metric)]
                    after = summaries[(dataset, backbone, "fair_ncl", attribute, metric)]
                    group_cell = lambda r: (f"{pair(r['baseline'], r['fair_ncl'])} "
                                            f"({scaled(r['delta'], True)})")
                    metric_rows.append(" & ".join([
                        "ML-1M" if dataset == "ml-1m" else "Taobao",
                        NAMES[backbone],
                        "G" if attribute == "gender" else "A",
                        group_cell(g0), group_cell(g1),
                        pair(float(before["signed_gap_mean"]), float(after["signed_gap_mean"]), True),
                        pair(float(before["absolute_gap_mean"]), float(after["absolute_gap_mean"])),
                        f"{100 * float(before['relative_gap_mean']):.1f}/"
                        f"{100 * float(after['relative_gap_mean']):.1f}",
                        pair(float(before["worst_group_mean"]), float(after["worst_group_mean"])),
                    ]))
        label = "tab:directional_" + metric.lower().replace("hitrate", "hr").replace("@", "")
        caption = (rf"Marginal-group {metric} by dataset, backbone, and attribute "
                   r"(G: gender; A: age). B/F denotes plain-backbone/Fair-NCL five-seed means; "
                   r"parentheses give F$-$B changes. Utilities, signed $D=M_0-M_1$, absolute "
                   r"$|D|$, and worst-group utility are in units of $10^{-3}$; relative gaps "
                   r"are percentages. Gaps and the worst-group value are calculated for each seed "
                   r"before averaging, so the worst group's identity may change. Full group counts, "
                   r"paired intervals, and exact tests are in Supplementary Data~S1.")
        appendix.append(latex_table(label, caption, "lllcccccc",
            r"Dataset & Backbone & Attr. & $M_0$: B/F ($\Delta$) & $M_1$: B/F ($\Delta$) & $D$: B/F & $|D|$: B/F & Rel. (\%): B/F & Worst: B/F",
            metric_rows))

    intersection_rows = []
    metric = "NDCG@10"
    for dataset in DATASETS:
        for backbone in BACKBONES:
            before = summaries[(dataset, backbone, "baseline", "gender_age", metric)]
            after = summaries[(dataset, backbone, "fair_ncl", "gender_age", metric)]
            cells = [pair(indexed[(dataset, backbone, "gender_age", metric, group)]["baseline"],
                          indexed[(dataset, backbone, "gender_age", metric, group)]["fair_ncl"])
                     for group in GROUPS["gender_age"]]
            intersection_rows.append(" & ".join([
                "ML-1M" if dataset == "ml-1m" else "Taobao",
                NAMES[backbone],
                *cells,
                pair(float(before["absolute_gap_mean"]), float(after["absolute_gap_mean"])),
                pair(float(before["worst_group_mean"]), float(after["worst_group_mean"])),
            ]))
    caption = (r"Intersectional gender--age NDCG@10 means before/after Fair-NCL, "
               r"in units of $10^{-3}$. The four cells $(g0a0,g0a1,g1a0,g1a1)$ contain "
               r"$(3338,988,1271,437)$ ML-1M users and $(1918,2660,17935,8820)$ Taobao "
               r"users. Gap is the seed-level maximum minus minimum, then averaged; Worst "
               r"is the averaged seed-level minimum. Paired cell changes and intervals are in "
               r"Supplementary Data~S1; no intersectional gap interval excludes zero.")
    appendix.append(latex_table("tab:intersection_group_values", caption, "llcccccc",
        r"Dataset & Backbone & $g0a0$: B/F & $g0a1$: B/F & $g1a0$: B/F & $g1a1$: B/F & Gap: B/F & Worst: B/F",
        intersection_rows))
    (TEX / "group_directional_appendix.tex").write_text("\n".join(appendix), encoding="utf-8")
    print(f"Wrote {len(rows)} group rows, one main table, and {len(appendix)} appendix tables")


if __name__ == "__main__":
    main()

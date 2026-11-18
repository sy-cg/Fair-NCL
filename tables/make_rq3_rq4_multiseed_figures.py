from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET


DATASET_ORDER = ["ML-1M", "Taobao"]
DATASET_RAW_TO_DISPLAY = {
    "ml-1m": "ML-1M",
    "taobao": "Taobao",
    "lastfm-1k": "LastFM-1K",
}
BACKBONE_ORDER = ["SASRec", "BERT4Rec", "GRU4Rec", "Caser"]
BACKBONE_RAW_TO_DISPLAY = {
    "sasrec": "SASRec",
    "bert4rec": "BERT4Rec",
    "gru4rec": "GRU4Rec",
    "caser": "Caser",
}
METHOD_DISPLAY = {
    "baseline": "Baseline",
    "fair_ncl": "Fair-NCL",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RQ3/RQ4 tables and PDF figures from multi-seed means "
            "without requiring pandas or matplotlib."
        )
    )
    parser.add_argument("--comparison-workbook", default="results/comparison_ml_taobao_3seed.xlsx")
    parser.add_argument(
        "--mechanism-workbook",
        default="results/mechanism_comparison_ml_taobao_3seed.xlsx",
        help="Optional mechanism summary workbook produced by summarize-mechanism.",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-rq3-dir", default="figures/rq3")
    parser.add_argument("--output-rq4-dir", default="figures/rq4")
    parser.add_argument("--paper-dir", default="els-cas-templates")
    parser.add_argument("--datasets", nargs="+", default=["ml-1m", "taobao"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 2026])
    parser.add_argument("--compile", action="store_true", help="Compile generated TikZ files to PDF.")
    parser.add_argument(
        "--strict-rq4-seeds",
        action="store_true",
        help="Fail if RQ4 mechanism diagnostics do not cover every requested seed.",
    )
    args = parser.parse_args()

    rq3_dir = Path(args.output_rq3_dir)
    rq4_dir = Path(args.output_rq4_dir)
    paper_dir = Path(args.paper_dir)
    rq3_dir.mkdir(parents=True, exist_ok=True)
    rq4_dir.mkdir(parents=True, exist_ok=True)

    dataset_filter = {normalize_dataset_name(name) for name in args.datasets}

    rq3_summary = build_rq3_from_workbook(
        Path(args.comparison_workbook),
        dataset_filter=dataset_filter,
    )
    write_dict_csv(rq3_summary, rq3_dir / "rq3_robustness_summary.csv")
    write_rq3_latex(rq3_summary, rq3_dir / "rq3_robustness_summary.tex")
    write_rq3_main_figure(rq3_summary, rq3_dir / "rq3_main_figure.tex")
    write_rq3_extended_figure(rq3_summary, rq3_dir / "rq3_extended_20_heatmaps.tex")

    mechanism_workbook = Path(args.mechanism_workbook) if args.mechanism_workbook else None
    if mechanism_workbook and mechanism_workbook.exists():
        rq4_full, rq4_summary, rq4_missing = build_rq4_from_workbook(
            mechanism_workbook,
            dataset_filter=dataset_filter,
            expected_seeds=set(args.seeds),
        )
    else:
        rq4_records = load_mechanism_records(
            Path(args.results_root),
            dataset_filter=dataset_filter,
            expected_seeds=set(args.seeds),
        )
        rq4_full, rq4_summary, rq4_missing = build_rq4_tables(rq4_records, expected_seeds=set(args.seeds))
    if rq4_missing:
        message = (
            "RQ4 mechanism diagnostics are missing for: "
            + "; ".join(rq4_missing[:12])
            + ("; ..." if len(rq4_missing) > 12 else "")
        )
        if args.strict_rq4_seeds:
            raise RuntimeError(message)
        print(f"[WARN] {message}")

    write_dict_csv(rq4_full, rq4_dir / "rq4_full_mechanism_baseline_fair.csv")
    write_dict_csv(rq4_summary, rq4_dir / "rq4_mechanism_summary.csv")
    write_rq4_summary_latex(rq4_summary, rq4_dir / "rq4_mechanism_summary.tex")
    write_rq4_full_latex(rq4_full, rq4_dir / "rq4_full_mechanism_baseline_fair.tex")
    write_rq4_main_figure(rq4_summary, rq4_dir / "rq4_main_mechanism_figure.tex")
    write_rq4_attribute_figure(rq4_summary, rq4_dir / "rq4_attribute_specific_heatmaps.tex")
    write_rq4_transform_figure(rq4_full, rq4_dir / "rq4_transform_diagnostics.tex")

    if args.compile:
        generated = [
            rq3_dir / "rq3_main_figure.tex",
            rq3_dir / "rq3_extended_20_heatmaps.tex",
            rq4_dir / "rq4_main_mechanism_figure.tex",
            rq4_dir / "rq4_attribute_specific_heatmaps.tex",
            rq4_dir / "rq4_transform_diagnostics.tex",
        ]
        for tex_path in generated:
            compile_tikz(tex_path)

        if paper_dir.exists():
            for pdf in [
                rq3_dir / "rq3_main_figure.pdf",
                rq3_dir / "rq3_extended_20_heatmaps.pdf",
                rq4_dir / "rq4_main_mechanism_figure.pdf",
                rq4_dir / "rq4_attribute_specific_heatmaps.pdf",
                rq4_dir / "rq4_transform_diagnostics.pdf",
            ]:
                if pdf.exists():
                    shutil.copy2(pdf, paper_dir / pdf.name)

    print(f"Saved RQ3 outputs to {rq3_dir}")
    print(f"Saved RQ4 outputs to {rq4_dir}")


# ---------------------------------------------------------------------------
# XLSX reading
# ---------------------------------------------------------------------------


def read_xlsx_sheet(path: Path, sheet_name: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_path = resolve_sheet_path(archive, sheet_name)
        root = ET.fromstring(archive.read(sheet_path))

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: List[List[str]] = []
    for row in root.findall(".//a:sheetData/a:row", ns):
        values: List[str] = []
        for cell in row.findall("a:c", ns):
            idx = cell_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = read_cell_value(cell, shared_strings, ns)
        rows.append(values)

    if not rows:
        return []
    header = rows[0]
    result: List[Dict[str, str]] = []
    for raw_row in rows[1:]:
        row = {header[i]: raw_row[i] if i < len(raw_row) else "" for i in range(len(header))}
        if any(str(value).strip() for value in row.values()):
            result.append(row)
    return result


def read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for item in root.findall("a:si", ns):
        strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))
    return strings


def resolve_sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

    for sheet in workbook.find("a:sheets", ns).findall("a:sheet", ns):
        if sheet.attrib.get("name") == sheet_name:
            target = rel_map[sheet.attrib[rel_ns]]
            if target.startswith("/"):
                return target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return f"xl/{target}"
    raise ValueError(f"Sheet not found in {archive.filename}: {sheet_name}")


def read_cell_value(cell: ET.Element, shared_strings: Sequence[str], ns: Dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", ns)
    if cell_type == "s" and value is not None:
        return shared_strings[int(value.text)]
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns))
    return value.text if value is not None else ""


def cell_index(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - 64
    return index - 1


# ---------------------------------------------------------------------------
# RQ3
# ---------------------------------------------------------------------------


def build_rq3_from_workbook(path: Path, dataset_filter: set[str]) -> List[Dict[str, object]]:
    rows = read_xlsx_sheet(path, "summary_all_metrics")
    selected = []
    for row in rows:
        dataset_raw = normalize_dataset_name(row.get("dataset", row.get("dataset_display", "")))
        method = row.get("method", "").strip()
        backbone = row.get("backbone", "").strip()
        if dataset_raw not in dataset_filter:
            continue
        if method not in {"baseline", "fair_ncl"}:
            continue
        selected.append((dataset_raw, backbone, method, row))

    by_key: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for dataset, backbone, method, row in selected:
        by_key[(dataset, backbone)][method] = row

    summary: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_raw = normalize_dataset_name(dataset)
        for backbone_display in BACKBONE_ORDER:
            backbone_raw = normalize_backbone_name(backbone_display)
            methods = by_key.get((dataset_raw, backbone_raw), {})
            if "baseline" not in methods or "fair_ncl" not in methods:
                continue
            base = methods["baseline"]
            fair = methods["fair_ncl"]
            out: Dict[str, object] = {
                "dataset": dataset,
                "backbone": backbone_display,
            }
            improved = 0
            total = 0
            for k in (10, 20):
                base_ndcg = numeric(get_metric(base, f"selected__NDCG@{k}"))
                fair_ndcg = numeric(get_metric(fair, f"selected__NDCG@{k}"))
                base_g_gap = numeric(get_metric(base, f"selected__gender_NDCG@{k}_Gap"))
                fair_g_gap = numeric(get_metric(fair, f"selected__gender_NDCG@{k}_Gap"))
                base_a_gap = numeric(get_metric(base, f"selected__age_group_NDCG@{k}_Gap"))
                fair_a_gap = numeric(get_metric(fair, f"selected__age_group_NDCG@{k}_Gap"))
                base_avg = mean_ignore_nan([base_g_gap, base_a_gap])
                fair_avg = mean_ignore_nan([fair_g_gap, fair_a_gap])

                out[f"baseline_ndcg@{k}"] = base_ndcg
                out[f"fair_ndcg@{k}"] = fair_ndcg
                out[f"delta_ndcg@{k}_pct"] = rel_change_pct(fair_ndcg, base_ndcg)
                out[f"baseline_gender_ndcg_gap@{k}"] = base_g_gap
                out[f"fair_gender_ndcg_gap@{k}"] = fair_g_gap
                out[f"baseline_age_ndcg_gap@{k}"] = base_a_gap
                out[f"fair_age_ndcg_gap@{k}"] = fair_a_gap
                out[f"baseline_avg_ndcg_gap@{k}"] = base_avg
                out[f"fair_avg_ndcg_gap@{k}"] = fair_avg
                out[f"avg_ndcg_gap_reduction@{k}_pct"] = reduction_pct(base_avg, fair_avg)

                for base_gap, fair_gap in ((base_g_gap, fair_g_gap), (base_a_gap, fair_a_gap)):
                    if not math.isnan(base_gap) and not math.isnan(fair_gap):
                        total += 1
                        improved += int(fair_gap < base_gap)

            out["improved_ndcg_gaps"] = improved
            out["total_ndcg_gaps"] = total
            out["improved_ndcg_gaps_label"] = f"{improved}/{total}"
            summary.append(out)
    return summary


def get_metric(row: Dict[str, str], base_name: str) -> str:
    for name in (f"{base_name}__mean", base_name):
        if name in row and row[name] != "":
            return row[name]
    return ""


def write_rq3_latex(summary: Sequence[Dict[str, object]], output_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Robustness summary of Fair-NCL across datasets and backbones. Utility change is measured by relative NDCG change over the corresponding backbone baseline, computed from three-seed mean values. Fairness improvement is measured by the relative reduction of the average NDCG-based group gap.}",
        r"\label{tab:rq3_robustness_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Backbone & $\Delta$NDCG@10 & $\Delta$NDCG@20 & GapRed@10 & GapRed@20 & Improved Gaps \\",
        r"\midrule",
    ]
    current = None
    for row in summary:
        if current is not None and row["dataset"] != current:
            lines.append(r"\midrule")
        current = row["dataset"]
        lines.append(
            f"{tex_escape(str(row['dataset']))} & {tex_escape(str(row['backbone']))} & "
            f"{format_pct(row['delta_ndcg@10_pct'])} & "
            f"{format_pct(row['delta_ndcg@20_pct'])} & "
            f"{format_pct(row['avg_ndcg_gap_reduction@10_pct'])} & "
            f"{format_pct(row['avg_ndcg_gap_reduction@20_pct'])} & "
            f"{row['improved_ndcg_gaps_label']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# RQ4
# ---------------------------------------------------------------------------

RQ4_METRIC_MAP = [
    ("Raw->Fair Cosine", "mechanism__representation__transform__cosine_mean"),
    ("Gender Probe BA Raw", "mechanism__gender__probe_raw__balanced_accuracy"),
    ("Gender Probe BA Fair", "mechanism__gender__probe_fair__balanced_accuracy"),
    ("Gender Probe BA Drop", "mechanism__gender__probe_balanced_accuracy_drop"),
    ("Age Probe BA Raw", "mechanism__age_group__probe_raw__balanced_accuracy"),
    ("Age Probe BA Fair", "mechanism__age_group__probe_fair__balanced_accuracy"),
    ("Age Probe BA Drop", "mechanism__age_group__probe_balanced_accuracy_drop"),
    ("Gender Sep Raw", "mechanism__gender__separability_raw__between_within_ratio"),
    ("Gender Sep Fair", "mechanism__gender__separability_fair__between_within_ratio"),
    ("Age Sep Raw", "mechanism__age_group__separability_raw__between_within_ratio"),
    ("Age Sep Fair", "mechanism__age_group__separability_fair__between_within_ratio"),
]


def build_rq4_from_workbook(
    path: Path,
    dataset_filter: set[str],
    expected_seeds: set[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    summary_rows = read_xlsx_sheet(path, "summary_long")
    raw_rows = read_xlsx_sheet(path, "raw_results")

    seed_sets: Dict[Tuple[str, str, str], set[int]] = defaultdict(set)
    for row in raw_rows:
        if row.get("mechanism_status") != "ok":
            continue
        dataset_raw = normalize_dataset_name(row.get("dataset", row.get("dataset_display", "")))
        method = row.get("method", "").strip()
        backbone_raw = normalize_backbone_name(row.get("backbone", row.get("backbone_display", "")))
        if dataset_raw not in dataset_filter or method not in {"baseline", "fair_ncl"}:
            continue
        seed = numeric(row.get("seed"))
        if not math.isnan(seed):
            seed_sets[(dataset_raw, backbone_raw, method)].add(int(seed))

    by_key: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in summary_rows:
        dataset_raw = normalize_dataset_name(row.get("dataset", row.get("dataset_display", "")))
        method = row.get("method", "").strip()
        backbone_raw = normalize_backbone_name(row.get("backbone", row.get("backbone_display", "")))
        if dataset_raw not in dataset_filter or method not in {"baseline", "fair_ncl"}:
            continue
        by_key[(dataset_raw, backbone_raw, method)] = row

    full_rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for dataset in DATASET_ORDER:
        dataset_raw = normalize_dataset_name(dataset)
        for backbone in BACKBONE_ORDER:
            backbone_raw = normalize_backbone_name(backbone)
            for method_raw, method_display in (("baseline", "Baseline"), ("fair_ncl", "Fair-NCL")):
                source = by_key.get((dataset_raw, backbone_raw, method_raw))
                seeds = seed_sets.get((dataset_raw, backbone_raw, method_raw), set())
                if source is None:
                    missing.append(f"{dataset}/{backbone}/{method_display}: all seeds")
                    continue
                metric_counts = [
                    int(numeric(source.get(f"{metric}__count")))
                    for _, metric in RQ4_METRIC_MAP
                    if not math.isnan(numeric(source.get(f"{metric}__count")))
                ]
                seed_count = len(seeds) if seeds else (min(metric_counts) if metric_counts else 0)
                absent = sorted(expected_seeds - seeds) if seeds else []
                if expected_seeds and (seed_count < len(expected_seeds) or absent):
                    if absent:
                        missing.append(f"{dataset}/{backbone}/{method_display}: missing {','.join(map(str, absent))}")
                    else:
                        missing.append(f"{dataset}/{backbone}/{method_display}: count {seed_count}/{len(expected_seeds)}")

                out: Dict[str, object] = {
                    "Dataset": dataset,
                    "Backbone": backbone,
                    "Method": method_display,
                    "seed_count": seed_count,
                    "seeds": ",".join(str(seed) for seed in sorted(seeds)),
                }
                for label, metric in RQ4_METRIC_MAP:
                    out[label] = numeric(source.get(f"{metric}__mean"))
                    out[f"{label}__std"] = numeric(source.get(f"{metric}__std"))
                full_rows.append(out)

    return full_rows, summarize_rq4_full_rows(full_rows), missing


def load_mechanism_records(
    results_root: Path,
    dataset_filter: set[str],
    expected_seeds: set[int],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in results_root.glob("*/*/mechanism_test_summary.json"):
        job_path = path.with_name("job.json")
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("phase") != "comparison":
            continue
        dataset_raw = normalize_dataset_name(str(job.get("dataset", "")))
        method = str(job.get("method", ""))
        backbone_raw = normalize_backbone_name(str(job.get("backbone", "")))
        seed = int(job.get("seed", -1))
        if dataset_raw not in dataset_filter:
            continue
        if method not in {"baseline", "fair_ncl"}:
            continue
        if backbone_raw not in {normalize_backbone_name(name) for name in BACKBONE_ORDER}:
            continue
        if expected_seeds and seed not in expected_seeds:
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        attrs = payload.get("attributes", {})
        gender = attrs.get("gender", {})
        age = attrs.get("age_group", {})
        transform = payload.get("representation", {}).get("transform", {})
        record = {
            "Dataset": DATASET_RAW_TO_DISPLAY.get(dataset_raw, dataset_raw),
            "Backbone": BACKBONE_RAW_TO_DISPLAY.get(backbone_raw, backbone_raw),
            "Method": METHOD_DISPLAY.get(method, method),
            "seed": seed,
            "Raw->Fair Cosine": numeric(transform.get("cosine_mean")),
            "Gender Probe BA Raw": numeric_nested(gender, "probe_raw", "balanced_accuracy"),
            "Gender Probe BA Fair": numeric_nested(gender, "probe_fair", "balanced_accuracy"),
            "Gender Sep Raw": numeric_nested(gender, "separability_raw", "between_within_ratio"),
            "Gender Sep Fair": numeric_nested(gender, "separability_fair", "between_within_ratio"),
            "Age Probe BA Raw": numeric_nested(age, "probe_raw", "balanced_accuracy"),
            "Age Probe BA Fair": numeric_nested(age, "probe_fair", "balanced_accuracy"),
            "Age Sep Raw": numeric_nested(age, "separability_raw", "between_within_ratio"),
            "Age Sep Fair": numeric_nested(age, "separability_fair", "between_within_ratio"),
        }
        record["Gender Probe BA Drop"] = record["Gender Probe BA Raw"] - record["Gender Probe BA Fair"]
        record["Age Probe BA Drop"] = record["Age Probe BA Raw"] - record["Age Probe BA Fair"]
        records.append(record)
    return sort_records(records, ["Dataset", "Backbone", "Method", "seed"])


def build_rq4_tables(
    records: Sequence[Dict[str, object]],
    expected_seeds: set[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["Dataset"]), str(record["Backbone"]), str(record["Method"]))].append(record)

    full_rows: List[Dict[str, object]] = []
    missing: List[str] = []
    metrics = [
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

    for dataset in DATASET_ORDER:
        for backbone in BACKBONE_ORDER:
            for method in ("Baseline", "Fair-NCL"):
                items = grouped.get((dataset, backbone, method), [])
                if not items:
                    missing.append(f"{dataset}/{backbone}/{method}: all seeds")
                    continue
                present = {int(item["seed"]) for item in items}
                absent = sorted(expected_seeds - present)
                if absent:
                    missing.append(f"{dataset}/{backbone}/{method}: missing {','.join(map(str, absent))}")
                out: Dict[str, object] = {
                    "Dataset": dataset,
                    "Backbone": backbone,
                    "Method": method,
                    "seed_count": len(present),
                    "seeds": ",".join(str(seed) for seed in sorted(present)),
                }
                for metric in metrics:
                    values = [float(item[metric]) for item in items if not math.isnan(float(item[metric]))]
                    out[metric] = mean_ignore_nan(values)
                    out[f"{metric}__std"] = sample_std(values)
                full_rows.append(out)

    return full_rows, summarize_rq4_full_rows(full_rows), missing


def summarize_rq4_full_rows(full_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    by_pair = {(r["Dataset"], r["Backbone"], r["Method"]): r for r in full_rows}
    for dataset in DATASET_ORDER:
        for backbone in BACKBONE_ORDER:
            base = by_pair.get((dataset, backbone, "Baseline"))
            fair = by_pair.get((dataset, backbone, "Fair-NCL"))
            if not base or not fair:
                continue
            base_gender_probe = float(base["Gender Probe BA Fair"])
            fair_gender_probe = float(fair["Gender Probe BA Fair"])
            base_age_probe = float(base["Age Probe BA Fair"])
            fair_age_probe = float(fair["Age Probe BA Fair"])
            base_gender_sep = float(base["Gender Sep Fair"])
            fair_gender_sep = float(fair["Gender Sep Fair"])
            base_age_sep = float(base["Age Sep Fair"])
            fair_age_sep = float(fair["Age Sep Fair"])

            base_avg_probe = mean_ignore_nan([base_gender_probe, base_age_probe])
            fair_avg_probe = mean_ignore_nan([fair_gender_probe, fair_age_probe])
            base_avg_sep = mean_ignore_nan([base_gender_sep, base_age_sep])
            fair_avg_sep = mean_ignore_nan([fair_gender_sep, fair_age_sep])

            row = {
                "dataset": dataset,
                "backbone": backbone,
                "baseline_seed_count": base["seed_count"],
                "fair_seed_count": fair["seed_count"],
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
                "fair_raw_to_fair_cosine": fair["Raw->Fair Cosine"],
                "fair_gender_probe_ba_drop": fair["Gender Probe BA Drop"],
                "fair_age_probe_ba_drop": fair["Age Probe BA Drop"],
                "fair_avg_probe_ba_drop": mean_ignore_nan([
                    float(fair["Gender Probe BA Drop"]),
                    float(fair["Age Probe BA Drop"]),
                ]),
            }
            row["improved_probe_dims"] = int(fair_gender_probe < base_gender_probe) + int(
                fair_age_probe < base_age_probe
            )
            row["improved_sep_dims"] = int(fair_gender_sep < base_gender_sep) + int(
                fair_age_sep < base_age_sep
            )
            row["improved_probe_label"] = f"{row['improved_probe_dims']}/2"
            row["improved_sep_label"] = f"{row['improved_sep_dims']}/2"
            summary.append(row)

    return summary


def write_rq4_summary_latex(summary: Sequence[Dict[str, object]], output_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Representation-level mechanism summary for Fair-NCL. ProbeRed and SepRed denote the relative reduction of average probe balanced accuracy and average sensitive-group separability compared with the corresponding backbone baseline. Values are computed from three-seed mean mechanism diagnostics.}",
        r"\label{tab:rq4_mechanism_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Backbone & ProbeRed & SepRed & Probe Dims & Sep Dims \\",
        r"\midrule",
    ]
    current = None
    for row in summary:
        if current is not None and row["dataset"] != current:
            lines.append(r"\midrule")
        current = row["dataset"]
        lines.append(
            f"{tex_escape(str(row['dataset']))} & {tex_escape(str(row['backbone']))} & "
            f"{format_pct(row['avg_probe_reduction_pct'])} & "
            f"{format_pct(row['avg_sep_reduction_pct'])} & "
            f"{row['improved_probe_label']} & {row['improved_sep_label']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_rq4_full_latex(full_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Full mechanism metrics for Baseline and Fair-NCL. Each cell reports three-seed mean values; standard deviations are retained in the source CSV.}",
        r"\label{tab:rq4_full_mechanism_baseline_fair}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Backbone & Method & Cosine & G-Probe Raw & G-Probe Fair & G-Probe Drop & A-Probe Raw & A-Probe Fair & A-Probe Drop & G-Sep Raw & G-Sep Fair & A-Sep Raw & A-Sep Fair \\",
        r"\midrule",
    ]
    current = None
    for row in full_rows:
        if current is not None and row["Dataset"] != current:
            lines.append(r"\midrule")
        current = row["Dataset"]
        fields = [
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
        values = []
        for field in fields:
            if field in {"Dataset", "Backbone", "Method"}:
                values.append(tex_escape(str(row[field])))
            else:
                values.append(format_float(row[field], 4))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# TikZ figures
# ---------------------------------------------------------------------------


def write_rq3_main_figure(summary: Sequence[Dict[str, object]], output_path: Path) -> None:
    body = [
        r"\begin{tikzpicture}[font=\sffamily]",
        r"\node[anchor=west,font=\bfseries] at (0,5.25) {a\quad Utility--fairness trade-off};",
    ]
    body.extend(scatter_panel(summary, 0.0, 0.0, "delta_ndcg@10_pct", "avg_ndcg_gap_reduction@10_pct"))
    body.append(r"\node[anchor=west,font=\bfseries] at (6.25,5.25) {b\quad Fairness robustness};")
    body.extend(heatmap_panel(summary, 6.25, 2.1, "avg_ndcg_gap_reduction@10_pct", "GapRed@10"))
    body.append(r"\node[anchor=west,font=\bfseries] at (12.2,5.25) {c\quad Utility change};")
    body.extend(heatmap_panel(summary, 12.2, 2.1, "delta_ndcg@10_pct", r"$\Delta$NDCG@10"))
    body.append(r"\end{tikzpicture}")
    output_path.write_text(tikz_document("\n".join(body)), encoding="utf-8")


def write_rq3_extended_figure(summary: Sequence[Dict[str, object]], output_path: Path) -> None:
    body = [
        r"\begin{tikzpicture}[font=\sffamily]",
        r"\node[anchor=west,font=\bfseries] at (0,5.2) {a\quad Avg. NDCG gap reduction@20};",
    ]
    body.extend(heatmap_panel(summary, 0.0, 2.0, "avg_ndcg_gap_reduction@20_pct", "GapRed@20"))
    body.append(r"\node[anchor=west,font=\bfseries] at (6.4,5.2) {b\quad $\Delta$NDCG@20};")
    body.extend(heatmap_panel(summary, 6.4, 2.0, "delta_ndcg@20_pct", r"$\Delta$NDCG@20"))
    body.append(r"\end{tikzpicture}")
    output_path.write_text(tikz_document("\n".join(body)), encoding="utf-8")


def write_rq4_main_figure(summary: Sequence[Dict[str, object]], output_path: Path) -> None:
    body = [
        r"\begin{tikzpicture}[font=\sffamily]",
        r"\node[anchor=west,font=\bfseries] at (0,5.25) {a\quad Group separability reduction};",
    ]
    body.extend(heatmap_panel(summary, 0.0, 2.1, "avg_sep_reduction_pct", "SepRed"))
    body.append(r"\node[anchor=west,font=\bfseries] at (6.25,5.25) {b\quad Probe leakage reduction};")
    body.extend(heatmap_panel(summary, 6.25, 2.1, "avg_probe_reduction_pct", "ProbeRed"))
    body.append(r"\node[anchor=west,font=\bfseries] at (12.8,5.25) {c\quad Reduction consistency};")
    body.extend(count_panel(summary, 12.8, 0.65))
    body.append(r"\end{tikzpicture}")
    output_path.write_text(tikz_document("\n".join(body)), encoding="utf-8")


def write_rq4_attribute_figure(summary: Sequence[Dict[str, object]], output_path: Path) -> None:
    panels = [
        ("gender_sep_reduction_pct", "a  Gender separability reduction"),
        ("age_sep_reduction_pct", "b  Age separability reduction"),
        ("gender_probe_reduction_pct", "c  Gender probe reduction"),
        ("age_probe_reduction_pct", "d  Age probe reduction"),
    ]
    positions = [(0, 5.9), (6.4, 5.9), (0, 0.6), (6.4, 0.6)]
    body = [r"\begin{tikzpicture}[font=\sffamily]"]
    for (col, title), (x, y) in zip(panels, positions):
        body.append(fr"\node[anchor=west,font=\bfseries] at ({x},{y + 3.15}) {{{tex_text(title)}}};")
        body.extend(heatmap_panel(summary, x, y, col, "Reduction"))
    body.append(r"\end{tikzpicture}")
    output_path.write_text(tikz_document("\n".join(body)), encoding="utf-8")


def write_rq4_transform_figure(full_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    body = [
        r"\begin{tikzpicture}[font=\sffamily]",
        r"\node[anchor=west,font=\bfseries] at (0,4.4) {a\quad Raw-to-fair representation cosine};",
    ]
    body.extend(strip_panel(full_rows, 0.0, 0.5, "Raw->Fair Cosine", 0.98, 1.02, "Cosine"))
    body.append(r"\node[anchor=west,font=\bfseries] at (7.2,4.4) {b\quad Raw-to-fair probe BA drop};")
    for row in full_rows:
        row["Avg Probe BA Drop"] = mean_ignore_nan([
            float(row["Gender Probe BA Drop"]),
            float(row["Age Probe BA Drop"]),
        ])
    body.extend(strip_panel(full_rows, 7.2, 0.5, "Avg Probe BA Drop", -0.02, 0.02, "Avg. BA drop"))
    body.append(r"\end{tikzpicture}")
    output_path.write_text(tikz_document("\n".join(body)), encoding="utf-8")


def tikz_document(body: str) -> str:
    return "\n".join(
        [
            r"\documentclass[tikz,border=2pt]{standalone}",
            r"\usepackage{amsmath}",
            r"\usetikzlibrary{shapes.geometric}",
            r"\definecolor{cSASRec}{HTML}{0072B2}",
            r"\definecolor{cBERT}{HTML}{D55E00}",
            r"\definecolor{cGRU}{HTML}{009E73}",
            r"\definecolor{cCaser}{HTML}{CC79A7}",
            r"\begin{document}",
            body,
            r"\end{document}",
            "",
        ]
    )


def heatmap_panel(
    rows: Sequence[Dict[str, object]],
    x0: float,
    y0: float,
    value_col: str,
    label: str,
) -> List[str]:
    values = [float(row[value_col]) for row in rows if not math.isnan(float(row[value_col]))]
    vmax = max([abs(v) for v in values] + [1e-6])
    cell_w = 1.1
    cell_h = 0.72
    lines: List[str] = []
    for j, backbone in enumerate(BACKBONE_ORDER):
        lines.append(
            fr"\node[rotate=25,anchor=south west,font=\scriptsize] "
            fr"at ({x0 + 1.23 + j * cell_w},{y0 + 1.55}) {{{backbone}}};"
        )
    for i, dataset in enumerate(DATASET_ORDER):
        y = y0 + (len(DATASET_ORDER) - 1 - i) * cell_h
        lines.append(fr"\node[anchor=east,font=\scriptsize] at ({x0 + 1.0},{y + 0.36}) {{{tex_escape(dataset)}}};")
        for j, backbone in enumerate(BACKBONE_ORDER):
            row = find_row(rows, dataset, backbone)
            value = float(row[value_col]) if row else math.nan
            color = diverging_hex(value, vmax)
            color_name = f"cell{abs(hash((x0, y0, value_col, dataset, backbone))) % 1000000}"
            lines.append(fr"\definecolor{{{color_name}}}{{HTML}}{{{color}}}")
            x = x0 + 1.1 + j * cell_w
            lines.append(fr"\filldraw[fill={color_name},draw=white,line width=0.5pt] ({x},{y}) rectangle ++({cell_w},{cell_h});")
            precision = 2 if 0 < abs(value) < 0.05 else 1
            lines.append(fr"\node[font=\scriptsize] at ({x + cell_w / 2:.3f},{y + cell_h / 2:.3f}) {{{format_float(value, precision)}}};")
    lines.append(fr"\node[anchor=west,font=\scriptsize] at ({x0 + 1.1},{y0 - 0.25}) {{{label} (\%)}};")
    return lines


def scatter_panel(
    rows: Sequence[Dict[str, object]],
    x0: float,
    y0: float,
    x_col: str,
    y_col: str,
) -> List[str]:
    width = 5.0
    height = 3.85
    x_values = [float(row[x_col]) for row in rows]
    y_values = [float(row[y_col]) for row in rows]
    x_min, x_max = padded_range(x_values + [0.0])
    y_min, y_max = padded_range(y_values + [0.0])

    lines = [
        fr"\draw[->,gray] ({x0},{y0}) -- ({x0 + width + 0.15},{y0});",
        fr"\draw[->,gray] ({x0},{y0}) -- ({x0},{y0 + height + 0.15});",
        fr"\node[anchor=north,font=\scriptsize] at ({x0 + width / 2},{y0 - 0.35}) {{$\Delta$NDCG@10 (\%)}};",
        fr"\node[rotate=90,anchor=south,font=\scriptsize] at ({x0 - 0.55},{y0 + height / 2}) {{Gap reduction@10 (\%)}};",
    ]
    zero_x = map_range(0.0, x_min, x_max, x0, x0 + width)
    zero_y = map_range(0.0, y_min, y_max, y0, y0 + height)
    lines.extend([
        fr"\draw[dashed,gray!70] ({zero_x},{y0}) -- ({zero_x},{y0 + height});",
        fr"\draw[dashed,gray!70] ({x0},{zero_y}) -- ({x0 + width},{zero_y});",
    ])
    for tick in [x_min, 0.0, x_max]:
        px = map_range(tick, x_min, x_max, x0, x0 + width)
        lines.append(fr"\node[anchor=north,font=\tiny] at ({px},{y0 - 0.05}) {{{format_float(tick, 1)}}};")
    for tick in [y_min, 0.0, y_max]:
        py = map_range(tick, y_min, y_max, y0, y0 + height)
        lines.append(fr"\node[anchor=east,font=\tiny] at ({x0 - 0.05},{py}) {{{format_float(tick, 1)}}};")

    for row in rows:
        x = map_range(float(row[x_col]), x_min, x_max, x0, x0 + width)
        y = map_range(float(row[y_col]), y_min, y_max, y0, y0 + height)
        color = tikz_backbone_color(str(row["backbone"]))
        if row["dataset"] == "ML-1M":
            lines.append(fr"\node[regular polygon,regular polygon sides=3,draw=white,fill={color},inner sep=2.0pt] at ({x},{y}) {{}};")
        else:
            lines.append(fr"\node[rectangle,draw=white,fill={color},minimum size=4.5pt,inner sep=0pt] at ({x},{y}) {{}};")
    lines.append(fr"\node[anchor=west,font=\tiny] at ({x0 + 3.35},{y0 + 3.55}) {{$\triangle$ ML-1M}};")
    lines.append(fr"\node[anchor=west,font=\tiny] at ({x0 + 3.35},{y0 + 3.25}) {{\rule{{0.45em}}{{0.45em}} Taobao}};")
    return lines


def count_panel(rows: Sequence[Dict[str, object]], x0: float, y0: float) -> List[str]:
    labels = ["G Sep", "A Sep", "G Probe", "A Probe"]
    keys = [
        "gender_sep_reduction_pct",
        "age_sep_reduction_pct",
        "gender_probe_reduction_pct",
        "age_probe_reduction_pct",
    ]
    values = [sum(1 for row in rows if float(row[key]) > 0) for key in keys]
    total = max(len(rows), 1)
    bar_w = 0.52
    max_h = 3.2
    lines = [
        fr"\draw[->,gray] ({x0},{y0}) -- ({x0 + 4.8},{y0});",
        fr"\draw[->,gray] ({x0},{y0}) -- ({x0},{y0 + max_h + 0.3});",
        fr"\node[rotate=90,anchor=south,font=\scriptsize] at ({x0 - 0.45},{y0 + 1.7}) {{Reduced settings}};",
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = x0 + 0.55 + i * 1.0
        h = value / total * max_h
        color = "blue!35" if "Sep" in label else "orange!35"
        lines.append(fr"\filldraw[fill={color},draw=white] ({x},{y0}) rectangle ++({bar_w},{h});")
        lines.append(fr"\node[font=\scriptsize] at ({x + bar_w / 2},{y0 + h + 0.18}) {{{value}/{total}}};")
        lines.append(fr"\node[anchor=north,font=\tiny] at ({x + bar_w / 2},{y0 - 0.08}) {{{label}}};")
    return lines


def strip_panel(
    rows: Sequence[Dict[str, object]],
    x0: float,
    y0: float,
    value_col: str,
    y_min: float,
    y_max: float,
    label: str,
) -> List[str]:
    width = 5.4
    height = 3.0
    method_x = {"Baseline": x0 + 1.7, "Fair-NCL": x0 + 3.7}
    jitter = {"SASRec": -0.24, "BERT4Rec": -0.08, "GRU4Rec": 0.08, "Caser": 0.24}
    lines = [
        fr"\draw[->,gray] ({x0},{y0}) -- ({x0 + width},{y0});",
        fr"\draw[->,gray] ({x0},{y0}) -- ({x0},{y0 + height + 0.2});",
        fr"\node[anchor=north,font=\scriptsize] at ({method_x['Baseline']},{y0 - 0.1}) {{Baseline}};",
        fr"\node[anchor=north,font=\scriptsize] at ({method_x['Fair-NCL']},{y0 - 0.1}) {{Fair-NCL}};",
        fr"\node[rotate=90,anchor=south,font=\scriptsize] at ({x0 - 0.45},{y0 + 1.5}) {{{label}}};",
    ]
    for tick in [y_min, (y_min + y_max) / 2, y_max]:
        y = map_range(tick, y_min, y_max, y0, y0 + height)
        lines.append(fr"\draw[gray!30] ({x0},{y}) -- ({x0 + width - 0.2},{y});")
        lines.append(fr"\node[anchor=east,font=\tiny] at ({x0 - 0.05},{y}) {{{format_float(tick, 2)}}};")
    for row in rows:
        method = str(row["Method"])
        if method not in method_x:
            continue
        value = float(row[value_col])
        x = method_x[method] + jitter.get(str(row["Backbone"]), 0.0)
        y = map_range(value, y_min, y_max, y0, y0 + height)
        color = tikz_backbone_color(str(row["Backbone"]))
        if row["Dataset"] == "ML-1M":
            lines.append(fr"\node[regular polygon,regular polygon sides=3,draw=white,fill={color},inner sep=2.0pt] at ({x},{y}) {{}};")
        else:
            lines.append(fr"\node[rectangle,draw=white,fill={color},minimum size=4.5pt,inner sep=0pt] at ({x},{y}) {{}};")
    return lines


def compile_tikz(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=str(tex_path.parent),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for suffix in [".aux", ".log"]:
        aux = tex_path.with_suffix(suffix)
        if aux.exists():
            aux.unlink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_dict_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_dataset_name(name: str) -> str:
    lowered = str(name).strip().lower()
    if lowered in {"ml-1m", "ml1m", "movie-lens", "movielens", "ml-1m"}:
        return "ml-1m"
    if lowered == "ml-1m":
        return "ml-1m"
    if lowered == "taobao":
        return "taobao"
    if lowered in {"lastfm", "lastfm-1k", "lastfm1k"}:
        return "lastfm-1k"
    return lowered


def normalize_backbone_name(name: str) -> str:
    lowered = str(name).strip().lower()
    mapping = {
        "sasrec": "sasrec",
        "bert4rec": "bert4rec",
        "gru4rec": "gru4rec",
        "caser": "caser",
    }
    return mapping.get(lowered, lowered)


def numeric(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else math.nan


def numeric_nested(mapping: Dict, first: str, second: str) -> float:
    value = mapping.get(first, {}).get(second)
    return numeric(value)


def mean_ignore_nan(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else math.nan


def sample_std(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if len(clean) < 2:
        return 0.0
    return statistics.stdev(clean)


def rel_change_pct(new_value: float, baseline_value: float) -> float:
    if math.isnan(new_value) or math.isnan(baseline_value) or abs(baseline_value) < 1e-12:
        return math.nan
    return (new_value - baseline_value) / baseline_value * 100.0


def reduction_pct(baseline_value: float, fair_value: float) -> float:
    if math.isnan(baseline_value) or math.isnan(fair_value) or abs(baseline_value) < 1e-12:
        return math.nan
    return (baseline_value - fair_value) / baseline_value * 100.0


def find_row(rows: Sequence[Dict[str, object]], dataset: str, backbone: str) -> Optional[Dict[str, object]]:
    for row in rows:
        if row.get("dataset") == dataset and row.get("backbone") == backbone:
            return row
    return None


def padded_range(values: Sequence[float]) -> Tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    lo = min(clean)
    hi = max(clean)
    if abs(hi - lo) < 1e-9:
        return lo - 1, hi + 1
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if abs(in_max - in_min) < 1e-12:
        return (out_min + out_max) / 2
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


def diverging_hex(value: float, vmax: float) -> str:
    if math.isnan(value):
        return "F2F2F2"
    t = min(abs(value) / vmax, 1.0)
    if value >= 0:
        base = (49, 130, 189)
    else:
        base = (203, 24, 29)
    white = (247, 247, 247)
    rgb = tuple(round(white[i] * (1 - t) + base[i] * t) for i in range(3))
    return "".join(f"{part:02X}" for part in rgb)


def tikz_backbone_color(backbone: str) -> str:
    return {
        "SASRec": "cSASRec",
        "BERT4Rec": "cBERT",
        "GRU4Rec": "cGRU",
        "Caser": "cCaser",
    }.get(backbone, "black")


def sort_records(rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> List[Dict[str, object]]:
    dataset_rank = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    backbone_rank = {name: idx for idx, name in enumerate(BACKBONE_ORDER)}
    method_rank = {"Baseline": 0, "Fair-NCL": 1}

    def key(row: Dict[str, object]):
        values = []
        for field in fields:
            value = row.get(field)
            if field in {"Dataset", "dataset"}:
                values.append(dataset_rank.get(str(value), len(dataset_rank)))
            elif field in {"Backbone", "backbone"}:
                values.append(backbone_rank.get(str(value), len(backbone_rank)))
            elif field == "Method":
                values.append(method_rank.get(str(value), len(method_rank)))
            else:
                values.append(value)
        return values

    return sorted(rows, key=key)


def format_pct(value) -> str:
    value = float(value)
    if math.isnan(value):
        return "--"
    return f"{value:.2f}\\%"


def format_float(value, digits: int) -> str:
    value = float(value)
    if math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def tex_text(text: str) -> str:
    return tex_escape(text)


if __name__ == "__main__":
    main()

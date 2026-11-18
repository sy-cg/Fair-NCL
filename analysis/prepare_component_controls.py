"""Prepare the cross-backbone component-control table from five-seed results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tables/supplementary/final_5seed_extended"
PLAN = ROOT / "experiments/revision_supplementary_final_extended_595.jsonl"
METHODS = ("baseline", "ncl_only", "random_aug", "similarity_aug", "ncl_sr_mce", "fair_ncl")
LABELS = ("Plain", "VICReg only", "Random + VICReg", "Local + VICReg", "NCL-SR-ID", "Fair-NCL")
BACKBONES = ("sasrec", "bert4rec", "gru4rec", "caser")


def read_csv(name: str) -> list[dict[str, str]]:
    with (SUMMARY / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    runs = {(r["dataset"], r["backbone"], r["method"]): r
            for r in read_csv("run_summary.csv")}
    gaps = {(r["dataset"], r["backbone"], r["method"], r["attribute"]): r
            for r in read_csv("group_summary.csv")
            if r["metric"] == "NDCG@10" and r["attribute"] in {"gender", "age_group"}}
    with PLAN.open(encoding="utf-8") as handle:
        jobs = [json.loads(line) for line in handle if line.strip()]
    local_pools = {}
    local_seeds = {}
    for job in jobs:
        if job["method"] not in {"similarity_aug", "ncl_sr_mce"}:
            continue
        params = job["params"]
        key = job["dataset"], job["backbone"], job["method"]
        pool = (
            "semantic_hybrid"
            if params.get("use_similarity_scores", False)
            and params.get("similarity_source", "semantic_hybrid") == "semantic_hybrid"
            else "cooccurrence"
        )
        if key in local_pools:
            assert local_pools[key] == pool
        local_pools[key] = pool
        local_seeds.setdefault(key, set()).add(job["seed"])
    assert len(local_pools) == 15 and all(len(seeds) == 5 for seeds in local_seeds.values())

    rows = []
    table_rows = []
    for dataset in ("ml-1m", "taobao"):
        for backbone in BACKBONES:
            cells = []
            for method in METHODS:
                if (dataset, backbone, method) not in runs:
                    assert (dataset, backbone, method) == ("ml-1m", "caser", "ncl_sr_mce")
                    cells.append("---")
                    continue
                run = runs[dataset, backbone, method]
                gender = gaps[dataset, backbone, method, "gender"]
                age = gaps[dataset, backbone, method, "age_group"]
                assert int(run["seed_count"]) == 5
                utility = float(run["NDCG@10_mean"])
                mean_gap = (float(gender["absolute_gap_mean"])
                            + float(age["absolute_gap_mean"])) / 2
                failed = method == 'ncl_sr_mce' and (dataset, backbone) == ('taobao', 'caser')
                if failed:
                    for job in jobs:
                        if (job['dataset'], job['backbone'], job['method']) == (dataset, backbone, method):
                            history = json.loads((ROOT / 'results_revision_kbs_v2' / dataset /
                                                  job['job_id'] / 'history.json').read_text(encoding='utf-8'))
                            assert sum(history.get('skipped_nonfinite_updates', [])) > 0
                            assert not any(k.startswith('train_') and isinstance(v, list) and v
                                           for k, v in history.items())
                rows.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "seed_count": 5,
                    "ndcg_10_mean": utility,
                    "ndcg_10_std": run["NDCG@10_std"],
                    "gender_ndcg_10_abs_gap_mean": gender["absolute_gap_mean"],
                    "gender_ndcg_10_abs_gap_std": gender["absolute_gap_std"],
                    "age_ndcg_10_abs_gap_mean": age["absolute_gap_mean"],
                    "age_ndcg_10_abs_gap_std": age["absolute_gap_std"],
                    "mean_abs_ndcg_10_gap": mean_gap,
                    "candidate_pool": local_pools[dataset, backbone, method]
                    if method in {"similarity_aug", "ncl_sr_mce"} else "",
                    "training_status": "zero_finite_updates" if failed else "trained",
                })
                cells.append("TF" if failed else f"{utility:.4f}/{mean_gap:.4f}")
            name = {"sasrec": "SASRec", "bert4rec": "BERT4Rec",
                    "gru4rec": "GRU4Rec", "caser": "Caser"}[backbone]
            table_rows.append(f"{dataset.upper() if dataset == 'ml-1m' else 'Taobao'} & {name} & "
                              + " & ".join(cells) + r" \\")
    assert len(rows) == 47

    output = ROOT / "tables/supplementary/reviewer_component_controls.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = ROOT / "els-cas-templates/source_data/table_source_data/Supplementary_Data_S6_component_controls.csv"
    source.write_bytes(output.read_bytes())

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Five-seed component controls across both datasets and four backbones. Each numeric cell reports mean NDCG@10 / mean of the absolute gender and age-group NDCG@10 gaps; higher utility and lower gap are preferable. NCL-SR-ID independently reimplements transferable NCL-SR components on shared ID backbones. Its ML-1M/Caser configuration did not qualify; TF marks zero effective training updates in all five Taobao/Caser runs. Failed runs are excluded from learned-performance comparisons; Caser remains in the main comparison. The local control uses semantic-hybrid candidates only for SASRec. Unrounded records and status are in Supplementary Data~S6.}",
        r"\label{tab:cross_backbone_components}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        "Dataset & Backbone & " + " & ".join(LABELS) + r" \\",
        r"\midrule",
        *table_rows[:4],
        r"\midrule",
        *table_rows[4:],
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table*}",
    ]
    (ROOT / "els-cas-templates/component_controls_main_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} control rows and cross-backbone LaTeX table")


if __name__ == "__main__":
    main()

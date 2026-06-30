#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize unified processed sequential recommendation datasets.

Expected input:
    processed_data/<dataset>/processed_data.pkl

Outputs:
    data_statistics_summary.csv
    data_statistics_summary.xlsx
    data_statistics_latex.txt
    group_user_counts.csv
    split_group_sample_counts.csv

Example:
    python scripts/summarize_processed_data.py \
        --processed-root processed_data \
        --datasets ml-1m taobao lastfm-1k \
        --output-dir tables/data_stats
"""

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


DATASET_LABELS = {
    "ml-1m": "ML-1M",
    "lastfm-1k": "LastFM-1K",
    "taobao": "Taobao",
}

GENDER_LABELS = {
    0: "Gender-0",
    1: "Gender-1",
}

AGE_LABELS = {
    0: "Age-0",
    1: "Age-1",
}


def load_processed_data(processed_root: str, dataset: str) -> Dict[str, Any]:
    path = os.path.join(processed_root, dataset, "processed_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def reconstruct_sequences_from_test(test_data: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """
    Reconstruct effective user sequences from test samples:
        full_effective_seq = test_input_seq + [test_target]

    Warning:
        input_seq may have been truncated by max_seq_len in preprocessing.
    """
    sequences = {}
    for sample in test_data:
        user_id = int(sample["user_id"])
        seq = list(sample.get("input_seq", [])) + [int(sample["target"])]
        sequences[user_id] = seq
    return sequences


def get_sequences(data: Dict[str, Any]) -> Tuple[Dict[int, List[int]], str]:
    """
    Prefer full saved sequences if available. Otherwise reconstruct from test_data.
    """
    if "sequences" in data and isinstance(data["sequences"], dict):
        seqs = {int(u): [int(x) for x in seq] for u, seq in data["sequences"].items()}
        return seqs, "saved_full_sequences"

    test_data = data.get("test_data", [])
    seqs = reconstruct_sequences_from_test(test_data)
    return seqs, "reconstructed_from_test_effective_sequences"


def safe_density(num_interactions: int, num_users: int, num_items: int) -> float:
    denom = num_users * num_items
    if denom <= 0:
        return np.nan
    return num_interactions / denom


def get_users_frame(data: Dict[str, Any]) -> pd.DataFrame:
    users = data.get("users")
    if isinstance(users, pd.DataFrame):
        users_df = users.copy()
    else:
        users_df = pd.DataFrame(users)

    required = {"user_id", "gender", "age_group"}
    missing = required - set(users_df.columns)
    if missing:
        raise ValueError(f"Missing required user columns: {missing}")

    users_df = users_df.drop_duplicates("user_id").copy()
    users_df["user_id"] = users_df["user_id"].astype(int)
    users_df["gender"] = users_df["gender"].astype(int)
    users_df["age_group"] = users_df["age_group"].astype(int)
    return users_df


def count_group_users(users_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []

    for attr in ["gender", "age_group"]:
        counts = users_df[attr].value_counts(dropna=False).sort_index()
        for group, count in counts.items():
            rows.append({
                "dataset": dataset,
                "dataset_display": DATASET_LABELS.get(dataset, dataset),
                "attribute": attr,
                "group": int(group),
                "num_users": int(count),
                "ratio": float(count / len(users_df)) if len(users_df) > 0 else np.nan,
            })

    cross = users_df.groupby(["gender", "age_group"], dropna=False).size().reset_index(name="num_users")
    cross["dataset"] = dataset
    cross["dataset_display"] = DATASET_LABELS.get(dataset, dataset)
    cross["ratio"] = cross["num_users"] / len(users_df) if len(users_df) > 0 else np.nan
    cross = cross[["dataset", "dataset_display", "gender", "age_group", "num_users", "ratio"]]

    group_rows = pd.DataFrame(rows)
    return group_rows, cross


def count_split_group_samples(data: Dict[str, Any], dataset: str) -> pd.DataFrame:
    rows = []

    for split in ["train_data", "val_data", "test_data"]:
        samples = data.get(split, [])
        split_name = split.replace("_data", "")

        if not samples:
            continue

        df = pd.DataFrame(samples)

        for attr in ["gender", "age_group"]:
            if attr not in df.columns:
                continue

            counts = df[attr].value_counts(dropna=False).sort_index()
            for group, count in counts.items():
                rows.append({
                    "dataset": dataset,
                    "dataset_display": DATASET_LABELS.get(dataset, dataset),
                    "split": split_name,
                    "attribute": attr,
                    "group": int(group),
                    "num_samples": int(count),
                    "ratio": float(count / len(df)) if len(df) > 0 else np.nan,
                })

    return pd.DataFrame(rows)


def summarize_one_dataset(data: Dict[str, Any], dataset: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_df = get_users_frame(data)
    sequences, seq_source = get_sequences(data)

    num_users = int(data.get("num_users", users_df["user_id"].nunique()))
    num_items = int(data.get("num_items", np.nan))

    train_data = data.get("train_data", [])
    val_data = data.get("val_data", [])
    test_data = data.get("test_data", [])

    seq_lengths = [len(seq) for seq in sequences.values()]
    num_interactions = int(np.sum(seq_lengths)) if seq_lengths else 0

    gender_user_counts = users_df["gender"].value_counts().to_dict()
    age_user_counts = users_df["age_group"].value_counts().to_dict()

    summary = {
        "dataset": dataset,
        "dataset_display": DATASET_LABELS.get(dataset, dataset),
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "interaction_source": seq_source,
        "density": safe_density(num_interactions, num_users, num_items),
        "avg_seq_len": float(np.mean(seq_lengths)) if seq_lengths else np.nan,
        "median_seq_len": float(np.median(seq_lengths)) if seq_lengths else np.nan,
        "min_seq_len": int(np.min(seq_lengths)) if seq_lengths else 0,
        "max_seq_len": int(np.max(seq_lengths)) if seq_lengths else 0,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "num_test_samples": len(test_data),
        "gender_0_users": int(gender_user_counts.get(0, 0)),
        "gender_1_users": int(gender_user_counts.get(1, 0)),
        "age_group_0_users": int(age_user_counts.get(0, 0)),
        "age_group_1_users": int(age_user_counts.get(1, 0)),
    }

    group_user_counts, gender_age_cross = count_group_users(users_df, dataset)
    split_group_counts = count_split_group_samples(data, dataset)

    return summary, group_user_counts, gender_age_cross, split_group_counts


def format_int(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(x):,}"


def format_float(x, digits: int = 4) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def make_latex_table(summary_df: pd.DataFrame) -> str:
    """
    Produce a compact paper-style LaTeX dataset statistics table.
    """
    rows = []
    for _, r in summary_df.iterrows():
        rows.append(
            f"{r['dataset_display']} & "
            f"{format_int(r['num_users'])} & "
            f"{format_int(r['num_items'])} & "
            f"{format_int(r['num_interactions'])} & "
            f"{format_float(r['avg_seq_len'], 2)} & "
            f"{format_float(r['density'] * 100, 4)}\\% & "
            f"{format_int(r['gender_0_users'])}/{format_int(r['gender_1_users'])} & "
            f"{format_int(r['age_group_0_users'])}/{format_int(r['age_group_1_users'])} \\\\"
        )

    body = "\n".join(rows)

    latex = rf"""
\begin{{table}}[t]
\centering
\small
\caption{{Statistics of the processed datasets. Density is computed as interactions divided by users times items. Gender and age columns report the number of users in group 0 and group 1.}}
\label{{tab:dataset_statistics}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Dataset & Users & Items & Interactions & Avg. Len. & Density & Gender 0/1 & Age 0/1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()
    return latex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ml-1m", "lastfm-1k", "taobao"],
        help="Datasets to summarize. Use dataset directory names.",
    )
    parser.add_argument("--output-dir", default=os.path.join("tables", "data_stats"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    group_user_frames = []
    gender_age_frames = []
    split_group_frames = []

    for dataset in args.datasets:
        print(f"[INFO] Summarizing {dataset} ...")
        data = load_processed_data(args.processed_root, dataset)
        summary, group_users, gender_age_cross, split_groups = summarize_one_dataset(data, dataset)

        summary_rows.append(summary)
        group_user_frames.append(group_users)
        gender_age_frames.append(gender_age_cross)
        split_group_frames.append(split_groups)

    summary_df = pd.DataFrame(summary_rows)
    group_user_df = pd.concat(group_user_frames, ignore_index=True) if group_user_frames else pd.DataFrame()
    gender_age_df = pd.concat(gender_age_frames, ignore_index=True) if gender_age_frames else pd.DataFrame()
    split_group_df = pd.concat(split_group_frames, ignore_index=True) if split_group_frames else pd.DataFrame()

    # Save CSV files.
    summary_csv = os.path.join(args.output_dir, "data_statistics_summary.csv")
    group_csv = os.path.join(args.output_dir, "group_user_counts.csv")
    gender_age_csv = os.path.join(args.output_dir, "gender_age_cross_counts.csv")
    split_group_csv = os.path.join(args.output_dir, "split_group_sample_counts.csv")

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    group_user_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    gender_age_df.to_csv(gender_age_csv, index=False, encoding="utf-8-sig")
    split_group_df.to_csv(split_group_csv, index=False, encoding="utf-8-sig")

    # Save Excel workbook.
    excel_path = os.path.join(args.output_dir, "data_statistics_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        group_user_df.to_excel(writer, sheet_name="group_user_counts", index=False)
        gender_age_df.to_excel(writer, sheet_name="gender_age_cross", index=False)
        split_group_df.to_excel(writer, sheet_name="split_group_samples", index=False)

    # Save LaTeX table.
    latex = make_latex_table(summary_df)
    latex_path = os.path.join(args.output_dir, "data_statistics_latex.txt")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    print("\n[INFO] Saved outputs:")
    print(f"  - {summary_csv}")
    print(f"  - {group_csv}")
    print(f"  - {gender_age_csv}")
    print(f"  - {split_group_csv}")
    print(f"  - {excel_path}")
    print(f"  - {latex_path}")

    print("\n[INFO] Compact summary:")
    print(summary_df[[
        "dataset_display",
        "num_users",
        "num_items",
        "num_interactions",
        "avg_seq_len",
        "density",
        "gender_0_users",
        "gender_1_users",
        "age_group_0_users",
        "age_group_1_users",
        "interaction_source",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
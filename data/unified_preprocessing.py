import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class DatasetBuildConfig:
    dataset: str
    raw_dir: str
    output_dir: str = "processed_data"
    max_seq_len: int = 100
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    age_threshold: int = 35
    rating_threshold: int = 4
    taobao_click_only: bool = True
    lastfm_item_level: str = "artist"
    max_rows: Optional[int] = None


class UnifiedSequentialPreprocessor:
    """Build a common sequential recommendation format for all datasets.

    Output format intentionally matches the experiment loader contract:
    train_data / val_data / test_data are lists of dictionaries with
    user_id, input_seq, target, gender, age_group.

    Real item ids are always 1..num_items and 0 is reserved for padding.
    """

    SUPPORTED_DATASETS = {"ml-1m", "lastfm-1k", "taobao"}

    def __init__(self, build_config: DatasetBuildConfig):
        dataset = build_config.dataset.lower()
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{build_config.dataset}'. "
                f"Choose from {sorted(self.SUPPORTED_DATASETS)}."
            )
        self.cfg = build_config
        self.cfg.dataset = dataset

    def build(self) -> Dict:
        interactions, users, items, metadata = self._load_raw_tables()
        interactions, users, items = self._filter_and_prepare(interactions, users, items)
        interactions, users, items, user_map, item_map = self._remap_ids(interactions, users, items)

        sequences = self._build_sequences(interactions)
        train_data, val_data, test_data = self._split_sequences(sequences, users)

        data = {
            "dataset": self.cfg.dataset,
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,

            # Save full, non-truncated user sequences for accurate dataset statistics.
            # Note: input_seq in train/val/test samples is truncated by max_seq_len,
            # but sequences here preserves the complete processed sequence.
            "sequences": sequences,

            "users": users,
            "movies": items,
            "items": items,
            "user_map": user_map,
            "item_map": item_map,
            "item_id_start": 1,
            "num_users": len(user_map),
            "num_items": len(item_map),
            "metadata": metadata,
            "build_config": asdict(self.cfg),
        }

        output_path = self._save(data)
        print(f"Saved unified processed data to {output_path}")
        return data

    def _load_raw_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        if self.cfg.dataset == "ml-1m":
            return self._load_ml_1m()
        if self.cfg.dataset == "lastfm-1k":
            return self._load_lastfm_1k()
        if self.cfg.dataset == "taobao":
            return self._load_taobao()
        raise AssertionError("unreachable")

    def _load_ml_1m(self):
        ratings = pd.read_csv(
            os.path.join(self.cfg.raw_dir, "ratings.dat"),
            sep="::",
            names=["raw_user_id", "raw_item_id", "rating", "timestamp"],
            engine="python",
            dtype={
                "raw_user_id": "int64",
                "raw_item_id": "int64",
                "rating": "int8",
                "timestamp": "int64",
            },
            nrows=self.cfg.max_rows,
        )
        ratings = ratings[ratings["rating"] >= self.cfg.rating_threshold].copy()

        users = pd.read_csv(
            os.path.join(self.cfg.raw_dir, "users.dat"),
            sep="::",
            names=["raw_user_id", "gender_raw", "age", "occupation", "zip_code"],
            engine="python",
            dtype={
                "raw_user_id": "int64",
                "age": "int64",
                "occupation": "int64",
            },
        )
        users["gender"] = users["gender_raw"].map({"M": 0, "F": 1})
        users["age_group"] = (users["age"] > self.cfg.age_threshold).astype("int64")

        items = pd.read_csv(
            os.path.join(self.cfg.raw_dir, "movies.dat"),
            sep="::",
            names=["raw_item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
            dtype={"raw_item_id": "int64"},
        )
        items["item_text"] = (
            items["title"].astype(str) + ". Genres: " + items["genres"].astype(str)
        )

        metadata = {
            "sequence_semantics": "true_timestamp_sequence",
            "positive_signal": f"rating>={self.cfg.rating_threshold}",
            "sensitive_attributes": ["gender", "age_group"],
            "report_sensitive_attributes": ["gender", "age_group"],
            "gender_mapping": {"M": 0, "F": 1},
            "age_group_rule": f"age>{self.cfg.age_threshold}",
        }
        return ratings, users, items, metadata

    def _load_lastfm_1k(self):
        profiles = pd.read_csv(
            os.path.join(self.cfg.raw_dir, "userid-profile.tsv"),
            sep="\t",
            header=0,
            dtype={
                "#id": "string",
                "gender": "string",
                "age": "string",
                "country": "string",
                "registered": "string",
            },
        )
        profiles = profiles.rename(
            columns={
                "#id": "raw_user_id",
                "gender": "gender_raw",
                "registered": "signup",
            }
        )
        profiles["age"] = pd.to_numeric(profiles["age"], errors="coerce")
        profiles = profiles.dropna(subset=["gender_raw", "age"]).copy()
        profiles = profiles[profiles["gender_raw"].isin(["m", "f"])].copy()
        profiles["gender"] = profiles["gender_raw"].map({"m": 0, "f": 1}).astype("int64")
        profiles["age_group"] = (profiles["age"] > self.cfg.age_threshold).astype("int64")

        listens = self._read_lastfm_listens(
            os.path.join(
                self.cfg.raw_dir,
                "userid-timestamp-artid-artname-traid-traname.tsv",
            )
        )

        timestamp = pd.to_datetime(listens["timestamp_raw"], errors="coerce", utc=True)
        valid_time = timestamp.notna()
        listens = listens.loc[valid_time].copy()
        timestamp = timestamp.loc[valid_time]
        listens["timestamp"] = timestamp.astype("int64") // 10**9

        if self.cfg.lastfm_item_level == "track":
            listens["raw_item_id"] = listens["track_mbid"].fillna("")
            missing = listens["raw_item_id"] == ""
            listens.loc[missing, "raw_item_id"] = (
                "track:"
                + listens.loc[missing, "artist_name"].astype(str)
                + "::"
                + listens.loc[missing, "track_name"].astype(str)
            )
            item_cols = ["raw_item_id", "artist_name", "track_name"]
            items = listens[item_cols].drop_duplicates("raw_item_id").copy()
            items["item_text"] = (
                items["artist_name"].astype(str)
                + " - "
                + items["track_name"].astype(str)
            )
        else:
            listens["raw_item_id"] = listens["artist_mbid"].fillna("")
            missing = listens["raw_item_id"] == ""
            listens.loc[missing, "raw_item_id"] = (
                "artist:" + listens.loc[missing, "artist_name"].astype(str)
            )
            items = listens[["raw_item_id", "artist_name"]].drop_duplicates("raw_item_id").copy()
            items["item_text"] = items["artist_name"].astype(str)

        listens = listens.drop_duplicates(["raw_user_id", "raw_item_id", "timestamp"]).copy()
        interactions = listens[["raw_user_id", "raw_item_id", "timestamp"]].copy()

        metadata = {
            "sequence_semantics": "true_timestamp_sequence",
            "positive_signal": "listening event",
            "item_level": self.cfg.lastfm_item_level,
            "sensitive_attributes": ["gender", "age_group"],
            "report_sensitive_attributes": ["gender", "age_group"],
            "gender_mapping": {"m": 0, "f": 1},
            "age_group_rule": f"age>{self.cfg.age_threshold}",
        }
        return interactions, profiles, items, metadata

    def _read_lastfm_listens(self, path: str) -> pd.DataFrame:
        columns = [
            "raw_user_id",
            "timestamp_raw",
            "artist_mbid",
            "artist_name",
            "track_mbid",
            "track_name",
        ]
        dtype = {
            "raw_user_id": "string",
            "timestamp_raw": "string",
            "artist_mbid": "string",
            "artist_name": "string",
            "track_mbid": "string",
            "track_name": "string",
        }
        read_kwargs = {
            "sep": "\t",
            "names": columns,
            "dtype": dtype,
            "nrows": self.cfg.max_rows,
            "on_bad_lines": "skip",
            "encoding_errors": "replace",
        }

        try:
            listens = pd.read_csv(path, engine="c", **read_kwargs)
        except Exception as exc:
            print(
                "Fast LastFM parser failed; retrying with the Python parser and "
                f"skipping malformed rows. Details: {exc}"
            )
            listens = pd.read_csv(path, engine="python", **read_kwargs)

        return listens

    def _load_taobao(self):
        users = pd.read_csv(os.path.join(self.cfg.raw_dir, "user_profile.csv"))
        users = users.rename(columns={"userid": "raw_user_id"})
        users["gender"] = users["final_gender_code"].map({1: 0, 2: 1})
        users["age_level"] = pd.to_numeric(users["age_level"], errors="coerce")
        users = users.dropna(subset=["gender", "age_level"]).copy()
        users["gender"] = users["gender"].astype("int64")
        users["age_group"] = (users["age_level"] >= 4).astype("int64")

        raw = pd.read_csv(
            os.path.join(self.cfg.raw_dir, "raw_sample.csv"),
            dtype={
                "user": "int64",
                "time_stamp": "int64",
                "adgroup_id": "int64",
                "pid": "string",
                "nonclk": "int8",
                "clk": "int8",
            },
            nrows=self.cfg.max_rows,
        )

        if self.cfg.taobao_click_only:
            raw = raw[raw["clk"] == 1].copy()

        interactions = raw.rename(
            columns={
                "user": "raw_user_id",
                "adgroup_id": "raw_item_id",
                "time_stamp": "timestamp",
            }
        )[["raw_user_id", "raw_item_id", "timestamp", "clk"]].copy()

        items = pd.read_csv(os.path.join(self.cfg.raw_dir, "ad_feature.csv"))
        items = items.rename(columns={"adgroup_id": "raw_item_id"})
        items["item_text"] = (
            "cate:"
            + items["cate_id"].astype(str)
            + " brand:"
            + items["brand"].astype(str)
            + " price:"
            + items["price"].astype(str)
        )

        metadata = {
            "sequence_semantics": "true_timestamp_sequence",
            "positive_signal": "clk=1" if self.cfg.taobao_click_only else "ad exposure with clicks retained",
            "sensitive_attributes": ["gender", "age_group"],
            "report_sensitive_attributes": ["gender", "age_group"],
            "gender_mapping": {"final_gender_code=1": 0, "final_gender_code=2": 1},
            "age_group_rule": "age_level>=4",
        }
        return interactions, users, items, metadata

    def _filter_and_prepare(self, interactions, users, items):
        interactions = interactions.dropna(subset=["raw_user_id", "raw_item_id", "timestamp"]).copy()
        users = users.dropna(subset=["raw_user_id", "gender", "age_group"]).copy()

        interactions = interactions[
            interactions["raw_user_id"].isin(set(users["raw_user_id"]))
        ].copy()

        if "raw_item_id" in items.columns:
            interactions = interactions[
                interactions["raw_item_id"].isin(set(items["raw_item_id"]))
            ].copy()

        prev_shape = None
        while True:
            user_counts = interactions["raw_user_id"].value_counts()
            item_counts = interactions["raw_item_id"].value_counts()

            valid_users = set(user_counts[user_counts >= self.cfg.min_user_interactions].index)
            valid_items = set(item_counts[item_counts >= self.cfg.min_item_interactions].index)

            interactions = interactions[
                interactions["raw_user_id"].isin(valid_users)
                & interactions["raw_item_id"].isin(valid_items)
            ].copy()

            shape = (
                interactions["raw_user_id"].nunique(),
                interactions["raw_item_id"].nunique(),
                len(interactions),
            )

            if shape == prev_shape:
                break

            prev_shape = shape
            print(
                f"k-core filtering: users={shape[0]}, "
                f"items={shape[1]}, interactions={shape[2]}"
            )

            if len(interactions) == 0:
                break

        users = users[users["raw_user_id"].isin(set(interactions["raw_user_id"]))].copy()
        items = items[items["raw_item_id"].isin(set(interactions["raw_item_id"]))].copy()

        if interactions.empty:
            raise ValueError(
                f"No interactions left after filtering for dataset={self.cfg.dataset}. "
                "Please check min_user_interactions, min_item_interactions, and raw data path."
            )

        return interactions, users, items

    def _remap_ids(self, interactions, users, items):
        raw_users = sorted(interactions["raw_user_id"].unique())
        raw_items = sorted(interactions["raw_item_id"].unique())

        user_map = {raw_id: idx for idx, raw_id in enumerate(raw_users)}
        item_map = {raw_id: idx for idx, raw_id in enumerate(raw_items, start=1)}

        interactions["user_id"] = interactions["raw_user_id"].map(user_map).astype("int64")
        interactions["movie_id"] = interactions["raw_item_id"].map(item_map).astype("int64")

        users["user_id"] = users["raw_user_id"].map(user_map).astype("int64")
        items["movie_id"] = items["raw_item_id"].map(item_map).astype("int64")

        return interactions, users, items, user_map, item_map

    def _build_sequences(self, interactions):
        interactions = interactions.sort_values(["user_id", "timestamp", "movie_id"])

        sequences = {}
        for user_id, group in tqdm(
            interactions.groupby("user_id"),
            desc=f"Build {self.cfg.dataset} sequences",
        ):
            seq = group["movie_id"].astype("int64").tolist()
            if len(seq) >= 3:
                sequences[int(user_id)] = seq

        if not sequences:
            raise ValueError(
                f"No valid user sequences were built for dataset={self.cfg.dataset}. "
                "Each sequence must contain at least 3 interactions."
            )

        return sequences

    def _split_sequences(self, sequences, users):
        user_info = users.set_index("user_id").to_dict("index")
        train_data, val_data, test_data = [], [], []

        for user_id, sequence in tqdm(
            sequences.items(),
            desc=f"Split {self.cfg.dataset} sequences",
        ):
            if user_id not in user_info or len(sequence) < 3:
                continue

            info = user_info[user_id]
            attrs = {
                "gender": int(info["gender"]),
                "age_group": int(info["age_group"]),
            }

            for i in range(1, len(sequence) - 2):
                train_data.append(
                    self._make_sample(
                        user_id=user_id,
                        input_seq=sequence[:i],
                        target=sequence[i],
                        attrs=attrs,
                    )
                )

            val_data.append(
                self._make_sample(
                    user_id=user_id,
                    input_seq=sequence[:-2],
                    target=sequence[-2],
                    attrs=attrs,
                )
            )
            test_data.append(
                self._make_sample(
                    user_id=user_id,
                    input_seq=sequence[:-1],
                    target=sequence[-1],
                    attrs=attrs,
                )
            )

        print(
            f"Split done: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )

        return train_data, val_data, test_data

    def _make_sample(self, user_id, input_seq, target, attrs):
        input_seq = input_seq[-self.cfg.max_seq_len:]

        sample = {
            "user_id": int(user_id),
            "input_seq": [int(x) for x in input_seq],
            "target": int(target),
        }
        sample.update(attrs)
        return sample

    def _save(self, data):
        dataset_dir = os.path.join(self.cfg.output_dir, self.cfg.dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        output_path = os.path.join(dataset_dir, "processed_data.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        summary = self._build_summary(data)

        summary_path = os.path.join(dataset_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return output_path

    def _build_summary(self, data):
        sequences = data.get("sequences", {})
        seq_lengths = [len(seq) for seq in sequences.values()]

        num_interactions = int(sum(seq_lengths)) if seq_lengths else 0
        num_users = int(data["num_users"])
        num_items = int(data["num_items"])

        density = (
            float(num_interactions / (num_users * num_items))
            if num_users > 0 and num_items > 0
            else 0.0
        )

        users = data.get("users")
        if isinstance(users, pd.DataFrame) and not users.empty:
            gender_user_counts = self._value_counts_as_dict(users, "gender")
            age_group_user_counts = self._value_counts_as_dict(users, "age_group")
            gender_age_user_counts = self._cross_counts_as_dict(users, ["gender", "age_group"])
        else:
            gender_user_counts = {}
            age_group_user_counts = {}
            gender_age_user_counts = {}

        summary = {
            "dataset": data["dataset"],
            "num_users": num_users,
            "num_items": num_items,
            "num_interactions": num_interactions,
            "density": density,
            "avg_seq_len": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
            "median_seq_len": float(np.median(seq_lengths)) if seq_lengths else 0.0,
            "min_seq_len": int(np.min(seq_lengths)) if seq_lengths else 0,
            "max_seq_len": int(np.max(seq_lengths)) if seq_lengths else 0,
            "num_train": int(len(data["train_data"])),
            "num_val": int(len(data["val_data"])),
            "num_test": int(len(data["test_data"])),
            "gender_user_counts": gender_user_counts,
            "age_group_user_counts": age_group_user_counts,
            "gender_age_user_counts": gender_age_user_counts,
            "metadata": data["metadata"],
            "build_config": data["build_config"],
        }

        return summary

    @staticmethod
    def _value_counts_as_dict(df: pd.DataFrame, column: str) -> Dict[str, int]:
        if column not in df.columns:
            return {}

        counts = df[column].value_counts(dropna=False).sort_index()
        result = {}
        for key, value in counts.items():
            if pd.isna(key):
                result["nan"] = int(value)
            else:
                result[str(int(key))] = int(value)
        return result

    @staticmethod
    def _cross_counts_as_dict(df: pd.DataFrame, columns) -> Dict[str, int]:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            return {}

        grouped = df.groupby(columns, dropna=False).size()
        result = {}

        for key, value in grouped.items():
            if not isinstance(key, tuple):
                key = (key,)

            key_parts = []
            for item in key:
                if pd.isna(item):
                    key_parts.append("nan")
                else:
                    key_parts.append(str(int(item)))

            result["|".join(key_parts)] = int(value)

        return result


def default_raw_dir(dataset: str) -> str:
    mapping = {
        "ml-1m": os.path.join("data", "ml-1m"),
        "lastfm-1k": os.path.join("data", "lastfm-dataset-1K"),
        "taobao": os.path.join("data", "Taobao"),
    }
    return mapping[dataset]


def main():
    parser = argparse.ArgumentParser(
        description="Build unified sequential recommendation datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(UnifiedSequentialPreprocessor.SUPPORTED_DATASETS),
        required=True,
    )
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default="processed_data")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: cap raw interaction rows.")
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--age-threshold", type=int, default=35)
    parser.add_argument("--rating-threshold", type=int, default=4)
    parser.add_argument("--lastfm-item-level", choices=["artist", "track"], default="artist")

    parser.add_argument(
        "--taobao-click-only",
        action="store_true",
        default=True,
        help="Use only clicked Taobao interactions. Enabled by default.",
    )
    parser.add_argument(
        "--taobao-include-nonclick",
        action="store_true",
        help="If set, keep Taobao exposure rows instead of click-only rows.",
    )

    args = parser.parse_args()

    taobao_click_only = args.taobao_click_only
    if args.taobao_include_nonclick:
        taobao_click_only = False

    cfg = DatasetBuildConfig(
        dataset=args.dataset,
        raw_dir=args.raw_dir or default_raw_dir(args.dataset),
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        max_seq_len=args.max_seq_len,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        age_threshold=args.age_threshold,
        rating_threshold=args.rating_threshold,
        taobao_click_only=taobao_click_only,
        lastfm_item_level=args.lastfm_item_level,
    )

    UnifiedSequentialPreprocessor(cfg).build()


if __name__ == "__main__":
    main()
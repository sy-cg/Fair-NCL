from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _resolve_num_items(num_items_or_items, data: List[Dict]) -> int:
    if isinstance(num_items_or_items, int):
        return int(num_items_or_items)
    if num_items_or_items is not None:
        try:
            return int(max(num_items_or_items))
        except ValueError:
            pass

    max_item = 0
    for sample in data:
        target = int(sample.get("target", 0))
        if target > max_item:
            max_item = target
        seq = sample.get("input_seq", [])
        if seq:
            seq_max = int(max(seq))
            if seq_max > max_item:
                max_item = seq_max
    return max_item


def _build_user_history_dict(*splits: Iterable[Dict]) -> Dict[int, set]:
    history: Dict[int, set] = {}
    for split in splits:
        for sample in split:
            user_id = int(sample["user_id"])
            user_history = history.setdefault(user_id, set())
            user_history.update(int(x) for x in sample.get("input_seq", []) if int(x) > 0)
            target = int(sample.get("target", 0))
            if target > 0:
                user_history.add(target)
    return history


def _max_item_from_splits(*splits: Iterable[Dict]) -> int:
    max_item = 0
    for split in splits:
        for sample in split:
            target = int(sample.get("target", 0))
            if target > max_item:
                max_item = target
            seq = sample.get("input_seq", [])
            if seq:
                seq_max = int(max(seq))
                if seq_max > max_item:
                    max_item = seq_max
    return max_item


def _batch_tensor(values, dtype=torch.long) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=dtype)

    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.dim() == 0:
            return torch.tensor([int(v.item()) for v in values], dtype=dtype)
        return torch.stack([v.to(dtype=dtype) for v in values], dim=0)

    if isinstance(first, np.ndarray):
        if first.ndim == 1 and first.size == 0:
            return torch.empty((len(values), 0), dtype=dtype)
        return torch.as_tensor(np.stack(values, axis=0), dtype=dtype)

    return torch.tensor(values, dtype=dtype)


class OptimizedMovieLensDataset(Dataset):
    """Compact dataset representation to reduce Python-object and worker memory."""

    def __init__(self,
                 data,
                 max_seq_len,
                 all_items,
                 user_history_dict,
                 num_negative=0,
                 cache_dir=None,
                 seed=42):
        self.max_seq_len = int(max_seq_len)
        self.cache_dir = cache_dir
        self.num_items = _resolve_num_items(all_items, data)
        self.user_history_dict = user_history_dict
        self.num_negative = int(num_negative)
        self.rng = np.random.default_rng(seed)

        self._build_compact_arrays(data)

    def _build_compact_arrays(self, data: List[Dict]) -> None:
        num_samples = len(data)
        id_dtype = np.int32 if self.num_items <= np.iinfo(np.int32).max else np.int64
        seq_len_dtype = np.int16 if self.max_seq_len <= np.iinfo(np.int16).max else np.int32

        self.user_ids = np.empty(num_samples, dtype=np.int32)
        self.targets = np.empty(num_samples, dtype=id_dtype)
        self.genders = np.empty(num_samples, dtype=np.int8)
        self.age_groups = np.empty(num_samples, dtype=np.int8)
        self.seq_lens = np.empty(num_samples, dtype=seq_len_dtype)
        self.input_seqs = np.zeros((num_samples, self.max_seq_len), dtype=id_dtype)

        for idx, sample in enumerate(data):
            seq = [int(x) for x in sample.get("input_seq", [])[-self.max_seq_len:] if int(x) > 0]
            seq_len = len(seq)
            if seq_len:
                self.input_seqs[idx, -seq_len:] = np.asarray(seq, dtype=id_dtype)

            self.user_ids[idx] = int(sample["user_id"])
            self.targets[idx] = int(sample["target"])
            self.genders[idx] = int(sample["gender"])
            self.age_groups[idx] = int(sample["age_group"])
            self.seq_lens[idx] = seq_len

    def __len__(self):
        return int(self.user_ids.shape[0])

    def _sample_negative_items(self, idx: int) -> np.ndarray:
        if self.num_negative <= 0:
            return np.empty((0,), dtype=np.int64)

        user_id = int(self.user_ids[idx])
        target = int(self.targets[idx])
        user_seen = self.user_history_dict.get(user_id, set())

        negatives: List[int] = []
        negative_set = set()
        max_available = max(0, self.num_items - len(user_seen) - (0 if target in user_seen else 1))
        target_count = min(self.num_negative, max_available) if max_available > 0 else 0

        attempts = 0
        max_attempts = max(64, self.num_negative * 16)
        while len(negatives) < target_count and attempts < max_attempts:
            candidate = int(self.rng.integers(1, self.num_items + 1))
            attempts += 1
            if candidate == target or candidate in user_seen or candidate in negative_set:
                continue
            negatives.append(candidate)
            negative_set.add(candidate)

        candidate = int(self.rng.integers(1, self.num_items + 1))
        while len(negatives) < target_count:
            if candidate > self.num_items:
                candidate = 1
            if candidate != target and candidate not in user_seen and candidate not in negative_set:
                negatives.append(candidate)
                negative_set.add(candidate)
            candidate += 1

        if len(negatives) < self.num_negative:
            fill_value = negatives[-1] if negatives else max(1, min(self.num_items, target + 1))
            negatives.extend([fill_value] * (self.num_negative - len(negatives)))

        return np.asarray(negatives, dtype=np.int64)

    def __getitem__(self, idx):
        return {
            "user_id": int(self.user_ids[idx]),
            "input_seq": self.input_seqs[idx],
            "target": int(self.targets[idx]),
            "gender": int(self.genders[idx]),
            "age_group": int(self.age_groups[idx]),
            "seq_len": int(self.seq_lens[idx]),
            "negative_items": self._sample_negative_items(idx),
        }


def custom_collate_fn(batch):
    """Build tensors in batch scope instead of storing per-sample tensors in RAM."""
    return {
        "user_id": _batch_tensor([sample["user_id"] for sample in batch], dtype=torch.long),
        "input_seq": _batch_tensor([sample["input_seq"] for sample in batch], dtype=torch.long),
        "target": _batch_tensor([sample["target"] for sample in batch], dtype=torch.long),
        "gender": _batch_tensor([sample["gender"] for sample in batch], dtype=torch.long),
        "age_group": _batch_tensor([sample["age_group"] for sample in batch], dtype=torch.long),
        "seq_len": _batch_tensor([sample["seq_len"] for sample in batch], dtype=torch.long),
        "negative_items": _batch_tensor([sample["negative_items"] for sample in batch], dtype=torch.long),
    }


class GPUOptimizedDataLoader:
    """Thin wrapper that applies conservative worker settings on Windows."""

    def __init__(self, dataset, batch_size, shuffle=True, config=None):
        self.config = config
        num_workers = max(0, int(getattr(config, "num_workers", 0)))
        persistent_workers = bool(getattr(config, "persistent_workers", False) and num_workers > 0)
        prefetch_factor = getattr(config, "prefetch_factor", None) if num_workers > 0 else None
        pin_memory = bool(getattr(config, "pin_memory", False) and getattr(config.device, "type", "cpu") == "cuda")

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=bool(shuffle),
            collate_fn=custom_collate_fn,
        )

    def __iter__(self):
        for batch in self.loader:
            yield self._move_batch_to_gpu(batch)

    def _move_batch_to_gpu(self, batch):
        if not isinstance(batch, dict):
            return batch

        return {
            key: value.to(
                self.config.device,
                non_blocking=bool(getattr(self.config, "non_blocking", True)),
            ) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def __len__(self):
        return len(self.loader)


def create_optimized_data_loaders(train_data, val_data, test_data, config):
    """Create legacy loaders with compact in-memory storage."""
    print("Creating optimized data loaders...")

    num_items = int(getattr(config, "num_items", 0) or _max_item_from_splits(train_data, val_data, test_data))
    user_history_dict = _build_user_history_dict(train_data, val_data, test_data)

    train_dataset = OptimizedMovieLensDataset(
        train_data,
        config.max_seq_len,
        num_items,
        user_history_dict,
        seed=getattr(config, "seed", 42),
        num_negative=0,
        cache_dir=config.cache_dir,
    )
    val_dataset = OptimizedMovieLensDataset(
        val_data,
        config.max_seq_len,
        num_items,
        user_history_dict,
        seed=getattr(config, "seed", 42) + 1,
        num_negative=getattr(config, "eval_num_negative", 0),
        cache_dir=config.cache_dir,
    )
    test_dataset = OptimizedMovieLensDataset(
        test_data,
        config.max_seq_len,
        num_items,
        user_history_dict,
        seed=getattr(config, "seed", 42) + 2,
        num_negative=getattr(config, "eval_num_negative", 0),
        cache_dir=config.cache_dir,
    )

    train_loader = GPUOptimizedDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, config=config)
    val_loader = GPUOptimizedDataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, config=config)
    test_loader = GPUOptimizedDataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, config=config)

    print("Data loaders created successfully:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    return train_loader, val_loader, test_loader


def create_research_data_loaders(train_data, val_data, test_data, config):
    """Create loaders for the refactored research pipeline."""
    num_items = int(getattr(config, "num_items", 0) or _max_item_from_splits(train_data, val_data, test_data))
    user_history_dict = _build_user_history_dict(train_data, val_data, test_data)

    train_dataset = OptimizedMovieLensDataset(
        train_data,
        config.max_seq_len,
        num_items,
        user_history_dict,
        seed=getattr(config, "seed", 42),
        num_negative=getattr(config, "train_num_negative", 1),
        cache_dir=config.cache_dir,
    )
    val_dataset = OptimizedMovieLensDataset(
        val_data,
        config.max_seq_len,
        num_items,
        user_history_dict,
        seed=getattr(config, "seed", 42) + 1,
        num_negative=getattr(config, "eval_num_negative", 0),
        cache_dir=config.cache_dir,
    )
    test_dataset = OptimizedMovieLensDataset(
        test_data,
        config.max_seq_len,
        num_items,
        user_history_dict,
        seed=getattr(config, "seed", 42) + 2,
        num_negative=getattr(config, "eval_num_negative", 0),
        cache_dir=config.cache_dir,
    )

    train_loader = GPUOptimizedDataLoader(train_dataset, config.batch_size, shuffle=True, config=config)
    val_loader = GPUOptimizedDataLoader(val_dataset, config.eval_batch_size, shuffle=False, config=config)
    test_loader = GPUOptimizedDataLoader(test_dataset, config.eval_batch_size, shuffle=False, config=config)
    return train_loader, val_loader, test_loader

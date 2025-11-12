import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import lru_cache
import pickle
import os


class OptimizedMovieLensDataset(Dataset):
    """GPU优化的MovieLens数据集"""

    def __init__(self, data, max_seq_len, all_items, user_history_dict, num_negative=99, cache_dir=None):
        self.data = data
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir
        self.all_items = list(set(all_items))
        self.user_history_dict = user_history_dict  # {user_id: set of seen items}
        self.num_negative = num_negative

        self._preprocess_sequences()

    def _preprocess_sequences(self):
        """预处理所有序列，优化后续访问速度"""
        print("Preprocessing sequences for GPU optimization...")

        self.preprocessed_data = []

        for sample in self.data:
            # 序列填充和截断
            input_seq = sample['input_seq']
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]

            # 填充到固定长度
            seq_len = len(input_seq)
            padded_seq = [0] * (self.max_seq_len - seq_len) + input_seq

            # 生成负样本（避免采样已看过的）
            user_id = sample['user_id']
            user_seen = self.user_history_dict.get(user_id, set())
            neg_candidates = list(set(self.all_items) - user_seen)

            neg_items = np.random.choice(neg_candidates, size=self.num_negative,
                                         replace=len(neg_candidates) < self.num_negative)

            processed_sample = {
                'user_id': user_id,
                'input_seq': torch.LongTensor(padded_seq),
                'target': torch.LongTensor([sample['target']]),
                'gender': torch.LongTensor([sample['gender']]),
                'age_group': torch.LongTensor([sample['age_group']]),
                'gender_age': torch.LongTensor([sample['gender_age']]),
                'seq_len': seq_len,
                'negative_items': torch.LongTensor(neg_items)
            }

            self.preprocessed_data.append(processed_sample)

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]


def custom_collate_fn(batch):
    batch_dict = {
        'user_id': [],
        'input_seq': [],
        'target': [],
        'gender': [],
        'age_group': [],
        'gender_age': [],
        'seq_len': [],
        'negative_items': []
    }

    for sample in batch:
        for key in batch_dict:
            batch_dict[key].append(sample[key])

    return {
        'user_id': torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in batch_dict['user_id']]),
        'input_seq': torch.stack(batch_dict['input_seq']),
        'target': torch.stack(batch_dict['target']).squeeze(1),
        'gender': torch.stack(batch_dict['gender']).squeeze(1),
        'age_group': torch.stack(batch_dict['age_group']).squeeze(1),
        'gender_age': torch.stack(batch_dict['gender_age']).squeeze(1),
        'seq_len': torch.LongTensor(batch_dict['seq_len']),
        'negative_items': torch.stack(batch_dict['negative_items'])
    }



class GPUOptimizedDataLoader:
    """GPU优化的数据加载器包装器"""

    def __init__(self, dataset, batch_size, shuffle=True, config=None):
        self.config = config

        # 使用优化的数据加载器参数和自定义collate函数
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers if config.num_workers > 0 else 0,  # 避免多进程问题
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            drop_last=True if shuffle else False,  # 训练时丢弃不完整批次
            collate_fn=custom_collate_fn  # 使用自定义collate函数
        )

    def __iter__(self):
        for batch in self.loader:
            # 批量移动到GPU
            gpu_batch = self._move_batch_to_gpu(batch)
            yield gpu_batch

    def _move_batch_to_gpu(self, batch):
        """高效地将整个批次移动到GPU"""
        if isinstance(batch, dict):
            gpu_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    gpu_batch[key] = value.to(self.config.device, non_blocking=self.config.non_blocking)
                else:
                    gpu_batch[key] = value
            return gpu_batch
        return batch

    def __len__(self):
        return len(self.loader)


def create_optimized_data_loaders(train_data, val_data, test_data, config):
    """创建GPU优化的数据加载器"""
    print("Creating optimized data loaders...")

    # 构建所有物品集合
    all_items = list({item for d in train_data + val_data + test_data for item in d['input_seq'] + [d['target']]})

    # 构建用户历史交互字典（仅从训练集采样）
    user_history_dict = {}
    for d in train_data:
        uid = d['user_id']
        user_history_dict.setdefault(uid, set()).update(d['input_seq'])

    # 创建数据集
    train_dataset = OptimizedMovieLensDataset(train_data, config.max_seq_len, all_items, user_history_dict, num_negative=0, cache_dir=config.cache_dir)
    val_dataset = OptimizedMovieLensDataset(val_data, config.max_seq_len, all_items, user_history_dict, cache_dir=config.cache_dir)
    test_dataset = OptimizedMovieLensDataset(test_data, config.max_seq_len, all_items, user_history_dict, cache_dir=config.cache_dir)

    # 创建优化的数据加载器
    train_loader = GPUOptimizedDataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config
    )

    val_loader = GPUOptimizedDataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        config=config
    )

    test_loader = GPUOptimizedDataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        config=config
    )

    return train_loader, val_loader, test_loader

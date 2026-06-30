"""
AFRL适配器 - 复用现有的数据集和数据加载器
"""

import torch
from .dataset import OptimizedMovieLensDataset, GPUOptimizedDataLoader


def create_afrl_compatible_loaders(train_data, val_data, test_data, config):
    """
    创建与AFRL兼容的数据加载器，复用现有的OptimizedMovieLensDataset

    这个函数确保AFRL可以直接使用您现有的数据集类，无需修改
    """
    print("Creating AFRL-compatible data loaders using existing dataset...")

    # 构建所有物品集合
    num_items = int(getattr(config, 'num_items', 0))
    if num_items <= 0:
        for split in (train_data, val_data, test_data):
            for sample in split:
                num_items = max(num_items, int(sample['target']))
                if sample['input_seq']:
                    num_items = max(num_items, max(int(item) for item in sample['input_seq']))

    # 构建用户历史交互字典
    user_history_dict = {}
    for split in (train_data, val_data, test_data):
        for sample in split:
            uid = sample['user_id']
            user_history_dict.setdefault(uid, set()).update(sample['input_seq'])
            user_history_dict[uid].add(sample['target'])

    # 直接使用您现有的OptimizedMovieLensDataset
    train_dataset = OptimizedMovieLensDataset(
        train_data, config.max_seq_len, num_items, user_history_dict,
        num_negative=0, cache_dir=config.cache_dir,
        seed=getattr(config, 'seed', 42)
    )

    val_dataset = OptimizedMovieLensDataset(
        val_data, config.max_seq_len, num_items, user_history_dict,
        num_negative=getattr(config, 'eval_num_negative', 0),
        cache_dir=config.cache_dir,
        seed=getattr(config, 'seed', 42) + 1
    )

    test_dataset = OptimizedMovieLensDataset(
        test_data, config.max_seq_len, num_items, user_history_dict,
        num_negative=getattr(config, 'eval_num_negative', 0),
        cache_dir=config.cache_dir,
        seed=getattr(config, 'seed', 42) + 2
    )

    # 创建数据加载器 - 使用您现有的GPUOptimizedDataLoader
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


class AFRLDatasetWrapper:
    """
    包装器类，使现有数据集与AFRL训练流程兼容
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.is_training = True  # AFRL需要这个属性

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 直接返回原数据集的数据
        return self.dataset[idx]


def verify_data_compatibility(data_loader, config):
    """
    验证数据是否与AFRL兼容
    """
    print("Verifying data compatibility with AFRL...")

    # 获取一个批次进行检查
    for batch in data_loader:
        required_fields = ['user_id', 'input_seq', 'target', 'gender', 'age_group']

        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Missing required field: {field}")

        # 检查属性值范围
        gender_values = batch['gender'].unique()
        age_values = batch['age_group'].unique()

        print(f"Gender values: {gender_values.cpu().numpy()}")
        print(f"Age group values: {age_values.cpu().numpy()}")

        # 验证值范围
        assert all(g in [0, 1] for g in gender_values), f"Invalid gender values: {gender_values}"
        assert all(a in [0, 1] for a in age_values), f"Invalid age_group values: {age_values}"

        print("✓ Data is compatible with AFRL")
        break

    return True

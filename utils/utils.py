import torch
import numpy as np
import random
import os
import pickle
from typing import Dict, Any
import json


def set_seed(seed: int = 42):
    """设置随机种子以保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config, save_path: str):
    """保存配置文件"""
    config_dict = {}
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_gpu_memory_info():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return "CUDA not available"

    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3

        info.append({
            'gpu_id': i,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved
        })

    return info


def create_directory_structure(base_dir: str):
    """创建完整的目录结构"""
    directories = [
        'processed_data',
        'cache',
        'models',
        'logs',
        'results',
        'figures'
    ]

    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


class EarlyStopping:
    """早停类"""

    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_gpu_utilization():
    """打印GPU使用情况"""
    if torch.cuda.is_available():
        print("GPU Utilization:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            total = props.total_memory / 1024 ** 3

            print(f"  GPU {i} ({props.name}):")
            print(f"    Allocated: {allocated:.1f}GB")
            print(f"    Reserved: {reserved:.1f}GB")
            print(f"    Total: {total:.1f}GB")
            print(f"    Free: {total - reserved:.1f}GB")
    else:
        print("CUDA not available")
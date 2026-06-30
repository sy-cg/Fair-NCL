import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from torch.cuda.amp import autocast


class BaseRecommenderModel(nn.Module, ABC):
    """
    推荐系统基础模型抽象类
    定义所有推荐模型必须实现的接口
    """

    def __init__(self, config):
        super(BaseRecommenderModel, self).__init__()
        self.config = config
        self.device = config.device
        self.num_items = config.num_items
        self.num_users = getattr(config, 'num_users', 0)

        # 模型基础参数
        self.hidden_units = config.hidden_units
        self.dropout_rate = config.dropout_rate

        # 训练相关
        self.training_step = 0
        self.epoch = 0

    @abstractmethod
    def forward(self, *args, **kwargs):
        """前向传播，子类必须实现"""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """预测函数，子类必须实现"""
        pass

    def compute_loss(self, *args, **kwargs):
        """损失计算，默认实现，子类可以重写"""
        raise NotImplementedError("Subclass should implement compute_loss")

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        获取用户嵌入（如果模型支持用户嵌入）
        默认实现返回None，子类可以重写
        """
        return None

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        获取物品嵌入
        子类应该重写此方法
        """
        if hasattr(self, 'item_emb'):
            return self.item_emb(item_ids)
        else:
            raise NotImplementedError("Subclass must implement get_item_embeddings")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'device': str(self.device),
            'hidden_units': self.hidden_units,
            'num_items': self.num_items,
            'num_users': self.num_users
        }

    def save_checkpoint(self, filepath: str, additional_info: Dict = None):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'model_info': self.get_model_info(),
            'training_step': self.training_step,
            'epoch': self.epoch
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, device: Optional[torch.device] = None):
        """加载模型检查点"""
        if device is None:
            device = self.device

        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

        if 'training_step' in checkpoint:
            self.training_step = checkpoint['training_step']
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']

        print(f"Model checkpoint loaded from {filepath}")
        return checkpoint

    def freeze_embeddings(self):
        """冻结嵌入层参数"""
        if hasattr(self, 'item_emb'):
            self.item_emb.weight.requires_grad = False
            print("Item embeddings frozen")

        if hasattr(self, 'user_emb'):
            self.user_emb.weight.requires_grad = False
            print("User embeddings frozen")

    def unfreeze_embeddings(self):
        """解冻嵌入层参数"""
        if hasattr(self, 'item_emb'):
            self.item_emb.weight.requires_grad = True
            print("Item embeddings unfrozen")

        if hasattr(self, 'user_emb'):
            self.user_emb.weight.requires_grad = True
            print("User embeddings unfrozen")

    def get_parameters_by_name(self, name_pattern: str):
        """根据名称模式获取参数"""
        matching_params = []
        for name, param in self.named_parameters():
            if name_pattern in name:
                matching_params.append((name, param))
        return matching_params

    def print_model_summary(self):
        """打印模型摘要"""
        info = self.get_model_info()

        print("=" * 60)
        print(f"Model Summary: {info['model_name']}")
        print("=" * 60)
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print(f"Device: {info['device']}")
        print(f"Hidden Units: {info['hidden_units']}")
        print(f"Number of Items: {info['num_items']}")
        print(f"Number of Users: {info['num_users']}")
        print("=" * 60)


class BaseSequentialModel(BaseRecommenderModel):
    """
    序列推荐模型基类
    为序列推荐模型提供通用功能
    """

    def __init__(self, config):
        super(BaseSequentialModel, self).__init__(config)
        self.max_seq_len = config.max_seq_len

    def create_padding_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """创建padding mask"""
        return (sequences == 0)

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建因果mask（下三角矩阵）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)

    def get_sequence_lengths(self, sequences: torch.Tensor) -> torch.Tensor:
        """获取序列的实际长度"""
        return torch.sum(sequences > 0, dim=1)

    def get_last_item_embeddings(self, sequence_embeddings: torch.Tensor,
                                 sequences: torch.Tensor) -> torch.Tensor:
        """获取序列中最后一个有效物品的嵌入"""
        seq_lengths = self.get_sequence_lengths(sequences)
        batch_size = sequences.size(0)

        # 获取最后一个有效位置的索引
        batch_indices = torch.arange(batch_size, device=self.device)
        last_indices = torch.clamp(seq_lengths - 1, min=0)

        # 提取最后一个有效位置的嵌入
        last_embeddings = sequence_embeddings[batch_indices, last_indices]

        return last_embeddings

    def truncate_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        """截断序列到最大长度"""
        if sequences.size(1) > self.max_seq_len:
            return sequences[:, -self.max_seq_len:]
        return sequences

    def pad_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        """填充序列到最大长度"""
        batch_size, seq_len = sequences.size()
        if seq_len < self.max_seq_len:
            padding = torch.zeros(batch_size, self.max_seq_len - seq_len,
                                  dtype=sequences.dtype, device=sequences.device)
            sequences = torch.cat([padding, sequences], dim=1)
        return sequences


class BaseMatrixFactorizationModel(BaseRecommenderModel):
    """
    矩阵分解模型基类
    为协同过滤类模型提供通用功能
    """

    def __init__(self, config):
        super(BaseMatrixFactorizationModel, self).__init__(config)

        # 用户和物品嵌入
        self.user_emb = nn.Embedding(self.num_users, self.hidden_units)
        self.item_emb = nn.Embedding(self.num_items + 1, self.hidden_units, padding_idx=0)

        # 偏置项
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items + 1, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """获取用户嵌入"""
        return self.user_emb(user_ids)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """获取物品嵌入"""
        return self.item_emb(item_ids)

    def compute_rating_prediction(self, user_ids: torch.Tensor,
                                  item_ids: torch.Tensor) -> torch.Tensor:
        """计算评分预测"""
        user_emb = self.get_user_embeddings(user_ids)
        item_emb = self.get_item_embeddings(item_ids)

        # 矩阵分解预测
        rating_pred = torch.sum(user_emb * item_emb, dim=-1)

        # 添加偏置项
        user_bias = self.user_bias(user_ids).squeeze(-1)
        item_bias = self.item_bias(item_ids).squeeze(-1)

        rating_pred = rating_pred + user_bias + item_bias + self.global_bias

        return rating_pred


class ModelRegistry:
    """模型注册表，用于管理不同的模型类"""

    _models = {}

    @classmethod
    def register(cls, name: str):
        """注册模型装饰器"""

        def decorator(model_class):
            cls._models[name.lower()] = model_class
            return model_class

        return decorator

    @classmethod
    def get_model(cls, name: str):
        """根据名称获取模型类"""
        name = name.lower()
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls):
        """列出所有注册的模型"""
        return list(cls._models.keys())


# 模型工厂函数
def create_model(model_name: str, config) -> BaseRecommenderModel:
    """
    模型工厂函数
    根据模型名称和配置创建模型实例
    """
    model_class = ModelRegistry.get_model(model_name)
    model = model_class(config)

    # 移动到指定设备
    model = model.to(config.device)

    # 如果支持模型编译，进行编译优化
    if hasattr(config, 'use_compile') and config.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            print(f"Model {model_name} compiled for optimization")
        except Exception as e:
            print(f"Model compilation failed: {e}, using uncompiled model")

    return model


# 导出的工具函数
def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def freeze_model_partially(model: nn.Module, freeze_patterns: list):
    """部分冻结模型参数"""
    frozen_count = 0
    for name, param in model.named_parameters():
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                break

    print(f"Frozen {frozen_count} parameters matching patterns: {freeze_patterns}")


def get_model_device(model: nn.Module) -> torch.device:
    """获取模型所在的设备"""
    return next(model.parameters()).device


def move_model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """将模型移动到指定设备"""
    return model.to(device)


def register_advanced_models():
    """延迟注册高级模型（AFRL和SM）"""
    try:
        # === 修复: 直接导入整个模块，而不是写死具体的类名 ===
        import models.afrl_model
        print("AFRL models registered successfully")
    except ImportError as e:
        print(f"Warning: Could not import AFRL models: {e}")

    try:
        # === 修复: 直接导入整个模块 ===
        import models.sm_models
        print("SM models registered successfully")
    except ImportError as e:
        print(f"Warning: Could not import SM models: {e}")


# 在模块级别调用注册函数，但使用延迟导入
def _lazy_register():
    """延迟注册函数"""
    import sys
    if 'models.afrl_model' not in sys.modules and 'sm_models' not in sys.modules:
        try:
            register_advanced_models()
        except:
            pass  # 忽略导入错误


# 设置模块级别的延迟加载
import atexit

atexit.register(_lazy_register)
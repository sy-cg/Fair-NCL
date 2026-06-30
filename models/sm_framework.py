"""
Separate Method (SM) Framework Implementation
基于论文 "Towards Personalized Fairness based on Causal Notion" 的SM方法实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import Dict, List, Set, Tuple, Any, Union  # 添加 Union
import numpy as np
from collections import defaultdict


class SMFilterModule(nn.Module):
    """SM方法的过滤器模块

    为每个敏感特征组合训练单独的过滤器函数
    """

    def __init__(self, embed_dim: int, sensitive_attributes: List[str],
                 dropout_rate: float = 0.1):
        super(SMFilterModule, self).__init__()

        self.embed_dim = embed_dim
        self.sensitive_attributes = sensitive_attributes
        self.dropout_rate = dropout_rate

        # 生成所有可能的敏感特征组合
        self.feature_combinations = self._generate_feature_combinations()

        # 为每个组合创建过滤器
        self.filters = nn.ModuleDict()
        for combo_key in self.feature_combinations:
            self.filters[combo_key] = self._create_filter_network()

        print(f"SM Framework initialized with {len(self.filters)} filter combinations:")
        for combo_key in self.feature_combinations:
            print(f"  - {combo_key}: {self._decode_combo_key(combo_key)}")

    def _generate_feature_combinations(self) -> List[str]:
        """生成所有可能的敏感特征组合"""
        combinations = []

        # 单个特征
        for attr in self.sensitive_attributes:
            combinations.append(attr)

        # 特征组合 (从2个到全部)
        for r in range(2, len(self.sensitive_attributes) + 1):
            for combo in itertools.combinations(self.sensitive_attributes, r):
                combo_key = '_'.join(sorted(combo))
                combinations.append(combo_key)

        return combinations

    def _decode_combo_key(self, combo_key: str) -> List[str]:
        """解码组合键为特征列表"""
        return combo_key.split('_')

    def _create_filter_network(self) -> nn.Module:
        """创建过滤器网络"""
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

    def forward(self, user_embeddings: torch.Tensor,
                sensitive_mask: Dict[str, bool]) -> torch.Tensor:
        """前向传播

        Args:
            user_embeddings: 用户嵌入 [batch_size, embed_dim]
            sensitive_mask: 敏感特征掩码 {attr_name: is_sensitive}

        Returns:
            过滤后的用户嵌入 [batch_size, embed_dim]
        """
        # 确定需要过滤的敏感特征组合
        target_attributes = [attr for attr, is_sensitive in sensitive_mask.items()
                             if is_sensitive and attr in self.sensitive_attributes]

        if not target_attributes:
            # 没有敏感特征需要过滤，返回原嵌入
            return user_embeddings

        # 生成组合键
        combo_key = '_'.join(sorted(target_attributes))

        if combo_key not in self.filters:
            # 如果没有对应的过滤器，使用最接近的过滤器
            combo_key = self._find_closest_filter(target_attributes)

        # 应用过滤器
        filtered_embeddings = self.filters[combo_key](user_embeddings)

        return filtered_embeddings

    def _find_closest_filter(self, target_attributes: List[str]) -> str:
        """找到最接近的过滤器"""
        target_set = set(target_attributes)
        best_match = None
        best_overlap = 0

        for combo_key in self.filters.keys():
            combo_attrs = set(self._decode_combo_key(combo_key))
            overlap = len(target_set.intersection(combo_attrs))

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = combo_key

        return best_match if best_match else list(self.filters.keys())[0]

    def get_filter_for_combination(self, attributes: List[str]) -> nn.Module:
        """获取特定组合的过滤器"""
        combo_key = '_'.join(sorted(attributes))
        return self.filters.get(combo_key, list(self.filters.values())[0])


class SMDiscriminatorModule(nn.Module):
    """SM方法的判别器模块

    为每个敏感特征创建判别器
    """

    def __init__(self, embed_dim: int, attribute_dims: Dict[str, int],
                 dropout_rate: float = 0.3):
        super(SMDiscriminatorModule, self).__init__()

        self.embed_dim = embed_dim
        self.attribute_dims = attribute_dims
        self.dropout_rate = dropout_rate

        # 为每个敏感特征创建判别器
        self.discriminators = nn.ModuleDict()
        for attr, num_classes in attribute_dims.items():
            self.discriminators[attr] = self._create_discriminator(num_classes)

    def _create_discriminator(self, num_classes: int) -> nn.Module:
        """创建判别器网络"""
        layers = []

        # 7层MLP结构 (按论文设置)
        layer_dims = [self.embed_dim, self.embed_dim, self.embed_dim // 2,
                      self.embed_dim // 4, self.embed_dim // 8,
                      self.embed_dim // 16, num_classes]

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # 最后一层不加激活函数
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, filtered_embeddings: torch.Tensor,
                target_attributes: List[str]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            filtered_embeddings: 过滤后的嵌入 [batch_size, embed_dim]
            target_attributes: 目标敏感特征列表

        Returns:
            各特征的预测结果字典
        """
        predictions = {}

        for attr in target_attributes:
            if attr in self.discriminators:
                predictions[attr] = self.discriminators[attr](filtered_embeddings)

        return predictions


"""
修复SM框架中的forward方法
主要修复批处理掩码的处理逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Any
import numpy as np


class SMFramework(nn.Module):
    """修复版的SM框架"""

    def __init__(self, base_model: nn.Module, config: Any):
        super(SMFramework, self).__init__()

        self.base_model = base_model
        self.config = config
        self.hidden_units = config.hidden_units
        self.sensitive_attributes = config.sensitive_attributes
        self.attribute_dims = getattr(config, 'attribute_dims', {
            'gender': 2, 'age_group': 2
        })

        # 初始化SM组件
        self.filter_module = SMFilterModule(
            embed_dim=self.hidden_units,
            sensitive_attributes=self.sensitive_attributes,
            dropout_rate=config.dropout_rate
        )

        self.discriminator_module = SMDiscriminatorModule(
            embed_dim=self.hidden_units,
            attribute_dims=self.attribute_dims,
            dropout_rate=0.3
        )

        # 超参数
        self.lambda_adv = getattr(config, 'sm_lambda', 1.0)

        print(f"SM Framework initialized with λ={self.lambda_adv}")

    def get_user_embedding(self, input_seq: torch.Tensor) -> torch.Tensor:
        """获取用户原始嵌入"""
        seq_emb = self.base_model.forward(input_seq).clone()  # [B, L, D]

        # 获取最后一个非padding位置的表示
        seq_lengths = torch.sum(input_seq > 0, dim=1)
        batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)
        last_indices = torch.clamp(seq_lengths - 1, min=0)

        user_emb = seq_emb[batch_indices, last_indices]  # [B, D]
        return user_emb

    def forward(self, input_seq: torch.Tensor,
                sensitive_mask: Union[Dict[str, bool], torch.Tensor, List[Dict[str, bool]]] = None):
        """
        修复版的前向传播，正确处理各种掩码格式

        Args:
            input_seq: 输入序列 [B, L]
            sensitive_mask: 敏感特征掩码，可以是：
                - None: 生成随机掩码
                - Dict[str, bool]: 单个掩码，应用到整个批次
                - List[Dict[str, bool]]: 批次中每个样本的掩码
                - torch.Tensor: [B, num_attrs]的张量
        """
        # 获取原始用户嵌入
        user_emb = self.get_user_embedding(input_seq)
        batch_size = user_emb.size(0)

        # 处理不同格式的sensitive_mask
        if sensitive_mask is None:
            # 生成随机掩码
            sensitive_mask = [self._generate_random_mask() for _ in range(batch_size)]

        elif isinstance(sensitive_mask, dict):
            # 单个字典掩码，扩展到整个批次
            sensitive_mask = [sensitive_mask] * batch_size

        elif isinstance(sensitive_mask, torch.Tensor):
            # Tensor格式，转换为list of dict
            if sensitive_mask.dim() == 1:
                # 1D tensor，扩展到批次
                sensitive_mask = sensitive_mask.unsqueeze(0).expand(batch_size, -1)

            # 转换为list of dict
            mask_list = []
            for i in range(batch_size):
                mask = {}
                for j, attr in enumerate(self.sensitive_attributes):
                    if j < sensitive_mask.size(1):
                        mask[attr] = bool(sensitive_mask[i, j].item())
                    else:
                        mask[attr] = False
                mask_list.append(mask)
            sensitive_mask = mask_list

        elif isinstance(sensitive_mask, list):
            # 已经是列表格式
            if len(sensitive_mask) == 0:
                # 空列表，生成随机掩码
                sensitive_mask = [self._generate_random_mask() for _ in range(batch_size)]
            elif len(sensitive_mask) == 1:
                # 只有一个元素，扩展到整个批次
                sensitive_mask = sensitive_mask * batch_size
            elif len(sensitive_mask) != batch_size:
                # 长度不匹配，扩展或截断
                if len(sensitive_mask) < batch_size:
                    # 循环扩展
                    extended_mask = []
                    for i in range(batch_size):
                        extended_mask.append(sensitive_mask[i % len(sensitive_mask)])
                    sensitive_mask = extended_mask
                else:
                    # 截断
                    sensitive_mask = sensitive_mask[:batch_size]

        # 确保sensitive_mask现在是一个长度为batch_size的列表
        assert isinstance(sensitive_mask, list), f"sensitive_mask should be list, got {type(sensitive_mask)}"
        assert len(sensitive_mask) == batch_size, f"mask length {len(sensitive_mask)} != batch_size {batch_size}"

        # 批量过滤
        filtered_emb_list = []
        for i in range(batch_size):
            emb = self.filter_module(user_emb[i].unsqueeze(0), sensitive_mask[i])
            filtered_emb_list.append(emb)

        # 确保列表不为空
        if len(filtered_emb_list) == 0:
            # 如果列表为空，返回原始嵌入
            filtered_emb = user_emb
        else:
            filtered_emb = torch.cat(filtered_emb_list, dim=0)

        return filtered_emb, user_emb

    def _generate_random_mask(self) -> Dict[str, bool]:
        """随机生成敏感特征掩码"""
        mask = {}
        for attr in self.sensitive_attributes:
            mask[attr] = np.random.random() > 0.5
        return mask

    def predict(self, input_seq: torch.Tensor, candidate_items: torch.Tensor = None,
                sensitive_mask: Union[Dict[str, bool], torch.Tensor, List[Dict[str, bool]]] = None) -> torch.Tensor:
        """
        预测函数

        Args:
            input_seq: 输入序列 [B, L]
            candidate_items: 候选物品 [B, N] 或 [B, 1]
            sensitive_mask: 敏感特征掩码
        """
        # 获取过滤后的嵌入
        filtered_emb, _ = self.forward(input_seq, sensitive_mask)

        # 使用过滤后的嵌入进行预测
        if candidate_items is None:
            # 预测所有物品
            item_emb = self.base_model.item_emb.weight
            logits = torch.matmul(filtered_emb, item_emb.transpose(0, 1))
        else:
            # 预测候选物品
            if candidate_items.dim() == 1:
                candidate_items = candidate_items.unsqueeze(1)

            candidate_emb = self.base_model.item_emb(candidate_items)  # [B, N, D]

            if candidate_emb.dim() == 3:
                # [B, N, D] -> [B, N]
                logits = torch.sum(
                    filtered_emb.unsqueeze(1) * candidate_emb,
                    dim=2
                )
            else:
                # [B, D] -> [B]
                logits = torch.sum(filtered_emb * candidate_emb, dim=1, keepdim=True)

        return logits

    def compute_sm_losses(self, batch: Dict[str, torch.Tensor],
                          filtered_emb: torch.Tensor,
                          original_emb: torch.Tensor,
                          sensitive_mask: List[Dict[str, bool]]) -> Dict[str, torch.Tensor]:
        """计算SM的损失函数"""
        losses = {}

        # 1. 推荐损失
        pos_items = batch['target']
        pos_emb = self.base_model.item_emb(pos_items)
        pos_scores = torch.sum(filtered_emb * pos_emb, dim=1)

        # 负采样推荐损失
        neg_items = batch.get('negative_items', None)
        if neg_items is not None and neg_items.size(1) > 0:
            neg_emb = self.base_model.item_emb(neg_items)  # [B, N, D]
            neg_scores = torch.sum(
                filtered_emb.unsqueeze(1) * neg_emb, dim=2
            )  # [B, N]
            neg_scores_max = torch.max(neg_scores, dim=1)[0]  # [B]
            rec_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores_max) + 1e-8).mean()
        else:
            # 简单回归损失
            rec_loss = F.mse_loss(pos_scores, torch.ones_like(pos_scores))

        losses['rec_loss'] = rec_loss

        # 2. 对抗损失（判别器损失）
        # 收集所有需要判别的属性
        all_target_attrs = set()
        for mask in sensitive_mask:
            for attr, is_sensitive in mask.items():
                if is_sensitive and attr in self.sensitive_attributes:
                    all_target_attrs.add(attr)

        if all_target_attrs:
            disc_predictions = self.discriminator_module(filtered_emb, list(all_target_attrs))

            disc_loss = 0
            for attr in all_target_attrs:
                if attr in disc_predictions and attr in batch:
                    attr_logits = disc_predictions[attr]
                    attr_labels = batch[attr]
                    attr_loss = F.cross_entropy(attr_logits, attr_labels)
                    disc_loss += attr_loss

            losses['disc_loss'] = disc_loss / len(all_target_attrs) if all_target_attrs else torch.tensor(0.0)

            # 总的对抗损失（过滤器试图最小化判别器的准确性）
            losses['adv_loss'] = -self.lambda_adv * losses['disc_loss']
        else:
            losses['disc_loss'] = torch.tensor(0.0, device=filtered_emb.device)
            losses['adv_loss'] = torch.tensor(0.0, device=filtered_emb.device)

        # 3. 总损失
        total_loss = losses['rec_loss'] + losses['adv_loss']
        losses['total_loss'] = total_loss

        return losses


# 辅助函数：创建SM兼容的掩码
def create_sm_compatible_mask(batch: Dict[str, torch.Tensor],
                              sensitive_attributes: List[str],
                              mask_prob: float = 0.5) -> List[Dict[str, bool]]:
    """为SM框架创建兼容的敏感特征掩码"""
    batch_size = batch['input_seq'].size(0)
    masks = []

    for i in range(batch_size):
        mask = {}
        for attr in sensitive_attributes:
            mask[attr] = np.random.random() < mask_prob
        masks.append(mask)

    return masks


# 工具函数
def analyze_sm_filter_usage(sm_model: SMFramework, data_loader, num_batches: int = 10):
    """分析SM过滤器使用情况"""
    filter_usage = defaultdict(int)
    total_samples = 0

    sm_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            batch_masks = create_sm_compatible_mask(
                batch, sm_model.sensitive_attributes
            )

            for mask in batch_masks:
                target_attrs = [attr for attr, is_sensitive in mask.items() if is_sensitive]
                if target_attrs:
                    combo_key = '_'.join(sorted(target_attrs))
                    filter_usage[combo_key] += 1
                else:
                    filter_usage['none'] += 1
                total_samples += 1

    print("SM Filter Usage Analysis:")
    print("-" * 40)
    for combo, count in sorted(filter_usage.items()):
        percentage = count / total_samples * 100
        print(f"  {combo:<15}: {count:>6} ({percentage:>5.1f}%)")
    print(f"  {'Total':<15}: {total_samples:>6}")

    return filter_usage

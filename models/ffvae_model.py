"""
Flexibly Fair VAE (FFVAE) Implementation
基于 "Flexibly Fair Representation Learning by Disentanglement" 论文实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.cuda.amp import autocast

from models.base_model import BaseSequentialModel, ModelRegistry


class FFVAEEncoder(nn.Module):
    """FFVAE编码器：将输入编码为非敏感潜在码z和敏感潜在码b"""

    def __init__(self, input_dim: int, z_dim: int, b_dims: List[int],
                 hidden_dims: List[int] = [512, 256]):
        super(FFVAEEncoder, self).__init__()

        self.z_dim = z_dim
        self.b_dims = b_dims
        self.total_b_dim = sum(b_dims)

        # 构建编码器网络
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        self.encoder_backbone = nn.Sequential(*layers)

        # z的均值和方差（VAE风格）
        self.z_mean = nn.Linear(prev_dim, z_dim)
        self.z_logvar = nn.Linear(prev_dim, z_dim)

        # b是确定性的（根据论文）
        self.b_head = nn.Linear(prev_dim, self.total_b_dim)

    def forward(self, x):
        h = self.encoder_backbone(x)

        # 非敏感潜在码z（随机）
        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)
        z = self.reparameterize(z_mean, z_logvar)

        # 敏感潜在码b（确定性）
        b = self.b_head(h)

        # 将b分割为各个敏感属性对应的维度
        b_list = []
        start = 0
        for b_dim in self.b_dims:
            b_list.append(b[:, start:start + b_dim])
            start += b_dim

        return z, b_list, z_mean, z_logvar

    def reparameterize(self, mean, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class FFVAEDecoder(nn.Module):
    """FFVAE解码器：从z和b重建输入x，并预测敏感属性a"""

    def __init__(self, z_dim: int, b_dims: List[int], output_dim: int,
                 attribute_class_dims: List[int] = None,
                 hidden_dims: List[int] = [256, 512]):
        super(FFVAEDecoder, self).__init__()

        self.z_dim = z_dim
        self.b_dims = b_dims
        self.total_b_dim = sum(b_dims)
        self.attribute_class_dims = attribute_class_dims or [2] * len(b_dims)

        # 重建网络 p(x|z,b)
        layers = []
        prev_dim = z_dim + self.total_b_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder_backbone = nn.Sequential(*layers)

        # 敏感属性预测器 p(a|b)
        # 每个敏感属性从对应的b维度预测
        self.attribute_predictors = nn.ModuleList()
        for i, b_dim in enumerate(b_dims):
            # 二分类敏感属性
            self.attribute_predictors.append(
                nn.Linear(b_dim, 1)  # 输出logit
            )

        self.attribute_predictors = nn.ModuleList([
            nn.Linear(b_dim, self.attribute_class_dims[i])
            for i, b_dim in enumerate(b_dims)
        ])

    def forward(self, z, b_list):
        # 合并z和所有b
        b_concat = torch.cat(b_list, dim=1)
        zb = torch.cat([z, b_concat], dim=1)

        # 重建x
        x_recon = self.decoder_backbone(zb)

        # 预测敏感属性
        a_logits = []
        for i, (b_i, predictor) in enumerate(zip(b_list, self.attribute_predictors)):
            logit = predictor(b_i)
            a_logits.append(logit)

        return x_recon, a_logits


class TotalCorrelationDiscriminator(nn.Module):
    """用于估计总相关性的判别器"""

    def __init__(self, z_dim: int, b_dims: List[int], hidden_dim: int = 256):
        super(TotalCorrelationDiscriminator, self).__init__()

        input_dim = z_dim + sum(b_dims)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, b_list):
        b_concat = torch.cat(b_list, dim=1)
        zb = torch.cat([z, b_concat], dim=1)
        return self.network(zb)


class FFVAEModel(nn.Module):
    """完整的FFVAE模型"""

    def __init__(self, config):
        super(FFVAEModel, self).__init__()

        self.config = config
        self.device = config.device

        # 模型参数
        self.input_dim = config.num_items  # 用于推荐系统
        self.z_dim = config.ffvae_z_dim if hasattr(config, 'ffvae_z_dim') else 64
        self.sensitive_attributes = config.sensitive_attributes
        self.num_sensitive = len(self.sensitive_attributes)
        self.attribute_dims = getattr(config, 'attribute_dims', {
            'gender': 2,
            'age_group': 2
        })

        # 每个敏感属性对应的b维度（默认每个属性1维）
        self.b_dims = [max(1, self.attribute_dims[attr]) for attr in self.sensitive_attributes]

        # 超参数
        self.alpha = getattr(config, 'ffvae_alpha', 1.0)  # 预测性权重
        self.beta = getattr(config, 'ffvae_beta', 1.0)  # 解耦权重

        # 构建模型组件
        self.encoder = FFVAEEncoder(
            input_dim=self.input_dim,
            z_dim=self.z_dim,
            b_dims=self.b_dims
        )

        self.decoder = FFVAEDecoder(
            z_dim=self.z_dim,
            b_dims=self.b_dims,
            output_dim=self.input_dim,
            attribute_class_dims=[self.attribute_dims[attr] for attr in self.sensitive_attributes]
        )

        self.tc_discriminator = TotalCorrelationDiscriminator(
            z_dim=self.z_dim,
            b_dims=self.b_dims
        )

        # 判别器优化器（独立的）
        self.disc_optimizer = None

    def encode(self, x):
        """编码输入"""
        return self.encoder(x)

    def decode(self, z, b_list):
        """解码潜在表示"""
        return self.decoder(z, b_list)

    def forward(self, x):
        """前向传播"""
        z, b_list, z_mean, z_logvar = self.encode(x)
        x_recon, a_logits = self.decode(z, b_list)
        return x_recon, a_logits, z, b_list, z_mean, z_logvar

    def get_fair_representation(self, x, sensitive_mask=None):
        """获取公平表示

        Args:
            x: 输入数据
            sensitive_mask: 布尔列表，指示哪些敏感属性需要去除
                          如果为None，则去除所有敏感属性

        Returns:
            fair_repr: 公平的表示
        """
        z, b_list, _, _ = self.encode(x)

        if sensitive_mask is None:
            # 去除所有敏感信息，只返回z
            return z
        else:
            # 选择性去除某些敏感属性
            fair_b_list = []
            for i, (b_i, is_sensitive) in enumerate(zip(b_list, sensitive_mask)):
                if is_sensitive:
                    # 用噪声替换敏感维度
                    noise = torch.randn_like(b_i)
                    fair_b_list.append(noise)
                else:
                    fair_b_list.append(b_i)

            # 返回z和处理后的b
            b_concat = torch.cat(fair_b_list, dim=1)
            fair_repr = torch.cat([z, b_concat], dim=1)
            return fair_repr

    def compute_loss(self, batch, disc_loss=None):
        """计算FFVAE损失"""
        x = self.get_sequence_embedding(batch['input_seq'])

        sensitive_attrs = []
        for attr in self.sensitive_attributes:
            if attr in batch:
                sensitive_attrs.append(batch[attr].long())

        # 前向传播
        x_recon, a_logits, z, b_list, z_mean, z_logvar = self.forward(x)

        # 1. 重建损失
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # 2. 预测损失（确保b包含敏感信息）
        pred_loss = 0
        for i, (a_logit, a_true) in enumerate(zip(a_logits, sensitive_attrs)):
            pred_loss += F.cross_entropy(a_logit, a_true)
        pred_loss = pred_loss / len(sensitive_attrs) if sensitive_attrs else 0

        # 3. KL散度损失
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = kl_loss / x.size(0)  # 平均到批次

        # 4. 总相关性损失（如果提供了判别器损失）
        tc_loss = disc_loss if disc_loss is not None else torch.tensor(0.0).to(self.device)

        # 总损失
        total_loss = recon_loss + self.alpha * pred_loss + kl_loss + self.beta * tc_loss

        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'pred_loss': pred_loss,
            'kl_loss': kl_loss,
            'tc_loss': tc_loss
        }

        return total_loss, loss_dict

    def get_sequence_embedding(self, input_seq):
        """获取序列的嵌入表示（适配推荐系统）"""
        # 将序列转换为multi-hot向量
        batch_size = input_seq.size(0)
        x = torch.zeros(batch_size, self.config.num_items).to(input_seq.device)

        for i in range(batch_size):
            valid_items = input_seq[i][input_seq[i] > 0]
            if len(valid_items) > 0:
                x[i, valid_items - 1] = 1.0  # -1因为物品ID从1开始

        return x

    def compute_tc_discriminator_loss(self, batch):
        """计算总相关性判别器的损失"""
        x = self.get_sequence_embedding(batch['input_seq'])
        batch_size = x.size(0)

        # 获取真实样本从q(z,b)
        with torch.no_grad():
            z, b_list, _, _ = self.encode(x)

        # 创建假样本从q(z)∏q(b_j)
        # 通过在批次内随机打乱各维度
        z_perm = z[torch.randperm(batch_size)]
        b_list_perm = []
        for b_i in b_list:
            b_list_perm.append(b_i[torch.randperm(batch_size)])

        # 判别器预测
        real_score = self.tc_discriminator(z, b_list)
        fake_score = self.tc_discriminator(z_perm, b_list_perm)

        # 二分类损失
        ones = torch.ones_like(real_score)
        zeros = torch.zeros_like(fake_score)

        disc_loss = F.binary_cross_entropy_with_logits(real_score, ones) + \
                    F.binary_cross_entropy_with_logits(fake_score, zeros)

        # 用于主模型的对抗损失（log密度比估计）
        with torch.no_grad():
            log_ratio = real_score - fake_score

        return disc_loss, log_ratio.mean()


class FFVAERecommender(BaseSequentialModel):
    """FFVAE推荐系统包装器"""

    def __init__(self, config):
        super(FFVAERecommender, self).__init__(config)

        self.ffvae = FFVAEModel(config)

        # 推荐头：从公平表示预测物品
        fair_repr_dim = self.ffvae.z_dim + sum(self.ffvae.b_dims)
        self.recommendation_head = nn.Sequential(
            nn.Linear(fair_repr_dim, config.hidden_units),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_units, config.num_items + 1)
        )

    def get_sequence_embedding(self, input_seq):
        """获取序列的嵌入表示（适配推荐系统）"""
        # 将序列转换为multi-hot向量
        batch_size = input_seq.size(0)
        x = torch.zeros(batch_size, self.config.num_items).to(input_seq.device)

        for i in range(batch_size):
            valid_items = input_seq[i][input_seq[i] > 0]
            if len(valid_items) > 0:
                x[i, valid_items - 1] = 1.0  # -1因为物品ID从1开始

        return x

    def forward(self, input_seq):
        """前向传播"""
        x = self.get_sequence_embedding(input_seq)
        z, b_list, _, _ = self.ffvae.encode(x)

        # 获取完整表示
        b_concat = torch.cat(b_list, dim=1)
        full_repr = torch.cat([z, b_concat], dim=1)

        return full_repr

    def predict(self, input_seq, candidate_items=None, fairness_mask=None):
        """预测（支持公平性调整）"""
        x = self.get_sequence_embedding(input_seq)

        # 获取公平表示
        if fairness_mask is not None:
            fair_repr = self.ffvae.get_fair_representation(x, fairness_mask)
        else:
            # 默认使用完整表示
            z, b_list, _, _ = self.ffvae.encode(x)
            b_concat = torch.cat(b_list, dim=1)
            fair_repr = torch.cat([z, b_concat], dim=1)

        # 预测
        logits = self.recommendation_head(fair_repr)

        if candidate_items is not None:
            # 只返回候选物品的分数
            batch_size = candidate_items.size(0)
            if candidate_items.dim() == 1:
                candidate_items = candidate_items.unsqueeze(1)

            scores = []
            for i in range(batch_size):
                item_scores = logits[i][candidate_items[i]]
                scores.append(item_scores)

            return torch.stack(scores)
        else:
            return logits

    def compute_loss(self, input_seq, target_items):
        """计算推荐损失"""
        x = self.get_sequence_embedding(input_seq)

        # 获取完整表示
        z, b_list, _, _ = self.ffvae.encode(x)
        b_concat = torch.cat(b_list, dim=1)
        full_repr = torch.cat([z, b_concat], dim=1)

        # 推荐预测
        logits = self.recommendation_head(full_repr)

        # 交叉熵损失
        rec_loss = F.cross_entropy(logits, target_items, ignore_index=0)

        return rec_loss


# 注册模型
@ModelRegistry.register('ffvae_rec')
class FFVAEForRecommendation(FFVAERecommender):
    """FFVAE推荐模型（用于实验）"""
    pass

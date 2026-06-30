"""
改进的AFRL模型实现
去除重复定义，使用afrl_components中的组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .base_model import ModelRegistry
from .afrl_components import (
    AFRL_Generator,
    AFRL_Discriminator,
    AFRL_CombineMLP,
    DebiasedCollaborativeEncoder
)


class AFRLWrapper(nn.Module):
    """AFRL框架包装器，可以与任何推荐模型结合"""

    def __init__(self, base_model, config):
        super(AFRLWrapper, self).__init__()

        self.base_model = base_model
        self.config = config
        self.hidden_units = config.hidden_units
        self.num_items = config.num_items

        # 敏感属性配置
        self.sensitive_attributes = config.sensitive_attributes
        self.attribute_dims = getattr(config, 'attribute_dims', {
            'gender': 2,
            'age_group': 2
        })

        # AFRL组件
        self._build_afrl_components()

        # 超参数
        self.beta = getattr(config, 'afrl_beta', 1.0)
        self.lambda_adv = getattr(config, 'afrl_lambda', 1.0)

        # 损失权重
        self.rec_weight = getattr(config, 'afrl_rec_weight', 1.0)
        self.align_weight = getattr(config, 'afrl_align_weight', 0.1)
        self.adv_weight = getattr(config, 'afrl_adv_weight', 0.1)
        self.recon_weight = getattr(config, 'afrl_recon_weight', 0.01)

    def _build_afrl_components(self):
        """构建AFRL组件"""
        # 属性生成器
        self.attribute_generators = nn.ModuleDict()
        for attr in self.sensitive_attributes:
            self.attribute_generators[attr] = AFRL_Generator(
                self.hidden_units,
                self.config.dropout_rate
            )

        # 去偏协作编码器
        self.debiased_encoder = DebiasedCollaborativeEncoder(
            self.hidden_units,
            self.config.dropout_rate
        )

        # 属性分类器
        self.attribute_classifiers = nn.ModuleDict()
        for attr in self.sensitive_attributes:
            self.attribute_classifiers[attr] = nn.Linear(
                self.hidden_units,
                self.attribute_dims[attr]
            )

        # 属性判别器
        self.attribute_discriminators = nn.ModuleDict()
        for attr in self.sensitive_attributes:
            self.attribute_discriminators[attr] = AFRL_Discriminator(
                self.hidden_units,
                self.attribute_dims[attr],
                self.config.dropout_rate
            )

        # 信息聚合模块
        self.combine_mlp = AFRL_CombineMLP(
            self.hidden_units,
            len(self.sensitive_attributes),
            self.config.dropout_rate
        )

    def get_user_embedding(self, input_seq):
        """获取用户的原始嵌入"""
        seq_emb = self.base_model.forward(input_seq)

        # 获取最后一个非padding位置的表示
        seq_lengths = torch.sum(input_seq > 0, dim=1)
        batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)
        last_indices = torch.clamp(seq_lengths - 1, min=0)

        user_emb = seq_emb[batch_indices, last_indices]
        return user_emb

    def generate_fair_embedding(self, user_emb, fairness_mask=None):
        """生成公平嵌入"""
        batch_size = user_emb.size(0)

        if fairness_mask is None:
            fairness_mask = self._generate_random_fairness_mask(batch_size)

        # 生成属性特定嵌入
        attribute_embeddings = []
        for attr in self.sensitive_attributes:
            z_attr = self.attribute_generators[attr](user_emb)
            attribute_embeddings.append(z_attr)

        # 生成去偏协作嵌入
        z_debiased = self.debiased_encoder(user_emb)

        # 聚合生成公平嵌入
        fair_emb = self.combine_mlp(z_debiased, attribute_embeddings, fairness_mask)

        return fair_emb, attribute_embeddings, z_debiased

    def _generate_random_fairness_mask(self, batch_size):
        """随机生成公平性需求掩码"""
        num_attrs = len(self.sensitive_attributes)
        mask = torch.rand(batch_size, num_attrs, device=self.config.device) > 0.5
        return mask.float()

    def forward(self, input_seq, fairness_mask=None):
        """前向传播"""
        user_emb = self.get_user_embedding(input_seq)
        fair_emb, attr_embs, debiased_emb = self.generate_fair_embedding(
            user_emb, fairness_mask
        )
        return fair_emb, attr_embs, debiased_emb, user_emb

    def predict(self, input_seq, candidate_items=None, fairness_mask=None):
        """预测函数"""
        fair_emb, _, _, _ = self.forward(input_seq, fairness_mask)

        if candidate_items is None:
            item_emb = self.base_model.item_emb.weight
            logits = torch.matmul(fair_emb, item_emb.transpose(0, 1))
        else:
            if candidate_items.dim() == 1:
                candidate_items = candidate_items.unsqueeze(1)

            candidate_emb = self.base_model.item_emb(candidate_items)

            if candidate_emb.dim() == 3:
                logits = torch.sum(
                    fair_emb.unsqueeze(1) * candidate_emb,
                    dim=2
                )
            else:
                logits = torch.sum(fair_emb * candidate_emb, dim=1, keepdim=True)

        return logits

    def compute_afrl_losses(self, batch, fair_emb, attr_embs, debiased_emb, user_emb):
        """计算AFRL的各项损失"""
        losses = {}

        # 1. 推荐损失
        pos_items = batch['target']
        pos_emb = self.base_model.item_emb(pos_items)
        pos_scores = torch.sum(fair_emb * pos_emb, dim=1)

        neg_items = batch.get('negative_items', None)
        if neg_items is not None and neg_items.size(1) > 0:
            neg_emb = self.base_model.item_emb(neg_items)
            if neg_emb.dim() == 3:
                neg_scores = torch.sum(
                    fair_emb.unsqueeze(1) * neg_emb, dim=2
                )
                neg_scores_max = torch.max(neg_scores, dim=1)[0]
                rec_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores_max) + 1e-8).mean()
            else:
                neg_scores = torch.sum(fair_emb * neg_emb, dim=1)
                rec_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        else:
            rec_loss = F.mse_loss(pos_scores, torch.ones_like(pos_scores))

        losses['rec_loss'] = rec_loss

        # 2. 信息对齐损失
        align_loss = 0
        for i, attr in enumerate(self.sensitive_attributes):
            z_attr = attr_embs[i]

            # L2正则化
            l2_loss = 0.5 * torch.mean(torch.sum(z_attr ** 2, dim=1))

            # 分类损失
            attr_logits = self.attribute_classifiers[attr](z_attr)
            attr_labels = batch[attr]
            cls_loss = F.cross_entropy(attr_logits, attr_labels)

            align_loss += l2_loss - self.beta * cls_loss

        losses['align_loss'] = align_loss / len(self.sensitive_attributes)

        # 3. 去偏损失（对抗训练）
        adv_loss = 0
        for attr in self.sensitive_attributes:
            gen_logits = self.attribute_discriminators[attr](debiased_emb)
            batch_size = debiased_emb.size(0)
            num_classes = self.attribute_dims[attr]
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=debiased_emb.device)
            gen_loss = F.cross_entropy(gen_logits, fake_labels)
            adv_loss += self.lambda_adv * gen_loss

        losses['adv_loss'] = adv_loss / len(self.sensitive_attributes)

        # 4. 重构损失
        recon_loss = F.mse_loss(debiased_emb, user_emb)
        losses['recon_loss'] = recon_loss

        # 总损失
        total_loss = (
                self.rec_weight * losses['rec_loss'] +
                self.align_weight * losses['align_loss'] +
                self.adv_weight * losses['adv_loss'] +
                self.recon_weight * losses['recon_loss']
        )
        losses['total_loss'] = total_loss

        return losses

    def update_discriminators(self, batch, debiased_emb):
        """更新判别器"""
        disc_losses = {}

        for attr in self.sensitive_attributes:
            disc = self.attribute_discriminators[attr]
            disc_logits = disc(debiased_emb.detach())
            disc_loss = F.cross_entropy(disc_logits, batch[attr])
            disc_losses[f'disc_{attr}_loss'] = disc_loss

        return disc_losses


@ModelRegistry.register('afrl_sasrec')
class AFRL_SASRec(AFRLWrapper):
    """AFRL + SASRec"""

    def __init__(self, config):
        from .sasrec import OptimizedSASRec
        base_model = OptimizedSASRec(config)
        super(AFRL_SASRec, self).__init__(base_model, config)


@ModelRegistry.register('afrl_bert4rec')
class AFRL_BERT4Rec(AFRLWrapper):
    """AFRL + BERT4Rec"""

    def __init__(self, config):
        from .bert4rec import OptimizedBERT4Rec
        base_model = OptimizedBERT4Rec(config)
        super(AFRL_BERT4Rec, self).__init__(base_model, config)

# === ⬇️ 新增：补充 Caser 和 GRU4Rec 的 AFRL 注册 ===

@ModelRegistry.register('afrl_caser')
class AFRL_Caser(AFRLWrapper):
    """AFRL + Caser"""
    def __init__(self, config):
        from .caser import Caser
        base_model = Caser(config)
        # 注意：这里不需要手动暴露 item_emb，也不需要重写 get_user_embedding 了！
        # 因为重构后的 Caser 已经和 SASRec 拥有完全一致的 3D 接口
        super(AFRL_Caser, self).__init__(base_model, config)


@ModelRegistry.register('afrl_gru4rec')
class AFRL_GRU4Rec(AFRLWrapper):
    """AFRL + GRU4Rec"""
    def __init__(self, config):
        from .gru4rec import GRU4Rec
        base_model = GRU4Rec(config)
        # 同样，不需要任何补丁
        super(AFRL_GRU4Rec, self).__init__(base_model, config)

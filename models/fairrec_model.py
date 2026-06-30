import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseSequentialModel, ModelRegistry, create_model


class FairRecDecomposition(nn.Module):
    """
    FairRec 分解模块
    负责将 Base Model 输出的用户向量分解为 Fair 和 Bias 部分
    """

    def __init__(self, config):
        super(FairRecDecomposition, self).__init__()
        self.hidden_dim = config.hidden_units
        self.sensitive_attributes = config.sensitive_attributes
        self.num_sensitive = len(self.sensitive_attributes)
        self.attribute_dims = getattr(config, 'attribute_dims', {
            'gender': 2,
            'age_group': 2
        })

        # 1. 投影层：将 Base Model 的输出映射到 2x 空间以便分割
        # input: [Batch, Hidden] -> output: [Batch, Hidden * 2]
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout_rate)
        )

        # 2. 偏见预测器 (Bias Predictor): 从 ub 预测敏感属性
        self.bias_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.attribute_dims[attr])
            ) for attr in self.sensitive_attributes
        ])

        # 3. 对抗判别器 (Adversary): 尝试从 uf 预测敏感属性
        self.adversaries = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.attribute_dims[attr])
            ) for attr in self.sensitive_attributes
        ])

    def forward(self, hidden_vector):
        """
        Args:
            hidden_vector: Base Model 输出的序列特征 [Batch, Hidden]
        Returns:
            uf: 公平表征
            ub: 偏见表征
        """
        # 投影并分割
        combined = self.projection(hidden_vector)
        # Split: 前半部分为 uf, 后半部分为 ub
        uf, ub = torch.split(combined, self.hidden_dim, dim=-1)
        return uf, ub


@ModelRegistry.register('fairrec')
class FairRecForRecommendation(BaseSequentialModel):
    """
    FairRec 完整推荐模型 (Base Model + Decomposition)
    """

    def __init__(self, config):
        super(FairRecForRecommendation, self).__init__(config)
        self.config = config

        # 1. 初始化 Base Model (SASRec, Caser, GRU4Rec 等)
        # 注意：我们需要 config.base_model_name，如果没有则默认 sasrec
        base_model_name = getattr(config, 'base_model_name', 'sasrec')
        print(f"FairRec: Using base model '{base_model_name}' for sequence encoding.")

        # 避免递归创建，我们需要临时修改 model_name 来创建基座
        temp_name = config.model_name
        config.model_name = base_model_name
        self.base_model = create_model(base_model_name, config)
        config.model_name = temp_name  # 恢复

        # 2. 初始化分解模块
        self.fairrec = FairRecDecomposition(config)

        # 3. 推荐头 (Prediction Head)
        # 仅使用 uf 进行预测
        self.recommendation_head = nn.Linear(config.hidden_units, config.num_items + 1, bias=False)

        # 权重初始化
        nn.init.xavier_normal_(self.recommendation_head.weight)

        # 超参数
        self.alpha = getattr(config, 'fairrec_alpha', 1.0)  # Bias Prediction Weight
        self.beta = getattr(config, 'fairrec_beta', 1.0)  # Adversarial Weight
        self.gamma = getattr(config, 'fairrec_gamma', 0.01)  # Orthogonal Weight

    def get_user_representation(self, input_seq):
        """获取用户表征 (Base Model Output)"""
        # 调用 Base Model 的 forward 获取序列特征
        seq_output = self.base_model(input_seq)

        # === 修复 2: 维度自适应 (适配 Caser/GRU4Rec) ===
        # 如果输出已经是 [Batch, Hidden] (如Caser)，直接返回
        if seq_output.dim() == 2:
            return seq_output

        # 如果输出是 [Batch, SeqLen, Hidden] (如SASRec)，取最后一个有效位
        # 简单处理：取最后一个时间步 (假设 Base Model 已经处理了 Masking)
        if isinstance(seq_output, tuple):
            seq_output = seq_output[0]

        # 动态获取最后一个有效 item 的 embedding
        valid_len = (input_seq > 0).sum(dim=1)  # [B]
        batch_indices = torch.arange(input_seq.shape[0], device=input_seq.device)
        last_indices = torch.clamp(valid_len - 1, min=0)

        user_emb = seq_output[batch_indices, last_indices, :]  # [B, H]
        return user_emb

    def forward(self, input_seq):
        """前向传播：仅用于获取 uf (推理用)"""
        h = self.get_user_representation(input_seq)
        uf, _ = self.fairrec(h)
        return uf

    def predict(self, input_seq, candidate_items=None, fairness_config='fair'):
        """
        预测接口
        """
        h = self.get_user_representation(input_seq)
        uf, ub = self.fairrec(h)

        if fairness_config == 'fair':
            repr_to_use = uf
        else:
            # 审计模式：使用包含偏见的信息 (uf + ub) 还原原始性能
            repr_to_use = uf + ub

        # 计算分数
        if candidate_items is None:
            # 全量预测
            logits = self.recommendation_head(repr_to_use)
        else:
            # 候选集预测
            # Recommendation head weight: [NumItems, H]
            # Candidate embeddings: [B, C, H] or [B, C] -> lookup

            # 这里稍微 tricky，因为 recommendation_head 是 Linear。
            # 我们需要手动做点积
            if hasattr(self.base_model, 'item_emb'):
                # 共享 Embedding 权重 (Tied Weights)
                item_emb = self.base_model.item_emb(candidate_items)  # [B, C, H]
                if item_emb.dim() == 3:
                    scores = torch.bmm(item_emb, repr_to_use.unsqueeze(2)).squeeze(2)  # [B, C]
                else:
                    scores = torch.sum(item_emb * repr_to_use, dim=1)
            else:
                # Fallback: 全量计算再 gather (效率低但通用)
                full_logits = self.recommendation_head(repr_to_use)
                scores = torch.gather(full_logits, 1, candidate_items)

            logits = scores

        return logits

    def compute_loss(self, input_seq, target_items=None):
        """
        === 修复 1: 统一接口兼容性 ===
        支持两种调用方式:
        1. compute_loss(batch_dict) -> 来自 train_fairrec.py
        2. compute_loss(input_seq, target) -> 来自 train.py (validation)
        """
        # 1. 参数解析
        if isinstance(input_seq, dict):
            batch = input_seq
            input_seq = batch['input_seq']
            target = batch['target']
            # 如果是字典调用，还需要计算额外的 fairrec loss
            return self._compute_full_loss(batch)
        else:
            # 验证集调用，只计算推荐 Loss
            target = target_items
            return self._compute_rec_only_loss(input_seq, target)

    def _compute_rec_only_loss(self, input_seq, target):
        """仅计算推荐损失 (用于验证集)"""
        uf = self.forward(input_seq)  # 只获取 uf

        if hasattr(self.base_model, 'item_emb'):
            # 如果 base model 有 item_emb，使用共享权重计算 logits
            target_emb = self.base_model.item_emb(target)
            # 这里简化计算，仅计算 target 的得分或者全量 softmax
            # 为保持一致性，使用 recommendation_head
            rec_logits = self.recommendation_head(uf)
        else:
            rec_logits = self.recommendation_head(uf)

        rec_loss = F.cross_entropy(rec_logits, target, ignore_index=0)
        return rec_loss

    def _compute_full_loss(self, batch):
        """完整的 FairRec 训练损失 (包含对抗损失等)"""
        input_seq = batch['input_seq']
        target = batch['target']

        # 1. 获取表征
        h = self.get_user_representation(input_seq)
        uf, ub = self.fairrec(h)  # 分解

        # 2. 推荐任务损失 (Main Task)
        rec_logits = self.recommendation_head(uf)
        rec_loss = F.cross_entropy(rec_logits, target, ignore_index=0)

        # 3. 辅助损失计算
        sensitive_labels = [batch[attr].long() for attr in self.config.sensitive_attributes]

        # Bias Prediction Loss (L_pred)
        pred_loss = 0
        for i, label in enumerate(sensitive_labels):
            logits = self.fairrec.bias_predictors[i](ub)
            pred_loss += F.cross_entropy(logits, label)

        # Adversarial Loss (L_adv)
        adv_loss = 0
        for i, label in enumerate(sensitive_labels):
            logits = self.fairrec.adversaries[i](uf)
            # 目标：让判别器输出 0.5
            uniform = torch.full_like(logits, 1.0 / logits.size(-1))
            adv_loss += F.kl_div(F.log_softmax(logits, dim=-1), uniform, reduction='batchmean')

        # Orthogonal Loss (L_ortho)
        ortho_loss = torch.mean(
            torch.abs(torch.sum(uf * ub, dim=1)) / (torch.norm(uf, dim=1) * torch.norm(ub, dim=1) + 1e-8))

        # 总损失
        total_loss = rec_loss + self.alpha * pred_loss + self.beta * adv_loss + self.gamma * ortho_loss

        return total_loss, {
            'total': total_loss,
            'rec': rec_loss,
            'pred': pred_loss,
            'adv': adv_loss,
            'ortho': ortho_loss
        }

    def compute_adversary_loss(self, batch):
        """计算判别器损失 (Step A: 更新 Discriminator)"""
        input_seq = batch['input_seq']

        with torch.no_grad():
            h = self.get_user_representation(input_seq)
            uf, _ = self.fairrec(h)

        adv_disc_loss = 0
        sensitive_labels = [batch[attr].long() for attr in self.config.sensitive_attributes]

        for i, label in enumerate(sensitive_labels):
            logits = self.fairrec.adversaries[i](uf)
            adv_disc_loss += F.cross_entropy(logits, label)

        return adv_disc_loss

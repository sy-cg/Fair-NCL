import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseSequentialModel, ModelRegistry


class CaserModel(nn.Module):
    """
    Caser 核心网络结构
    """

    def __init__(self, num_users, num_items, config):
        super(CaserModel, self).__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.dims = config.hidden_units

        # 1. 嵌入层
        self.item_emb = nn.Embedding(num_items + 1, self.dims, padding_idx=0)

        # 用户嵌入 (兼容性处理)
        self.use_user_emb = hasattr(config, 'use_user_emb') and config.use_user_emb and num_users > 0
        if self.use_user_emb:
            self.user_emb = nn.Embedding(num_users, self.dims)

        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # 2. 卷积配置 (提供默认值以增强鲁棒性)
        self.filter_sizes = getattr(config, 'filter_sizes', [2, 3, 4])
        self.num_filters = getattr(config, 'num_filters', 64)
        self.num_vertical_filters = getattr(config, 'num_vertical_filters', 4)

        # 3. 垂直卷积
        self.conv_v = nn.Conv2d(1, self.num_vertical_filters, (self.max_seq_len, 1))

        # 4. 水平卷积
        self.conv_h = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (h, self.dims))
            for h in self.filter_sizes
        ])

        # 5. 全连接层
        self.fc_dim_v = self.num_vertical_filters * self.dims
        self.fc_dim_h = self.num_filters * len(self.filter_sizes)

        fc_input_dim = self.fc_dim_v + self.fc_dim_h
        if self.use_user_emb:
            fc_input_dim += self.dims

        self.fc_dropout = nn.Dropout(config.dropout_rate)
        self.fc_layer = nn.Linear(fc_input_dim, self.dims)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.item_emb.weight)
        if self.use_user_emb:
            nn.init.xavier_normal_(self.user_emb.weight)

        nn.init.xavier_uniform_(self.conv_v.weight)
        for conv in self.conv_h:
            nn.init.xavier_uniform_(conv.weight)

        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.constant_(self.fc_layer.bias, 0)

    def forward(self, input_seq, user_id=None):
        # input_seq: [Batch, SeqLen]
        item_embs = self.item_emb(input_seq)
        item_embs = self.emb_dropout(item_embs)

        # [Batch, 1, SeqLen, Hidden]
        x = item_embs.unsqueeze(1)

        # Vertical Conv
        out_v = self.conv_v(x)
        out_v = out_v.view(out_v.size(0), -1)

        # Horizontal Convs
        out_hs = []
        for conv in self.conv_h:
            conv_out = conv(x).squeeze(3)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, dim=1)

        # Concat
        z = torch.cat([out_v, out_h], dim=1)

        if self.use_user_emb and user_id is not None:
            u_emb = self.user_emb(user_id)
            z = torch.cat([z, u_emb], dim=1)
        elif self.use_user_emb:
            # Fallback if user_id missing
            u_emb = torch.zeros(z.size(0), self.dims, device=z.device)
            z = torch.cat([z, u_emb], dim=1)

        z = self.fc_dropout(z)
        z = self.fc_layer(z)
        z = F.relu(z)

        return z


@ModelRegistry.register('caser')
class Caser(BaseSequentialModel):
    def __init__(self, config):
        super(Caser, self).__init__(config)
        self.model = CaserModel(self.num_users, self.num_items, config)

        # === 核心修复 1：向外暴露 item_emb 属性 ===
        # 让 NCL 等消融框架能够正确找到 self.item_emb 计算对比损失
        self.item_emb = self.model.item_emb

    def forward(self, input_seq, user_id=None):
        # 原生输出: [Batch, Hidden]
        z = self.model(input_seq, user_id)

        # === 核心修复 2：伪装成 3D 序列输出 ===
        # 通过 expand 零成本增加一个序列维度变成 [Batch, SeqLen, Hidden]
        # 这样无论外部 NCL 框架如何进行三维切片索引，提取到的始终是正确的 z 张量
        seq_len = input_seq.size(1)
        z_3d = z.unsqueeze(1).expand(-1, seq_len, -1)
        return z_3d

    def predict(self, input_seq, candidate_items=None, user_id=None):
        # 预测时直接调用内部 model 获取 2D 向量，保证最高运行效率
        z = self.model(input_seq, user_id)

        if candidate_items is None:
            # 全量预测
            item_emb = self.model.item_emb.weight
            logits = torch.matmul(z, item_emb.transpose(0, 1))
        else:
            # 候选集预测
            candidate_emb = self.model.item_emb(candidate_items)
            if candidate_emb.dim() == 2:
                logits = torch.sum(z * candidate_emb, dim=1)
            else:
                logits = torch.matmul(z.unsqueeze(1), candidate_emb.transpose(-1, -2)).squeeze(1)

        return logits

    def compute_loss(self, input_seq, target_items):
        # 尝试从 input_seq 所在的 batch 中推断 user_id
        logits = self.predict(input_seq, user_id=None)

        # 处理 target 维度
        if target_items.dim() == 2:
            target_items = target_items[:, -1]

        loss = F.cross_entropy(logits, target_items, ignore_index=0)
        return loss
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .base_model import BaseSequentialModel, ModelRegistry


class GRU4RecModel(nn.Module):
    def __init__(self, num_items, hidden_units, num_layers, dropout_rate, device):
        super(GRU4RecModel, self).__init__()
        self.device = device
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.out_dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            init.xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, input_seq):
        seqs = self.item_emb(input_seq)
        seqs = self.emb_dropout(seqs)
        output, _ = self.gru(seqs)
        output = self.out_dropout(output)
        return output


@ModelRegistry.register('gru4rec')
class GRU4Rec(BaseSequentialModel):
    def __init__(self, config):
        super(GRU4Rec, self).__init__(config)
        # 兼容配置
        self.num_layers = getattr(config, 'num_layers', getattr(config, 'num_blocks', 2))

        self.model = GRU4RecModel(
            self.num_items,
            self.hidden_units,
            self.num_layers,
            self.dropout_rate,
            self.device
        )

        # === 核心修复：向外暴露 item_emb 属性 ===
        # 防止 NCL 获取负样本嵌入时引发 AttributeError
        self.item_emb = self.model.item_emb

    def forward(self, input_seq):
        # GRU4Rec 本身原生就是输出 3D 张量 [Batch, SeqLen, Hidden]，这里无需伪装
        return self.model(input_seq)

    def predict(self, input_seq, candidate_items=None):
        # 1. 获取序列所有时间步输出 [Batch, SeqLen, Hidden]
        seq_output = self.model(input_seq)

        # 2. 正确获取最后一个有效物品的 Hidden State
        seq_lengths = torch.sum(input_seq > 0, dim=1)  # [Batch]
        batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)
        last_indices = torch.clamp(seq_lengths - 1, min=0)

        # [Batch, Hidden]
        last_emb = seq_output[batch_indices, last_indices]

        # 3. 计算 Logits
        if candidate_items is None:
            item_emb = self.model.item_emb.weight
            logits = torch.matmul(last_emb, item_emb.transpose(0, 1))
        else:
            candidate_emb = self.model.item_emb(candidate_items)
            if candidate_emb.dim() == 2:
                logits = torch.sum(last_emb * candidate_emb, dim=1)
            else:
                logits = torch.matmul(last_emb.unsqueeze(1), candidate_emb.transpose(-1, -2)).squeeze(1)

        return logits

    def compute_loss(self, input_seq, target_items):
        logits = self.predict(input_seq)

        # 确保 target 是 Next Item [Batch]
        if target_items.dim() == 2:
            target_items = target_items[:, -1]  # 取序列最后一个作为 Target

        loss = F.cross_entropy(logits, target_items, ignore_index=0)
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.cuda.amp import autocast


class OptimizedBertMultiHeadAttention(nn.Module):
    """GPU优化的BERT双向注意力机制"""

    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(OptimizedBertMultiHeadAttention, self).__init__()
        assert hidden_units % num_heads == 0

        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.head_dim = hidden_units // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 使用单个线性层进行QKV投影
        self.qkv_linear = nn.Linear(hidden_units, hidden_units * 3, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_units, hidden_units)

    def forward(self, inputs, mask=None):
        batch_size, seq_len, _ = inputs.size()

        # 一次性计算QKV
        qkv = self.qkv_linear(inputs)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用padding mask
        if mask is not None:
            scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e4)

        # 双向注意力，不使用因果mask
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        output = self.out_proj(context)
        output += inputs  # 残差连接

        return output


class OptimizedBertTransformerBlock(nn.Module):
    """GPU优化的BERT Transformer块"""

    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(OptimizedBertTransformerBlock, self).__init__()
        self.attention = OptimizedBertMultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.feed_forward = OptimizedPointWiseFeedForward(hidden_units, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, inputs, mask=None):
        attention_output = self.attention(self.layer_norm1(inputs), mask)
        feed_forward_output = self.feed_forward(self.layer_norm2(attention_output))
        return feed_forward_output

from .base_model import BaseSequentialModel, ModelRegistry

@ModelRegistry.register('bert4rec')
class OptimizedBERT4Rec(BaseSequentialModel):
    """GPU优化的BERT4Rec模型"""

    def __init__(self, config):
        super(OptimizedBERT4Rec, self).__init__(config)

        self.config = config
        self.num_items = config.num_items
        self.max_seq_len = config.max_seq_len
        self.hidden_units = config.hidden_units
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.mask_prob = getattr(config, 'mask_prob', 0.15)

        # 特殊token
        self.mask_token = self.num_items + 1
        self.pad_token = 0

        # 嵌入层
        self.item_emb = nn.Embedding(
            self.num_items + 2, self.hidden_units, padding_idx=0
        )  # +2 for pad and mask
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_units)
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)

        # Transformer块 (双向)
        self.transformer_blocks = nn.ModuleList([
            OptimizedBertTransformerBlock(self.hidden_units, self.num_heads, self.dropout_rate)
            for _ in range(self.num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # 输出层
        self.out = nn.Linear(self.hidden_units, self.num_items + 1)  # +1 for mask token

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """优化的权重初始化"""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size()

        # Padding mask
        mask = (input_seq == 0)

        # 位置嵌入
        positions = torch.arange(seq_len, device=input_seq.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # 项目嵌入
        seq_emb = self.item_emb(input_seq)
        pos_emb = self.pos_emb(positions)

        # 添加位置嵌入
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer块 (双向)
        for transformer in self.transformer_blocks:
            seq_emb = transformer(seq_emb, mask)

        seq_emb = self.layer_norm(seq_emb)

        return seq_emb

    def predict(self, input_seq, candidate_items=None):
        """优化的预测函数"""
        seq_emb = self.forward(input_seq)

        # 获取最后一个非padding位置对应的表示
        seq_lengths = torch.sum(input_seq > 0, dim=1)  # [B]
        last_indices = torch.clamp(seq_lengths - 1, min=0)
        batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)

        # [B, H] 每个用户的最后一个token表示
        last_hidden = seq_emb[batch_indices, last_indices]

        if candidate_items is None:
            logits = self.out(last_hidden)  # [B, num_items+1]
        else:
            # 只预测候选物品
            candidate_emb = self.item_emb(candidate_items)  # [N, H]
            if candidate_emb.dim() == 2:
                logits = torch.matmul(last_hidden, candidate_emb.T)  # [B, N]
            else:
                logits = torch.matmul(last_hidden.unsqueeze(1), candidate_emb.transpose(-1, -2)).squeeze(1)

        assert logits.ndim == 2, f"Predictions shape should be [B, N], but got {logits.shape}"
        return logits

    def mask_sequence_optimized(self, input_seq):
        """GPU优化的序列masking"""
        batch_size, seq_len = input_seq.size()

        masked_seq = input_seq.clone()
        labels = torch.full_like(input_seq, -100)  # -100 表示不计算损失

        # 向量化的masking操作
        for i in range(batch_size):
            valid_positions = (input_seq[i] != 0).nonzero(as_tuple=True)[0]

            if len(valid_positions) > 0:
                # 随机选择masking位置
                num_mask = max(1, int(len(valid_positions) * self.mask_prob))
                mask_positions = valid_positions[torch.randperm(len(valid_positions))[:num_mask]]

                for pos in mask_positions:
                    original_item = input_seq[i, pos].item()
                    labels[i, pos] = original_item

                    # 80%概率用mask token替换，10%随机替换，10%保持不变
                    prob = random.random()
                    if prob < 0.8:
                        masked_seq[i, pos] = self.mask_token
                    elif prob < 0.9:
                        masked_seq[i, pos] = random.randint(1, self.num_items)

        return masked_seq, labels

    def compute_loss(self, input_seq, target_items=None):
        """优化的损失计算"""
        if target_items is not None:
            # 用于next item prediction
            logits = self.predict(input_seq)  # [B, num_items + 1]
            loss = F.cross_entropy(logits, target_items.long(), ignore_index=0)

        else:
            # 用于masked language model训练
            masked_seq, labels = self.mask_sequence_optimized(input_seq)
            logits = self.predict(masked_seq)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).long(),
                ignore_index=-100
            )

        return loss


# 导入优化的前馈网络
from .sasrec import OptimizedPointWiseFeedForward

# 为了向后兼容，保持原名称
BERT4Rec = OptimizedBERT4Rec
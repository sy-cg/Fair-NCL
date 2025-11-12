import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast

from .base_model import BaseSequentialModel, ModelRegistry


class OptimizedPointWiseFeedForward(nn.Module):
    """GPU优化的前馈网络"""

    def __init__(self, hidden_units, dropout_rate):
        super(OptimizedPointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)  # 使用inplace以节省内存
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs  # 残差连接
        return outputs


class OptimizedMultiHeadAttention(nn.Module):
    """GPU优化的多头注意力"""

    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(OptimizedMultiHeadAttention, self).__init__()
        assert hidden_units % num_heads == 0

        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.head_dim = hidden_units // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 使用单个线性层进行QKV投影以提高效率
        self.qkv_linear = nn.Linear(hidden_units, hidden_units * 3, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_units, hidden_units)

    def forward(self, inputs, mask=None):
        batch_size, seq_len, _ = inputs.size()

        # 一次性计算QKV
        qkv = self.qkv_linear(inputs)  # [B, L, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用mask
        if mask is not None:
            scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e4)

        # 因果mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=inputs.device), diagonal=1).bool()
        scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        # Softmax和dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力
        context = torch.matmul(attention, v)  # [B, H, L, D_h]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 输出投影
        output = self.out_proj(context)
        output += inputs  # 残差连接

        return output


class OptimizedTransformerBlock(nn.Module):
    """GPU优化的Transformer块"""

    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(OptimizedTransformerBlock, self).__init__()
        self.attention = OptimizedMultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.feed_forward = OptimizedPointWiseFeedForward(hidden_units, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, inputs, mask=None):
        # Pre-norm架构，更稳定
        attention_output = self.attention(self.layer_norm1(inputs), mask)
        feed_forward_output = self.feed_forward(self.layer_norm2(attention_output))
        return feed_forward_output

@ModelRegistry.register('sasrec')
class OptimizedSASRec(BaseSequentialModel):
#class OptimizedSASRec(nn.Module):
    """GPU优化的SASRec模型"""

    def __init__(self, config):
        super(OptimizedSASRec, self).__init__(config)

        self.config = config
        self.num_items = config.num_items
        self.max_seq_len = config.max_seq_len
        self.hidden_units = config.hidden_units
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate

        # 嵌入层
        self.item_emb = nn.Embedding(
            self.num_items + 1, self.hidden_units, padding_idx=0
        )
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_units)
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)

        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            OptimizedTransformerBlock(self.hidden_units, self.num_heads, self.dropout_rate)
            for _ in range(self.num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_units, eps=1e-8)

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

        # 位置嵌入 - 优化版本
        positions = torch.arange(seq_len, device=input_seq.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # 项目嵌入
        seq_emb = self.item_emb(input_seq)
        pos_emb = self.pos_emb(positions)

        # 添加位置嵌入
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer块
        for transformer in self.transformer_blocks:
            seq_emb = transformer(seq_emb, mask)

        seq_emb = self.layer_norm(seq_emb)

        return seq_emb

    def predict(self, input_seq, candidate_items=None):
        """优化的预测函数"""
        seq_emb = self.forward(input_seq)  # [batch_size, seq_len, hidden_units]

        # 获取最后一个非padding位置的表示
        seq_lengths = torch.sum(input_seq > 0, dim=1)  # [batch_size]
        batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)
        last_indices = torch.clamp(seq_lengths - 1, min=0)

        last_emb = seq_emb[batch_indices, last_indices]  # [batch_size, hidden_units]

        if candidate_items is None:
            # 预测所有物品
            item_emb = self.item_emb.weight[1:]  # 排除padding
            logits = torch.matmul(last_emb, item_emb.transpose(0, 1))
        else:
            # 预测特定候选物品
            candidate_emb = self.item_emb(candidate_items)
            if candidate_emb.dim() == 2:  # [batch_size, hidden_units]
                logits = torch.sum(last_emb * candidate_emb, dim=1)
            else:  # [batch_size, num_candidates, hidden_units]
                logits = torch.matmul(last_emb.unsqueeze(1), candidate_emb.transpose(-1, -2)).squeeze(1)

        # Debug 强化检查
        assert logits.dim() == 2, f"logits 应为二维张量 [B, N]，但当前为 {logits.shape}"

        return logits

    def compute_loss(self, input_seq, target_items):
        """优化的损失计算"""
        logits = self.predict(input_seq)

        # 标签平滑
        if hasattr(self.config, 'label_smoothing') and self.config.label_smoothing > 0:
            loss = self._label_smoothing_loss(logits, target_items, self.config.label_smoothing)
        else:
            loss = F.cross_entropy(logits, target_items, ignore_index=0)

        return loss

    def _label_smoothing_loss(self, logits, targets, smoothing=0.1):
        """标签平滑损失"""
        confidence = 1.0 - smoothing
        log_probs = F.log_softmax(logits, dim=-1)

        nll_loss = F.nll_loss(log_probs, targets, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)

        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


# 为了向后兼容，保持原名称
SASRec = OptimizedSASRec
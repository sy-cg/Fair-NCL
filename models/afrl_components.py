import torch
import torch.nn as nn
import torch.nn.functional as F


class AFRL_Generator(nn.Module):
    """AFRL属性特定嵌入生成器"""

    def __init__(self, embed_dim, dropout_rate=0.1):
        super(AFRL_Generator, self).__init__()

        self.embed_dim = embed_dim

        # 6层MLP结构
        self.generator_network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim * 4, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim * 8, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 8, embed_dim * 4, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.02)

    def forward(self, x):
        return self.generator_network(x)


class AFRL_Discriminator(nn.Module):
    """AFRL属性判别器"""

    def __init__(self, embed_dim, num_classes, dropout_rate=0.1):
        super(AFRL_Discriminator, self).__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 2层MLP
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)

    def forward(self, embeddings):
        return self.network(embeddings)

    def predict_proba(self, embeddings):
        logits = self.forward(embeddings)
        return F.log_softmax(logits, dim=1)


class AFRL_CombineMLP(nn.Module):
    """AFRL信息聚合模块"""

    def __init__(self, embed_dim, num_attributes, dropout_rate=0.1):
        super(AFRL_CombineMLP, self).__init__()

        self.embed_dim = embed_dim
        self.num_attributes = num_attributes

        # 输入维度：去偏嵌入 + 属性嵌入
        input_dim = embed_dim * (num_attributes + 1)

        self.network = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 8, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.02)

    def forward(self, debiased_emb, attribute_embs, fairness_mask):
        """
        Args:
            debiased_emb: 去偏协作嵌入 [batch_size, embed_dim]
            attribute_embs: 属性嵌入列表 [(batch_size, embed_dim), ...]
            fairness_mask: 公平性需求掩码 [batch_size, num_attributes]
        """
        # 连接所有属性嵌入
        attr_embs_concat = torch.cat(attribute_embs, dim=1)  # [batch_size, num_attrs * embed_dim]

        # 应用公平性掩码
        # 将fairness_mask扩展到嵌入维度
        mask_expanded = fairness_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        mask_expanded = mask_expanded.reshape(fairness_mask.size(0), -1)

        masked_attr_embs = attr_embs_concat * mask_expanded

        # 连接去偏嵌入和掩码后的属性嵌入
        combined = torch.cat([debiased_emb, masked_attr_embs], dim=1)

        # 通过MLP生成最终公平嵌入
        fair_emb = self.network(combined)

        return fair_emb


class DebiasedCollaborativeEncoder(nn.Module):
    """去偏协作信号编码器"""

    def __init__(self, embed_dim, dropout_rate=0.1):
        super(DebiasedCollaborativeEncoder, self).__init__()

        self.embed_dim = embed_dim

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.02)

    def forward(self, x):
        # 编码并保持残差连接
        encoded = self.encoder(x)
        return encoded + x  # 残差连接
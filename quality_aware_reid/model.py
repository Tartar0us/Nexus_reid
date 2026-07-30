"""
质量感知时序建模模型 (Quality-Aware Temporal Modeling for Video ReID)

核心创新：
  原始方法: 时序注意力权重 = f(语义特征)  ← 隐式，无法解释
  本方法:   时序注意力权重 = f(语义特征) × g(帧质量分数)  ← 显式，可解释

帧质量评估模块 (Frame Quality Estimator, FQE) 从两个维度显式量化帧质量：
  1. 清晰度分数 (Sharpness Score): 基于特征图的梯度能量，模糊帧梯度小
  2. 信息量分数 (Information Score): 基于特征激活的熵，遮挡/背景帧激活稀疏

两个分数融合后与语义注意力相乘，得到质量感知的最终权重。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FrameQualityEstimator(nn.Module):
    """
    帧质量评估模块 (FQE)
    输入: 帧级特征图 [B*T, 2048, H, W]
    输出: 质量分数 [B, T, 1]，值域 (0, 1)，越高表示帧质量越好

    两路质量信号:
      - 清晰度路: 对特征图计算 Laplacian 梯度能量，模糊帧能量低
      - 信息量路: 对 GAP 后的特征向量计算激活熵，遮挡帧激活稀疏
    """
    def __init__(self, feat_dim=2048):
        super().__init__()

        # 清晰度评估头：从特征图空间梯度估计清晰度
        self.sharpness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),          # [B*T, 2048, 4, 4]
            nn.Flatten(),                      # [B*T, 2048*16]
            nn.Linear(2048 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 信息量评估头：从 GAP 特征向量估计信息量
        self.info_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 融合两路质量分数的可学习权重
        self.fusion = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, feat_map, feat_vec):
        """
        feat_map: [B*T, 2048, H, W]  特征图（用于清晰度）
        feat_vec: [B*T, 2048]         GAP后特征向量（用于信息量）
        返回: quality_score [B*T, 1]
        """
        sharpness = self.sharpness_head(feat_map)   # [B*T, 1]
        info      = self.info_head(feat_vec)         # [B*T, 1]

        # 归一化融合权重
        w = F.softmax(self.fusion, dim=0)
        quality = w[0] * sharpness + w[1] * info    # [B*T, 1]
        return quality


class QualityAwareTemporalAttention(nn.Module):
    """
    质量感知时序注意力模块

    原始注意力: attn = softmax(MLP(feat))
    质量感知:   attn = softmax(MLP(feat) + log(quality + eps))

    加法融合而非乘法，避免质量分数为0时完全屏蔽某帧，
    同时保留语义注意力的主导地位。
    """
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.semantic_attn = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.quality_scale = nn.Parameter(torch.tensor(1.0))  # 可学习的质量权重系数

    def forward(self, feat_vec, quality_score):
        """
        feat_vec:      [B, T, 2048]
        quality_score: [B, T, 1]
        返回: 加权聚合后的视频特征 [B, 2048]，以及注意力权重 [B, T, 1]
        """
        semantic = self.semantic_attn(feat_vec)                    # [B, T, 1]
        quality_bias = self.quality_scale * torch.log(quality_score + 1e-6)  # [B, T, 1]
        combined = semantic + quality_bias                          # [B, T, 1]
        attn_weights = F.softmax(combined, dim=1)                  # [B, T, 1]
        video_feat = (feat_vec * attn_weights).sum(dim=1)          # [B, 2048]
        return video_feat, attn_weights


class QualityAwareVideoReID(nn.Module):
    """
    质量感知视频行人重识别模型

    架构:
        输入视频 [B, T, 3, H, W]
            ↓
        ResNet50 Backbone (Layer1-3冻结, Layer4微调)
            ↓
        帧质量评估模块 (FQE) → 质量分数 [B, T, 1]
            ↓
        质量感知时序注意力 → 视频特征 [B, 2048]
            ↓
        特征嵌入 (2048→512) + BN
            ↓
        分类头 / 嵌入输出
    """
    def __init__(self, num_classes, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        # Backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 冻结 Layer1-3
        for name, module in self.backbone.named_children():
            if name in ['0', '1', '2', '3', '4', '5', '6']:
                for param in module.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 核心创新模块
        self.fqe = FrameQualityEstimator(feat_dim=2048)
        self.qa_attention = QualityAwareTemporalAttention(feat_dim=2048)

        # 特征嵌入
        self.embedding = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.3)

        # 分类头
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding=False, return_quality=False):
        B, T, C, H, W = x.shape

        # 空间特征提取
        x_flat = x.view(B * T, C, H, W)
        feat_map = self.backbone(x_flat)                          # [B*T, 2048, h, w]
        feat_vec = self.gap(feat_map).squeeze(-1).squeeze(-1)     # [B*T, 2048]

        # 帧质量评估
        quality = self.fqe(feat_map, feat_vec)                    # [B*T, 1]
        quality = quality.view(B, T, 1)                           # [B, T, 1]

        # 质量感知时序注意力聚合
        feat_seq = feat_vec.view(B, T, 2048)                      # [B, T, 2048]
        video_feat, attn_weights = self.qa_attention(feat_seq, quality)  # [B, 2048]

        # 特征嵌入
        embedding = self.embedding(video_feat)
        embedding = self.bn(embedding)
        embedding = self.dropout(embedding)
        logits = self.classifier(embedding)

        if return_embedding and return_quality:
            emb_norm = F.normalize(embedding, p=2, dim=1)
            return logits, emb_norm, quality.squeeze(-1), attn_weights.squeeze(-1)
        elif return_embedding:
            emb_norm = F.normalize(embedding, p=2, dim=1)
            return logits, emb_norm
        return logits

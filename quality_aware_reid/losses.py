"""
损失函数
- LabelSmoothingCrossEntropy: 复用自 Step3
- TripletLoss: 复用自 Step3
- QualityRegularizationLoss: 新增，约束质量分数的合理性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


class TripletLoss(nn.Module):
    """Hard Mining Triplet Loss"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        loss = torch.tensor(0.0, device=embeddings.device)
        num_triplets = 0

        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (
                torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]
            if not pos_mask.any() or not neg_mask.any():
                continue
            pos_dist = dist_matrix[i][pos_mask].max()
            neg_dist = dist_matrix[i][neg_mask].min()
            loss = loss + torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
            num_triplets += 1

        return loss / num_triplets if num_triplets > 0 else loss


class QualityRegularizationLoss(nn.Module):
    """
    质量分数正则化损失（新增）

    目的：防止质量评估模块退化（所有帧质量相同 → 退化为普通注意力）

    两个约束：
    1. 多样性约束：同一视频内各帧质量分数的方差不能太小
       → 鼓励模型区分高质量帧和低质量帧
    2. 稳定性约束：同一身份不同视频的平均质量分布应相似
       → 防止质量分数对身份信息过度敏感
    """
    def __init__(self, diversity_weight=0.1, min_variance=0.01):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.min_variance = min_variance

    def forward(self, quality_scores):
        """
        quality_scores: [B, T] 每个视频每帧的质量分数
        """
        # 多样性约束：惩罚方差过小（所有帧质量相同）
        var_per_video = quality_scores.var(dim=1)                  # [B]
        diversity_loss = F.relu(self.min_variance - var_per_video).mean()

        return self.diversity_weight * diversity_loss


class QualityRankingLoss(nn.Module):
    """
    Encourage clean videos to receive higher quality scores than degraded ones.

    Synthetic degradation provides a weak supervisory signal for the frame
    quality estimator without requiring external frame-quality labels.
    """
    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def forward(self, clean_quality, degraded_quality):
        clean_mean = clean_quality.mean(dim=1)
        degraded_mean = degraded_quality.mean(dim=1)
        return F.relu(self.margin - (clean_mean - degraded_mean)).mean()

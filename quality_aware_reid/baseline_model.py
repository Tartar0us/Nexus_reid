"""
Baseline video ReID models used for ablation under the same evaluator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SemanticAttentionVideoReID(nn.Module):
    """
    Step3-style video ReID baseline.

    Frames are encoded by ResNet50, aggregated by semantic temporal attention,
    then projected to a 512-d embedding for classification/retrieval.
    """
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        for name, module in self.backbone.named_children():
            if name in ["0", "1", "2", "3", "4", "5", "6"]:
                for param in module.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.embedding = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding=False):
        bsz, seq_len, channels, height, width = x.shape
        x = x.view(bsz * seq_len, channels, height, width)
        feat = self.backbone(x)
        feat = self.gap(feat).view(bsz, seq_len, 2048)

        attn = self.attention(feat)
        attn = torch.softmax(attn, dim=1)
        video_feat = (feat * attn).sum(dim=1)

        embedding = self.embedding(video_feat)
        embedding = self.bn(embedding)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, F.normalize(embedding, p=2, dim=1)
        return logits


class MeanPoolingVideoReID(nn.Module):
    """
    Mean-pooling video ReID baseline.

    This is the simplest temporal aggregation ablation: all sampled frames
    contribute equally after ResNet50 frame encoding.
    """
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        for name, module in self.backbone.named_children():
            if name in ["0", "1", "2", "3", "4", "5", "6"]:
                for param in module.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding=False):
        bsz, seq_len, channels, height, width = x.shape
        x = x.view(bsz * seq_len, channels, height, width)
        feat = self.backbone(x)
        feat = self.gap(feat).view(bsz, seq_len, 2048)
        video_feat = feat.mean(dim=1)

        embedding = self.embedding(video_feat)
        embedding = self.bn(embedding)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, F.normalize(embedding, p=2, dim=1)
        return logits

"""
Tensor-level synthetic degradation for quality supervision.

Inputs are normalized ImageNet tensors with shape [B, T, 3, H, W]. The helpers
temporarily denormalize to [0, 1], apply degradation, and normalize back.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def denormalize(videos: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(videos.device, videos.dtype)
    std = IMAGENET_STD.to(videos.device, videos.dtype)
    return (videos * std + mean).clamp(0.0, 1.0)


def normalize(videos: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(videos.device, videos.dtype)
    std = IMAGENET_STD.to(videos.device, videos.dtype)
    return (videos - mean) / std


def blur(videos: torch.Tensor, severity: float) -> torch.Tensor:
    bsz, seq_len, channels, height, width = videos.shape
    kernel = 3 if severity < 0.67 else 5
    pad = kernel // 2
    flat = videos.view(bsz * seq_len, channels, height, width)
    flat = F.avg_pool2d(flat, kernel_size=kernel, stride=1, padding=pad)
    return flat.view(bsz, seq_len, channels, height, width)


def darken(videos: torch.Tensor, severity: float) -> torch.Tensor:
    factor = max(0.1, 1.0 - 0.75 * severity)
    return (videos * factor).clamp(0.0, 1.0)


def occlude(videos: torch.Tensor, severity: float) -> torch.Tensor:
    out = videos.clone()
    bsz, seq_len, _, height, width = out.shape
    side = int((height * width * 0.35 * severity) ** 0.5)
    side = max(8, min(side, height, width))
    for b_idx in range(bsz):
        for t_idx in range(seq_len):
            top = random.randint(0, max(0, height - side))
            left = random.randint(0, max(0, width - side))
            out[b_idx, t_idx, :, top:top + side, left:left + side] = 0.0
    return out


def degrade_videos(videos: torch.Tensor, mode: str = "mixed", severity: float = 0.5) -> torch.Tensor:
    severity = min(1.0, max(0.0, float(severity)))
    raw = denormalize(videos)

    if mode == "none" or severity == 0:
        degraded = raw
    elif mode == "blur":
        degraded = blur(raw, severity)
    elif mode == "brightness":
        degraded = darken(raw, severity)
    elif mode == "occlusion":
        degraded = occlude(raw, severity)
    elif mode == "mixed":
        degraded = raw
        for op in random.sample([blur, darken, occlude], k=random.randint(1, 3)):
            degraded = op(degraded, severity)
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

    return normalize(degraded.clamp(0.0, 1.0))

"""
质量感知视频行人重识别 - 训练脚本

改进点对比 Step3:
  Step3: 时序注意力权重 = softmax(MLP(语义特征))
  本方法: 时序注意力权重 = softmax(MLP(语义特征) + λ·log(质量分数))
         + QualityRegularizationLoss 防止质量模块退化

训练策略与 Step3 完全一致（Warmup+Cosine, AdamW, 梯度裁剪）
新增: quality_reg_weight 控制质量正则化损失的权重
"""
import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import QualityAwareVideoReID
from degradation import degrade_videos
from losses import (
    LabelSmoothingCrossEntropy,
    QualityRankingLoss,
    QualityRegularizationLoss,
    TripletLoss,
)
from dataset import MARSDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train quality-aware video person ReID on MARS.")
    parser.add_argument("--data-root", required=True,
                        help="Path to MARS bbox_train directory.")
    parser.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "models"),
                        help="Directory for checkpoints, config, and metrics.")
    parser.add_argument("--num-classes", type=int, default=625)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=0.05)
    parser.add_argument("--fusion-mode", choices=["additive_log", "multiplicative"],
                        default="additive_log")
    parser.add_argument("--disable-quality-bias", action="store_true",
                        help="Use semantic attention while still training/exporting FQE scores.")
    parser.add_argument("--quality-rank-weight", type=float, default=0.0,
                        help="Weight for synthetic degradation quality ranking loss.")
    parser.add_argument("--degradation-mode",
                        choices=["none", "blur", "brightness", "occlusion", "mixed"],
                        default="mixed")
    parser.add_argument("--degradation-severity", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup_epochs:
            lr = self.base_lr * (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_scale', 1.0)
        self.current_epoch += 1
        return lr


def train(args=None):
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    save_json(os.path.join(args.save_dir, "config.json"), vars(args))

    # ── 数据集 ───────────────────────────────────────────────
    train_ds = MARSDataset(args.data_root, seq_len=args.seq_len, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=True)

    # ── 模型 ─────────────────────────────────────────────────
    model = QualityAwareVideoReID(
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        fusion_mode=args.fusion_mode,
        use_quality_bias=not args.disable_quality_bias,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total/1e6:.1f}M | 可训练: {trainable/1e6:.1f}M")

    # ── 损失函数 ──────────────────────────────────────────────
    ce_crit      = LabelSmoothingCrossEntropy(smoothing=0.1)
    tri_crit     = TripletLoss(margin=0.3)
    quality_crit = QualityRegularizationLoss(diversity_weight=args.quality_weight)
    rank_crit    = QualityRankingLoss(margin=0.05)

    # ── 优化器（分层学习率，与 Step3 一致）────────────────────
    params = [
        {'params': model.backbone[7].parameters(),
         'lr_scale': 0.1},                          # Layer4 低学习率
        {'params': model.fqe.parameters(),
         'lr_scale': 1.0},                          # 质量评估模块
        {'params': model.qa_attention.parameters(),
         'lr_scale': 1.0},                          # 质量感知注意力
        {'params': model.embedding.parameters(),
         'lr_scale': 1.0},
        {'params': model.bn.parameters(),
         'lr_scale': 1.0},
        {'params': model.classifier.parameters(),
         'lr_scale': 1.0},
    ]
    optimizer = optim.AdamW(
        [{'params': p['params'], 'lr': args.base_lr * p['lr_scale'],
          'lr_scale': p['lr_scale']} for p in params],
        weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.base_lr, args.min_lr)

    # ── 训练循环 ──────────────────────────────────────────────
    print("=" * 65)
    print("开始训练 - 质量感知视频行人重识别")
    print("=" * 65)

    best_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        model.train()
        current_lr = scheduler.step()
        total_loss = total_ce = total_tri = total_qual = total_rank = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", ncols=120)
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, embeddings, quality_scores, _ = model(
                videos, return_embedding=True, return_quality=True)

            ce_loss   = ce_crit(logits, labels)
            tri_loss  = tri_crit(embeddings, labels)
            qual_loss = quality_crit(quality_scores)
            rank_loss = torch.tensor(0.0, device=device)
            if args.quality_rank_weight > 0:
                degraded = degrade_videos(
                    videos,
                    mode=args.degradation_mode,
                    severity=args.degradation_severity,
                )
                _, _, degraded_quality, _ = model(
                    degraded, return_embedding=True, return_quality=True)
                rank_loss = rank_crit(quality_scores, degraded_quality)

            loss = (
                ce_loss
                + args.triplet_weight * tri_loss
                + qual_loss
                + args.quality_rank_weight * rank_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += ce_loss.item()
            total_tri  += tri_loss.item() if isinstance(tri_loss, torch.Tensor) else tri_loss
            total_qual += qual_loss.item()
            total_rank += rank_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'tri': f'{tri_loss.item() if isinstance(tri_loss, torch.Tensor) else tri_loss:.4f}',
                'qual': f'{qual_loss.item():.4f}',
                'rank': f'{rank_loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        n = len(train_loader)
        avg = total_loss / n
        print(f"\nEpoch [{epoch+1}/{args.epochs}] "
              f"loss={avg:.4f} ce={total_ce/n:.4f} "
              f"tri={total_tri/n:.4f} qual={total_qual/n:.4f} "
              f"rank={total_rank/n:.4f} lr={current_lr:.2e}")

        history.append({'epoch': epoch+1, 'loss': avg,
                        'ce': total_ce/n, 'tri': total_tri/n,
                        'qual': total_qual/n, 'rank': total_rank/n})

        if avg < best_loss:
            best_loss = avg
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': vars(args),
                        'loss': best_loss},
                       os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ✅ 保存最优模型 (loss={best_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'model_epoch{epoch+1}.pth'))
        print()

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))

    save_json(os.path.join(args.save_dir, 'train_history.json'), history)
    save_json(os.path.join(args.save_dir, 'metrics.json'), {
        "best_train_loss": best_loss,
        "epochs": args.epochs,
        "final_epoch": history[-1] if history else None,
    })

    print("=" * 65)
    print(f"训练完成！最优 loss: {best_loss:.4f}")
    print("=" * 65)


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()

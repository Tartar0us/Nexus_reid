"""
Train baseline video ReID models for ablation.

Supported baselines:
  - mean_pooling
  - semantic_attention
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_model import MeanPoolingVideoReID, SemanticAttentionVideoReID
from dataset import MARSDataset
from losses import LabelSmoothingCrossEntropy, TripletLoss
from train import WarmupCosineScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train video ReID ablation baselines.")
    parser.add_argument("--data-root", required=True,
                        help="Path to MARS bbox_train directory.")
    parser.add_argument("--model-type", choices=["mean_pooling", "semantic_attention"],
                        required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=625)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
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


def build_baseline(model_type, num_classes):
    if model_type == "mean_pooling":
        return MeanPoolingVideoReID(num_classes=num_classes)
    if model_type == "semantic_attention":
        return SemanticAttentionVideoReID(num_classes=num_classes)
    raise ValueError(f"Unknown baseline model type: {model_type}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    save_json(os.path.join(args.save_dir, "config.json"), vars(args))

    train_ds = MARSDataset(args.data_root, seq_len=args.seq_len, is_train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = build_baseline(args.model_type, args.num_classes).to(device)
    ce_crit = LabelSmoothingCrossEntropy(smoothing=0.1)
    tri_crit = TripletLoss(margin=0.3)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.base_lr, args.min_lr)

    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        current_lr = scheduler.step()
        total_loss = total_ce = total_tri = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]", ncols=120)
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, embeddings = model(videos, return_embedding=True)
            ce_loss = ce_crit(logits, labels)
            tri_loss = tri_crit(embeddings, labels)
            loss = ce_loss + args.triplet_weight * tri_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_tri += tri_loss.item() if isinstance(tri_loss, torch.Tensor) else tri_loss
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "tri": f"{tri_loss.item() if isinstance(tri_loss, torch.Tensor) else tri_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            })

        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        row = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "ce": total_ce / n_batches,
            "tri": total_tri / n_batches,
        }
        history.append(row)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] loss={row['loss']:.4f} "
            f"ce={row['ce']:.4f} tri={row['tri']:.4f} lr={current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
                "loss": best_loss,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"Saved best model: loss={best_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    save_json(os.path.join(args.save_dir, "train_history.json"), history)
    save_json(os.path.join(args.save_dir, "metrics.json"), {
        "best_train_loss": best_loss,
        "epochs": args.epochs,
        "final_epoch": history[-1] if history else None,
    })


if __name__ == "__main__":
    main()

"""
Export frame-level quality scores and attention weights for qualitative analysis.

This tool is intentionally lightweight: it writes JSON records that can later be
used to draw paper figures or inspect failure cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval_mars_official import load_checkpoint
from model import QualityAwareVideoReID


def build_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def collect_tracklets(bbox_root: Path, limit: int):
    tracklets = []
    for pid_dir in sorted(p for p in bbox_root.iterdir() if p.is_dir()):
        by_tid = {}
        for image_path in sorted(pid_dir.glob("*.jpg")):
            tid = image_path.name[6:11]
            by_tid.setdefault(tid, []).append(image_path)
        for tid, frames in sorted(by_tid.items()):
            if frames:
                tracklets.append({
                    "pid": pid_dir.name,
                    "tracklet_id": tid,
                    "frames": frames,
                })
                if len(tracklets) >= limit:
                    return tracklets
    return tracklets


def sample_frames(frames: list[Path], seq_len: int):
    if len(frames) >= seq_len:
        indices = np.linspace(0, len(frames) - 1, seq_len).round().astype(int)
        return [frames[i] for i in indices]
    return frames + [frames[-1]] * (seq_len - len(frames))


def load_video(frames: list[Path], transform):
    tensors = []
    for path in frames:
        with path.open("rb") as f:
            image = Image.open(f).convert("RGB")
        tensors.append(transform(image))
    return torch.stack(tensors).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Export quality and attention weights.")
    parser.add_argument("--bbox-root", type=Path, required=True,
                        help="Path to MARS bbox_train or bbox_test PID folders.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=625)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = QualityAwareVideoReID(
        num_classes=args.num_classes, seq_len=args.seq_len).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    transform = build_transform()
    records = []

    with torch.no_grad():
        for tracklet in collect_tracklets(args.bbox_root, args.limit):
            frames = sample_frames(tracklet["frames"], args.seq_len)
            video = load_video(frames, transform).to(device)
            _, _, quality, attention = model(
                video, return_embedding=True, return_quality=True)

            records.append({
                "pid": tracklet["pid"],
                "tracklet_id": tracklet["tracklet_id"],
                "frame_paths": [str(path) for path in frames],
                "quality": [float(x) for x in quality.squeeze(0).cpu().tolist()],
                "attention": [float(x) for x in attention.squeeze(0).cpu().tolist()],
            })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "bbox_root": str(args.bbox_root),
            "seq_len": args.seq_len,
            "records": records,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved quality visualization records: {args.output_json}")


if __name__ == "__main__":
    main()

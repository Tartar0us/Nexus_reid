"""
MARS official-protocol evaluation under synthetic frame degradation.

Use this for robustness experiments after normal official-protocol evaluation.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

from eval_mars_official import (
    MARSOfficialTrackletDataset,
    build_model,
    evaluate_cmc_map,
    extract_features,
    load_checkpoint,
    load_mars_test_tracklets,
    load_query_indices,
    pairwise_euclidean,
    save_metrics,
    split_query_gallery,
)


class Degrade:
    def __init__(self, mode: str, severity: float, seed: int):
        self.mode = mode
        self.severity = severity
        self.rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.mode == "none":
            return image
        if self.mode == "blur":
            radius = max(0.1, 3.0 * self.severity)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        if self.mode == "brightness":
            factor = max(0.1, 1.0 - 0.75 * self.severity)
            return ImageEnhance.Brightness(image).enhance(factor)
        if self.mode == "occlusion":
            return self._occlude(image)
        if self.mode == "mixed":
            choice = self.rng.choice(["blur", "brightness", "occlusion"])
            return Degrade(choice, self.severity, self.rng.randint(0, 10**9))(image)
        raise ValueError(f"Unknown degradation mode: {self.mode}")

    def _occlude(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = width * height
        side = int((area * 0.35 * self.severity) ** 0.5)
        side = max(8, min(side, width, height))
        x0 = self.rng.randint(0, max(0, width - side))
        y0 = self.rng.randint(0, max(0, height - side))
        image = image.copy()
        image.paste((0, 0, 0), (x0, y0, x0 + side, y0 + side))
        return image


class DegradedMARSOfficialTrackletDataset(MARSOfficialTrackletDataset):
    def __init__(self, tracklets, seq_len, degradation):
        super().__init__(tracklets, seq_len)
        self.degradation = degradation

    def __getitem__(self, index):
        tracklet = self.tracklets[index]
        tensors = []
        for path in self.sample_frames(tracklet.frame_paths):
            try:
                with path.open("rb") as f:
                    img = Image.open(f).convert("RGB")
                img = self.degradation(img)
                tensors.append(self.transform(img))
            except Exception:
                tensors.append(torch.zeros(3, 256, 128))

        return (
            torch.stack(tensors),
            tracklet.pid,
            tracklet.camid,
            tracklet.tracklet_index,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MARS under synthetic degradation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["quality_aware", "semantic_attention", "mean_pooling"],
                        default="quality_aware")
    parser.add_argument("--degradation", choices=["none", "blur", "brightness", "occlusion", "mixed"],
                        default="mixed")
    parser.add_argument("--severity", type=float, default=0.5,
                        help="Degradation strength in [0, 1].")
    parser.add_argument("--num-classes", type=int, default=625)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path,
                        help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    severity = min(1.0, max(0.0, args.severity))
    device = torch.device(args.device)

    tracklets = load_mars_test_tracklets(args.data_root)
    query_indices = load_query_indices(args.data_root)
    query_tracklets, gallery_tracklets = split_query_gallery(tracklets, query_indices)
    degradation = Degrade(args.degradation, severity, args.seed)

    query_loader = DataLoader(
        DegradedMARSOfficialTrackletDataset(query_tracklets, args.seq_len, degradation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    gallery_loader = DataLoader(
        DegradedMARSOfficialTrackletDataset(gallery_tracklets, args.seq_len, degradation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model_type, args.num_classes, args.seq_len).to(device)
    load_checkpoint(model, args.checkpoint, device)

    q_feat, q_pids, q_camids = extract_features(model, query_loader, device)
    g_feat, g_pids, g_camids = extract_features(model, gallery_loader, device)
    distmat = pairwise_euclidean(q_feat, g_feat)
    cmc, mAP, valid_queries = evaluate_cmc_map(
        distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20)

    print("=" * 70)
    print("MARS Degraded Official Protocol Results")
    print("=" * 70)
    print(f"Model      : {args.model_type}")
    print(f"Degradation: {args.degradation} severity={severity:.2f}")
    print(f"Valid queries: {valid_queries}/{len(q_pids)}")
    print(f"Rank-1 : {cmc[0] * 100:.2f}%")
    print(f"Rank-5 : {cmc[4] * 100:.2f}%")
    print(f"Rank-10: {cmc[9] * 100:.2f}%")
    print(f"Rank-20: {cmc[19] * 100:.2f}%")
    print(f"mAP    : {mAP * 100:.2f}%")

    metrics = {
        "protocol": "MARS official query/gallery degraded",
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "degradation": args.degradation,
        "severity": severity,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_classes": args.num_classes,
        "query_tracklets": len(query_tracklets),
        "gallery_tracklets": len(gallery_tracklets),
        "valid_queries": int(valid_queries),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "rank20": float(cmc[19]),
        "mAP": float(mAP),
    }
    if args.output_json:
        save_metrics(args.output_json, metrics)
        print(f"Saved metrics: {args.output_json}")


if __name__ == "__main__":
    main()

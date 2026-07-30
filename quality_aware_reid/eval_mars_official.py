"""
Official-protocol MARS evaluation for video person ReID.

Expected MARS layout:
    <data_root>/
        bbox_train/
        bbox_test/
        info/
            test_name.txt
            tracks_test_info.mat
            query_IDX.mat

The official metadata defines test tracklets and query indices. This script
uses all non-query test tracklets as gallery and filters same-pid same-camera
matches during CMC/mAP computation.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from baseline_model import MeanPoolingVideoReID, SemanticAttentionVideoReID
from model import QualityAwareVideoReID


@dataclass(frozen=True)
class Tracklet:
    frame_paths: tuple[Path, ...]
    pid: int
    camid: int
    tracklet_index: int


def build_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def read_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def as_1d_int_array(value) -> np.ndarray:
    arr = np.asarray(value).squeeze()
    return arr.astype(np.int64).reshape(-1)


def load_mars_test_tracklets(data_root: Path) -> list[Tracklet]:
    info_dir = data_root / "info"
    bbox_test = data_root / "bbox_test"
    names = read_names(info_dir / "test_name.txt")

    mat = loadmat(info_dir / "tracks_test_info.mat")
    if "track_test_info" in mat:
        track_info = mat["track_test_info"]
    elif "tracks_test_info" in mat:
        track_info = mat["tracks_test_info"]
    else:
        keys = ", ".join(k for k in mat.keys() if not k.startswith("__"))
        raise KeyError(f"Cannot find test track info in .mat. Keys: {keys}")

    tracklets: list[Tracklet] = []
    for idx, row in enumerate(np.asarray(track_info)):
        start, end, pid, camid = [int(x) for x in row[:4]]
        frame_names = names[start - 1:end]
        frame_paths = tuple(bbox_test / name[:4] / name for name in frame_names)
        tracklets.append(Tracklet(
            frame_paths=frame_paths,
            pid=pid,
            camid=camid,
            tracklet_index=idx,
        ))
    return tracklets


def load_query_indices(data_root: Path) -> np.ndarray:
    mat = loadmat(data_root / "info" / "query_IDX.mat")
    if "query_IDX" not in mat:
        keys = ", ".join(k for k in mat.keys() if not k.startswith("__"))
        raise KeyError(f"Cannot find query_IDX in .mat. Keys: {keys}")
    # MARS query indices are 1-based tracklet indices.
    return as_1d_int_array(mat["query_IDX"]) - 1


class MARSOfficialTrackletDataset(Dataset):
    def __init__(self, tracklets: Iterable[Tracklet], seq_len: int):
        self.tracklets = list(tracklets)
        self.seq_len = seq_len
        self.transform = build_transform()

    def __len__(self):
        return len(self.tracklets)

    def sample_frames(self, frame_paths: tuple[Path, ...]) -> tuple[Path, ...]:
        if len(frame_paths) >= self.seq_len:
            indices = np.linspace(0, len(frame_paths) - 1, self.seq_len).round().astype(int)
            return tuple(frame_paths[i] for i in indices)
        return frame_paths + (frame_paths[-1],) * (self.seq_len - len(frame_paths))

    def __getitem__(self, index):
        tracklet = self.tracklets[index]
        tensors = []
        for path in self.sample_frames(tracklet.frame_paths):
            try:
                with path.open("rb") as f:
                    img = Image.open(f).convert("RGB")
                tensors.append(self.transform(img))
            except Exception:
                tensors.append(torch.zeros(3, 256, 128))

        return (
            torch.stack(tensors),
            tracklet.pid,
            tracklet.camid,
            tracklet.tracklet_index,
        )


def split_query_gallery(tracklets: list[Tracklet], query_indices: np.ndarray):
    query_set = set(int(i) for i in query_indices.tolist())
    query = [t for t in tracklets if t.tracklet_index in query_set]
    gallery = [t for t in tracklets if t.tracklet_index not in query_set]
    return query, gallery


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def build_model(model_type: str, num_classes: int, seq_len: int,
                fusion_mode: str = "additive_log", use_quality_bias: bool = True):
    if model_type == "quality_aware":
        return QualityAwareVideoReID(
            num_classes=num_classes,
            seq_len=seq_len,
            fusion_mode=fusion_mode,
            use_quality_bias=use_quality_bias,
        )
    if model_type == "semantic_attention":
        return SemanticAttentionVideoReID(num_classes=num_classes)
    if model_type == "mean_pooling":
        return MeanPoolingVideoReID(num_classes=num_classes)
    raise ValueError(f"Unknown model type: {model_type}")


def extract_features(model, loader, device):
    model.eval()
    features, pids, camids = [], [], []
    with torch.no_grad():
        for videos, pid, camid, _ in tqdm(loader, desc="Extract", ncols=100):
            videos = videos.to(device, non_blocking=True)
            _, embedding = model(videos, return_embedding=True)
            features.append(embedding.cpu())
            pids.append(pid.numpy())
            camids.append(camid.numpy())
    return (
        torch.cat(features, dim=0).numpy(),
        np.concatenate(pids, axis=0),
        np.concatenate(camids, axis=0),
    )


def pairwise_euclidean(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    q2 = np.sum(np.square(query_features), axis=1, keepdims=True)
    g2 = np.sum(np.square(gallery_features), axis=1, keepdims=True).T
    dist = q2 + g2 - 2.0 * np.matmul(query_features, gallery_features.T)
    return np.sqrt(np.maximum(dist, 0.0))


def evaluate_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    if distmat.shape[1] < max_rank:
        max_rank = distmat.shape[1]

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_queries = 0

    for q_idx in range(distmat.shape[0]):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_queries += 1

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        all_ap.append((precision * raw_cmc).sum() / num_rel)

    if num_valid_queries == 0:
        raise RuntimeError("No valid query has matching gallery tracklets.")

    cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / num_valid_queries
    mAP = float(np.mean(all_ap))
    return cmc, mAP, num_valid_queries


def save_metrics(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MARS with official query/gallery protocol.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to MARS root containing bbox_test and info.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Model checkpoint path.")
    parser.add_argument("--model-type", choices=["quality_aware", "semantic_attention", "mean_pooling"],
                        default="quality_aware",
                        help="Model architecture to instantiate for the checkpoint.")
    parser.add_argument("--num-classes", type=int, default=625)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--fusion-mode", choices=["additive_log", "multiplicative"],
                        default="additive_log",
                        help="Quality-aware fusion mode; only used for model-type quality_aware.")
    parser.add_argument("--disable-quality-bias", action="store_true",
                        help="Disable quality bias for quality_aware ablation.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path,
                        help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tracklets = load_mars_test_tracklets(args.data_root)
    query_indices = load_query_indices(args.data_root)
    query_tracklets, gallery_tracklets = split_query_gallery(tracklets, query_indices)

    print(f"MARS official split: query={len(query_tracklets)} gallery={len(gallery_tracklets)}")
    print(f"Device: {device}")

    query_loader = DataLoader(
        MARSOfficialTrackletDataset(query_tracklets, args.seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    gallery_loader = DataLoader(
        MARSOfficialTrackletDataset(gallery_tracklets, args.seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model_type,
        args.num_classes,
        args.seq_len,
        args.fusion_mode,
        not args.disable_quality_bias,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)

    print("Extracting query features...")
    q_feat, q_pids, q_camids = extract_features(model, query_loader, device)
    print("Extracting gallery features...")
    g_feat, g_pids, g_camids = extract_features(model, gallery_loader, device)

    distmat = pairwise_euclidean(q_feat, g_feat)
    cmc, mAP, valid_queries = evaluate_cmc_map(
        distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20)

    print("=" * 70)
    print("MARS Official Protocol Results")
    print("=" * 70)
    print(f"Valid queries: {valid_queries}/{len(q_pids)}")
    print(f"Rank-1 : {cmc[0] * 100:.2f}%")
    print(f"Rank-5 : {cmc[4] * 100:.2f}%")
    print(f"Rank-10: {cmc[9] * 100:.2f}%")
    print(f"Rank-20: {cmc[19] * 100:.2f}%")
    print(f"mAP    : {mAP * 100:.2f}%")

    metrics = {
        "protocol": "MARS official query/gallery",
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "seq_len": args.seq_len,
        "fusion_mode": args.fusion_mode if args.model_type == "quality_aware" else "",
        "use_quality_bias": (not args.disable_quality_bias) if args.model_type == "quality_aware" else "",
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

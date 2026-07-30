"""
Report model parameter counts and inference speed for paper tables.

This intentionally avoids optional FLOPs dependencies so it can run in a clean
PyTorch environment. If a FLOPs package is added later, this script can become
the single place that reports params, FLOPs, latency, and FPS.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval_mars_official import build_model


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark(model, videos, warmup: int, repeats: int, device: torch.device):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(videos, return_embedding=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(videos, return_embedding=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / repeats
    throughput = videos.size(0) * repeats / elapsed
    return latency_ms, throughput


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Report model params and inference speed.")
    parser.add_argument("--model-type",
                        choices=["quality_aware", "semantic_attention", "mean_pooling"],
                        default="quality_aware")
    parser.add_argument("--num-classes", type=int, default=625)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fusion-mode", choices=["additive_log", "multiplicative"],
                        default="additive_log")
    parser.add_argument("--disable-quality-bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(
        args.model_type,
        args.num_classes,
        args.seq_len,
        args.fusion_mode,
        not args.disable_quality_bias,
    ).to(device)

    total, trainable = count_parameters(model)
    videos = torch.randn(
        args.batch_size,
        args.seq_len,
        3,
        args.height,
        args.width,
        device=device,
    )
    latency_ms, throughput = benchmark(model, videos, args.warmup, args.repeats, device)

    report = {
        "model_type": args.model_type,
        "num_classes": args.num_classes,
        "seq_len": args.seq_len,
        "input_size": [args.seq_len, 3, args.height, args.width],
        "batch_size": args.batch_size,
        "fusion_mode": args.fusion_mode if args.model_type == "quality_aware" else "",
        "use_quality_bias": (not args.disable_quality_bias) if args.model_type == "quality_aware" else "",
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_m": total / 1e6,
        "trainable_params_m": trainable / 1e6,
        "latency_ms_per_batch": latency_ms,
        "throughput_videos_per_sec": throughput,
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }

    print("=" * 70)
    print("Model Complexity / Speed Report")
    print("=" * 70)
    print(f"Model              : {report['model_type']}")
    print(f"Total params       : {report['total_params_m']:.2f}M")
    print(f"Trainable params   : {report['trainable_params_m']:.2f}M")
    print(f"Latency per batch  : {report['latency_ms_per_batch']:.2f} ms")
    print(f"Throughput         : {report['throughput_videos_per_sec']:.2f} videos/s")
    print(f"Device             : {report['device']}")

    if args.output_json:
        save_json(args.output_json, report)
        print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()

"""Benchmark inference FPS for any registered backbone.

Usage:
    python tools/benchmark_inference_fps.py configs/dinov2_plnet.yaml --runs 100 --warmup 20
    python tools/benchmark_inference_fps.py configs/plnet.yaml --runs 100 --warmup 20
"""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hawp.fsl.config import cfg as base_cfg
from hawp.fsl.model.build import build_model


def _autocast(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def benchmark(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model = model.to(device).eval()

    h = cfg.DATASETS.IMAGE.HEIGHT
    w = cfg.DATASETS.IMAGE.WIDTH
    channels = 3

    amp_enabled = bool(getattr(cfg.MODEL.ENHANCEMENTS, "AMP", False)) or args.amp

    dummy_image = torch.randn(1, channels, h, w, device=device)
    dummy_ann = [{"width": w, "height": h, "filename": "benchmark_dummy.jpg"}]

    print(f"Model:      {cfg.MODEL.NAME}")
    print(f"Device:     {device}")
    print(f"Input:      (1, {channels}, {h}, {w})")
    print(f"AMP:        {amp_enabled}")
    print(f"Warmup:     {args.warmup} runs")
    print(f"Benchmark:  {args.runs} runs")
    print()

    with torch.no_grad():
        for i in range(args.warmup):
            with _autocast(amp_enabled):
                _ = model(dummy_image, annotations=dummy_ann)
            if device.type == "cuda":
                torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    with torch.no_grad():
        for i in range(args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with _autocast(amp_enabled):
                _ = model(dummy_image, annotations=dummy_ann)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    peak_mem_mb = 0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    fps_list = [1.0 / t for t in latencies]
    mean_fps = mean(fps_list)
    std_fps = stdev(fps_list) if len(fps_list) > 1 else 0.0
    mean_latency_ms = mean(latencies) * 1000
    std_latency_ms = stdev(latencies) * 1000 if len(latencies) > 1 else 0.0

    param_total = sum(p.numel() for p in model.parameters())
    param_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "model": cfg.MODEL.NAME,
        "device": str(device),
        "input_size": [1, channels, h, w],
        "amp": amp_enabled,
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
        "mean_fps": round(mean_fps, 2),
        "std_fps": round(std_fps, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        "params_total": param_total,
        "params_trainable": param_trainable,
    }

    print(f"FPS:          {mean_fps:.2f} +/- {std_fps:.2f}")
    print(f"Latency:      {mean_latency_ms:.2f} +/- {std_latency_ms:.2f} ms")
    print(f"Peak GPU:     {peak_mem_mb:.1f} MB")
    print(f"Params total: {param_total:,}")
    print(f"Params train: {param_trainable:,}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference FPS")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup runs")
    parser.add_argument("--amp", action="store_true", help="Force AMP on")
    parser.add_argument("--save-json", type=str, default=None, help="Save results to JSON")
    parser.add_argument("opts", nargs="*", default=[], help="Extra YACS overrides")
    args = parser.parse_args()

    cfg = base_cfg.clone()
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    benchmark(cfg, args)


if __name__ == "__main__":
    main()

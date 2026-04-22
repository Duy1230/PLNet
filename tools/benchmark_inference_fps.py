"""
Benchmark inference FPS for PLNet model variants.

Measures pure inference speed using randomly-initialised weights on
real wireframe test images.  No checkpoint required.

Usage:
    python tools/benchmark_inference_fps.py configs/plnet.yaml
    python tools/benchmark_inference_fps.py configs/enhanced_plnet_crossattn.yaml
    python tools/benchmark_inference_fps.py --compare configs/plnet.yaml configs/enhanced_plnet_crossattn.yaml
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hawp.fsl.config import cfg as base_cfg
from hawp.fsl.config.paths_catalog import DatasetCatalog
from hawp.fsl.model.build import build_model


def _autocast_context(device, enabled=False, amp_dtype="fp16"):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def _load_cfg(config_path):
    cfg = base_cfg.clone()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg


def _prepare_test_images(cfg, max_images=200):
    """Load and preprocess wireframe test images to tensors."""
    test_name = cfg.DATASETS.TEST[0]
    dargs = DatasetCatalog.get(test_name)
    img_dir = dargs["args"]["root"]
    ann_file = dargs["args"]["ann_file"]

    with open(ann_file, "r") as f:
        annotations = json.load(f)

    img_h = cfg.DATASETS.IMAGE.HEIGHT
    img_w = cfg.DATASETS.IMAGE.WIDTH
    ann_h = cfg.DATASETS.TARGET.HEIGHT
    ann_w = cfg.DATASETS.TARGET.WIDTH

    model_name = cfg.MODEL.NAME
    uses_raw = model_name in {"PointLine", "EnhancedPointLine"}

    images = []
    metas = []
    for ann in annotations[:max_images]:
        path = str(Path(img_dir) / ann["filename"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float64)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        orig_h, orig_w = ann["height"], ann["width"]
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0)

        tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        if not uses_raw:
            pixel_mean = cfg.DATASETS.IMAGE.PIXEL_MEAN
            pixel_std = cfg.DATASETS.IMAGE.PIXEL_STD
            if cfg.DATASETS.IMAGE.TO_255:
                tensor = tensor * 255.0
            for c in range(3):
                tensor[c] = (tensor[c] - pixel_mean[c]) / pixel_std[c]

        images.append(tensor)
        metas.append({
            "filename": ann["filename"],
            "height": ann_h,
            "width": ann_w,
        })

    return images, metas


@torch.no_grad()
def benchmark_model(cfg, images, metas, warmup=50, measured=200, use_amp=False, amp_dtype="fp16"):
    device = torch.device(cfg.MODEL.DEVICE)
    if use_amp and device.type != "cuda":
        use_amp = False
    model = build_model(cfg).to(device)
    model.eval()

    n_imgs = len(images)
    n_iters = warmup + measured

    for i in range(warmup):
        idx = i % n_imgs
        img = images[idx].unsqueeze(0).to(device, non_blocking=True)
        ann = [metas[idx]]
        with _autocast_context(device, enabled=use_amp, amp_dtype=amp_dtype):
            _ = model(img, ann)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies_ms = []
    stage_times = {
        "time_backbone": [],
        "time_proposal": [],
        "time_matching": [],
        "time_verification": [],
    }

    for i in range(measured):
        idx = i % n_imgs
        img = images[idx].unsqueeze(0).to(device, non_blocking=True)
        ann = [metas[idx]]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with _autocast_context(device, enabled=use_amp, amp_dtype=amp_dtype):
            output, extra_info = model(img, ann)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)
        for k in stage_times:
            if k in extra_info:
                stage_times[k].append(extra_info[k] * 1000.0)

    avg_ms = mean(latencies_ms)
    fps = 1000.0 / avg_ms
    std_ms = stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    result = {
        "model": cfg.MODEL.NAME,
        "device": str(device),
        "images_measured": measured,
        "avg_latency_ms": round(avg_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "fps": round(fps, 1),
        "amp": bool(use_amp),
        "amp_dtype": amp_dtype,
        "total_params": param_count,
        "trainable_params": trainable,
        "peak_mem_mb": round(peak_mem, 1),
    }

    for k, vals in stage_times.items():
        if vals:
            result[f"{k}_ms"] = round(mean(vals), 2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference FPS for PLNet variants."
    )
    parser.add_argument(
        "configs", nargs="+", type=str,
        help="Path(s) to YAML config file(s)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="When two configs given, print side-by-side comparison",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--save-json", type=str, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP autocast")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP autocast")
    parser.set_defaults(amp=None)
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Autocast dtype when --amp is enabled",
    )

    args = parser.parse_args()
    results = []

    for config_path in args.configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")

        cfg = _load_cfg(config_path)
        images, metas = _prepare_test_images(cfg, max_images=args.max_images)
        print(f"Loaded {len(images)} test images")

        if cfg.MODEL.DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        cfg_amp = bool(getattr(getattr(cfg.MODEL, "ENHANCEMENTS", object()), "AMP", False))
        use_amp = cfg_amp if args.amp is None else bool(args.amp)

        result = benchmark_model(
            cfg, images, metas,
            warmup=args.warmup,
            measured=args.steps,
            use_amp=use_amp,
            amp_dtype=args.amp_dtype,
        )
        results.append(result)

        print(f"\nModel: {result['model']}")
        print(f"  FPS:              {result['fps']}")
        print(f"  Avg latency:      {result['avg_latency_ms']:.2f} ms")
        print(f"  P50 latency:      {result['p50_latency_ms']:.2f} ms")
        print(f"  P95 latency:      {result['p95_latency_ms']:.2f} ms")
        print(f"  Total params:     {result['total_params']:,}")
        print(f"  Trainable params: {result['trainable_params']:,}")
        print(f"  Peak VRAM:        {result['peak_mem_mb']:.1f} MB")
        for k in ["time_backbone_ms", "time_proposal_ms",
                   "time_matching_ms", "time_verification_ms"]:
            if k in result:
                print(f"  {k:22s} {result[k]:.2f}")

    if args.compare and len(results) == 2:
        a, b = results
        speedup = a["avg_latency_ms"] / max(b["avg_latency_ms"], 1e-6)
        fps_gain = b["fps"] / max(a["fps"], 1e-6)
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"  {a['model']:30s} -> {b['model']}")
        print(f"  Latency: {a['avg_latency_ms']:.1f} ms -> {b['avg_latency_ms']:.1f} ms  ({speedup:.2f}x)")
        print(f"  FPS:     {a['fps']:.1f} -> {b['fps']:.1f}  ({fps_gain:.2f}x)")
        param_ratio = b["trainable_params"] / max(a["trainable_params"], 1)
        print(f"  Params:  {a['trainable_params']:,} -> {b['trainable_params']:,}  ({param_ratio:.2f}x)")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

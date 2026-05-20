"""Benchmark inference FPS for any registered backbone.

Usage:
    python tools/benchmark_inference_fps.py configs/dinov2_plnet.yaml --runs 100 --warmup 20
    python tools/benchmark_inference_fps.py configs/plnet.yaml --runs 100 --warmup 20
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, stdev

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hawp.fsl.config import cfg as base_cfg
from hawp.fsl.benchmark import AVAILABLE_DATASETS
from hawp.fsl.dataset import build_test_dataset
from hawp.fsl.model.build import build_model


def _autocast(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(cfg, args):
    if args.device:
        return torch.device(args.device)
    cfg_device = str(getattr(cfg.MODEL, "DEVICE", "cuda"))
    if cfg_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled(cfg, args, device):
    config_amp = bool(getattr(cfg.MODEL.ENHANCEMENTS, "AMP", False))
    enabled = config_amp if args.amp is None else bool(args.amp)
    return bool(enabled and device.type == "cuda")


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return None
    ckpt_path = Path(ckpt_path).resolve()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = _coerce_state_dict_shapes(state_dict, model.state_dict())
    model.load_state_dict(state_dict)
    return str(ckpt_path)


def _coerce_state_dict_shapes(checkpoint_state, model_state):
    coerced = dict(checkpoint_state)
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is None or not isinstance(value, torch.Tensor):
            continue
        if value.shape == target.shape:
            continue
        if (
            value.ndim == 2
            and target.ndim == 4
            and target.shape[-2:] == (1, 1)
            and value.shape == target.shape[:2]
        ):
            coerced[key] = value.reshape(target.shape)
    return coerced


def _latency_stats(latencies):
    fps_list = [1.0 / t for t in latencies]
    return {
        "mean_fps": mean(fps_list),
        "std_fps": stdev(fps_list) if len(fps_list) > 1 else 0.0,
        "mean_latency_ms": mean(latencies) * 1000,
        "std_latency_ms": stdev(latencies) * 1000 if len(latencies) > 1 else 0.0,
    }


def _parameter_counts(model):
    return {
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def benchmark_dummy(cfg, args):
    device = _resolve_device(cfg, args)
    model = build_model(cfg)
    checkpoint_path = _load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    h = cfg.DATASETS.IMAGE.HEIGHT
    w = cfg.DATASETS.IMAGE.WIDTH
    channels = 3

    amp_enabled = _amp_enabled(cfg, args, device)
    runs = args.runs if args.runs is not None else 100

    dummy_image = torch.randn(1, channels, h, w, device=device)
    dummy_ann = [{"width": w, "height": h, "filename": "benchmark_dummy.jpg"}]

    print(f"Model:      {cfg.MODEL.NAME}")
    print(f"Mode:       synthetic dummy image")
    print(f"Checkpoint: {checkpoint_path or 'random initialization'}")
    print(f"Device:     {device}")
    print(f"Input:      (1, {channels}, {h}, {w})")
    print(f"AMP:        {amp_enabled}")
    print(f"Warmup:     {args.warmup} runs")
    print(f"Benchmark:  {runs} runs")
    print()

    with torch.no_grad():
        for i in range(args.warmup):
            with _autocast(device, amp_enabled):
                _ = model(dummy_image, annotations=dummy_ann)
            _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    with torch.no_grad():
        for i in range(runs):
            _synchronize(device)
            t0 = time.perf_counter()

            with _autocast(device, amp_enabled):
                _ = model(dummy_image, annotations=dummy_ann)

            _synchronize(device)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    peak_mem_mb = 0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    stats = _latency_stats(latencies)
    counts = _parameter_counts(model)

    results = {
        "model": cfg.MODEL.NAME,
        "mode": "dummy",
        "checkpoint": checkpoint_path,
        "device": str(device),
        "input_size": [1, channels, h, w],
        "amp": amp_enabled,
        "warmup_runs": args.warmup,
        "benchmark_runs": runs,
        "mean_fps": round(stats["mean_fps"], 2),
        "std_fps": round(stats["std_fps"], 2),
        "mean_latency_ms": round(stats["mean_latency_ms"], 2),
        "std_latency_ms": round(stats["std_latency_ms"], 2),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        **counts,
    }

    print(f"FPS:          {stats['mean_fps']:.2f} +/- {stats['std_fps']:.2f}")
    print(f"Latency:      {stats['mean_latency_ms']:.2f} +/- {stats['std_latency_ms']:.2f} ms")
    print(f"Peak GPU:     {peak_mem_mb:.1f} MB")
    print(f"Params total: {counts['params_total']:,}")
    print(f"Params train: {counts['params_trainable']:,}")

    if args.save_json:
        _save_json(args.save_json, results)

    return results


def _collect_real_samples(cfg, device, args):
    test_datasets = build_test_dataset(cfg)
    if len(test_datasets) != 1:
        raise RuntimeError(f"Expected one test dataset, got {len(test_datasets)}")
    dataset_name, dataloader = test_datasets[0]

    max_images = args.max_images if args.max_images and args.max_images > 0 else None
    samples = []
    for images, annotations in dataloader:
        samples.append((images, annotations))
        if max_images is not None and len(samples) >= max_images:
            break
    return dataset_name, samples


def benchmark_real_images(cfg, args):
    if args.ckpt is None:
        raise ValueError("--ckpt is required when --real-images is used")

    device = _resolve_device(cfg, args)
    model = build_model(cfg)
    checkpoint_path = _load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()
    if args.dataset == "york":
        model.topk_junctions = 512

    amp_enabled = _amp_enabled(cfg, args, device)
    dataset_name, samples = _collect_real_samples(cfg, device, args)
    if not samples:
        raise RuntimeError("No benchmark samples were loaded")

    measured_images = args.runs if args.runs is not None and args.runs > 0 else len(samples)
    repeats = max(1, int(args.repeat))

    print(f"Model:      {cfg.MODEL.NAME}")
    print(f"Mode:       real images")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {dataset_name}")
    print(f"Device:     {device}")
    print(f"Images:     {measured_images} per repeat ({len(samples)} cached)")
    print(f"Repeats:    {repeats}")
    print(f"AMP:        {amp_enabled}")
    print(f"Warmup:     {args.warmup} runs")
    print()

    with torch.no_grad():
        for i in range(args.warmup):
            images, annotations = samples[i % len(samples)]
            images = images.to(device, non_blocking=True)
            with _autocast(device, amp_enabled):
                _ = model(images, annotations=annotations)
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    repeat_results = []
    all_latencies = []
    with torch.no_grad():
        for repeat_idx in range(repeats):
            latencies = []
            for i in range(measured_images):
                images, annotations = samples[i % len(samples)]
                images = images.to(device, non_blocking=True)
                _synchronize(device)
                t0 = time.perf_counter()
                with _autocast(device, amp_enabled):
                    _ = model(images, annotations=annotations)
                _synchronize(device)
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
            elapsed = sum(latencies)
            fps = measured_images / elapsed if elapsed > 0 else 0.0
            repeat_results.append(
                {
                    "repeat": repeat_idx + 1,
                    "num_images": measured_images,
                    "total_time_sec": round(elapsed, 4),
                    "fps": round(fps, 2),
                    "mean_latency_ms": round(mean(latencies) * 1000, 2),
                }
            )
            all_latencies.extend(latencies)

    repeat_fps = [item["fps"] for item in repeat_results]
    mean_fps = mean(repeat_fps)
    std_fps = stdev(repeat_fps) if len(repeat_fps) > 1 else 0.0
    mean_latency_ms = mean(all_latencies) * 1000
    std_latency_ms = stdev(all_latencies) * 1000 if len(all_latencies) > 1 else 0.0

    peak_mem_mb = 0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    counts = _parameter_counts(model)

    results = {
        "model": cfg.MODEL.NAME,
        "mode": "real_images",
        "checkpoint": checkpoint_path,
        "dataset_alias": args.dataset,
        "dataset_name": dataset_name,
        "device": str(device),
        "amp": amp_enabled,
        "warmup_runs": args.warmup,
        "benchmark_runs": measured_images,
        "repeat": repeats,
        "mean_fps": round(mean_fps, 2),
        "std_fps": round(std_fps, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        "per_repeat": repeat_results,
        **counts,
    }

    print(f"FPS:          {mean_fps:.2f} +/- {std_fps:.2f}")
    print(f"Latency:      {mean_latency_ms:.2f} +/- {std_latency_ms:.2f} ms")
    print(f"Peak GPU:     {peak_mem_mb:.1f} MB")
    print(f"Params total: {counts['params_total']:,}")
    print(f"Params train: {counts['params_trainable']:,}")

    if args.save_json:
        _save_json(args.save_json, results)

    return results


def _save_json(path, results):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference FPS")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of measured runs/images. Defaults to 100 in dummy mode and the full split in real-image mode.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup runs")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Force AMP on")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Force AMP off")
    parser.set_defaults(amp=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional model checkpoint")
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="Benchmark real test images with checkpointed detector outputs and post-processing.",
    )
    parser.add_argument(
        "--dataset",
        default="wireframe",
        choices=sorted(AVAILABLE_DATASETS.keys()),
        help="Dataset alias used with --real-images",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat measured real-image benchmark")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on cached real images. By default all images in the split are cached.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu")
    parser.add_argument("--save-json", type=str, default=None, help="Save results to JSON")
    parser.add_argument("opts", nargs="*", default=[], help="Extra YACS overrides")
    args = parser.parse_args()

    cfg = base_cfg.clone()
    cfg.merge_from_file(args.config)
    if args.real_images:
        cfg.DATASETS.TEST = (AVAILABLE_DATASETS[args.dataset],)
    if args.device is not None:
        cfg.MODEL.DEVICE = args.device
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.real_images:
        benchmark_real_images(cfg, args)
    else:
        benchmark_dummy(cfg, args)


if __name__ == "__main__":
    main()

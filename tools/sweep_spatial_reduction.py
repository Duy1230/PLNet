"""
Benchmark matrix for cross-attention efficiency variants.

Compares:
1) Baseline PLNet / Enhanced without cross-attention
2) MHA vs SDPA attention backends
3) AvgPool vs learned reduction
4) Optional early proposal pruning
across several spatial reduction factors.
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


def _load_cfg(config_path, overrides=None):
    cfg = base_cfg.clone()
    cfg.merge_from_file(config_path)
    if overrides:
        cfg.defrost()
        for key_path, value in overrides.items():
            node = cfg
            parts = key_path.split(".")
            for part in parts[:-1]:
                node = getattr(node, part)
            setattr(node, parts[-1], value)
        cfg.freeze()
    return cfg


def _prepare_test_images(cfg, max_images=200):
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

    images, metas = [], []
    for ann in annotations[:max_images]:
        path = str(Path(img_dir) / ann["filename"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float64)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        if not uses_raw:
            pixel_mean = cfg.DATASETS.IMAGE.PIXEL_MEAN
            pixel_std = cfg.DATASETS.IMAGE.PIXEL_STD
            if cfg.DATASETS.IMAGE.TO_255:
                tensor = tensor * 255.0
            for c in range(3):
                tensor[c] = (tensor[c] - pixel_mean[c]) / pixel_std[c]
        images.append(tensor)
        metas.append({"filename": ann["filename"], "height": ann_h, "width": ann_w})
    return images, metas


def _autocast_context(device, enabled=False, amp_dtype="fp16"):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


@torch.no_grad()
def benchmark_once(cfg, images, metas, warmup=50, measured=200, use_amp=False, amp_dtype="fp16"):
    device = torch.device(cfg.MODEL.DEVICE)
    if use_amp and device.type != "cuda":
        use_amp = False
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg).to(device)
    model.eval()
    n_imgs = len(images)

    for i in range(warmup):
        idx = i % n_imgs
        img = images[idx].unsqueeze(0).to(device, non_blocking=True)
        with _autocast_context(device, enabled=use_amp, amp_dtype=amp_dtype):
            _ = model(img, [metas[idx]])

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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with _autocast_context(device, enabled=use_amp, amp_dtype=amp_dtype):
            output, extra_info = model(img, [metas[idx]])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        for key in stage_times:
            if key in extra_info:
                stage_times[key].append(extra_info[key] * 1000.0)

    avg = mean(latencies_ms)
    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "avg_latency_ms": round(avg, 2),
        "std_latency_ms": round(stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0, 2),
        "p50_latency_ms": round(sorted(latencies_ms)[len(latencies_ms) // 2], 2),
        "p95_latency_ms": round(sorted(latencies_ms)[int(len(latencies_ms) * 0.95)], 2),
        "fps": round(1000.0 / avg, 1),
        "amp": bool(use_amp),
        "amp_dtype": amp_dtype,
        "total_params": param_count,
        "trainable_params": trainable,
        "peak_mem_mb": round(peak_mem, 1),
    }
    for key, vals in stage_times.items():
        if vals:
            result[f"{key}_ms"] = round(mean(vals), 2)
    return result


def _build_matrix(base_cfg_path, include_sr1=False):
    reductions = [1, 2, 4, 8] if include_sr1 else [2, 4, 8]
    experiments = [
        {
            "label": "Baseline PLNet",
            "config": str(REPO_ROOT / "configs" / "plnet.yaml"),
            "overrides": None,
            "tokens": "N/A",
            "variant": "baseline",
        },
        {
            "label": "Enhanced no-cross-attn",
            "config": base_cfg_path,
            "overrides": {"MODEL.ENHANCEMENTS.USE_CROSS_ATTENTION": False},
            "tokens": "0",
            "variant": "no_cross_attn",
        },
    ]

    for sr in reductions:
        token_h = max(1, 128 // sr)
        token_w = max(1, 128 // sr)
        tokens = token_h * token_w
        common = {
            "MODEL.ENHANCEMENTS.USE_CROSS_ATTENTION": True,
            "MODEL.ENHANCEMENTS.CROSS_ATTN_SPATIAL_REDUCTION": sr,
            "MODEL.ENHANCEMENTS.EARLY_PROPOSAL_TOPK": 0,
            "MODEL.ENHANCEMENTS.CROSS_ATTN_FORCE_FLASH": False,
        }

        experiments.append(
            {
                "label": f"SR={sr} mha+avgpool",
                "config": base_cfg_path,
                "overrides": {
                    **common,
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_IMPL": "mha",
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_REDUCTION_IMPL": "avgpool",
                },
                "tokens": str(tokens),
                "variant": "mha_avgpool",
            }
        )
        experiments.append(
            {
                "label": f"SR={sr} sdpa+avgpool",
                "config": base_cfg_path,
                "overrides": {
                    **common,
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_IMPL": "sdpa",
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_REDUCTION_IMPL": "avgpool",
                },
                "tokens": str(tokens),
                "variant": "sdpa_avgpool",
            }
        )
        experiments.append(
            {
                "label": f"SR={sr} sdpa+learned",
                "config": base_cfg_path,
                "overrides": {
                    **common,
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_IMPL": "sdpa",
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_REDUCTION_IMPL": "learned",
                },
                "tokens": str(tokens),
                "variant": "sdpa_learned",
            }
        )
        experiments.append(
            {
                "label": f"SR={sr} sdpa+learned+prune512",
                "config": base_cfg_path,
                "overrides": {
                    **common,
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_IMPL": "sdpa",
                    "MODEL.ENHANCEMENTS.CROSS_ATTN_REDUCTION_IMPL": "learned",
                    "MODEL.ENHANCEMENTS.EARLY_PROPOSAL_TOPK": 512,
                    "MODEL.ENHANCEMENTS.EARLY_PROPOSAL_LENGTH_WEIGHT": 0.05,
                },
                "tokens": str(tokens),
                "variant": "sdpa_learned_prune",
            }
        )
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Cross-attention benchmark matrix.")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--include-sr1", action="store_true", help="Include SR=1 rows")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Autocast dtype when AMP is enabled",
    )
    args = parser.parse_args()

    enhanced_cfg = str(REPO_ROOT / "configs" / "enhanced_plnet_crossattn.yaml")
    cfg_for_images = _load_cfg(enhanced_cfg)
    images, metas = _prepare_test_images(cfg_for_images, max_images=args.max_images)
    print(f"Loaded {len(images)} test images\n")

    experiments = _build_matrix(enhanced_cfg, include_sr1=args.include_sr1)
    results = []

    for exp in experiments:
        print("=" * 72)
        print(f"  {exp['label']}")
        print("=" * 72)

        cfg = _load_cfg(exp["config"], overrides=exp["overrides"])
        if cfg.MODEL.NAME == "PointLine":
            imgs, ms = _prepare_test_images(cfg, max_images=args.max_images)
        else:
            imgs, ms = images, metas

        res = benchmark_once(
            cfg,
            imgs,
            ms,
            warmup=args.warmup,
            measured=args.steps,
            use_amp=bool(args.amp),
            amp_dtype=args.amp_dtype,
        )
        res["label"] = exp["label"]
        res["variant"] = exp["variant"]
        res["tokens"] = exp["tokens"]
        res["sr"] = (
            exp["overrides"].get("MODEL.ENHANCEMENTS.CROSS_ATTN_SPATIAL_REDUCTION")
            if exp["overrides"]
            else None
        )
        res["attn_impl"] = (
            exp["overrides"].get("MODEL.ENHANCEMENTS.CROSS_ATTN_IMPL")
            if exp["overrides"]
            else "none"
        )
        res["reduction_impl"] = (
            exp["overrides"].get("MODEL.ENHANCEMENTS.CROSS_ATTN_REDUCTION_IMPL")
            if exp["overrides"]
            else "none"
        )
        res["proposal_topk"] = (
            exp["overrides"].get("MODEL.ENHANCEMENTS.EARLY_PROPOSAL_TOPK", 0)
            if exp["overrides"]
            else 0
        )
        results.append(res)

        print(f"  FPS:            {res['fps']}")
        print(f"  Avg latency:    {res['avg_latency_ms']:.2f} ms")
        print(f"  P95 latency:    {res['p95_latency_ms']:.2f} ms")
        print(f"  Backbone:       {res.get('time_backbone_ms', 0):.2f} ms")
        print(f"  Proposal:       {res.get('time_proposal_ms', 0):.2f} ms")
        print(f"  Matching:       {res.get('time_matching_ms', 0):.2f} ms")
        print(f"  Verification:   {res.get('time_verification_ms', 0):.2f} ms")
        print(f"  Peak VRAM:      {res['peak_mem_mb']:.1f} MB")
        print()

    print("\n" + "=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    hdr = (
        f"{'Variant':<33} {'SR':>3} {'Tokens':>7} {'Attn':>6} {'Reduce':>8} "
        f"{'Prune':>6} {'AMP':>5} {'FPS':>6} {'Lat':>7} {'P95':>7} {'VRAM':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['variant']:<33} {str(r['sr']):>3} {r['tokens']:>7} "
            f"{str(r['attn_impl']):>6} {str(r['reduction_impl']):>8} "
            f"{int(r['proposal_topk']):>6} {str(r['amp']):>5} {r['fps']:>6.1f} "
            f"{r['avg_latency_ms']:>7.1f} {r['p95_latency_ms']:>7.1f} {r['peak_mem_mb']:>7.1f}"
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

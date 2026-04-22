import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from hawp.base.utils.comm import to_device
from hawp.fsl.config import cfg as base_cfg
from hawp.fsl.dataset import build_train_dataset
from hawp.fsl.model.build import build_model
from hawp.fsl.solver import make_optimizer


class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        return sum(self.loss_weights[k] * loss_dict[k] for k in self.loss_weights.keys())


def _autocast(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _configure_backend(mode, tf32=False):
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if mode == "baseline":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _build_cfg(config_path, mode):
    cfg = base_cfg.clone()
    cfg.merge_from_file(config_path)
    cfg.defrost()

    if mode == "baseline":
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.GRAD_ACCUM_STEPS = 3
        cfg.MODEL.ENHANCEMENTS.AMP = False
        cfg.MODEL.ENHANCEMENTS.TORCH_COMPILE = False
        cfg.DATALOADER.PIN_MEMORY = False
        cfg.DATALOADER.PERSISTENT_WORKERS = False
        cfg.DATALOADER.PREFETCH_FACTOR = 2
        cfg.DATALOADER.PRECOMPUTE_HAFM = False
    elif mode == "optimized":
        cfg.SOLVER.IMS_PER_BATCH = 6
        cfg.SOLVER.GRAD_ACCUM_STEPS = 1
        cfg.MODEL.ENHANCEMENTS.AMP = True
        cfg.DATALOADER.PIN_MEMORY = True
        cfg.DATALOADER.PERSISTENT_WORKERS = True
        cfg.DATALOADER.PREFETCH_FACTOR = 2
        cfg.DATALOADER.PRECOMPUTE_HAFM = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cfg.freeze()
    return cfg


def _maybe_compile_model(model, cfg):
    compile_enabled = bool(getattr(cfg.MODEL.ENHANCEMENTS, "TORCH_COMPILE", False))
    compile_mode = str(getattr(cfg.MODEL.ENHANCEMENTS, "COMPILE_MODE", "default"))
    if not compile_enabled:
        return model, False
    if not hasattr(torch, "compile"):
        return model, False
    try:
        model = torch.compile(model, mode=compile_mode)
        return model, True
    except Exception:
        return model, False


def run_speed_benchmark(config_path, mode, steps, warmup, tf32):
    cfg = _build_cfg(config_path, mode)
    device = torch.device(cfg.MODEL.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    _configure_backend(mode, tf32=tf32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg).to(device)
    model, compiled = _maybe_compile_model(model, cfg)
    model.train()

    optimizer = make_optimizer(cfg, model)
    loss_reducer = LossReducer(cfg)
    train_loader = build_train_dataset(cfg)
    loader_iter = iter(train_loader)

    grad_accum = max(int(getattr(cfg.SOLVER, "GRAD_ACCUM_STEPS", 1)), 1)
    use_amp = bool(getattr(cfg.MODEL.ENHANCEMENTS, "AMP", False) and device.type == "cuda")
    scaler = _grad_scaler(use_amp)
    optimizer.zero_grad(set_to_none=True)

    measured_data_ms = []
    measured_step_ms = []
    measured_fwd_ms = []
    measured_bwd_ms = []
    measured_opt_ms = []

    total_steps = warmup + steps
    for step_idx in range(total_steps):
        t_data0 = time.perf_counter()
        batch, loader_iter = _next_batch(loader_iter, train_loader)
        data_ms = (time.perf_counter() - t_data0) * 1000.0

        images, annotations = batch
        images = images.to(device, non_blocking=True)
        annotations = to_device(annotations, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with _autocast(use_amp):
            loss_dict, _ = model(images, annotations)
            total_loss = loss_reducer(loss_dict)
            backward_loss = total_loss / grad_accum
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        if scaler.is_enabled():
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        if ((step_idx + 1) % grad_accum) == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()

        if step_idx >= warmup:
            measured_data_ms.append(data_ms)
            measured_fwd_ms.append((t1 - t0) * 1000.0)
            measured_bwd_ms.append((t2 - t1) * 1000.0)
            measured_opt_ms.append((t3 - t2) * 1000.0)
            measured_step_ms.append((t3 - t0) * 1000.0)

    step_ms_avg = mean(measured_step_ms)
    batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
    samples_per_sec = (batch_size * 1000.0) / max(step_ms_avg, 1e-6)

    return {
        "mode": mode,
        "compiled": compiled,
        "device": str(device),
        "amp": use_amp,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "dataloader_precompute_hafm": bool(getattr(cfg.DATALOADER, "PRECOMPUTE_HAFM", False)),
        "data_ms_avg": mean(measured_data_ms),
        "forward_ms_avg": mean(measured_fwd_ms),
        "backward_ms_avg": mean(measured_bwd_ms),
        "optimizer_ms_avg": mean(measured_opt_ms),
        "step_ms_avg": step_ms_avg,
        "steps_per_sec": 1000.0 / max(step_ms_avg, 1e-6),
        "samples_per_sec": samples_per_sec,
        "peak_mem_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PLNet training speed (baseline vs optimized)."
    )
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["baseline", "optimized", "both"],
        help="Which benchmark mode to run",
    )
    parser.add_argument("--steps", type=int, default=30, help="Measured steps per mode")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps per mode")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32")
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to write benchmark JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    modes = ["baseline", "optimized"] if args.mode == "both" else [args.mode]
    results = []

    for mode in modes:
        result = run_speed_benchmark(
            config_path=args.config,
            mode=mode,
            steps=args.steps,
            warmup=args.warmup,
            tf32=args.tf32,
        )
        results.append(result)
        print(f"\n[{mode}]")
        print(json.dumps(result, indent=2))

    if len(results) == 2:
        baseline, optimized = results
        speedup = baseline["step_ms_avg"] / max(optimized["step_ms_avg"], 1e-6)
        throughput_gain = optimized["samples_per_sec"] / max(
            baseline["samples_per_sec"], 1e-6
        )
        comparison = {
            "step_time_speedup_x": speedup,
            "throughput_gain_x": throughput_gain,
        }
        print("\n[comparison]")
        print(json.dumps(comparison, indent=2))
    else:
        comparison = None

    if args.save_json:
        payload = {"results": results, "comparison": comparison}
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved benchmark JSON to: {output_path}")


if __name__ == "__main__":
    main()


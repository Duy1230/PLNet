import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from torch.profiler import ProfilerActivity, profile, record_function


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


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_mem_mb(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _autocast(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_train_step(model, optimizer, loss_reducer, batch, device, use_amp, scaler):
    images, annotations = batch
    images = images.to(device, non_blocking=True)
    annotations = to_device(annotations, device)

    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync_if_cuda(device)

    t0 = time.perf_counter()
    with _autocast(use_amp):
        loss_dict, _ = model(images, annotations)
        total_loss = loss_reducer(loss_dict)
    _sync_if_cuda(device)
    t1 = time.perf_counter()

    if scaler is not None and scaler.is_enabled():
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
    _sync_if_cuda(device)
    t2 = time.perf_counter()

    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    _sync_if_cuda(device)
    t3 = time.perf_counter()

    return {
        "forward_ms": (t1 - t0) * 1000.0,
        "backward_ms": (t2 - t1) * 1000.0,
        "optimizer_ms": (t3 - t2) * 1000.0,
        "peak_mem_mb": _cuda_mem_mb(device),
        "loss": float(total_loss.detach().item()),
    }


def run_operator_profile(model, optimizer, loss_reducer, batch, device, use_amp, scaler, topk):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    images, annotations = batch
    images = images.to(device, non_blocking=True)
    annotations = to_device(annotations, device)

    optimizer.zero_grad(set_to_none=True)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("forward_pass"):
            with _autocast(use_amp):
                loss_dict, _ = model(images, annotations)
                total_loss = loss_reducer(loss_dict)
        with record_function("backward_pass"):
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
        with record_function("optimizer_step"):
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        prof.step()

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print("\n==== Top Operators by Self Time ====")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=topk))
    print("\n==== Top Operators by Memory ====")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=topk))


def next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def parse_args():
    parser = argparse.ArgumentParser(description="Profile PLNet training bottlenecks")
    parser.add_argument("config", type=str, help="Path to yaml config")
    parser.add_argument("--steps", type=int, default=6, help="Measured steps")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Enable AMP while profiling")
    parser.add_argument(
        "--profile-ops",
        action="store_true",
        help="Run torch.profiler and print top expensive ops",
    )
    parser.add_argument("--topk", type=int, default=25, help="Rows in profiler tables")
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save summary JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = base_cfg.clone()
    cfg.merge_from_file(args.config)

    if args.device is not None:
        cfg.MODEL.DEVICE = args.device

    if cfg.MODEL.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    device = torch.device(cfg.MODEL.DEVICE)
    model = build_model(cfg).to(device)
    model.train()

    optimizer = make_optimizer(cfg, model)
    loss_reducer = LossReducer(cfg)

    config_amp = bool(getattr(cfg.MODEL.ENHANCEMENTS, "AMP", False))
    use_amp = bool((args.amp or config_amp) and device.type == "cuda")
    scaler = _grad_scaler(use_amp)

    train_loader = build_train_dataset(cfg)
    loader_iter = iter(train_loader)

    data_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    peak_mems = []
    losses = []

    print(
        f"Profiling with device={device}, amp={use_amp}, warmup={args.warmup}, steps={args.steps}"
    )
    print(f"Config: {args.config}")

    end_t = time.perf_counter()
    total_steps = args.warmup + args.steps
    for step_idx in range(total_steps):
        batch, loader_iter = next_batch(loader_iter, train_loader)
        data_ms = (time.perf_counter() - end_t) * 1000.0
        metrics = run_train_step(
            model=model,
            optimizer=optimizer,
            loss_reducer=loss_reducer,
            batch=batch,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        end_t = time.perf_counter()

        if step_idx >= args.warmup:
            data_times.append(data_ms)
            forward_times.append(metrics["forward_ms"])
            backward_times.append(metrics["backward_ms"])
            optimizer_times.append(metrics["optimizer_ms"])
            peak_mems.append(metrics["peak_mem_mb"])
            losses.append(metrics["loss"])

        print(
            f"step={step_idx+1}/{total_steps} "
            f"data={data_ms:.1f}ms forward={metrics['forward_ms']:.1f}ms "
            f"backward={metrics['backward_ms']:.1f}ms opt={metrics['optimizer_ms']:.1f}ms "
            f"mem={metrics['peak_mem_mb']:.1f}MB loss={metrics['loss']:.4f}"
        )

    summary = {
        "device": str(device),
        "amp": use_amp,
        "data_ms_avg": mean(data_times) if data_times else 0.0,
        "forward_ms_avg": mean(forward_times) if forward_times else 0.0,
        "backward_ms_avg": mean(backward_times) if backward_times else 0.0,
        "optimizer_ms_avg": mean(optimizer_times) if optimizer_times else 0.0,
        "step_ms_avg": (
            (mean(data_times) + mean(forward_times) + mean(backward_times) + mean(optimizer_times))
            if data_times
            else 0.0
        ),
        "peak_mem_mb_max": max(peak_mems) if peak_mems else 0.0,
        "loss_avg": mean(losses) if losses else 0.0,
    }

    print("\n==== Profile Summary (averaged over measured steps) ====")
    print(json.dumps(summary, indent=2))

    if args.profile_ops:
        batch, loader_iter = next_batch(loader_iter, train_loader)
        run_operator_profile(
            model=model,
            optimizer=optimizer,
            loss_reducer=loss_reducer,
            batch=batch,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            topk=args.topk,
        )

    if args.save_json:
        output_path = Path(args.save_json)
        os.makedirs(output_path.parent, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary JSON to: {output_path}")


if __name__ == "__main__":
    main()

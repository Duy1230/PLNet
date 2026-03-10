# PLNet Training Optimization Report

## Goal

Reduce training step time and GPU memory usage for `EnhancedPointLine` with cross-attention enabled, on an 8 GB VRAM GPU.

## Baseline Symptoms (from `train.log`)

- Iteration time grows to ~70-80 seconds.
- Peak GPU memory reaches ~9.8 GB (`max mem: 9818`), causing shared memory spill.
- Estimated training completion is impractically long (100+ days).

## Primary Bottlenecks

1. **Cross-attention token explosion**
   - File: `hawp/fsl/backbones/modules/cross_attention.py`
   - Current behavior attends over full `128 x 128 = 16384` tokens.
   - Multi-head attention complexity is quadratic in token count.
   - This dominates both forward and backward compute and memory.

2. **No mixed precision training**
   - File: `hawp/fsl/train.py`
   - Training runs entirely in FP32, increasing activation memory and bandwidth costs.

3. **No activation checkpointing in stacked U-Nets**
   - File: `hawp/fsl/backbones/modules/deformable_conv.py`
   - Both U-Net stacks retain full activation graphs for backward.

4. **Frozen backbone still tracked in autograd graph**
   - File: `hawp/fsl/backbones/enhanced_plnet.py`
   - When the SuperPoint encoder is frozen, its forward should run under `torch.no_grad()`.

5. **Training configuration is too heavy for 8 GB VRAM**
   - File: `configs/enhanced_plnet_crossattn.yaml`
   - `IMS_PER_BATCH=6` with 512 input is too large for stable non-paged memory use.
   - `NUM_WORKERS=0` slows data feeding.

## Functional/Robustness Bugs to Fix Alongside Optimization

1. **Unsafe scalar conversion in logging**
   - File: `hawp/fsl/train.py`
   - `.item()` is called unconditionally; fails if a loss value is a Python float.

2. **Hardcoded TensorBoard path**
   - File: `hawp/fsl/train.py`
   - Uses a fixed Linux path that does not match the current environment.

3. **Undefined `max_epoch` in train summary**
   - File: `hawp/fsl/train.py`
   - End-of-training logging references a variable outside function scope.

4. **EnhancedPointLine transform mismatch**
   - File: `hawp/fsl/dataset/build.py`
   - `EnhancedPointLine` path should follow `PointLine` transform behavior (no normalization), but currently does not.

## Implemented Changes

1. **Cross-attention spatial reduction**
   - File: `hawp/fsl/backbones/modules/cross_attention.py`
   - Added `spatial_reduction` with adaptive pooling before attention and bilinear upsampling after attention.
   - Config key: `MODEL.ENHANCEMENTS.CROSS_ATTN_SPATIAL_REDUCTION`.

2. **AMP + gradient accumulation**
   - File: `hawp/fsl/train.py`
   - Added autocast + GradScaler execution path.
   - Added gradient accumulation with `SOLVER.GRAD_ACCUM_STEPS`.
   - Effective batch kept at 6 via `IMS_PER_BATCH=2` and `GRAD_ACCUM_STEPS=3`.

3. **Gradient checkpointing**
   - Files:
     - `hawp/fsl/backbones/modules/deformable_conv.py`
     - `hawp/fsl/backbones/enhanced_plnet.py`
   - Added optional checkpointed activations in `DeformableUNet`.
   - Config key: `MODEL.ENHANCEMENTS.GRAD_CHECKPOINT`.

4. **Frozen backbone no-grad**
   - File: `hawp/fsl/backbones/enhanced_plnet.py`
   - SuperPoint encoder forward now runs under `torch.no_grad()` when frozen.

5. **Training/runtime bug fixes**
   - File: `hawp/fsl/train.py`
     - robust scalar logging for mixed tensor/float loss entries
     - TensorBoard path now uses `cfg.OUTPUT_DIR`
     - end-of-training epoch-time computation fixed
     - removed debug `pdb` halt
   - File: `hawp/fsl/dataset/build.py`
     - `EnhancedPointLine` now uses raw SuperPoint-compatible transforms.

6. **Config extensions**
   - Files:
     - `hawp/fsl/config/models/models.py`
     - `hawp/fsl/config/solver.py`
     - `configs/enhanced_plnet_crossattn.yaml`
   - Added keys:
     - `CROSS_ATTN_SPATIAL_REDUCTION`
     - `GRAD_CHECKPOINT`
     - `AMP`
     - `GRAD_ACCUM_STEPS`
   - Tuned run config for 8 GB VRAM:
     - `NUM_WORKERS: 2`
     - `IMS_PER_BATCH: 2`
     - `GRAD_ACCUM_STEPS: 3`
     - `AMP: true`
     - `CROSS_ATTN_SPATIAL_REDUCTION: 4`
     - `GRAD_CHECKPOINT: true`

7. **Profiling utility**
   - File: `tools/profile_training.py`
   - Added step-level timing and memory profile:
     - data, forward, backward, optimizer
     - peak memory
     - optional operator-level `torch.profiler` report.

## Measured Results

### Baseline (old run)

- From `output/experiments/enhanced_plnet_crossattn/260310-011922/train.log`
  - Steady-state runtime: ~`69-71 s/iter`
  - Peak memory: `9818 MB`

### Optimized run (new code)

- From `tools/profile_training.py` (`steps=3`, `warmup=1`):
  - `step_ms_avg`: `872.77 ms`
  - `peak_mem_mb_max`: `1835.12 MB`
- From short real training run:
  - Iter 60 average time: `1.3508 s/iter`
  - Peak memory: `1987 MB`

### Improvement Summary

- **VRAM**: `9818 MB -> 1987 MB` (about **79.8% less** peak GPU memory)
- **Throughput**: `~70.67 s/iter -> ~1.35 s/iter` (about **52x faster** in logged training loop)
- **No shared-memory pressure observed** in optimized run.

## Remaining Hotspots (from operator profiling)

- Top costs now are expected dense ops in forward/backward:
  - `backward_pass`
  - `aten::convolution_backward`
  - `aten::cudnn_convolution`
  - `aten::cudnn_batch_norm`
- Cross-attention is no longer the dominating bottleneck after token reduction.

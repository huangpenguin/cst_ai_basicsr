# CI SwinIR debug job 1h timeout

## Symptom

`train_cylinderblock_swinir_debug` logged model creation and `Start training from epoch: 0, iter: 0`, then no iteration logs before GitLab killed the job with SIGTERM at the 1h timeout.

## Root cause

1. YAML `num_gpu: auto` resolved to 4 on the runner while `CI_DEBUG_NPROC=1` (non-distributed). BasicSR wrapped the network in `DataParallel` with `batch_size_per_gpu=1`.
2. `CUDA_LAUNCH_BLOCKING=1` forced synchronous CUDA kernels. SwinIR attention blocks became impractically slow on the first forward/backward pass.

## Fix

- Force `num_gpu=1` in `scripts/ci_swinir_cylinderblock_debug.py` `--force_yml` overrides.
- Remove default `CUDA_LAUNCH_BLOCKING=1` from the debug CI job; opt in via `CI_DEBUG_CUDA_SYNC=1` when debugging CUDA errors.

## Prevention

Short CI smoke jobs that use single-process training must pin `num_gpu=1` when the YAML uses `num_gpu: auto`. Do not enable `CUDA_LAUNCH_BLOCKING` unless actively debugging async CUDA failures.

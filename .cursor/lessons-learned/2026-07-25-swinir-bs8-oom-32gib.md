# SwinIR grayDN: batch_size_per_gpu=8 OOMs on 32GiB GPUs

## Symptom

Formal 4-GPU CI train (`train_paired_tiff_swinir`) failed on the first `optimize_parameters` with `torch.OutOfMemoryError` (~31.65 GiB used / 31.73 GiB, need +90 MiB) across ranks 0–3.

## Cause

`batch_size_per_gpu: 8` with SwinIR-M (`depths` 6×6, `embed_dim` 180, `gt_size` 128) saturates a 32GiB GPU even alone. Concurrent local eval on the same host can make it worse, but bs=8 is already over budget.

## Fix

Use `batch_size_per_gpu: 4` (same as the working phantom YAML). Effective global batch = 4 × `num_gpu`.

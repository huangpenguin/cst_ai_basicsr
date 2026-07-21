# Aligned Paired-TIFF SwinIR Denoising

This workflow fine-tunes grayscale SwinIR on already aligned LQ/GT uint16 TIFF
pairs. It performs no rotation, resize, crop, or other geometric preprocessing.

## Storage layout

The two NFS mounts have different purposes:

| GPU host | CI container | Purpose |
|---|---|---|
| `/mnt/data/cst_ai` | `/data/cst_ai` | Shared, read-only source data |
| `/mnt/home/huang/cst_ai` | `/home/huang/cst_ai` | Metadata, models, logs, and personal outputs |

Put a new dataset in:

```text
/mnt/data/cst_ai/datasets/<dataset_name>/
  LQ/*.tif
  GT/*.tif
```

Every LQ file must have a same-name GT file. Both must be 2D uint16 TIFFs with
the same dimensions and pixel-level alignment.

## 1. Validate and split

The preparation step writes only small metadata files. It does not copy or
modify images.

```bash
cd /home/huang/code/cst_ai/BasicSR

uv run python scripts/prepare_paired_tiff_dataset.py \
  --source-root /mnt/data/cst_ai/datasets/<dataset_name> \
  --output-root /mnt/home/huang/cst_ai/datasets/<dataset_name> \
  --split-mode tail \
  --val-fraction 0.1
```

Use `--dry-run` to validate without writing metadata. Use `--split-mode random
--seed 0` only when adjacent filenames are not neighboring CT slices.

Generated files:

```text
/mnt/home/huang/cst_ai/datasets/<dataset_name>/
  train.txt
  val.txt
  manifest.csv
  normalization.json
  dataset.json
```

## 2. Train directly

The training config is:

`options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml`

For GitLab CI, set `PAIRED_DATASET_NAME` in `.gitlab/ci/train.yml` (current
default: `aligned_drive`), then run:

1. `train_paired_tiff_swinir_debug`
2. `train_paired_tiff_swinir`

CI reads images directly from `/data/cst_ai/datasets/<dataset_name>` and split
metadata from `/home/huang/cst_ai/datasets/<dataset_name>`. It never invokes an
image preprocessor.

Local single-GPU example:

```bash
DATASET_NAME=<dataset_name>
uv run python scripts/download_swinir_release.py --only noise25
uv run python basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml \
  --force_yml \
  "num_gpu=1" \
  "datasets:train:dataroot_lq=/mnt/data/cst_ai/datasets/${DATASET_NAME}/LQ" \
  "datasets:train:dataroot_gt=/mnt/data/cst_ai/datasets/${DATASET_NAME}/GT" \
  "datasets:train:meta_info_file=/mnt/home/huang/cst_ai/datasets/${DATASET_NAME}/train.txt" \
  "datasets:train:normalization=/mnt/home/huang/cst_ai/datasets/${DATASET_NAME}/normalization.json" \
  "datasets:val:dataroot_lq=/mnt/data/cst_ai/datasets/${DATASET_NAME}/LQ" \
  "datasets:val:dataroot_gt=/mnt/data/cst_ai/datasets/${DATASET_NAME}/GT" \
  "datasets:val:meta_info_file=/mnt/home/huang/cst_ai/datasets/${DATASET_NAME}/val.txt" \
  "datasets:val:normalization=/mnt/home/huang/cst_ai/datasets/${DATASET_NAME}/normalization.json"
```

Persistent CI output:

`/mnt/home/huang/cst_ai/ci_outputs/<pipeline_id>/`

## 3. Evaluate a trained model

The generic TIFF inference command writes preview panels and, when GT is
provided, `metrics.csv` with LQ/output SNR, SNR gain, PSNR, and SSIM:

```bash
uv run python scripts/infer_swinir_graydn_tiff.py \
  --model /mnt/home/huang/cst_ai/ci_outputs/<pipeline_id>/experiments/<experiment>/models/net_g_latest.pth \
  --inputs /mnt/data/cst_ai/datasets/<dataset_name>/LQ \
  --gt-dir /mnt/data/cst_ai/datasets/<dataset_name>/GT \
  --normalization /mnt/home/huang/cst_ai/datasets/<dataset_name>/normalization.json \
  --output-dir /mnt/home/huang/cst_ai/outputs/<dataset_name>
```

## SNR baseline

The default remains SwinIR-M with the official grayDN noise25 initialization.
The default pixel loss is MSE because it directly optimizes error power used by
PSNR/reference SNR. Controlled loss comparisons require only one override:

```bash
# Existing baseline
"train:pixel_opt:type=L1Loss"

# Robust alternative
"train:pixel_opt:type=CharbonnierLoss"
```

Validation reports reference SNR, PSNR, and SSIM. Keep a fixed physical
normalization range; never use per-image percentile stretching for quantitative
comparison.

Do not change architecture for the first aligned-data baseline. Add a small
DnCNN or residual U-Net comparison only after the SwinIR baseline is stable.
Consider Restormer/NAFNet or 2.5D input only if blind-test SNR plateaus,
structures become visibly blurred, or adjacent-slice information is required.

## Archived previous experiment

The old geometry-dependent Cylinderblock experiment is retained at:

`/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/`

It contains the final model, exact config, manifest, normalization, logs,
metrics, report, and `SHA256SUMS`.

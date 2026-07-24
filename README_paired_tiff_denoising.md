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

Edit paths / `name` / `num_gpu` / hyperparams in that YAML (Chinese comments
explain each field). CI only points at it via `TRAIN_OPT` in
`.gitlab/ci/train.yml`, then run:

1. `train_paired_tiff_swinir_debug`
2. `train_paired_tiff_swinir`

CI loads dataset paths from the YAML (container paths under `/data/...` and
`/home/...`). It never invokes an image preprocessor.

Local example (paths already in YAML; optional single-GPU override):

```bash
uv run python scripts/download_swinir_release.py --only noise25
uv run python basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml \
  --force_yml "num_gpu=1"
```

Persistent CI output:

`/mnt/home/huang/cst_ai/ci_outputs/<pipeline_id>/`

During a CI train job, TensorBoard is started automatically on port `6006`
(logdir = that run's `tb_logger/`) and stopped in `after_script`. From a Mac:

```bash
ssh -L 6006:localhost:6006 <user>@<gpu-host>
# open http://localhost:6006
```

Requires the GPU runner `network_mode = "host"` (or a published host port).
After the job ends you can still re-serve events from NFS:

```bash
bash scripts/serve_tensorboard.sh /mnt/home/huang/cst_ai/ci_outputs/latest/tb_logger
```

## 3. Evaluate a trained model

The generic TIFF inference command writes preview panels and, when GT is
provided, `metrics.csv` with LQ/output SNR, SNR gain, PSNR, and SSIM:

```bash
uv run python scripts/infer_swinir_graydn_tiff.py \
  --model /mnt/home/huang/cst_ai/experiments/<dataset_name>/<experiment>/models/net_g_latest.pth \
  --baseline-model experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth \
  --lq-dir /mnt/data/cst_ai/datasets/<dataset_name>/LQ \
  --gt-dir /mnt/data/cst_ai/datasets/<dataset_name>/GT \
  --meta-info /mnt/home/huang/cst_ai/datasets/<dataset_name>/val.txt \
  --pick-indices 0 202 405 608 \
  --normalization /mnt/home/huang/cst_ai/datasets/<dataset_name>/normalization.json \
  --output-dir /mnt/home/huang/cst_ai/outputs/<dataset_name>_preft_vs_ft
```

With `--baseline-model`, each `*_compare.png` is **LQ | PreFT | FT | GT**.

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

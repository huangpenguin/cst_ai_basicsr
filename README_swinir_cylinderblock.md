# Cylinderblock SwinIR Paired Denoising

This setup fine-tunes the pretrained SwinIR grayscale denoising model on real
low-dose/high-dose Cylinderblock X-ray TIFF pairs. It does not use the official
SwinIR synthetic Gaussian-noise data flow.

## Data Selection

Raw data is expected under `/mnt/data/cst_ai/data_0607` by default.

The preprocessing script uses two Cylinderblock scans:

- `260520_2-Cylinderblock_data/260520_cylinderblock image data`
- `260527_4-Cylinderblock_data/260527_cylinderblock image data`

For each scan, `0-300kV_100uA/XY` is treated as GT. Low-current folders are
paired with GT by the same `data_XXXX.tif` slice index. Only slices
`data_0130.tif` through `data_0750.tif` are used.

The 10uA and 20uA folders are excluded because they are too noisy for this
round. The selected noisy conditions are:

- 260520: `30uA`, `50uA`, `60uA`, `70uA`, `80uA`, `90uA` at 110kV
- 260527: `30uA`, `40uA`, `50uA`, `60uA` at 150kV

This produces 6210 paired images before train/val splitting.

## Preprocess

Generate the paired dataset with:

```bash
python scripts/preprocess_cylinderblock_swinir.py \
  --data-root /mnt/data/cst_ai/data_0607 \
  --output-root /home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired \
  --overwrite
```

For a quick validation without writing files:

```bash
python scripts/preprocess_cylinderblock_swinir.py --dry-run
```

The output structure is:

```text
swinir_cylinderblock_paired/
  train/LQ/*.tif
  train/GT/*.tif
  val/LQ/*.tif
  val/GT/*.tif
  manifest.csv
  normalization.json
```

The raw LQ TIFFs are `1280x1280`; GT TIFFs are `1530x1530`. Preprocessing center
crops GT to `1280x1280`, then writes matching basenames in `LQ` and `GT` so
BasicSR folder pairing is unambiguous.

LQ files are copied by default so the derived dataset is self-contained inside
the CI `/home/${GITLAB_USER_LOGIN}` mount. Repeated GT files are hardlinked by
default after the first cropped copy is written.

By default, validation uses the tail slice block rather than random slice
sampling. This reduces leakage from adjacent slices between train and val.

## Normalization

The default normalization is a fixed global uint16 range:

```json
{
  "method": "uint16_full",
  "raw_min": 0.0,
  "raw_max": 65535.0,
  "output_range": [0.0, 1.0],
  "apply_to_lq": true,
  "apply_to_gt": true,
  "source_dtype": "uint16"
}
```

This keeps the broad physical density range and does not percentile-clip
saturated pixels, streaks, artifacts, or noise. The paired dataset reader maps
raw uint16 values to float tensors with this range at training time.

If you later want a tighter fixed range after inspecting statistics, run:

```bash
python scripts/preprocess_cylinderblock_swinir.py \
  --stats-only \
  --norm-mode global_dataset
```

Only switch to `global_dataset` if you also regenerate `normalization.json` and
use the same normalization for inference.

## Training

Download the SwinIR grayscale denoising pretrained model and run BasicSR
training:

```bash
uv run python scripts/download_swinir_release.py --only 004_grayDN
uv run python basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_cylinderblock.yml \
  --force_yml \
  datasets:train:dataroot_lq=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/train/LQ \
  datasets:train:dataroot_gt=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/train/GT \
  datasets:train:normalization=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/normalization.json \
  datasets:val:dataroot_lq=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/val/LQ \
  datasets:val:dataroot_gt=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/val/GT \
  datasets:val:normalization=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/normalization.json
```

For distributed training:

```bash
uv run python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=4331 \
  basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_cylinderblock.yml \
  --launcher pytorch \
  --auto_resume \
  --force_yml \
  datasets:train:dataroot_lq=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/train/LQ \
  datasets:train:dataroot_gt=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/train/GT \
  datasets:train:normalization=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/normalization.json \
  datasets:val:dataroot_lq=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/val/LQ \
  datasets:val:dataroot_gt=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/val/GT \
  datasets:val:normalization=/home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired/normalization.json
```

The GitLab manual job `train_cylinderblock_swinir_graydn` uses the same yml and
defaults to `/home/${GITLAB_USER_LOGIN}/cst_ai/datasets/swinir_cylinderblock_paired`.

For debugging CUDA / shape issues without a long DDP run, use the manual CI job
`train_cylinderblock_swinir_debug` (see `scripts/ci_swinir_cylinderblock_debug.py`).
It preprocesses only a few slices, runs single-GPU training for ~30 iterations
with `CUDA_LAUNCH_BLOCKING=1`, and uploads logs under
`results/cylinderblock_swinir_debug/`.

## Implementation Notes

Use `PairedTiffImageDataset` for this dataset. The standard `PairedImageDataset`
uses an 8-bit image path and `/255` normalization, which is not safe for these
uint16 TIFF files.

The training config uses:

- `model_type: SwinIRModel`
- `scale: 1`
- `network_g.in_chans: 1`
- `network_g.upsampler: ''`
- `path.pretrain_network_g: experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth`

The pretrained model was trained for synthetic Gaussian denoising. These
Cylinderblock pairs contain real low-dose noise, artifacts, and scan-condition
differences, so start with a small learning rate and monitor validation slices
visually as well as by PSNR/SSIM.

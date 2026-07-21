# Data mount decisions

## NFS layout (2026-06)

Autofs on main_gpu:

| Host | Container | Mode | Purpose |
|------|-----------|------|---------|
| `/mnt/data/<PROJECT_NAME>/` | `/data/<PROJECT_NAME>/` | ro | Raw / shared team datasets |
| `/mnt/home/<user>/<PROJECT_NAME>/` | `/home/<user>/<PROJECT_NAME>/` | rw | Personal outputs / derived data |

Runner mounts (all users, no per-user config):

```toml
volumes = ["/cache", "/mnt/data:/data:ro", "/mnt/home:/home:rw"]
```

- `/cache` on the runner is **not** used for datasets or training outputs (see `scripts/ci_storage.py`).
- GitLab `cache:` in CI only reuses `.venv/`, `.cache/uv/`, and pretrained weights between jobs.

Env vars: `DATA_DIR`, `PROJECT_NAME`, `WORK_DIR`, `CST_STORAGE_USER`, optional `DATASET_DIR`, `OUTPUT_DIR`.

CI training outputs (container `/home/<CST_STORAGE_USER>/cst_ai/`, host `/mnt/home/<CST_STORAGE_USER>/cst_ai/`).
Default `CST_STORAGE_USER=huang` in `.gitlab/ci/train.yml` (not `GITLAB_USER_LOGIN`).

## Unified GPU image (2026-07)

Dev Container and GitLab CI use the **same** image tag `basicsr-mlops-gpu:2.6.0-cu124` from `Dockerfile` (base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`).

- Image layer: OS tools + `uv` only (granularity 1).
- Python deps: `scripts/uv-bootstrap.sh` → `.venv` (same as devcontainer `postCreateCommand`).
- gpu01 one-time: `bash scripts/build_mlops_image.sh`; CI `pull_policy: if-not-present`.

Devcontainer / docker-dev host exports: `DATA_MOUNT_SOURCE=/mnt/data`, `HOME_MOUNT_SOURCE=/mnt/home`. Dev Container + compose: see `.devcontainer/README.md`.

## Aligned paired-TIFF workflow (2026-07)

- Source data is already pixel-aligned and lives at `/mnt/data/cst_ai/datasets/<name>/{LQ,GT}`.
- Pair files must have the same filename, dimensions, and uint16 grayscale dtype.
- `scripts/prepare_paired_tiff_dataset.py` validates pairs and writes only split/normalization metadata under `/mnt/home/<user>/cst_ai/datasets/<name>`.
- No rotation, resize, crop, resampling, or image copy is allowed in the preparation step.
- CI reads source images from `/data/cst_ai/datasets/<name>` and metadata from `/home/<user>/cst_ai/datasets/<name>`.
- Set `PAIRED_DATASET_NAME` to switch datasets; run `train_paired_tiff_swinir_debug` before `train_paired_tiff_swinir`.
- Current dataset: `aligned_drive` (Z-filtered via `scripts/folder_roles.yaml` `slice_ranges`: 5-1 275–623, 5-2 185–740, 5-3/5-4 185–745).
- Training config: `options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml`.
- Monitoring: CI train jobs auto-start/stop TensorBoard on port `6006` via `scripts/ci_tensorboard.sh` (logdir = run `tb_logger/`); Mac views with `ssh -L 6006:localhost:6006`. Runner needs `network_mode = "host"` (see `docs/use-cases/gpu-runner.zh-CN.md`). Manual re-serve: `scripts/serve_tensorboard.sh …/tb_logger`. wandb optional (`WANDB_API_KEY`).

BasicSR: `PROJECT_NAME=cst_ai`, raw backup at `/mnt/data/cst_ai/`.

Helper: `scripts/data_paths.py`.

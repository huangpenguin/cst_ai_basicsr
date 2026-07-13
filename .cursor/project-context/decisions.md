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

## Cylinderblock preprocess geometry (2026-06)

- GT: `0-300kV_100uA/XY`, 1530×1530, no transform.
- LQ: rotate CW per batch (`260520` 193°, `260527` 91°), `nearest` edge fill, bilinear resize to 1530.
- Script: `scripts/preprocess_cylinderblock_swinir.py`; previews: `scripts/preview_cylinderblock_preprocess.py`.
- CI does **not** preprocess; dataset must exist at `/home/${CST_STORAGE_USER}/cst_ai/datasets/swinir_cylinderblock_paired` before `train_cylinderblock_swinir_graydn`.
- Short CI repro job: `train_cylinderblock_swinir_debug` (tiny subset, ~minutes).
- Weights & Biases: optional via GitLab `WANDB_API_KEY`; syncs TensorBoard when `logger.wandb.project` is set.

BasicSR: `PROJECT_NAME=cst_ai`, raw backup at `/mnt/data/cst_ai/`.

Helper: `scripts/data_paths.py`.

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

Env vars: `DATA_DIR`, `PROJECT_NAME`, `WORK_DIR`, optional `DATASET_DIR`, `OUTPUT_DIR`.

Devcontainer host exports: `DATA_MOUNT_SOURCE=/mnt/data`, `HOME_MOUNT_SOURCE=/mnt/home`.

BasicSR: `PROJECT_NAME=cst_ai`, raw backup at `/mnt/data/cst_ai/`.

Helper: `scripts/data_paths.py`.

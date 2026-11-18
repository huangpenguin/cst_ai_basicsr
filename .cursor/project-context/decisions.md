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

- GitLab `cache:` for quality jobs only reuses `.venv/` and `.cache/uv/` on GitLab.com shared runners.
- Training outputs live on the host at `/mnt/home/<user>/cst_ai/`, started directly from `options/train/` YAML. CI does not launch training (removed 2026-11-09).

Env vars: `DATA_DIR`, `PROJECT_NAME`, `WORK_DIR`, `CST_STORAGE_USER`, optional `DATASET_DIR`, `OUTPUT_DIR`.

Training writes to host `/mnt/home/<CST_STORAGE_USER>/cst_ai/` (container path `/home/<CST_STORAGE_USER>/cst_ai/` when using the Dev Container). `CST_STORAGE_USER` defaults to `huang`, not `GITLAB_USER_LOGIN`.

## Repo location / multi-host (2026-09)

- Single working copy on NFS: `/mnt/home/huang/code/cst_ai/BasicSR` (gpu01 `/home/huang/code/cst_ai/BasicSR` is a symlink to it). gpu01 runs training directly; a second GPU host runs the Gradio demo from the same path.
- venv is per-host on local disk via `UV_PROJECT_ENVIRONMENT=$HOME/.local/share/venvs/basicsr` (zsh `chpwd` hook scoped to the repo); never a `.venv` on NFS.
- Remotes: `gitlab` (primary), `github` (private mirror `huangpenguin/cst_ai_basicsr`), `all` (dual push), `upstream` (XPixelGroup, fetch only). Git write ops on one host at a time.
- Details: `docs/infrastructure.md` §2.4.

## Unified GPU image (2026-07)

Dev Container uses image tag `basicsr-mlops-gpu:2.6.0-cu124` from `Dockerfile` (base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`). GitLab CI only runs Ruff/Pyright on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`.

- Image layer: OS tools + `uv` only (granularity 1).
- Python deps: `scripts/env/uv-bootstrap.sh` → `.venv` (same as devcontainer `postCreateCommand`).
- gpu01 one-time: `bash scripts/env/build_mlops_image.sh`; CI `pull_policy: if-not-present`.

Devcontainer / docker-dev host exports: `DATA_MOUNT_SOURCE=/mnt/data`, `HOME_MOUNT_SOURCE=/mnt/home`. Dev Container + compose: see `.devcontainer/README.md`.

## Aligned paired-TIFF workflow (2026-07)

- Source data is already pixel-aligned and lives at `/mnt/data/cst_ai/datasets/<name>/{LQ,GT}`.
- Pair files must have the same filename, dimensions, and uint16 grayscale dtype.
- `scripts/data/prepare_paired_tiff_dataset.py` validates pairs and writes only split/normalization metadata under `/mnt/home/<user>/cst_ai/datasets/<name>`.
- No rotation, resize, crop, resampling, or image copy is allowed in the preparation step.
- CI reads source images from `/data/cst_ai/datasets/<name>` and metadata from `/home/<user>/cst_ai/datasets/<name>`.
- Formal training checkpoints live in a **stable** path from YAML `path.experiments_root` (e.g. `/home/<user>/cst_ai/experiments/aligned_drive/`) so `--auto_resume` survives new pipelines; TB/run markers still use `ci_outputs/<pipeline_id>/`.
- Training semantics (name, paths, `num_gpu`, port, timeout, hyperparams) live in the train YAML. Launch with `uv run python -m torch.distributed.run --standalone --nproc_per_node=4 basicsr/train.py -opt <yaml> --launcher pytorch` on gpu01. CI does not select or start training.
- `scripts/ci/ci_seed_resume.py --opt <yaml>` copies the newest `*.state` + `net_g_*.pth` from prior `ci_outputs` into the stable experiments root before train.
- Switch datasets / experiment identity by editing the YAML (paths + `name`), then run `uv run python -m torch.distributed.run --standalone --nproc_per_node=4 basicsr/train.py -opt <yaml> --launcher pytorch` on gpu01.
- Current dataset (phantom): `aligned_drive` (Z-filtered via `scripts/data/folder_roles.yaml` `slice_ranges`: 5-1 275–623, 5-2 185–740, 5-3/5-4 185–745). Note: `val.txt` is almost entirely group 5-4 because `split_mode=tail` sorts `4-…` filenames last.
- Cylinder retrain dataset: `cylinder_260520_gt90ua` (GT=`1-110kV_90uA`, skip `0-300kV_100uA`, slices 130–750, hardlink flat LQ/GT). Meta + `SPLIT_RECORD.md` under `/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/`. Eval-only holdout: `cylinder_260527_eval_gt60ua` (not for training).
- Training config (round 1 phantom): `options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml`.
- Training config (round 2 phantom): `options/train/SwinIR/train_SwinIR_grayDN_paired_tiff_bs8_30k.yml`.
- Training config (cylinder GT90µA): `options/train/SwinIR/train_SwinIR_grayDN_cylinder_gt90ua_bs8_30k.yml` — `batch_size_per_gpu: 4` (bs=8 OOM'd ~31.65GiB/GPU). Do not overwrite `aligned_drive` or `archive/cylinderblock_*`.
- Mix dataset: `cylinder_phantom_mix` — hardlink merge of `cylinder_260520_gt90ua` (`cyl__`) + `aligned_drive` (`ph__`); stratified domain×µA train/val; meta under `/mnt/home/huang/cst_ai/datasets/cylinder_phantom_mix/`. Built by `scripts/data/build_cylinder_phantom_mix.py`.
- Training config (mix): `options/train/SwinIR/train_SwinIR_grayDN_cylinder_phantom_mix_bs4_30k.yml` — former CI default; kept for reuse. Run it directly with `basicsr/train.py -opt`.
- ALcast dataset: `alcast_150kv_gt66ma` — raw `/mnt/data/cst_ai/datasets/alcast_150kv_raw/` (Drive folder `1U1yvwC1AQoa2AzRGtPDf5rk8mJZ-rP8W`); flat hardlink `/mnt/data/cst_ai/datasets/alcast_150kv_gt66ma/`; meta `/mnt/home/huang/cst_ai/datasets/alcast_150kv_gt66ma/`. GT=`1_ALcast_150kV66mA_data`, LQ=`4/7/17/33mA` (roles: `scripts/data/folder_roles_alcast_150kv_gt66ma.yaml`). User QC slice range **70–820** (inclusive); blanks outside excluded. Stratified holdout: **8 test + 8 val per current** (32 each); remainder train. Train YAML has **`val: ~` (no online DDP val)** — same pattern as TSFDAN; avoids train-CI blocking. `val.txt` kept on disk unused during train; **formal eval after train**: ALcast `test.txt` **plus prior benches** (`aligned_drive` / cylinder / mix). YAML: `options/train/SwinIR/train_SwinIR_grayDN_alcast_gt66ma_bs4_30k.yml`. Train on gpu01 with `basicsr/train.py -opt` that file. The old manual CI jobs `train_paired_tiff_swinir_debug` → `train_paired_tiff_swinir` were removed on 2026-11-09.
- TSFDAN dual-branch + LoRA mix FT: arch `TSFDANSwinIR` / model `TSFDANSwinIRModel`; pretrain `experiments/pretrained_models/TSFDAN/SwinTransformer_288800iter.ckpt` (Wasserdawn/TSFDAN; not SwinIR-M); YAML `options/train/SwinIR/train_TSFDAN_lora_cylinder_phantom_mix_bs4_10k.yml` with **`val: ~` (no online DDP val)**; experiments_root `/home/huang/cst_ai/experiments/cylinder_phantom_mix_tsfdan_lora`. Train with `basicsr/train.py -opt` that YAML; metrics come from a separate eval run. Infer: `scripts/infer/infer_tsfdan_tiff.py`, `scripts/eval/run_eval_benchmark_tsfdan.sh`; tag `tsfdan_lora_mix`.
- Fixed eval benchmark v1: `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v1/` (kept).
- Fixed eval benchmark v2: `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2/` — leakage-first; 260527 stratified by µA; 260520 from safe val pool only. Build via `scripts/eval/build_eval_benchmark.py` (single entrypoint; `--output-root`).
- Eval infer / table: `scripts/eval/run_eval_benchmark_model.sh`, `scripts/eval/fill_eval_benchmark_table.py --root … --preset v1|v2` (no version-suffixed duplicates).
- Traditional CV baselines on the same hardlink bench: `scripts/eval/run_eval_benchmark_cv.py` → `/mnt/home/huang/cst_ai/outputs/eval_benchmark_cv/` (reuses v2 pairs; does not copy images).
- Post-hoc panel enrich (EPI + residual maps + large-font metrics, no re-infer): `scripts/eval/enrich_eval_benchmark_panels.py`. `--mode singles-fixed` writes four PPT PNGs per sample with a shared `[0, 1]` window (uint16 0–65535; no compare). CNR deferred (no defect ROIs).
- Experiment lab notes (params / I/O / commands / user prompts): `docs/experiments.md` (mirror under `/mnt/home/huang/cst_ai/outputs/_lab_notes/`).
- PreFT vs FT panels: `scripts/infer/infer_swinir_graydn_tiff.py --baseline-model …` → `LQ|PreFT|FT|GT` (enriched panels add ResPreFT|ResFT).
- Old cylinder archive weights: `/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/net_g_latest.pth` (read-only).
- Monitoring: CI train jobs auto-start/stop TensorBoard on port `6006` via `scripts/ci/ci_tensorboard.sh` (logdir = run `tb_logger/`); Mac views with `ssh -L 6006:localhost:6006`. Runner needs `network_mode = "host"` (see `docs/infrastructure.md`). Manual re-serve: `scripts/ci/serve_tensorboard.sh …/tb_logger`. wandb optional (`WANDB_API_KEY`).

BasicSR: `PROJECT_NAME=cst_ai`, raw backup at `/mnt/data/cst_ai/`.

Helper: `scripts/data/data_paths.py`.

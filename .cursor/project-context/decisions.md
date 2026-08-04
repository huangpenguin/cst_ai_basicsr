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
- Formal training checkpoints live in a **stable** path from YAML `path.experiments_root` (e.g. `/home/<user>/cst_ai/experiments/aligned_drive/`) so `--auto_resume` survives new pipelines; TB/run markers still use `ci_outputs/<pipeline_id>/`.
- Training semantics (name, paths, `num_gpu`, port, timeout, hyperparams) live in the train YAML; CI only sets `TRAIN_OPT` and launches. `scripts/ci_emit_opt_env.py` exports env vars from that YAML.
- `scripts/ci_seed_resume.py --opt <yaml>` copies the newest `*.state` + `net_g_*.pth` from prior `ci_outputs` into the stable experiments root before train.
- Switch datasets / experiment identity by editing the YAML (paths + `name`); run `train_paired_tiff_swinir_debug` before `train_paired_tiff_swinir`.
- Current dataset (phantom): `aligned_drive` (Z-filtered via `scripts/folder_roles.yaml` `slice_ranges`: 5-1 275–623, 5-2 185–740, 5-3/5-4 185–745). Note: `val.txt` is almost entirely group 5-4 because `split_mode=tail` sorts `4-…` filenames last.
- Cylinder retrain dataset: `cylinder_260520_gt90ua` (GT=`1-110kV_90uA`, skip `0-300kV_100uA`, slices 130–750, hardlink flat LQ/GT). Meta + `SPLIT_RECORD.md` under `/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/`. Eval-only holdout: `cylinder_260527_eval_gt60ua` (not for training).
- Training config (round 1 phantom): `options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml`.
- Training config (round 2 phantom): `options/train/SwinIR/train_SwinIR_grayDN_paired_tiff_bs8_30k.yml`.
- Training config (cylinder GT90µA): `options/train/SwinIR/train_SwinIR_grayDN_cylinder_gt90ua_bs8_30k.yml` — `batch_size_per_gpu: 4` (bs=8 OOM'd ~31.65GiB/GPU). Do not overwrite `aligned_drive` or `archive/cylinderblock_*`.
- Mix dataset: `cylinder_phantom_mix` — hardlink merge of `cylinder_260520_gt90ua` (`cyl__`) + `aligned_drive` (`ph__`); stratified domain×µA train/val; meta under `/mnt/home/huang/cst_ai/datasets/cylinder_phantom_mix/`. Built by `scripts/build_cylinder_phantom_mix.py`.
- Training config (mix): `options/train/SwinIR/train_SwinIR_grayDN_cylinder_phantom_mix_bs4_30k.yml` — CI `TRAIN_OPT` points here. Manual CI start.
- TSFDAN dual-branch + LoRA mix FT: arch `TSFDANSwinIR` / model `TSFDANSwinIRModel`; pretrain `experiments/pretrained_models/TSFDAN/SwinTransformer_288800iter.ckpt` (Wasserdawn/TSFDAN; not SwinIR-M); YAML `options/train/SwinIR/train_TSFDAN_lora_cylinder_phantom_mix_bs4_10k.yml`; experiments_root `/home/huang/cst_ai/experiments/cylinder_phantom_mix_tsfdan_lora`. CI: `.gitlab/ci/train_tsfdan.yml` manual jobs `train_tsfdan_lora_mix_debug` → `train_tsfdan_lora_mix` → `eval_tsfdan_lora_mix` (does not change default `TRAIN_OPT`). Infer scripts: `scripts/infer_tsfdan_tiff.py`, `scripts/run_eval_benchmark_tsfdan.sh`; tag `tsfdan_lora_mix`. Swap experiment by editing that YAML (or `TSFDAN_TRAIN_OPT` / `TSFDAN_EVAL_TAG`).
- Fixed eval benchmark v1: `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v1/` (kept).
- Fixed eval benchmark v2: `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2/` — leakage-first; 260527 stratified by µA; 260520 from safe val pool only. Build via `scripts/build_eval_benchmark.py` (single entrypoint; `--output-root`).
- Eval infer / table: `scripts/run_eval_benchmark_model.sh`, `scripts/fill_eval_benchmark_table.py --root … --preset v1|v2` (no version-suffixed duplicates).
- Traditional CV baselines on the same hardlink bench: `scripts/run_eval_benchmark_cv.py` → `/mnt/home/huang/cst_ai/outputs/eval_benchmark_cv/` (reuses v2 pairs; does not copy images).
- Post-hoc panel enrich (EPI + residual maps + large-font metrics, no re-infer): `scripts/enrich_eval_benchmark_panels.py`. CNR deferred (no defect ROIs).
- Experiment lab notes (params / I/O / commands / user prompts): `docs/lab-notes/` (mirror under `/mnt/home/huang/cst_ai/outputs/_lab_notes/`).
- PreFT vs FT panels: `scripts/infer_swinir_graydn_tiff.py --baseline-model …` → `LQ|PreFT|FT|GT` (enriched panels add ResPreFT|ResFT).
- Old cylinder archive weights: `/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/net_g_latest.pth` (read-only).
- Monitoring: CI train jobs auto-start/stop TensorBoard on port `6006` via `scripts/ci_tensorboard.sh` (logdir = run `tb_logger/`); Mac views with `ssh -L 6006:localhost:6006`. Runner needs `network_mode = "host"` (see `docs/use-cases/gpu-runner.zh-CN.md`). Manual re-serve: `scripts/serve_tensorboard.sh …/tb_logger`. wandb optional (`WANDB_API_KEY`).

BasicSR: `PROJECT_NAME=cst_ai`, raw backup at `/mnt/data/cst_ai/`.

Helper: `scripts/data_paths.py`.

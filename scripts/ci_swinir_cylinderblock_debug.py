#!/usr/bin/env python3
"""GitLab CI debug job for Cylinderblock SwinIR gray-dn training.

Runs a short, single-GPU training session on a tiny preprocessed subset so CUDA /
shape / dataset issues surface in minutes instead of after hours of full DDP
training.

Typical CI usage: manually play job ``train_cylinderblock_swinir_debug``.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ci_storage  # noqa: E402

PREPROCESS_SCRIPT = REPO_ROOT / "scripts" / "preprocess_cylinderblock_swinir.py"
TRAIN_SCRIPT = REPO_ROOT / "basicsr" / "train.py"
TRAIN_OPT = REPO_ROOT / "options" / "train" / "SwinIR" / "train_SwinIR_grayDN_cylinderblock.yml"
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_swinir_release.py"
ARTIFACT_MARKER = REPO_ROOT / ".ci_experiments_root"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def resolve_storage_root() -> Path:
    return ci_storage.resolve_storage_root(project_dir=REPO_ROOT)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n>>> {printable}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.check_call(cmd, cwd=REPO_ROOT, env=merged)


def ensure_debug_dataset(
    *,
    storage_root: Path,
    raw_data_root: Path,
    max_slices: int,
    overwrite: bool,
) -> Path:
    dataset_dir = storage_root / "datasets" / "swinir_cylinderblock_debug"
    required = [
        dataset_dir / "train" / "LQ",
        dataset_dir / "train" / "GT",
        dataset_dir / "val" / "LQ",
        dataset_dir / "val" / "GT",
        dataset_dir / "normalization.json",
    ]
    if not overwrite and all(path.exists() for path in required):
        print(f"Reusing existing debug dataset: {dataset_dir}")
        return dataset_dir

    if not raw_data_root.is_dir():
        raise SystemExit(
            f"ERROR: raw data root not found: {raw_data_root}\n"
            "Mount /mnt/data on the GPU runner and ensure TRAIN_RAW_DATA_ROOT is visible."
        )

    cmd = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--data-root",
        str(raw_data_root),
        "--output-root",
        str(dataset_dir),
        "--max-slices",
        str(max_slices),
        "--overwrite",
    ]
    run(cmd)
    return dataset_dir


def summarize_manifest(dataset_dir: Path) -> None:
    manifest = dataset_dir / "manifest.csv"
    if not manifest.is_file():
        print(f"WARNING: manifest missing: {manifest}")
        return

    sizes: Counter[tuple[int, int]] = Counter()
    with manifest.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            sizes[(int(row["width"]), int(row["height"]))] += 1
    print(f"manifest image sizes: {dict(sizes)}")


def validate_tiff_sizes(dataset_dir: Path, *, window_size: int) -> None:
    bad: list[str] = []
    for split in ("train", "val"):
        for kind in ("LQ", "GT"):
            folder = dataset_dir / split / kind
            if not folder.is_dir():
                raise SystemExit(f"ERROR: missing folder: {folder}")

            sizes: Counter[tuple[int, int]] = Counter()
            for path in sorted(folder.glob("*.tif")):
                from PIL import Image

                with Image.open(path) as image:
                    sizes[image.size] += 1

            print(f"{split}/{kind}: files={sum(sizes.values())} sizes={dict(sizes)}")
            for width, height in sizes:
                if width % window_size != 0 or height % window_size != 0:
                    bad.append(f"{split}/{kind} {width}x{height} (not divisible by window_size={window_size})")

    if bad:
        details = "\n  - ".join(bad)
        raise SystemExit(f"ERROR: dataset images incompatible with SwinIR window_size:\n  - {details}")


def build_train_command(
    *,
    dataset_dir: Path,
    experiments_root: Path,
    total_iter: int,
    batch_size_per_gpu: int,
    nproc: int,
    master_port: int,
) -> list[str]:
    yml_overrides: list[str] = [
        f"datasets:train:dataroot_lq={dataset_dir / 'train' / 'LQ'}",
        f"datasets:train:dataroot_gt={dataset_dir / 'train' / 'GT'}",
        f"datasets:train:normalization={dataset_dir / 'normalization.json'}",
        f"datasets:val:dataroot_lq={dataset_dir / 'val' / 'LQ'}",
        f"datasets:val:dataroot_gt={dataset_dir / 'val' / 'GT'}",
        f"datasets:val:normalization={dataset_dir / 'normalization.json'}",
        f"path:experiments_root={experiments_root}",
        "path:resume_state=~",
        f"train:total_iter={total_iter}",
        f"datasets:train:batch_size_per_gpu={batch_size_per_gpu}",
        "datasets:train:num_worker_per_gpu=0",
        # Single-GPU debug: avoid DataParallel when num_gpu:auto sees all GPUs.
        "num_gpu=1",
        # Full-resolution val on large TIFFs OOMs on 32GB GPUs; skip CI smoke validation.
        "val=~",
    ]

    train_args = [
        str(TRAIN_SCRIPT),
        "-opt",
        str(TRAIN_OPT),
        "--debug",
        "--force_yml",
        *yml_overrides,
    ]

    if nproc > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={master_port}",
            *train_args,
            "--launcher",
            "pytorch",
        ]
    return [sys.executable, *train_args]


def stage_artifacts(experiments_root: Path) -> None:
    results_dir = REPO_ROOT / "results" / "cylinderblock_swinir_debug"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not experiments_root.is_dir():
        print(f"WARNING: experiments root missing, skip artifact staging: {experiments_root}")
        return
    shutil.copytree(experiments_root, results_dir, dirs_exist_ok=True)
    print(f"Staged artifacts to {results_dir}")


def print_config(
    *,
    storage_root: Path,
    dataset_dir: Path,
    experiments_root: Path,
    raw_data_root: Path,
    max_slices: int,
    total_iter: int,
    nproc: int,
) -> None:
    print("=== SwinIR Cylinderblock CI debug ===")
    print(f"storage_root={storage_root}")
    print(f"dataset_dir={dataset_dir}")
    print(f"experiments_root={experiments_root}")
    print(f"raw_data_root={raw_data_root}")
    print(f"max_slices={max_slices}")
    print(f"total_iter={total_iter}")
    print(f"nproc={nproc}")
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', 'unset')}")


def main(argv: Iterable[str] | None = None) -> None:
    _ = argv
    if os.environ.get("CI_DEBUG_CUDA_LAUNCH_BLOCKING", "0") == "1":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    max_slices = _env_int("CI_DEBUG_MAX_SLICES", 5)
    total_iter = _env_int("CI_DEBUG_TOTAL_ITER", 30)
    nproc = _env_int("CI_DEBUG_NPROC", 1)
    master_port = _env_int("CI_DEBUG_MASTER_PORT", 4332)
    batch_size = _env_int("CI_DEBUG_BATCH_SIZE", 1)
    window_size = _env_int("CI_DEBUG_WINDOW_SIZE", 8)
    overwrite_dataset = os.environ.get("CI_DEBUG_OVERWRITE_DATASET", "0") == "1"

    project_name = os.environ.get("PROJECT_NAME", "cst_ai")
    raw_data_root = Path(os.environ.get("TRAIN_RAW_DATA_ROOT", f"/data/{project_name}/data_0607"))

    storage_root = resolve_storage_root()
    paths = ci_storage.prepare_run_outputs(storage_root, debug=True, dataset_subdir="swinir_cylinderblock_debug")
    ci_storage.apply_tb_logger_env(paths)
    ci_storage.write_markers(REPO_ROOT, paths)
    ci_storage.print_paths(paths)

    experiments_root = paths.experiments_root
    ARTIFACT_MARKER.write_text(str(experiments_root), encoding="utf-8")

    dataset_dir = ensure_debug_dataset(
        storage_root=storage_root,
        raw_data_root=raw_data_root,
        max_slices=max_slices,
        overwrite=overwrite_dataset,
    )
    summarize_manifest(dataset_dir)
    validate_tiff_sizes(dataset_dir, window_size=window_size)

    print_config(
        storage_root=storage_root,
        dataset_dir=dataset_dir,
        experiments_root=experiments_root,
        raw_data_root=raw_data_root,
        max_slices=max_slices,
        total_iter=total_iter,
        nproc=nproc,
    )

    run(["uv", "pip", "install", "-e", ".", "--no-deps"])
    run([sys.executable, str(DOWNLOAD_SCRIPT), "--only", "004_grayDN"])

    train_cmd = build_train_command(
        dataset_dir=dataset_dir,
        experiments_root=experiments_root,
        total_iter=total_iter,
        batch_size_per_gpu=batch_size,
        nproc=nproc,
        master_port=master_port,
    )
    run(train_cmd)
    stage_artifacts(experiments_root)
    print("CI SwinIR debug training finished successfully.")


if __name__ == "__main__":
    main()

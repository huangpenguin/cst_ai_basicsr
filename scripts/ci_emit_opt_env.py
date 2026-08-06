#!/usr/bin/env python3
"""Emit shell exports from a BasicSR train YAML (single source of truth for CI).

Usage::

    eval "$(python scripts/ci_emit_opt_env.py --opt options/train/.../foo.yml)"
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True, help="Path to train YAML")
    return parser.parse_args()


def _require(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    trail: list[str] = []
    for key in keys:
        trail.append(key)
        if not isinstance(cur, dict) or key not in cur:
            raise SystemExit(f"Missing YAML key: {':'.join(trail)}")
        cur = cur[key]
    return cur


def _dataset_subdir_from_dataroot(dataroot: str) -> str:
    """'/data/cst_ai/datasets/aligned_drive/LQ' -> 'aligned_drive'."""
    parts = Path(dataroot).parts
    try:
        idx = parts.index("datasets")
    except ValueError as exc:
        raise SystemExit(f"Cannot derive dataset subdir from dataroot: {dataroot}") from exc
    if idx + 1 >= len(parts):
        raise SystemExit(f"Cannot derive dataset subdir from dataroot: {dataroot}")
    return parts[idx + 1]


def _nproc(num_gpu: Any) -> int:
    if isinstance(num_gpu, int):
        if num_gpu < 1:
            raise SystemExit(f"num_gpu must be >= 1, got {num_gpu}")
        return num_gpu
    if num_gpu == "auto":
        try:
            import torch

            count = int(torch.cuda.device_count())
        except Exception:
            count = 0
        if count < 1:
            raise SystemExit("num_gpu is auto but no CUDA device is visible; set num_gpu to an int in YAML")
        return count
    raise SystemExit(f"Unsupported num_gpu value: {num_gpu!r} (use int or auto)")


def emit_env(opt_path: Path) -> dict[str, str]:
    opt = yaml.safe_load(opt_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict):
        raise SystemExit(f"Invalid train YAML: {opt_path}")

    name = _require(opt, "name")
    experiments_root = _require(opt, "path", "experiments_root")
    train_lq = _require(opt, "datasets", "train", "dataroot_lq")
    train_gt = _require(opt, "datasets", "train", "dataroot_gt")
    train_meta = _require(opt, "datasets", "train", "meta_info_file")
    train_norm = _require(opt, "datasets", "train", "normalization")
    # Optional online-val dataset block (TSFDAN train YAMLs may omit it; eval is a separate CI job).
    datasets = opt.get("datasets") or {}
    val_opt = datasets.get("val") if isinstance(datasets, dict) else None
    if isinstance(val_opt, dict):
        val_lq = _require(opt, "datasets", "val", "dataroot_lq")
        val_gt = _require(opt, "datasets", "val", "dataroot_gt")
        val_meta = _require(opt, "datasets", "val", "meta_info_file")
        val_norm = _require(opt, "datasets", "val", "normalization")
    else:
        val_lq = train_lq
        val_gt = train_gt
        val_meta = train_meta
        val_norm = train_norm
    port = _require(opt, "dist_params", "port")
    timeout_min = _require(opt, "dist_params", "timeout_min")
    num_gpu = _require(opt, "num_gpu")

    dataset_subdir = _dataset_subdir_from_dataroot(str(train_lq))
    source_dir = str(Path(str(train_lq)).parent)
    dataset_dir = str(Path(str(train_meta)).parent)

    return {
        "TRAIN_EXP_NAME": str(name),
        "TRAIN_EXPERIMENTS_ROOT": str(experiments_root),
        "PAIRED_DATASET_NAME": dataset_subdir,
        "TRAIN_SOURCE_DIR": source_dir,
        "TRAIN_DATASET_DIR": dataset_dir,
        "TRAIN_DATAROOT_LQ": str(train_lq),
        "TRAIN_DATAROOT_GT": str(train_gt),
        "TRAIN_META_TRAIN": str(train_meta),
        "TRAIN_META_VAL": str(val_meta),
        "TRAIN_NORM": str(train_norm),
        "TRAIN_VAL_DATAROOT_LQ": str(val_lq),
        "TRAIN_VAL_DATAROOT_GT": str(val_gt),
        "TRAIN_VAL_NORM": str(val_norm),
        "TRAIN_NPROC_PER_NODE": str(_nproc(num_gpu)),
        "TRAIN_MASTER_PORT": str(port),
        "TRAIN_DIST_TIMEOUT_MIN": str(timeout_min),
    }


def main() -> None:
    args = parse_args()
    opt_path = args.opt.expanduser().resolve()
    if not opt_path.is_file():
        raise SystemExit(f"Train YAML not found: {opt_path}")
    for key, value in emit_env(opt_path).items():
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()

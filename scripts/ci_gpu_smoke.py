#!/usr/bin/env python3
"""Minimal GPU CI smoke: verify DATA_DIR mount, fetch gray_dn weights, run one inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from data_paths import data_root, default_battery_tiff_dir


def _require_data_mount() -> tuple[Path, Path]:
    mount = data_root()
    print(f"DATA_DIR={mount}")
    if not mount.is_dir():
        raise SystemExit(
            "DATA_DIR is missing inside the CI job container.\n"
            "Configure the GPU runner config.toml, for example:\n"
            '  volumes = ["/cache", "/mnt/data:/data:ro"]\n'
            "The left-hand path must exist on the runner host (Docker daemon), not an NFS export URI."
        )

    battery_dir = default_battery_tiff_dir()
    print(f"dataset_dir={battery_dir}")
    if not battery_dir.is_dir():
        raise SystemExit(
            f"Battery TIFF directory not found: {battery_dir}\n"
            "Expected layout under /data/<PROJECT_NAME>/, e.g.:\n"
            "  data_0607/260520_1-battery_data/260520_battery image tiff data\n"
            "Or set CI variable DATASET_DIR to an absolute path visible in the job container."
        )

    tifs = sorted(battery_dir.glob("*.tif"))
    if not tifs:
        raise SystemExit(f"No *.tif files found in {battery_dir}")
    return battery_dir, tifs[0]


def _download_gray_dn_models() -> None:
    subprocess.check_call(
        [
            sys.executable,
            str(SCRIPTS / "download_swinir_release.py"),
            "--only",
            "004_grayDN",
        ],
        cwd=ROOT,
    )


def _run_single_inference(sample_tif: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            str(SCRIPTS / "smoke_test_swinir_gray_dn_battery.py"),
            "--devices",
            "0",
            "--noise-levels",
            "15",
            "--images",
            sample_tif.name,
        ],
        cwd=ROOT,
    )


def main() -> None:
    import torch

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the CI job container.")

    _, sample_tif = _require_data_mount()
    print(f"sample_tif={sample_tif}")
    _download_gray_dn_models()
    _run_single_inference(sample_tif)
    print("CI GPU smoke passed.")


if __name__ == "__main__":
    main()

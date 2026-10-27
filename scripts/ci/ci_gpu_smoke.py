#!/usr/bin/env python3
"""Minimal GPU CI smoke: verify DATA_DIR mount, fetch gray_dn weights, run one inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS / "data"))

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
            str(SCRIPTS / "download" / "download_swinir_release.py"),
            "--only",
            "004_grayDN",
        ],
        cwd=ROOT,
    )


def _run_single_inference(sample_tif: Path) -> None:
    out_dir = ROOT / "results" / "ci_gpu_smoke"
    subprocess.check_call(
        [
            sys.executable,
            str(SCRIPTS / "infer" / "infer_swinir_graydn_tiff.py"),
            "--model",
            str(ROOT / "experiments" / "pretrained_models" / "SwinIR" / "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth"),
            "--inputs",
            str(sample_tif),
            "--output-dir",
            str(out_dir),
            "--output-format",
            "tiff16",
            "--device",
            "cuda:0",
        ],
        cwd=ROOT,
    )
    produced = out_dir / f"{sample_tif.stem}_ft.tif"
    if not produced.is_file():
        raise SystemExit(f"Smoke inference did not produce {produced}")


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

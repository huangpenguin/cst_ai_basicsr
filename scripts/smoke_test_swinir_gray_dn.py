#!/usr/bin/env python3
"""Multi-GPU smoke test for SwinIR gray_dn pretrained models on TIFF data."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from battery_tiff_io import read_battery_tiff, save_gray_png, to_model_tensor
from data_paths import data_root, default_cylinder_tiff_dir
from swinir_gray_dn_infer import GRAY_DN_MODELS, infer_gray, load_gray_dn_model

DEFAULT_MODEL_DIR = ROOT / "experiments/pretrained_models/SwinIR"
DEFAULT_INPUT = default_cylinder_tiff_dir()
DEFAULT_OUTPUT = ROOT / "results/swinir_gray_dn_cylinder_smoke"


def _run_gpu_worker(
    gpu_id: int,
    tasks: list[tuple[int, str, str]],
    model_dir: str,
    output_dir: str,
    p_low: float,
    p_high: float,
) -> list[str]:
    import torch

    device = torch.device(f"cuda:{gpu_id}")
    out_root = Path(output_dir)
    model_root = Path(model_dir)
    logs: list[str] = []
    current_noise: int | None = None
    model = None

    for noise_level, model_name, image_path_str in tasks:
        image_path = Path(image_path_str)
        if noise_level != current_noise:
            model_path = model_root / model_name
            model = load_gray_dn_model(model_path, device)
            current_noise = noise_level

        assert model is not None
        stem = image_path.stem
        out_dir = out_root / f"noise{noise_level}"
        out_dir.mkdir(parents=True, exist_ok=True)

        img = read_battery_tiff(image_path, p_low=p_low, p_high=p_high)
        denoised = infer_gray(model, img, device, to_model_tensor)

        save_gray_png(img, out_dir / f"{stem}_input.png")
        save_gray_png(denoised, out_dir / f"{stem}_SwinIR_grayDN.png")
        logs.append(f"gpu{gpu_id} noise{noise_level} {stem}")

    return logs


def parse_devices(spec: str) -> list[int]:
    devices = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not devices:
        raise ValueError("At least one GPU id is required.")
    return devices


def build_task_buckets(
    image_paths: list[Path],
    devices: list[int],
    models: dict[int, str] | None = None,
) -> dict[int, list[tuple[int, str, str]]]:
    model_map = models or GRAY_DN_MODELS
    all_tasks: list[tuple[int, str, str]] = [
        (noise_level, model_name, str(image_path))
        for noise_level, model_name in model_map.items()
        for image_path in image_paths
    ]
    buckets: dict[int, list[tuple[int, str, str]]] = {gpu_id: [] for gpu_id in devices}
    for idx, task in enumerate(all_tasks):
        gpu_id = devices[idx % len(devices)]
        buckets[gpu_id].append(task)
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SwinIR gray_dn smoke test across multiple GPUs.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated CUDA device ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Optional TIFF basenames; default uses all *.tif in input-dir.",
    )
    parser.add_argument(
        "--noise-levels",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of gray_dn noise levels, e.g. 15 or 15 25.",
    )
    args = parser.parse_args()

    devices = parse_devices(args.devices)
    print(f"devices={devices}")
    print(f"DATA_DIR={data_root()}")
    print(f"input_dir={args.input_dir}")
    print(f"output_dir={args.output_dir}")

    if args.images:
        image_paths = [args.input_dir / name for name in args.images]
    else:
        image_paths = sorted(args.input_dir.glob("*.tif"))

    if not image_paths:
        raise FileNotFoundError(f"No TIFF images found in {args.input_dir}")

    selected_models = GRAY_DN_MODELS
    if args.noise_levels is not None:
        missing = [level for level in args.noise_levels if level not in GRAY_DN_MODELS]
        if missing:
            raise ValueError(f"Unknown noise levels: {missing}; available: {sorted(GRAY_DN_MODELS)}")
        selected_models = {level: GRAY_DN_MODELS[level] for level in args.noise_levels}

    for model_name in selected_models.values():
        model_path = args.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    buckets = build_task_buckets(image_paths, devices, selected_models)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(devices)) as pool:
        async_results = []
        for gpu_id in devices:
            tasks = buckets[gpu_id]
            if not tasks:
                continue
            async_results.append(
                pool.apply_async(
                    _run_gpu_worker,
                    args=(
                        gpu_id,
                        tasks,
                        str(args.model_dir),
                        str(args.output_dir),
                        args.p_low,
                        args.p_high,
                    ),
                )
            )

        for result in async_results:
            for line in result.get():
                print(line)

    print(f"\nSmoke test finished. Results: {args.output_dir}")


if __name__ == "__main__":
    main()

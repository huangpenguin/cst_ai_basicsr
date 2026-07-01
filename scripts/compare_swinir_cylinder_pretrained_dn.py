#!/usr/bin/env python3
"""Preprocess Cylinderblock LQ slices and compare SwinIR grayDN noise15 vs noise50."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cylinderblock_tiff_io import (  # noqa: E402
    array_to_u16_bytes,
    read_tiff,
    resize_u16,
    rotate_u16_cw,
    u16_bytes_to_array,
    write_u16_tiff,
)
from swinir_gray_dn_infer import GRAY_DN_MODELS, infer_gray, load_gray_dn_model  # noqa: E402

DEFAULT_DATA_ROOT = Path("/mnt/data/cst_ai/data_0607")
DEFAULT_MODEL_DIR = ROOT / "experiments/pretrained_models/SwinIR"
DEFAULT_OUTPUT = ROOT / "results/swinir_graydn_pretrained_compare"
GT_SIZE = 1530
RAW_MIN = 0.0
RAW_MAX = 65535.0


@dataclass(frozen=True)
class SampleSpec:
    batch: str
    label: str
    lq_src: Path
    gt_src: Path
    rotation_deg_cw: float


def build_samples(data_root: Path, slice_index: int) -> tuple[SampleSpec, ...]:
    stem = f"{slice_index:04d}"
    return (
        SampleSpec(
            batch="260520",
            label=f"260520_110kV_90uA_slice{stem}",
            lq_src=data_root
            / "260520_2-Cylinderblock_data/260520_cylinderblock image data/1-110kV_90uA/XY"
            / f"data_{stem}.tif",
            gt_src=data_root
            / "260520_2-Cylinderblock_data/260520_cylinderblock image data/0-300kV_100uA/XY"
            / f"data_{stem}.tif",
            rotation_deg_cw=193.0,
        ),
        SampleSpec(
            batch="260527",
            label=f"260527_150kV_60uA_slice{stem}",
            lq_src=data_root
            / "260527_4-Cylinderblock_data/260527_cylinderblock image data/1-150kV_60uA/XY"
            / f"data_{stem}.tif",
            gt_src=data_root
            / "260527_4-Cylinderblock_data/260527_cylinderblock image data/0-300kV_100uA/XY"
            / f"data_{stem}.tif",
            rotation_deg_cw=91.0,
        ),
    )


def preprocess_lq(lq_path: Path, rotation_deg_cw: float) -> np.ndarray:
    arr = u16_bytes_to_array(read_tiff(lq_path))
    rotated = rotate_u16_cw(arr, rotation_deg_cw)
    return resize_u16(rotated, GT_SIZE, GT_SIZE)


def load_gt_u16(gt_path: Path) -> np.ndarray:
    return u16_bytes_to_array(read_tiff(gt_path))


def u16_to_model_tensor(arr: np.ndarray) -> torch.Tensor:
    normalized = np.clip((arr.astype(np.float32) - RAW_MIN) / (RAW_MAX - RAW_MIN), 0.0, 1.0)
    chw = np.expand_dims(normalized, axis=0)
    return torch.from_numpy(chw).float().unsqueeze(0)


def to_u8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_gray_png_from_float(img_hw: np.ndarray, path: Path) -> None:
    out = (np.clip(img_hw, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)


def save_gray_png_from_u16(arr: np.ndarray, path: Path, lo: float, hi: float) -> None:
    Image.fromarray(to_u8(arr, lo, hi), mode="L").save(path)


def make_panel(
    lq_u16: np.ndarray,
    denoised_15: np.ndarray,
    denoised_50: np.ndarray,
    gt_u16: np.ndarray,
    title: str,
) -> Image.Image:
    lo = float(min(lq_u16.min(), gt_u16.min()))
    hi = float(max(lq_u16.max(), gt_u16.max()))
    cols = [
        to_u8(lq_u16, lo, hi),
        (np.clip(denoised_15, 0, 1) * 255).astype(np.uint8),
        (np.clip(denoised_50, 0, 1) * 255).astype(np.uint8),
        to_u8(gt_u16, lo, hi),
    ]
    labels = ["LQ preprocessed", "SwinIR noise15", "SwinIR noise50", "GT 300kV"]
    gap = 8
    h, w = cols[0].shape
    total_w = w * len(cols) + gap * (len(cols) - 1)
    panel = np.full((h + 40, total_w), 32, dtype=np.uint8)
    x = 0
    for col in cols:
        panel[40:, x : x + w] = col
        x += w + gap
    img = Image.fromarray(panel, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title, fill=(255, 220, 120))
    x = 0
    for label in labels:
        draw.text((x + 8, h + 44), label, fill=(200, 200, 200))
        x += w + gap
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slice", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--noise-levels",
        type=int,
        nargs="+",
        default=[15, 50],
        help="grayDN pretrained noise levels to compare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models: dict[int, torch.nn.Module] = {}
    for level in args.noise_levels:
        if level not in GRAY_DN_MODELS:
            raise ValueError(f"Unknown noise level {level}; available: {sorted(GRAY_DN_MODELS)}")
        model_path = args.model_dir / GRAY_DN_MODELS[level]
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path} (run download_swinir_release.py)")
        models[level] = load_gray_dn_model(model_path, device)
        print(f"Loaded noise{level}: {model_path.name}")

    for sample in build_samples(args.data_root, args.slice):
        for path in (sample.lq_src, sample.gt_src):
            if not path.is_file():
                raise FileNotFoundError(path)

        out_dir = args.output_dir / sample.label
        out_dir.mkdir(parents=True, exist_ok=True)

        lq_u16 = preprocess_lq(sample.lq_src, sample.rotation_deg_cw)
        gt_u16 = load_gt_u16(sample.gt_src)
        lq_tiff = out_dir / f"{sample.label}_lq_preprocessed.tif"
        write_u16_tiff(lq_tiff, array_to_u16_bytes(lq_u16), GT_SIZE, GT_SIZE)

        denoised: dict[int, np.ndarray] = {}
        for level, model in models.items():
            denoised[level] = infer_gray(model, lq_u16, device, u16_to_model_tensor)
            save_gray_png_from_float(denoised[level], out_dir / f"{sample.label}_denoise_noise{level}.png")

        lo = float(min(lq_u16.min(), gt_u16.min()))
        hi = float(max(lq_u16.max(), gt_u16.max()))
        save_gray_png_from_u16(lq_u16, out_dir / f"{sample.label}_lq_preprocessed.png", lo, hi)
        save_gray_png_from_u16(gt_u16, out_dir / f"{sample.label}_gt.png", lo, hi)

        if 15 in denoised and 50 in denoised:
            title = f"{sample.label} | rot {sample.rotation_deg_cw:.0f}° | pretrained grayDN"
            panel = make_panel(lq_u16, denoised[15], denoised[50], gt_u16, title)
            panel.save(out_dir / f"{sample.label}_panel_noise15_vs_noise50.png")

        print(f"Wrote {out_dir}")

    print(f"Done. Results: {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate preprocess alignment previews (panel + overlay) for Cylinderblock slices."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_swinir_cylinder_pretrained_dn import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    GT_SIZE,
    LQ_SIZE,
    build_samples,
    preprocess_lq,
    to_u8,
)
from cylinderblock_tiff_io import read_tiff, rotate_u16_cw, u16_bytes_to_array  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results/preprocess_preview"


@dataclass(frozen=True)
class PreviewPaths:
    raw: Path
    rotated: Path
    aligned: Path
    gt: Path
    overlay: Path
    panel: Path


def make_panel(lq_raw: np.ndarray, lq_aligned: np.ndarray, gt: np.ndarray, title: str) -> Image.Image:
    lo = int(min(lq_raw.min(), lq_aligned.min(), gt.min()))
    hi = int(max(lq_raw.max(), lq_aligned.max(), gt.max()))
    raw_u8 = to_u8(lq_raw, lo, hi)
    aligned_u8 = to_u8(lq_aligned, lo, hi)
    gt_u8 = to_u8(gt, lo, hi)
    raw_padded = np.zeros((GT_SIZE, GT_SIZE), dtype=np.uint8)
    t = (GT_SIZE - LQ_SIZE) // 2
    raw_padded[t : t + LQ_SIZE, t : t + LQ_SIZE] = raw_u8
    gap = 8
    total_w = GT_SIZE * 3 + gap * 2
    panel = np.full((GT_SIZE + 40, total_w), 32, dtype=np.uint8)
    panel[40:, 0:GT_SIZE] = raw_padded
    panel[40:, GT_SIZE + gap : 2 * GT_SIZE + gap] = aligned_u8
    panel[40:, 2 * (GT_SIZE + gap) : 2 * (GT_SIZE + gap) + GT_SIZE] = gt_u8
    img = Image.fromarray(panel, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    for x, label in zip(
        [GT_SIZE // 2, GT_SIZE + gap + GT_SIZE // 2, 2 * (GT_SIZE + gap) + GT_SIZE // 2],
        ["LQ raw 1280", "LQ rot+resize 1530", "GT 1530"],
        strict=True,
    ):
        draw.text((x - 70, 8), label, fill=(220, 220, 220))
    draw.text((8, 8), title, fill=(255, 220, 120))
    return img


def write_preview(
    sample_label: str,
    rotation_deg_cw: float,
    lq_path: Path,
    gt_path: Path,
    out_dir: Path,
    slice_index: int,
) -> PreviewPaths:
    lq_raw = u16_bytes_to_array(read_tiff(lq_path))
    gt = u16_bytes_to_array(read_tiff(gt_path))
    rotated = rotate_u16_cw(lq_raw, rotation_deg_cw)
    aligned = preprocess_lq(lq_path, rotation_deg_cw)

    out_dir.mkdir(parents=True, exist_ok=True)
    lo = int(min(lq_raw.min(), aligned.min(), gt.min()))
    hi = int(max(lq_raw.max(), aligned.max(), gt.max()))

    paths = PreviewPaths(
        raw=out_dir / f"{sample_label}_01_lq_raw_1280.png",
        rotated=out_dir / f"{sample_label}_02_lq_rot{int(rotation_deg_cw)}_1280.png",
        aligned=out_dir / f"{sample_label}_03_lq_rot_resize_1530.png",
        gt=out_dir / f"{sample_label}_04_gt_1530.png",
        overlay=out_dir / f"{sample_label}_05_overlay_gt_green_lq_red.png",
        panel=out_dir / f"{sample_label}_panel.png",
    )
    Image.fromarray(to_u8(lq_raw, lo, hi)).save(paths.raw)
    Image.fromarray(to_u8(rotated, lo, hi)).save(paths.rotated)
    Image.fromarray(to_u8(aligned, lo, hi)).save(paths.aligned)
    Image.fromarray(to_u8(gt, lo, hi)).save(paths.gt)

    thr = max(lo, np.percentile(aligned, 5))
    mask = (aligned > thr) & (gt > thr)
    overlay = np.zeros((GT_SIZE, GT_SIZE, 3), dtype=np.uint8)
    overlay[..., 1] = to_u8(gt, lo, hi)
    red = to_u8(aligned, lo, hi)
    overlay[mask, 0] = red[mask]
    overlay[mask, 2] = red[mask] // 2
    Image.fromarray(overlay).save(paths.overlay)

    title = f"{sample_label} | rot {rotation_deg_cw:.0f}° nearest-edge | slice {slice_index:04d}"
    make_panel(lq_raw, aligned, gt, title).save(paths.panel)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slice", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for sample in build_samples(args.data_root.expanduser().resolve(), args.slice):
        out_dir = args.output_root / sample.label
        paths = write_preview(
            sample.label,
            sample.rotation_deg_cw,
            sample.lq_src,
            sample.gt_src,
            out_dir,
            args.slice,
        )
        print(f"Wrote {paths.panel}")


if __name__ == "__main__":
    main()

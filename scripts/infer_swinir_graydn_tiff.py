#!/usr/bin/env python3
"""Run SwinIR grayDN inference on TIFF folders; save PNG + optional LQ|OUT|GT panels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from swinir_gray_dn_infer import load_gray_dn_model  # noqa: E402

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402
from basicsr.metrics.snr import calculate_snr  # noqa: E402


def read_u16_gray(path: Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D TIFF, got {img.shape}: {path}")
    return img


def normalize_u16(arr: np.ndarray, raw_min: float, raw_max: float) -> np.ndarray:
    return np.clip((arr.astype(np.float32) - raw_min) / (raw_max - raw_min), 0.0, 1.0)


def calculate_reference_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    prediction_hwc = prediction[..., None] * 255.0
    ground_truth_hwc = ground_truth[..., None] * 255.0
    return {
        "snr": calculate_snr(prediction_hwc, ground_truth_hwc, crop_border=0),
        "psnr": calculate_psnr(prediction_hwc, ground_truth_hwc, crop_border=0),
        "ssim": calculate_ssim(prediction_hwc, ground_truth_hwc, crop_border=0),
    }


def to_tensor(img_hw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_hw[None, None, ...]).float()


def infer_tiled(
    model: torch.nn.Module,
    img_hw: np.ndarray,
    device: torch.device,
    *,
    tile: int = 0,
    overlap: int = 32,
    window_size: int = 8,
) -> np.ndarray:
    img = to_tensor(img_hw).to(device)
    with torch.no_grad():
        if tile <= 0:
            _, _, h, w = img.size()
            mod_h = (window_size - h % window_size) % window_size
            mod_w = (window_size - w % window_size) % window_size
            padded = F.pad(img, (0, mod_w, 0, mod_h), "reflect")
            out = model(padded)
            return out[:, :, :h, :w].squeeze().float().cpu().clamp_(0, 1).numpy()

        _, _, h, w = img.size()
        tile_sz = min(tile, h, w)
        tile_sz = tile_sz - (tile_sz % window_size) or window_size
        stride = max(tile_sz - overlap, window_size)
        output = torch.zeros((1, 1, h, w), device=device)
        weight = torch.zeros_like(output)
        h_idxs = list(range(0, max(h - tile_sz, 0) + 1, stride))
        w_idxs = list(range(0, max(w - tile_sz, 0) + 1, stride))
        if not h_idxs or h_idxs[-1] != h - tile_sz:
            h_idxs.append(max(h - tile_sz, 0))
        if not w_idxs or w_idxs[-1] != w - tile_sz:
            w_idxs.append(max(w - tile_sz, 0))
        for hi in h_idxs:
            for wi in w_idxs:
                patch = img[:, :, hi : hi + tile_sz, wi : wi + tile_sz]
                ph, pw = patch.shape[-2:]
                mod_h = (window_size - ph % window_size) % window_size
                mod_w = (window_size - pw % window_size) % window_size
                patch = F.pad(patch, (0, mod_w, 0, mod_h), "reflect")
                out = model(patch)[:, :, :ph, :pw]
                output[:, :, hi : hi + ph, wi : wi + pw].add_(out)
                weight[:, :, hi : hi + ph, wi : wi + pw].add_(1.0)
        return (output / weight).squeeze().float().cpu().clamp_(0, 1).numpy()


def to_u8(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if lo is None or hi is None:
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_panel(lq: np.ndarray, out: np.ndarray, gt: np.ndarray | None, path: Path, title: str) -> None:
    if gt is not None:
        lo = float(min(lq.min(), gt.min()))
        hi = float(max(lq.max(), gt.max()))
        cols = [to_u8(lq, lo, hi), to_u8(out), to_u8(gt, lo, hi)]
        labels = ["LQ", "FT", "GT"]
    else:
        lo, hi = float(lq.min()), float(lq.max())
        cols = [to_u8(lq, lo, hi), to_u8(out)]
        labels = ["LQ", "FT"]
    h, w = cols[0].shape
    canvas = Image.new("L", (w * len(cols), h + 28), 0)
    draw_bg = Image.new("L", (w * len(cols), 28), 40)
    canvas.paste(draw_bg, (0, 0))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for i, (col, lab) in enumerate(zip(cols, labels, strict=True)):
        canvas.paste(Image.fromarray(col, mode="L"), (i * w, 28))
        draw.text((i * w + 8, 6), f"{title} | {lab}", fill=220)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--raw-min", type=float, default=0.0)
    parser.add_argument("--raw-max", type=float, default=65535.0)
    parser.add_argument("--tile", type=int, default=0, help="0 = full-frame")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.normalization and args.normalization.is_file():
        meta = json.loads(args.normalization.read_text(encoding="utf-8"))
        args.raw_min = float(meta.get("raw_min", args.raw_min))
        args.raw_max = float(meta.get("raw_max", args.raw_max))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_gray_dn_model(args.model, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for item in args.inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.tif")) + sorted(item.glob("*.tiff")))
        elif item.is_file():
            paths.append(item)
        else:
            raise SystemExit(f"Missing input: {item}")

    metric_rows: list[dict[str, str | float]] = []
    for path in paths:
        raw = read_u16_gray(path)
        lq = normalize_u16(raw, args.raw_min, args.raw_max)
        print(f"infer {path.name} shape={raw.shape}", flush=True)
        out = infer_tiled(model, lq, device, tile=args.tile)
        stem = path.stem
        Image.fromarray(to_u8(lq, float(lq.min()), float(lq.max())), mode="L").save(args.output_dir / f"{stem}_lq.png")
        Image.fromarray(to_u8(out), mode="L").save(args.output_dir / f"{stem}_ft.png")
        gt_img = None
        if args.gt_dir is not None:
            gt_path = args.gt_dir / path.name
            if gt_path.is_file():
                gt_img = normalize_u16(read_u16_gray(gt_path), args.raw_min, args.raw_max)
                Image.fromarray(to_u8(gt_img, float(gt_img.min()), float(gt_img.max())), mode="L").save(
                    args.output_dir / f"{stem}_gt.png"
                )
                lq_metrics = calculate_reference_metrics(lq, gt_img)
                output_metrics = calculate_reference_metrics(out, gt_img)
                metric_rows.append(
                    {
                        "filename": path.name,
                        "lq_snr": lq_metrics["snr"],
                        "output_snr": output_metrics["snr"],
                        "snr_gain": output_metrics["snr"] - lq_metrics["snr"],
                        "lq_psnr": lq_metrics["psnr"],
                        "output_psnr": output_metrics["psnr"],
                        "lq_ssim": lq_metrics["ssim"],
                        "output_ssim": output_metrics["ssim"],
                    }
                )
        save_panel(lq, out, gt_img, args.output_dir / f"{stem}_panel.png", stem)
        print(f"  wrote {stem}_panel.png", flush=True)

    if metric_rows:
        metrics_path = args.output_dir / "metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Wrote reference metrics: {metrics_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run TSFDAN dual-branch inference on TIFF folders; save PNG + comparison panels.

Panels:
  - single model: LQ | OUT | GT
  - with --baseline-model (微调前): LQ | PreFT | FT | GT
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from tsfdan_infer_helpers import infer_tiled, load_tsfdan_model  # noqa: E402

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


def to_u8(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if lo is None or hi is None:
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_panel(
    columns: list[tuple[str, np.ndarray]],
    path: Path,
    title: str,
    *,
    display_lo: float | None = None,
    display_hi: float | None = None,
) -> None:
    if not columns:
        raise ValueError("columns must be non-empty")
    if display_lo is None or display_hi is None:
        stack = np.stack([img for _, img in columns], axis=0)
        display_lo = float(stack.min())
        display_hi = float(stack.max())
    cols_u8 = [to_u8(img, display_lo, display_hi) for _, img in columns]
    labels = [lab for lab, _ in columns]
    h, w = cols_u8[0].shape
    canvas = Image.new("L", (w * len(cols_u8), h + 28), 0)
    canvas.paste(Image.new("L", (w * len(cols_u8), 28), 40), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, (col, lab) in enumerate(zip(cols_u8, labels, strict=True)):
        canvas.paste(Image.fromarray(col, mode="L"), (i * w, 28))
        draw.text((i * w + 8, 6), f"{title} | {lab}", fill=220)
    canvas.save(path)


def resolve_input_paths(inputs: list[Path], *, lq_dir: Path | None, meta_info: Path | None) -> list[Path]:
    if meta_info is not None:
        if lq_dir is None:
            raise SystemExit("--meta-info requires --lq-dir")
        names = [line.strip() for line in meta_info.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [lq_dir / name for name in names]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"Missing LQ files from meta-info (showing up to 5): {missing[:5]}")
        return paths

    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.tif")) + sorted(item.glob("*.tiff")))
        elif item.is_file():
            paths.append(item)
        else:
            raise SystemExit(f"Missing input: {item}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Finetuned (or primary) checkpoint")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=None,
        help="Optional pre-finetune TSFDAN checkpoint; panel becomes LQ|PreFT|FT|GT",
    )
    parser.add_argument("--inputs", type=Path, nargs="*", default=[], help="LQ files and/or folders")
    parser.add_argument("--lq-dir", type=Path, default=None)
    parser.add_argument("--meta-info", type=Path, default=None)
    parser.add_argument("--pick-indices", type=int, nargs="+", default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--raw-min", type=float, default=0.0)
    parser.add_argument("--raw-max", type=float, default=65535.0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.normalization and args.normalization.is_file():
        meta = json.loads(args.normalization.read_text(encoding="utf-8"))
        args.raw_min = float(meta.get("raw_min", args.raw_min))
        args.raw_max = float(meta.get("raw_max", args.raw_max))

    if not args.inputs and args.meta_info is None:
        raise SystemExit("Provide --inputs and/or --meta-info")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_tsfdan_model(args.model, device)
    baseline = load_tsfdan_model(args.baseline_model, device) if args.baseline_model is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_input_paths(list(args.inputs), lq_dir=args.lq_dir, meta_info=args.meta_info)
    if args.pick_indices is not None:
        paths = [paths[i] for i in args.pick_indices]

    metric_rows: list[dict[str, str | float]] = []
    for path in paths:
        raw = read_u16_gray(path)
        lq = normalize_u16(raw, args.raw_min, args.raw_max)
        print(f"infer {path.name} shape={raw.shape}", flush=True)
        out = infer_tiled(model, lq, device, tile=args.tile, overlap=args.tile_overlap)
        pre_out = None
        if baseline is not None:
            pre_out = infer_tiled(baseline, lq, device, tile=args.tile, overlap=args.tile_overlap)
        stem = path.stem
        Image.fromarray(to_u8(lq, float(lq.min()), float(lq.max())), mode="L").save(args.output_dir / f"{stem}_lq.png")
        Image.fromarray(to_u8(out), mode="L").save(args.output_dir / f"{stem}_ft.png")
        if pre_out is not None:
            Image.fromarray(to_u8(pre_out), mode="L").save(args.output_dir / f"{stem}_preft.png")

        gt_img = None
        row: dict[str, str | float] = {"filename": path.name}
        if args.gt_dir is not None:
            gt_path = args.gt_dir / path.name
            if gt_path.is_file():
                gt_img = normalize_u16(read_u16_gray(gt_path), args.raw_min, args.raw_max)
                Image.fromarray(to_u8(gt_img, float(gt_img.min()), float(gt_img.max())), mode="L").save(
                    args.output_dir / f"{stem}_gt.png"
                )
                lq_metrics = calculate_reference_metrics(lq, gt_img)
                ft_metrics = calculate_reference_metrics(out, gt_img)
                row.update(
                    {
                        "lq_snr": lq_metrics["snr"],
                        "ft_snr": ft_metrics["snr"],
                        "ft_snr_gain": ft_metrics["snr"] - lq_metrics["snr"],
                        "lq_psnr": lq_metrics["psnr"],
                        "ft_psnr": ft_metrics["psnr"],
                        "lq_ssim": lq_metrics["ssim"],
                        "ft_ssim": ft_metrics["ssim"],
                    }
                )
                if pre_out is not None:
                    pre_metrics = calculate_reference_metrics(pre_out, gt_img)
                    row.update(
                        {
                            "preft_snr": pre_metrics["snr"],
                            "preft_psnr": pre_metrics["psnr"],
                            "preft_ssim": pre_metrics["ssim"],
                            "ft_minus_preft_snr": ft_metrics["snr"] - pre_metrics["snr"],
                        }
                    )
                metric_rows.append(row)

        columns: list[tuple[str, np.ndarray]] = [("LQ", lq)]
        if pre_out is not None:
            columns.append(("PreFT", pre_out))
        columns.append(("FT", out))
        if gt_img is not None:
            columns.append(("GT", gt_img))
        ref = [lq] + ([gt_img] if gt_img is not None else [])
        lo = float(min(x.min() for x in ref))
        hi = float(max(x.max() for x in ref))
        save_panel(columns, args.output_dir / f"{stem}_compare.png", stem, display_lo=lo, display_hi=hi)
        print(f"  wrote {stem}_compare.png", flush=True)
        if pre_out is not None and gt_img is not None:
            print(
                f"  SNR  LQ={row.get('lq_snr'):.3f}  PreFT={row.get('preft_snr'):.3f}  "
                f"FT={row.get('ft_snr'):.3f}  Δ(FT-PreFT)={row.get('ft_minus_preft_snr'):.3f}",
                flush=True,
            )

    if metric_rows:
        metrics_path = args.output_dir / "metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Wrote reference metrics: {metrics_path}")


if __name__ == "__main__":
    main()

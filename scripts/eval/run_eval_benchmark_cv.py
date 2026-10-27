#!/usr/bin/env python3
"""在固定 eval_benchmark 上跑传统 CV 去噪基线，写对比图 / metrics / timing。

复用已有 hardlink 评测树（默认 eval_benchmark_v2 的 pairs/ + manifests/），
不复制图像。输出到独立目录（默认 …/outputs/eval_benchmark_cv/）。

面向 3D 重建 X 射线的 XOY-2D 切片：重建噪声近似相关/近似高斯，
边缘与匀质区域并存——对比高斯/中值/双边/NLM/TV/小波等经典方法。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402
from basicsr.metrics.snr import calculate_snr  # noqa: E402

COLUMNS = [
    "battery_260520",
    "battery_260527",
    "cylinder_260520_val20",
    "cylinder_260527_holdout20",
    "phantom_aligned_drive_val20",
]

MethodFn = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2"),
        help="已有 hardlink 评测根目录（含 pairs/ manifests/）",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/outputs/eval_benchmark_cv"),
        help="CV 基线输出根目录",
    )
    p.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="只跑指定列；默认全部 5 列",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="只跑指定方法名；默认全部",
    )
    p.add_argument(
        "--max-per-column",
        type=int,
        default=0,
        help="每列最多 N 张（0=全部）；快速试跑用",
    )
    return p.parse_args()


def read_u16_gray(path: Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D TIFF, got {img.shape}: {path}")
    return img


def load_norm(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["raw_min"]), float(data["raw_max"])


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


def to_u8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_panel(
    columns: list[tuple[str, np.ndarray]],
    path: Path,
    title: str,
    *,
    display_lo: float,
    display_hi: float,
) -> None:
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


def method_gaussian(x: np.ndarray) -> np.ndarray:
    # 轻度平滑基线；易糊边
    return cv2.GaussianBlur(x, (5, 5), 1.0)


def method_median(x: np.ndarray) -> np.ndarray:
    u8 = (np.clip(x, 0, 1) * 255.0).round().astype(np.uint8)
    return cv2.medianBlur(u8, 5).astype(np.float32) / 255.0


def method_bilateral(x: np.ndarray) -> np.ndarray:
    # 保边滤波；对 CT 切片边缘友好
    u8 = (np.clip(x, 0, 1) * 255.0).round().astype(np.uint8)
    out = cv2.bilateralFilter(u8, d=9, sigmaColor=50, sigmaSpace=50)
    return out.astype(np.float32) / 255.0


def method_nlm(x: np.ndarray) -> np.ndarray:
    # OpenCV Fast NLM；重建噪声近似高斯时常用
    u8 = (np.clip(x, 0, 1) * 255.0).round().astype(np.uint8)
    out = cv2.fastNlMeansDenoising(u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return out.astype(np.float32) / 255.0


def method_tv(x: np.ndarray) -> np.ndarray:
    # Total Variation (Chambolle)：分段光滑先验，CT 常用
    return denoise_tv_chambolle(x, weight=0.08, channel_axis=None).astype(np.float32)


def method_wavelet(x: np.ndarray) -> np.ndarray:
    # BayesShrink 小波软阈值
    out = denoise_wavelet(
        x,
        channel_axis=None,
        convert2ycbcr=False,
        method="BayesShrink",
        mode="soft",
        rescale_sigma=True,
    )
    return np.clip(out.astype(np.float32), 0.0, 1.0)


METHODS: dict[str, MethodFn] = {
    "gaussian": method_gaussian,
    "median": method_median,
    "bilateral": method_bilateral,
    "nlm": method_nlm,
    "tv": method_tv,
    "wavelet": method_wavelet,
}


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_cell(rows: list[dict[str, str]], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(r[key]) for r in rows) / len(rows)


def write_summary(output_root: Path, methods: list[str], columns: list[str]) -> None:
    lines = [
        "# 传统 CV 基线（eval_benchmark_v2 hardlink 复用）",
        "",
        "图像假设：3D 重建 X 射线的 XOY-2D 切片；重建噪声近似相关/近似高斯，结构多为分段光滑 + 边缘。",
        "",
        "面板布局：`LQ | CV | GT`。单位 dB（SNR/PSNR）；SSIM∈[0,1]。",
        "",
        "## 方法说明",
        "",
        "| 方法 | 意图 |",
        "|------|------|",
        "| gaussian | 轻度各向同性平滑基线（易糊边） |",
        "| median | 抗脉冲；对重建条纹帮助有限 |",
        "| bilateral | 保边平滑 |",
        "| nlm | Fast Non-Local Means（经典 AWGN 去噪） |",
        "| tv | Total Variation / Chambolle（分段光滑） |",
        "| wavelet | BayesShrink 小波软阈值 |",
        "",
        "## 列 × 方法：平均 SNR (Δ vs LQ) / PSNR / SSIM / 单图耗时",
        "",
    ]
    header = "| 列 | " + " | ".join(methods) + " |"
    sep = "|----|" + "|".join(["----:" for _ in methods]) + "|"
    lines.extend([header, sep])

    for col in columns:
        cells = [col]
        for method in methods:
            metrics_path = output_root / "metrics" / f"{method}__{col}.csv"
            if not metrics_path.is_file():
                cells.append("TBD")
                continue
            rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
            snr = mean_cell(rows, "out_snr")
            dsnr = mean_cell(rows, "out_minus_lq_snr")
            psnr = mean_cell(rows, "out_psnr")
            ssim = mean_cell(rows, "out_ssim")
            sec = mean_cell(rows, "seconds")
            if snr is None or dsnr is None or psnr is None or ssim is None or sec is None:
                cells.append("TBD")
            else:
                cells.append(f"{snr:.2f} (Δ{dsnr:+.2f}) / {psnr:.1f} / {ssim:.3f} / {sec:.2f}s")
        lines.append("| " + " | ".join(cells) + " |")

    timing_path = output_root / "timing_summary.csv"
    if timing_path.is_file():
        lines.extend(["", "## 方法总耗时", "", "见 `timing_summary.csv`。", ""])

    lines.extend(
        [
            "",
            "### 粗判提示",
            "- ΔSNR≈0 或为负：过度平滑 / 参数不适合该域。",
            "- 耗时：gaussian/median/bilateral ≪ nlm/tv/wavelet。",
            "- 本脚本为粗基线，未做 per-column 参数搜索。",
            "",
        ]
    )
    (output_root / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bench = args.bench_root
    out_root = args.output_root
    columns = args.columns or COLUMNS
    methods = args.methods or list(METHODS.keys())
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; choose from {list(METHODS)}")

    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    (out_root / "panels").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    timing_rows: list[dict[str, object]] = []

    for method in methods:
        fn = METHODS[method]
        for col in columns:
            names = read_manifest(bench / "manifests" / f"{col}.txt")
            if args.max_per_column > 0:
                names = names[: args.max_per_column]
            pair = bench / "pairs" / col
            raw_min, raw_max = load_norm(pair / "normalization.json")
            panel_dir = out_root / "panels" / method / col
            panel_dir.mkdir(parents=True, exist_ok=True)
            metric_rows: list[dict[str, object]] = []
            print(f"=== {method} / {col} (n={len(names)}) ===", flush=True)
            col_t0 = time.perf_counter()
            for name in names:
                stem = Path(name).stem
                lq_u16 = read_u16_gray(pair / "LQ" / name)
                gt_u16 = read_u16_gray(pair / "GT" / name)
                lq = normalize_u16(lq_u16, raw_min, raw_max)
                gt = normalize_u16(gt_u16, raw_min, raw_max)
                t0 = time.perf_counter()
                out = np.clip(fn(lq), 0.0, 1.0)
                dt = time.perf_counter() - t0
                lq_m = calculate_reference_metrics(lq, gt)
                out_m = calculate_reference_metrics(out, gt)
                lo = float(min(lq.min(), gt.min()))
                hi = float(max(lq.max(), gt.max()))
                save_panel(
                    [("LQ", lq), (method, out), ("GT", gt)],
                    panel_dir / f"{stem}_compare.png",
                    stem,
                    display_lo=lo,
                    display_hi=hi,
                )
                row = {
                    "name": name,
                    "seconds": f"{dt:.6f}",
                    "lq_snr": f"{lq_m['snr']:.6f}",
                    "out_snr": f"{out_m['snr']:.6f}",
                    "out_minus_lq_snr": f"{out_m['snr'] - lq_m['snr']:.6f}",
                    "out_psnr": f"{out_m['psnr']:.6f}",
                    "out_ssim": f"{out_m['ssim']:.6f}",
                }
                metric_rows.append(row)
                print(
                    f"  {name}  {dt:.3f}s  SNR {lq_m['snr']:.2f}->{out_m['snr']:.2f} "
                    f"(Δ{out_m['snr'] - lq_m['snr']:+.2f})",
                    flush=True,
                )
            col_dt = time.perf_counter() - col_t0
            metrics_csv = out_root / "metrics" / f"{method}__{col}.csv"
            with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "name",
                        "seconds",
                        "lq_snr",
                        "out_snr",
                        "out_minus_lq_snr",
                        "out_psnr",
                        "out_ssim",
                    ],
                )
                writer.writeheader()
                writer.writerows(metric_rows)
            mean_sec = sum(float(r["seconds"]) for r in metric_rows) / len(metric_rows) if metric_rows else 0.0
            mean_dsnr = (
                sum(float(r["out_minus_lq_snr"]) for r in metric_rows) / len(metric_rows) if metric_rows else 0.0
            )
            timing_rows.append(
                {
                    "method": method,
                    "column": col,
                    "n": len(metric_rows),
                    "total_seconds": f"{col_dt:.3f}",
                    "mean_seconds_per_image": f"{mean_sec:.4f}",
                    "mean_delta_snr": f"{mean_dsnr:.4f}",
                }
            )
            print(f"  DONE col total={col_dt:.1f}s mean/img={mean_sec:.3f}s", flush=True)

    timing_csv = out_root / "timing_summary.csv"
    with timing_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "column",
                "n",
                "total_seconds",
                "mean_seconds_per_image",
                "mean_delta_snr",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    write_summary(out_root, methods, columns)
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# eval_benchmark_cv",
                "",
                "传统 CV 去噪基线，复用 `eval_benchmark_v2` 的 hardlink pairs（不另拷图像）。",
                "",
                "生成命令：",
                "",
                "```bash",
                "uv run python scripts/eval/run_eval_benchmark_cv.py",
                "```",
                "",
                "输出：`panels/<method>/<column>/*_compare.png`、`metrics/`、"
                "`timing_summary.csv`、`comparison_table.md`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary → {out_root / 'comparison_table.md'}", flush=True)


if __name__ == "__main__":
    main()
